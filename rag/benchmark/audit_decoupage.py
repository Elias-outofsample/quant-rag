"""Audit de la découpe — reconnaissance en lecture seule, à signature ``10390927db``.

Ce n'est **pas** le chantier découpage : c'est sa reconnaissance. Rien n'est créé, rien
n'est ré-embarqué, rien n'est écrit en production, aucun appel LLM. Le script lit
``.cache/chunk-index-v5-10390927db.pkl`` — le corpus **tel qu'il est servi**, overlay
appliqué et documents retirés exclus, vérifié sur 2 000 overrides — et classe les chunks.

Chaque classe porte un **propriétaire du remède**, et c'est là tout l'objet :

    chunker            un plafond ou une règle de frontière répare la classe
    filtre ingestion   la classe ne doit pas être indexée du tout ; la couper l'aggrave
    re-parse           le flux de texte est déjà faux en amont du chunker

Les détecteurs sont des heuristiques. Chacun est **échantillonné à la main** (``--classe``)
et sa précision est rapportée dans le rapport d'audit : un compte sans précision mesurée
serait un chiffre décoratif.

    .venv/bin/python rag/benchmark/audit_decoupage.py --prevalence
    .venv/bin/python rag/benchmark/audit_decoupage.py --classe bibliographie --n 8
    .venv/bin/python rag/benchmark/audit_decoupage.py --anatomie
    .venv/bin/python rag/benchmark/audit_decoupage.py --plafond 1500
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import pickle
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
RACINE = HERE.parent.parent
SIGNATURE = "10390927db"
INDEX = HERE / ".cache" / f"chunk-index-v5-{SIGNATURE}.pkl"
QUESTIONS = HERE / "questions-v3.jsonl"
RESULTATS = HERE / f"results-answer-gap-{SIGNATURE}.json"
RETRIEVALS = HERE / ".cache" / f"router-retrievals-{SIGNATURE}.json"

#: Seuil du filtre de service (``quant_rag.MIN_CHARACTERS``). Un chunk plus court n'est
#: jamais servi — il pèse quand même dans l'index et dans le classement du pool.
MIN_SERVI = 250

PROPRIETAIRE = {
    "bibliographie": "filtre ingestion",
    "boilerplate": "filtre ingestion",
    "legende_table": "filtre ingestion",
    "sommaire": "filtre ingestion",
    "dechiquete": "re-parse",
    "demesure": "chunker",
    "coupure_debut": "chunker",
    "coupure_fin": "chunker",
    "entrelacement": "re-parse",
}


def charger() -> dict:
    return pickle.load(INDEX.open("rb"))["chunks"]


# ------------------------------------------------------------------- détecteurs

_REF = re.compile(r"^\s*(?:\[\d+\]|\(\d+\)|\d+\.)?\s*[A-Z][A-Za-z'’À-ÿ-]+,\s*"
                  r"(?:[A-Z]\.\s*)+", re.M)
_ET_AL = re.compile(r"\bet\s+al\.?", re.I)
_ANNEE = re.compile(r"\b(?:19|20)\d{2}\b")
_COURRIEL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
_INSTITUT = re.compile(r"\b(University|Universit[eé]|Department|School of|Institute|College|"
                       r"NBER|CNRS|Laborator(?:y|ies)|Faculty|Politecnico|Ecole|École)\b")
_ADRESSE = re.compile(r"\b(?:\d{4,5}\s+[A-Z][a-z]+|[A-Z][a-z]+,\s*[A-Z]{2}\s+\d{5}|"
                      r"\d+\s+[A-Z][\w.]*\s+(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Drive))\b")
_PUCE = re.compile(r"^\s*(?:[•▪◦*–—-]|\(\w\)|\d+[.)])\s+\S")
_TITRE_ETIQ = re.compile(r"^\s*(?:[A-Z]|[IVX]+|\d+(?:\.\d+)*)[.)]\s+[A-Z]")
_SOMMAIRE = re.compile(r"\.{3,}\s*\d+\s*$|\s…\s*\d+\s*$", re.M)
_FIN_OK = tuple(".?!:;\"'»)]}’”")
_BALISE_IND = re.compile(r"</?su[bp]>")
#: Densité de balises indice/exposant au-delà de laquelle le texte n'est plus du texte.
#: La distribution est **bimodale avec un trou** : p90 = 0,81 balise pour 1 000 caractères,
#: p98 = 79,8, et presque rien entre les deux. Le seuil est donc posé dans le vide — il vaut
#: 20, il vaudrait 10 ou 50 pour le même compte. Ce n'est pas un réglage, c'est une frontière.
DENSITE_DECHIQUETE = 20.0


def _lignes(texte: str) -> list[str]:
    return [l for l in texte.splitlines() if l.strip()]


def est_bibliographie(c: dict) -> bool:
    """Un bloc de références : beaucoup de lignes « Nom, I. » ou « et al. » + année."""
    t = c["text"]
    lignes = _lignes(t)
    if len(lignes) < 4:
        return False
    # L'échantillon a réfuté la première règle (« 50 % de lignes qui ressemblent à des
    # références ») : elle classait des sections *related work*, qui sont de la prose dense
    # en citations. Une vraie bibliographie est faite de lignes **courtes**, et presque
    # toutes en sont.
    courtes = [l for l in lignes if len(l) < 300]
    refs = sum(1 for l in courtes if _REF.match(l))
    if refs >= 6 and refs / len(lignes) >= 0.70:
        return True
    # Le cas « REFERENCES / REFERENCES / REFERENCES » : le mot répété *est* le bloc.
    return len(re.findall(r"\bREFERENCES\b|\bBIBLIOGRAPHY\b", t)) >= 3


def est_boilerplate(c: dict) -> bool:
    """Appareil non scientifique : affiliations, avertissements juridiques, licences.

    La classe s'appelait ``affiliation``. L'échantillon a montré qu'elle n'attrape pas ce
    que ce nom dit : sur quatre, deux sont un avertissement juridique J.P. Morgan et une
    note de licence de données Bloomberg, deux sont de la prose ordinaire. Elle est
    renommée pour ce qu'elle mesure, et sa précision mesurée — **2 sur 4** — est publiée
    avec son compte. À douze chunks, sa prévalence est négligeable dans les deux lectures.
    """
    t = c["text"]
    phrases = len(re.findall(r"[.!?](?:\s|$)", t))
    densite = phrases / max(len(t) / 500, 1)
    # L'échantillon a réfuté le comptage pondéré : « FTSE Russell is a global provider of
    # benchmarks » et un résumé d'article le franchissaient sur les seuls mots d'institution.
    # Seuls un courriel ou une adresse postale sont des marqueurs non ambigus de front-matter.
    durs = len(_COURRIEL.findall(t)) + len(_ADRESSE.findall(t))
    return durs >= 2 and densite < 4 and len(_INSTITUT.findall(t)) >= 1


def _corps_de_table(t: str) -> str:
    """Le tableau moins sa légende : ce qui reste quand on retire « Table: … »."""
    sans = re.sub(r"^\s*Table:.*$", "", t, count=1, flags=re.M)
    return sans.strip()


def _cellules_pleines(corps: str) -> int:
    """Cellules de données non vides, lignes de séparation exclues."""
    n = 0
    for ligne in corps.splitlines():
        if "|" not in ligne or set(ligne.strip()) <= set("|- :"):
            continue
        n += sum(1 for cel in ligne.split("|") if cel.strip())
    return n


def est_legende_table(c: dict) -> bool:
    """Une légende de tableau dont le corps ne porte presque rien.

    Le premier détecteur mesurait la longueur du corps ; il laissait passer le cas qui a
    motivé l'audit — ``chunk-1e5183074f182b21``, 262 caractères, classé **premier** à
    cosine 0,804 sur sa requête, dont la grille de six lignes ne contient que des cellules
    vides. Ce qui compte n'est pas la taille du corps mais **ce qu'il porte**.
    """
    if c["content_type"] != "table":
        return False
    corps = _corps_de_table(c["text"])
    # La règle « corps < 150 caractères » a été retirée : l'échantillon a montré qu'elle
    # classait de vrais tableaux, simplement petits (« Calls | Include borrow »). Un petit
    # tableau reste un tableau. Ne reste que le cas sans appel :
    # une table réduite à son en-tête : après la ligne de séparation, aucune ligne
    # ne porte de donnée. Le compte de cellules pleines (< 8) avait été essayé d'abord
    # et l'échantillon l'a montré trop large — il classait des tableaux réels dont les
    # cellules portent de la prose.
    lignes = [l for l in corps.splitlines() if "|" in l]
    sep = next((i for i, l in enumerate(lignes) if set(l.strip()) <= set("|- :")), None)
    if sep is None:
        return False
    donnees = [l for l in lignes[sep + 1:] if any(cel.strip() for cel in l.split("|"))]
    return not donnees


def est_figure_en_table(c: dict) -> bool:
    """RETIRÉ du jeu de détecteurs — conservé pour montrer pourquoi.

    Hypothèse : « Table: Figure 3: … » signalerait une figure OCRisée en tableau. Le
    détecteur rendait **699 chunks, 3,2 %**. L'échantillon de cinq l'a réfuté en cinq cas
    sur cinq : dans ce corpus, les manuels intitulent leurs *vrais* tableaux « Figure 15 »,
    « EXHIBIT 5 », « Figure 68 » — la convention de légende n'est pas un défaut de parse.

    Deux discriminants ont été essayés sur le cas qui a motivé l'audit
    (``chunk-1e5183074f182b21``) contre cinq vrais tableaux : la part de cellules vides
    (31 % contre 17 % pour un vrai tableau) et la densité de chiffres (2,6 % contre 1,9 %
    pour un tableau qualitatif). **Aucun ne sépare.** La classe existe — ce chunk est bien
    une figure dégradée, classée première à cosine 0,804 — mais elle n'est pas mesurable
    par heuristique ici, et un compte non mesurable n'entre pas dans un tableau de
    prévalence.
    """
    if c["content_type"] != "table":
        return False
    return bool(re.match(r"\s*Table:\s*(?:Figure|Fig\.?|Exhibit|Chart|Panel)\b",
                         c["text"], re.I))


def est_dechiquete(c: dict) -> bool:
    """Le texte est haché en balises ``<sub>``/``<sup>`` au milieu des mots.

    Trouvé **par l'échantillonnage**, pas prévu : ``Th<sub>e</sub> d<sub>es</sub>i<sub>gn</sub>``
    pour « The design ». L'OCR a enveloppé des fragments de mots dans des balises d'indice,
    et le flux de tokens est détruit — donc l'embedding aussi. Aucun découpage ne répare
    cela : le chunk le plus court d'un texte déchiqueté reste déchiqueté.
    """
    t = c["text"]
    if 1000 * len(_BALISE_IND.findall(t)) / max(len(t), 1) < DENSITE_DECHIQUETE:
        return False
    # L'échantillon a montré le faux positif : une liste d'auteurs
    # ``Meng Li<sup>a,b,c</sup>, Xiaohua Yang<sup>a,b,c</sup>`` est dense en balises sans
    # être déchiquetée. Ce qui distingue le déchiquetage, c'est que le texte *entre* les
    # balises est lui aussi en miettes — des segments de un à trois caractères, là où une
    # liste d'auteurs laisse des noms entiers.
    segments = [seg for seg in _BALISE_IND.split(t) if seg.strip()]
    if len(segments) < 6:
        return False
    return statistics.median(len(seg.strip()) for seg in segments) <= 4


def est_sommaire(c: dict) -> bool:
    return len(_SOMMAIRE.findall(c["text"])) >= 5


def est_demesure(c: dict, seuil: int = 4000) -> bool:
    return len(c["text"]) > seuil


def est_coupure_debut(c: dict) -> bool:
    """Le chunk commence au milieu d'une phrase.

    On ne regarde que le premier caractère *alphabétique* et seulement si le chunk
    commence par de la prose : une formule, une puce, un titre ou une ligne de tableau
    ne sont pas des coupures de phrase.
    """
    if c["content_type"] == "heading" or est_dechiquete(c):
        return False
    t = c["text"].lstrip()
    if not t or t[0] in "$|•▪◦*#" or _PUCE.match(t) or _TITRE_ETIQ.match(t):
        return False
    # Une URL, un courriel ou un chemin en tête ne sont pas une phrase coupée.
    if re.match(r"(?:https?://|www\.|[\w.+-]+@)", t):
        return False
    m = re.search(r"[A-Za-z]", t[:80])
    return bool(m and t[m.start()].islower())


def est_coupure_fin(c: dict) -> bool:
    """Le chunk finit au milieu d'une phrase.

    Un ``heading`` est exclu : un titre n'a pas à finir par un point, et le premier
    échantillon en comptait trois sur cinq. Un chunk qui finit sur une formule, une ligne
    de tableau ou un nombre ne l'est pas non plus.
    """
    if c["content_type"] == "heading" or est_dechiquete(c):
        return False
    t = c["text"].rstrip()
    if not t or t.endswith("$$") or t.endswith("|") or t[-1].isdigit():
        return False
    return t[-1] not in _FIN_OK


def est_entrelacement(c: dict) -> tuple[bool, bool]:
    """Deux colonnes d'un PDF tressées : puces et prose qui alternent.

    Retourne (soupçon, corroboré). Le soupçon est l'alternance ; la corroboration est
    un désaccord entre le champ ``section`` et le titre que le texte porte lui-même.
    """
    lignes = _lignes(c["text"])
    if len(lignes) < 6:
        return False, False
    genres = ["puce" if _PUCE.match(l) else ("prose" if len(l) >= 120 else "autre")
              for l in lignes]
    utiles = [g for g in genres if g != "autre"]
    transitions = sum(1 for a, b in zip(utiles, utiles[1:]) if a != b)
    soupcon = (genres.count("puce") >= 3 and genres.count("prose") >= 3 and transitions >= 3)
    corrobore = False
    if soupcon:
        premier = lignes[0].strip()
        if _TITRE_ETIQ.match(premier) and len(premier) < 80:
            queue = (c.get("section") or "").split(">")[-1].strip()
            corrobore = bool(queue) and premier.lower() not in queue.lower()
    return soupcon, corrobore


DETECTEURS = {
    "bibliographie": est_bibliographie,
    "boilerplate": est_boilerplate,
    "legende_table": est_legende_table,
    "sommaire": est_sommaire,
    "dechiquete": est_dechiquete,
    "demesure": est_demesure,
    "coupure_debut": est_coupure_debut,
    "coupure_fin": est_coupure_fin,
    "entrelacement": lambda c: est_entrelacement(c)[0],
}


def classer(chunks: dict) -> dict[str, list[str]]:
    par_classe: dict[str, list[str]] = {k: [] for k in DETECTEURS}
    par_classe["entrelacement_corrobore"] = []
    for cid, c in chunks.items():
        for nom, f in DETECTEURS.items():
            if f(c):
                par_classe[nom].append(cid)
        if est_entrelacement(c)[1]:
            par_classe["entrelacement_corrobore"].append(cid)
    return par_classe


# ------------------------------------------------------------------------ étapes

def _distribution(chunks: dict, ids: list[str]) -> str:
    if not ids:
        return "—"
    lg = sorted(len(chunks[c]["text"]) for c in ids)
    return (f"médiane {statistics.median(lg):>5.0f} · p90 {lg[9*len(lg)//10]:>5} · "
            f"max {lg[-1]:>5}")


def step_prevalence(chunks: dict) -> None:
    par_classe = classer(chunks)
    n = len(chunks)
    print(f"\n  corpus {SIGNATURE} — {n} chunks servis-ou-indexés\n")
    print(f"  {'classe':<24} {'propriétaire':<18} {'n':>6} {'part':>7}   longueurs")
    print(f"  {'-'*24} {'-'*18} {'-'*6} {'-'*7}   {'-'*38}")
    for nom in ("bibliographie", "boilerplate", "legende_table", "sommaire",
                "dechiquete", "demesure", "coupure_debut", "coupure_fin", "entrelacement"):
        ids = par_classe[nom]
        print(f"  {nom:<24} {PROPRIETAIRE[nom]:<18} {len(ids):>6} {100*len(ids)/n:>6.1f} %   "
              f"{_distribution(chunks, ids)}")
    corr = par_classe["entrelacement_corrobore"]
    print(f"  {'  dont corroboré':<24} {'':<18} {len(corr):>6} {100*len(corr)/n:>6.1f} %")

    parasites = set().union(*(set(par_classe[k]) for k in
                              ("bibliographie", "boilerplate", "legende_table", "sommaire")))
    servables = {c for c in parasites
                 if chunks[c]["content_type"] != "heading"
                 and len(chunks[c]["text"].strip()) >= MIN_SERVI}
    print(f"\n  parasites (union des quatre classes) : {len(parasites)} "
          f"({100*len(parasites)/n:.1f} %)")
    print(f"    dont franchissant le filtre de service (>= {MIN_SERVI} car., non-titre) : "
          f"{len(servables)} ({100*len(servables)/n:.1f} %)")
    coupures = set(par_classe["coupure_debut"]) | set(par_classe["coupure_fin"])
    print(f"  coupures de phrase (début ou fin) : {len(coupures)} ({100*len(coupures)/n:.1f} %)")


def step_classe(chunks: dict, nom: str, k: int) -> None:
    ids = classer(chunks)[nom]
    print(f"\n  {nom} — {len(ids)} chunks. Échantillon de {min(k, len(ids))}, "
          f"pris à pas régulier pour ne pas ne montrer que le début :\n")
    pas = max(1, len(ids) // max(k, 1))
    for cid in ids[::pas][:k]:
        c = chunks[cid]
        extrait = " ".join(c["text"].split())[:230]
        print(f"  · {cid} [{c['content_type']}] {len(c['text'])} car. p.{c['page_start']}")
        print(f"    section : {(c.get('section') or '—')[:70]}")
        print(f"    {extrait}…\n")


def step_anatomie(chunks: dict) -> None:
    res = json.loads(RESULTATS.read_text(encoding="utf-8"))
    qs = {"v3/" + json.loads(l)["qid"]: json.loads(l) for l in QUESTIONS.open(encoding="utf-8")}
    par_classe = classer(chunks)
    n = len(chunks)
    rates = [r["key"] for r in res["par_question"]]
    doc_hit = [r["key"] for r in res["par_question"] if r["doc_hit"]]
    docs_rates = {d for k in rates for d in (qs[k].get("gold_documents") or [])}
    docs_hit = {d for k in doc_hit for d in (qs[k].get("gold_documents") or [])}

    def part(ids: list[str], docs: set[str]) -> tuple[int, int, float]:
        dedans = [c for c in ids if chunks[c]["document_id"] in docs]
        total = sum(1 for c in chunks.values() if c["document_id"] in docs)
        return len(dedans), total, 100 * len(dedans) / max(total, 1)

    print(f"\n  documents d'or des 26 ratés : {len(docs_rates)}")
    print(f"  documents d'or des 17 doc-présent : {len(docs_hit)}\n")
    print(f"  {'classe':<24} {'corpus':>8}   {'26 ratés':>16}   {'17 doc-présent':>16}")
    print(f"  {'-'*24} {'-'*8}   {'-'*16}   {'-'*16}")
    for nom in DETECTEURS:
        ids = par_classe[nom]
        a, ta, pa = part(ids, docs_rates)
        b, tb, pb = part(ids, docs_hit)
        print(f"  {nom:<24} {100*len(ids)/n:>6.1f} %   {a:>5}/{ta:<5} {pa:>5.1f} %   "
              f"{b:>5}/{tb:<5} {pb:>5.1f} %")

    # Le chunk d'or lui-même porte-t-il un défaut ?
    print("\n  Et les chunks d'or eux-mêmes :")
    ors = {g for k in qs for g in qs[k].get("gold_chunks", []) if g in chunks}
    ors_rates = {g for k in rates for g in qs[k].get("gold_chunks", []) if g in chunks}
    for nom in DETECTEURS:
        s = set(par_classe[nom])
        print(f"    {nom:<22} tous les ors {len(ors & s):>3}/{len(ors)}   "
              f"ors des ratés {len(ors_rates & s):>2}/{len(ors_rates)}")


def step_plafond(chunks: dict, plafond: int) -> None:
    import math
    par_classe = classer(chunks)
    parasites = set().union(*(set(par_classe[k]) for k in
                              ("bibliographie", "boilerplate", "legende_table", "sommaire")))
    touches = [c for c, v in chunks.items() if len(v["text"]) > plafond]
    n = len(chunks)
    apres = sum(max(1, math.ceil(len(v["text"]) / plafond)) for v in chunks.values())
    print(f"\n  plafond de {plafond} caractères\n")
    print(f"  chunks touchés (plus longs que le plafond) : {len(touches)} "
          f"({100*len(touches)/n:.1f} %)")
    print(f"  chunks après découpe : {apres} (×{apres/n:.2f})")
    print(f"\n  ce que le plafond fait des classes parasites — il ne les retire pas, il les coupe :")
    print(f"  {'classe':<24} {'n':>6} {'touchés':>8} {'part':>7} {'morceaux produits':>18}")
    for nom in ("bibliographie", "boilerplate", "legende_table", "sommaire"):
        ids = par_classe[nom]
        t = [c for c in ids if len(chunks[c]["text"]) > plafond]
        morceaux = sum(max(1, math.ceil(len(chunks[c]["text"]) / plafond)) for c in ids)
        print(f"  {nom:<24} {len(ids):>6} {len(t):>8} {100*len(t)/max(len(ids),1):>6.1f} % "
              f"{morceaux:>18}")
    total_par = sum(max(1, math.ceil(len(chunks[c]["text"]) / plafond)) for c in parasites)
    print(f"\n  parasites : {len(parasites)} chunks aujourd'hui → {total_par} morceaux après "
          f"découpe (+{total_par - len(parasites)})")
    print(f"  soit {100*total_par/apres:.1f} % de l'index découpé, contre "
          f"{100*len(parasites)/n:.1f} % de l'index actuel.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prevalence", action="store_true")
    p.add_argument("--classe")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--anatomie", action="store_true")
    p.add_argument("--plafond", type=int)
    a = p.parse_args()
    chunks = charger()
    if a.prevalence:
        step_prevalence(chunks)
    if a.classe:
        step_classe(chunks, a.classe, a.n)
    if a.anatomie:
        step_anatomie(chunks)
    if a.plafond:
        step_plafond(chunks, a.plafond)


if __name__ == "__main__":
    main()
