"""Combien de tableaux servis ont perdu leur en-tête à l'import — et ce que cela change au texte.

Le défaut
----------
Un tableau trop grand pour un chunk est découpé en fragments. Seul le premier porte la
rangée d'en-tête ; les suivants ne sont qu'une suite de lignes de données. La passe de
conversion du corpus le sait et **recopie l'en-tête de la première partie** sur chaque
fragment (``convert_tables.convert_all``, via la chaîne ``previous_chunk_id``) : sans cela,
un fragment servi est un tableau de chiffres dont les colonnes n'ont plus de nom.

À l'**import**, cette recopie n'a jamais eu lieu. ``convert_tables.part_chains()`` lisait
``data/processed/ingested``, or l'étape 4 d'``apply_delivery`` (les tableaux) s'exécute
**avant** l'étape 8 (la promotion) : les chunks de la livraison en cours n'y sont pas encore.
La chaîne était donc vide, ``first_part`` rendait le fragment lui-même, et l'en-tête valait
toujours ``None``. Le témoin est dans les rapports de lot : ``fragments_with_inherited_header``
vaut **0** à chaque import.

Second défaut, dans le même geste : ``apply_delivery.convert_tables_for`` construisait ses
payloads sans ``parent_id``. Même avec la chaîne, la garde de ``first_part`` — « le fragment
précédent appartient-il au même tableau ? » — aurait comparé ``None`` à ``None`` et accepté
n'importe quel voisin.

Ce que ce module fait, et ce qu'il ne fait pas
------------------------------------------------
Il **mesure**, il ne répare rien. Les documents importés sont aujourd'hui dans
``data/processed/ingested`` : leur chaîne de fragments est donc lisible, et l'on peut
calculer ce que l'étape 4 **aurait** rendu si elle avait vu la livraison. Le résultat est
écrit **à côté** de l'overlay servi, jamais à sa place — corriger ``tables-markdown-v1.json``
changerait la signature du corpus, donc le nom de l'index, donc le gel.

Le calcul n'est pas une réimplémentation : il appelle ``convert_tables.convert_all`` avec
la chaîne en argument. Une copie de la règle divergerait de la production, et c'est
précisément le genre d'écart que ce recensement existe pour mesurer.

    .venv/bin/python rag/tables/recenser_en_tetes.py
    .venv/bin/python rag/tables/recenser_en_tetes.py --json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "rag" / "tables"))
sys.path.insert(0, str(ROOT / "src"))

import convert_tables as ct  # noqa: E402
import corpus_overlay  # noqa: E402

PROCESSED = ROOT / "data" / "processed" / "ingested"
REGISTRE = ROOT / "rag" / "ingestion" / "registry-v1.json"
CANDIDAT = HERE / "candidat-en-tetes-2026-09-09.json"
RAPPORT = HERE / "recensement-en-tetes-2026-09-09.json"


class Point:
    """La forme que ``convert_all`` attend — un objet à ``.payload``."""

    def __init__(self, payload: dict):
        self.payload = payload


def documents_importes() -> dict[str, dict]:
    """Les documents entrés par une livraison, par opposition à ceux de l'export figé.

    Le critère est celui du registre lui-même : ``delivery.id`` différent de l'identifiant
    de la livraison de base. On ne devine pas la provenance, on la lit.
    """
    sys.path.insert(0, str(ROOT / "rag" / "ingestion"))
    import registry as reg

    base = reg.BASELINE["id"]
    donnees = json.loads(REGISTRE.read_text(encoding="utf-8"))
    return {entree["document_id"]: entree for entree in donnees["documents"]
            if entree.get("status") == "active"
            and (entree.get("delivery") or {}).get("id") not in (None, base)}


def chunks_du_document(entree: dict) -> list[dict]:
    fichier = PROCESSED / entree["folder"] / "chunks.jsonl"
    if not fichier.exists():
        return []
    return [json.loads(l) for l in fichier.read_text(encoding="utf-8").splitlines() if l.strip()]


def payload_de(chunk: dict) -> dict:
    """Les champs dont ``convert_all`` se sert — **``parent_id`` compris**.

    C'est l'oubli du second défaut : sans ``parent_id``, la garde de ``first_part`` ne
    distingue plus deux tableaux voisins.
    """
    return {cle: chunk.get(cle) for cle in
            ("chunk_id", "document_id", "text", "title_path", "content_type",
             "part", "chapter", "section", "parent_id", "previous_chunk_id")}


def cellules(ligne: str) -> list[str]:
    return [c.strip() for c in ligne.strip().strip("|").split("|")]


def surtout_numerique(cells: list[str]) -> bool:
    """Une rangée de chiffres, donc des données et non un en-tête."""
    utiles = [c for c in cells if c]
    if not utiles:
        return False
    chiffres = sum(1 for c in utiles
                   if re.fullmatch(r"[-+(]?[\d.,%\s×x*/^()\-–−]+\)?", c))
    return chiffres >= 0.6 * len(utiles)


def classer_ecart(texte_servi: str, texte_candidat: str) -> str:
    """L'héritage aide-t-il, ou non ? Trois cas, et il faut les compter séparément.

    Publier « 1 066 tableaux réparés » serait faux : l'en-tête hérité n'est pas toujours une
    rangée de noms de colonnes bien découpée. On distingue donc le cas où le fragment servi
    commence par une rangée de **données** — l'héritage est alors un gain net — de celui où
    il porte déjà quelque chose qui ressemble à un en-tête, et de celui où l'en-tête hérité
    tient en **une seule cellule**.

    ⚠ La troisième classe s'appelait ``entete_herite_quasi_vide`` et son commentaire disait
    « c'est du bruit ». **C'était faux**, et le 9 septembre 2026 la mesure l'a montré : ces
    en-têtes d'une cellule sont des **cellules fusionnées** — les noms de colonnes agglomérés
    par l'OCR de MinerU, ou un titre de tableau sur toute la largeur. Longueur moyenne
    **9,1 mots** ; seuls **4,7 %** répètent la légende. Les hériter est un **gain** : sans
    eux, le fragment est une grille de valeurs anonymes. Le nom de la classe est donc
    descriptif, plus interprétatif — voir ``analyse_entetes_une_cellule``.
    """
    lignes_s = [l for l in texte_servi.splitlines() if l.startswith("|")]
    lignes_c = [l for l in texte_candidat.splitlines() if l.startswith("|")]
    if not lignes_s or not lignes_c:
        return "autre"
    heritee = cellules(lignes_c[0])
    if sum(1 for c in heritee if c) <= 1:
        return "entete_herite_en_une_cellule"
    return ("donnees_sans_entete" if surtout_numerique(cellules(lignes_s[0]))
            else "portait_deja_un_entete")


def analyse_entetes_une_cellule(entrees: list[dict]) -> dict:
    """Que contiennent vraiment les en-têtes hérités d'**une seule cellule** ?

    Le contrôle qui a renversé une conclusion publiée. Ils avaient été décrits comme « du
    bruit », sur la seule foi de leur forme. En les lisant, ce sont des cellules fusionnées :

        ['Incident (year) Main chain /Nature of Est. value Approx.token(s) attack stolen /
          created immediate(USD, at time) price response', '', '', '', '', '']

    — les six noms de colonnes d'un tableau, que l'OCR n'a pas su séparer. Les retirer
    priverait le fragment de toute indication sur ce que ses valeurs désignent.

    On mesure donc trois choses : leur longueur, la part qui **répète la légende** déjà
    présente au-dessus du tableau — la seule duplication réelle —, et la part très courte.
    """
    from parsing.table_markdown import header_of

    servi = json.loads(corpus_overlay.TABLES.read_text(encoding="utf-8"))["chunks"]
    normaliser = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()
    total = duplique = courts = mots = 0
    exemples = []
    for entree in entrees:
        chunks = chunks_du_document(entree)
        par_id = {c["chunk_id"]: c for c in chunks}
        for chunk in chunks:
            if chunk.get("content_type") != "table" or not est_un_fragment(chunk, par_id):
                continue
            entete = header_of(premiere_partie(chunk, par_id).get("text") or "")
            pleines = [c for c in (entete or []) if str(c).strip()]
            if len(pleines) > 1:
                continue
            total += 1
            texte = pleines[0] if pleines else ""
            mots += len(texte.split())
            courts += len(texte.split()) <= 3
            rendu = servi.get(chunk["chunk_id"]) or ""
            legende = rendu.split("\n\n")[0] if rendu.startswith("Table:") else ""
            if texte and normaliser(texte) in normaliser(legende):
                duplique += 1
            elif len(exemples) < 5:
                exemples.append(texte[:160])
    return {"fragments": total,
            "repetent_la_legende": duplique,
            "part_repetent_la_legende": round(100 * duplique / total, 1) if total else None,
            "trois_mots_ou_moins": courts,
            "mots_en_moyenne": round(mots / total, 1) if total else None,
            "exemples": exemples,
            "verdict": "cellules fusionnées : noms de colonnes agglomérés ou titre de "
                       "tableau. Les hériter est un GAIN ; les retirer priverait le "
                       "fragment de toute indication sur ses valeurs."}


def premiere_partie(chunk: dict, par_id: dict[str, dict]) -> dict:
    """Remonter la chaîne jusqu'au premier fragment — la règle de ``convert_all.first_part``."""
    tete, vus = chunk, {chunk["chunk_id"]}
    while True:
        precedent = par_id.get(tete.get("previous_chunk_id") or "")
        if (not precedent or precedent["chunk_id"] in vus
                or precedent.get("content_type") != "table"
                or precedent.get("parent_id") != tete.get("parent_id")):
            return tete
        vus.add(precedent["chunk_id"])
        tete = precedent


def taux_entete_vide(entrees: list[dict]) -> dict:
    """Le TÉMOIN : sur ces documents, quelle part des fragments hérite d'un en-tête vide ?

    Il faut ce chiffre sur les 256 documents d'**origine**, où l'héritage a déjà eu lieu :
    sans lui, on ne saurait pas si l'imperfection de la règle est introduite par la
    réparation ou si elle préexiste dans le corpus servi.
    """
    from parsing.table_markdown import header_of

    total = vide = 0
    for entree in entrees:
        chunks = chunks_du_document(entree)
        par_id = {c["chunk_id"]: c for c in chunks}
        for chunk in chunks:
            if chunk.get("content_type") != "table" or not est_un_fragment(chunk, par_id):
                continue
            entete = header_of(premiere_partie(chunk, par_id).get("text") or "")
            total += 1
            if not entete or sum(1 for c in entete if str(c).strip()) <= 1:
                vide += 1
    return {"fragments": total, "entete_quasi_vide": vide,
            "part": round(100 * vide / total, 1) if total else None}


def extraits(ecarts: list[dict], servi: dict, candidat: dict, combien: int) -> list[dict]:
    """Quelques cas lisibles, un par classe et par ordre de taille — pour qu'un relecteur
    puisse juger sur pièce sans rejouer l'instrument."""
    par_classe: dict[str, list[dict]] = {}
    for ecart in sorted(ecarts, key=lambda e: e["caracteres_servis"]):
        par_classe.setdefault(ecart["classe"], []).append(ecart)
    quota = max(1, combien // max(len(par_classe), 1))
    sortis = []
    for classe in sorted(par_classe):
        for ecart in par_classe[classe][:quota]:
            cid = ecart["chunk_id"]
            sortis.append({"chunk_id": cid, "classe": classe,
                           "caracteres_ajoutes": ecart["caracteres_ajoutes"],
                           "servi": servi[cid][:600], "candidat": candidat[cid][:600]})
    return sortis


def est_un_fragment(chunk: dict, par_id: dict[str, dict]) -> bool:
    """Un fragment est un tableau dont le précédent est un tableau du **même** parent."""
    precedent = par_id.get(chunk.get("previous_chunk_id") or "")
    return bool(precedent
                and precedent.get("content_type") == "table"
                and precedent.get("parent_id") == chunk.get("parent_id"))


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parseur.add_argument("--json", action="store_true")
    parseur.add_argument("--candidat", type=Path, default=CANDIDAT)
    parseur.add_argument("--rapport", type=Path, default=RAPPORT)
    parseur.add_argument("--analyse-entetes", action="store_true",
                         help="n'analyser que les en-têtes hérités d'une seule cellule, sur "
                              "TOUT le corpus actif — le contrôle qui a renversé la "
                              "conclusion publiée le 8 septembre")
    parseur.add_argument("--extraits", type=int, default=20,
                         help="nombre de cas rendus avec leur texte avant/après")
    args = parseur.parse_args()

    if args.analyse_entetes:
        tous = json.loads(REGISTRE.read_text(encoding="utf-8"))["documents"]
        actifs = [e for e in tous if e.get("status") == "active"]
        analyse = analyse_entetes_une_cellule(actifs)
        (HERE / "analyse-entetes-une-cellule-2026-09-09.json").write_text(
            json.dumps(analyse, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(analyse, ensure_ascii=False, indent=1))
        return

    servi = json.loads(corpus_overlay.TABLES.read_text(encoding="utf-8"))["chunks"]
    importes = documents_importes()
    print(f"{len(importes)} documents importés (registre, delivery.id hors livraison de base)")

    candidat: dict[str, str] = {}
    par_document: dict[str, dict] = {}
    ecarts: list[dict] = []
    tableaux_total = fragments_total = 0

    for document_id, entree in sorted(importes.items()):
        chunks = chunks_du_document(entree)
        if not chunks:
            continue
        par_id = {c["chunk_id"]: c for c in chunks}
        tableaux = [c for c in chunks if c.get("content_type") == "table"]
        if not tableaux:
            continue
        fragments = [c for c in tableaux if est_un_fragment(c, par_id)]
        tableaux_total += len(tableaux)
        fragments_total += len(fragments)

        # LA production, avec la chaîne que l'étape 4 n'avait pas.
        points = [Point(payload_de(c)) for c in tableaux]
        converti, _ = ct.convert_all(points, previous=ct.part_chains(chunks))

        changes = []
        for chunk in tableaux:
            cid = chunk["chunk_id"]
            neuf = converti.get(cid)
            ancien = servi.get(cid)
            if neuf is None or ancien is None or neuf == ancien:
                continue
            candidat[cid] = neuf
            changes.append(cid)
            ecarts.append({
                "chunk_id": cid, "document_id": document_id,
                "folder": entree["folder"],
                "fragment": est_un_fragment(chunk, par_id),
                "caracteres_servis": len(ancien),
                "caracteres_candidats": len(neuf),
                "caracteres_ajoutes": len(neuf) - len(ancien),
                "classe": classer_ecart(ancien, neuf),
                # Un chunk-tableau NON éligible est dans l'overlay mais n'est pas un point
                # de la collection : le corriger ne change rien à ce qui est servi. La
                # distinction a été trouvée en appliquant la correction — 1 066 chunks
                # d'overlay pour 989 points servis — et publiée le 9 septembre 2026 à côté
                # du chiffre d'origine.
                "servi": chunk.get("rag_eligible") is True,
            })
        if changes:
            par_document[document_id] = {"folder": entree["folder"],
                                         "titre": entree.get("title"),
                                         "tableaux": len(tableaux),
                                         "fragments": len(fragments),
                                         "chunks_changes": len(changes)}

    ajouts = [e["caracteres_ajoutes"] for e in ecarts]
    classes: dict[str, int] = {}
    for ecart in ecarts:
        classes[ecart["classe"]] = classes.get(ecart["classe"], 0) + 1
    sys.path.insert(0, str(ROOT / "rag" / "ingestion"))
    import registry as reg

    base = reg.BASELINE["id"]
    tous = json.loads(REGISTRE.read_text(encoding="utf-8"))["documents"]
    origine = [e for e in tous if e.get("status") == "active"
               and (e.get("delivery") or {}).get("id") in (None, base)]
    servis_total = (json.loads(REGISTRE.read_text(encoding="utf-8"))
                    .get("totals", {}).get("active_chunks"))
    rapport = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signature_corpus": corpus_overlay.signature(),
        "documents_importes": len(importes),
        "documents_avec_tableaux": sum(1 for d in importes
                                       if any(c.get("content_type") == "table"
                                              for c in chunks_du_document(importes[d]))),
        "chunks_tableaux_importes": tableaux_total,
        "fragments_importes": fragments_total,
        "chunks_dont_le_markdown_changerait": len(ecarts),
        "dont_servis": sum(1 for e in ecarts if e["servi"]),
        "dont_non_eligibles": sum(1 for e in ecarts if not e["servi"]),
        "documents_concernes": len(par_document),
        "caracteres_ajoutes": {
            "total": sum(ajouts),
            "mediane": round(statistics.median(ajouts), 1) if ajouts else 0,
            "max": max(ajouts) if ajouts else 0,
            "min": min(ajouts) if ajouts else 0,
        },
        "classement_des_ecarts": classes,
        # LE TÉMOIN. Sans lui, on ne saurait pas si l'héritage imparfait est un défaut
        # introduit par la réparation ou une propriété du corpus servi depuis toujours.
        "temoin_documents_origine": taux_entete_vide(origine),
        "temoin_documents_importes": taux_entete_vide(list(importes.values())),
        # LA part qui compte : celle des chunks réellement servis. Compter les chunks
        # d'overlay surestime — 1 066 contre 989, soit 4,081 % annoncés pour 3,786 % réels.
        "part_des_chunks_servis": (round(100 * sum(1 for e in ecarts if e["servi"]) / servis_total, 3)
                                   if servis_total else None),
        "part_des_chunks_servis_en_comptant_l_overlay": (round(100 * len(ecarts) / servis_total, 3)
                                                         if servis_total else None),
        "chunks_servis": servis_total,
        "par_document": par_document,
        # La liste quantitative est complète — c'est la preuve. Les EXTRAITS de texte, eux,
        # sont limités à vingt cas : les mille autres feraient un demi-mégaoctet dans un
        # dépôt dont le .git pèse déjà 3 Go, pour une information qui se relit avec
        # `--extraits` ou en rejouant l'instrument.
        "ecarts": ecarts,
        "extraits": extraits(ecarts, servi, candidat, args.extraits),
    }
    args.candidat.write_text(json.dumps(
        {"version": "candidat-en-tetes-2026-09-09",
         "note": "Ce que l'étape 4 d'apply_delivery AURAIT rendu si elle avait vu la "
                 "livraison en vol. Écrit À CÔTÉ de tables-markdown-v1.json, jamais à sa "
                 "place : corriger l'overlay servi changerait la signature du corpus.",
         "signature_corpus_au_moment_du_calcul": corpus_overlay.signature(),
         "chunks": candidat}, ensure_ascii=False, indent=1), encoding="utf-8")
    args.rapport.write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")

    if args.json:
        print(json.dumps({k: v for k, v in rapport.items() if k not in ("ecarts", "par_document")},
                         ensure_ascii=False, indent=1))
        return
    print(f"  {tableaux_total} chunks-tableaux dans ces documents, "
          f"dont {fragments_total} fragments")
    servis_changes = sum(1 for e in ecarts if e["servi"])
    print(f"  {len(ecarts)} chunks dont le Markdown changerait, "
          f"dans {len(par_document)} document(s)")
    print(f"      dont SERVIS (points de la collection) : {servis_changes}")
    print(f"      dont non éligibles (overlay seulement) : {len(ecarts) - servis_changes}")
    for nom, nombre in sorted(classes.items(), key=lambda x: -x[1]):
        print(f"      {nom:26} {nombre:5}  ({100 * nombre / max(len(ecarts), 1):.1f} %)")
    for etiquette, cle in (("origine", "temoin_documents_origine"),
                           ("importés", "temoin_documents_importes")):
        temoin = rapport[cle]
        print(f"  témoin {etiquette:9} : {temoin['fragments']} fragments, "
              f"en-tête d'une cellule {temoin['entete_quasi_vide']} ({temoin['part']} %)")
    if ajouts:
        print(f"  caractères ajoutés : médiane {statistics.median(ajouts):.0f}, "
              f"max {max(ajouts)}, total {sum(ajouts)}")
    print(f"  part des {servis_total} chunks servis : "
          f"{100 * servis_changes / servis_total:.3f} %" if servis_total else "")
    print(f"  candidat : {args.candidat.relative_to(ROOT)}")
    print(f"  rapport  : {args.rapport.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
