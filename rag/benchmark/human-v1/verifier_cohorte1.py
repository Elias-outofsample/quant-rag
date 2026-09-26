"""Auto-audit de la cohorte 1 — avant de demander du temps humain, et non après.

Six contrôles, et deux d'entre eux existent pour répondre aux deux objections les plus
fortes qu'on puisse faire à ce lot :

    offsets     chaque offset désigne encore l'extrait, et chaque empreinte tient
    pages       la page déclarée est celle du bloc où l'offset tombe — vérifiée, pas recopiée
    gold        aucun fait ne peut entrer dans un score : tous sont candidats
    diversite   les dix situations exigées sont couvertes, et par des documents différents
    fuite       **objection 1 — auto-référentialité** : les questions recopient-elles leurs sources ?
    nouveaute   **objection 2 — réutilisation du banc** : une question du banc v3 est-elle reprise ?

La fuite lexicale est mesurée sur le même instrument **et sur la même base** — le texte du
chunk servi — pour la cohorte 1 et pour le banc v3. La première version comparait les
extraits ancrés de la cohorte aux chunks entiers de v3 : deux vocabulaires sources de taille
double, donc un contrôle qui **validait par construction**. Corrigé le 7 septembre 2026 après
la revue contradictoire (rapport E) ; le contrôle échoue désormais, et c'est le bon résultat :
0,500 contre 0,333, p = 0,002.

    .venv/bin/python rag/benchmark/human-v1/verifier_cohorte1.py
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
ROOT = BENCHMARK.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(BENCHMARK.parent))

import offsets as mod_offsets  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

LOT = HERE / "items-cohorte1-v0.jsonl"
INGESTED = ROOT / "data" / "processed" / "ingested"

#: Les dix situations que la cohorte doit couvrir, telles que le cahier des charges les nomme.
SITUATIONS = {
    "restriction_de_modele", "limite_methodologique", "calibration_estimation_fenetre",
    "biais_de_backtest_leakage", "resultat_chiffre_tableau_legende",
    "deux_passages_meme_document", "plusieurs_documents", "divergence_entre_sources",
    "reponse_partielle_attendue", "abstention_attendue",
}

#: Mots vides anglais, volontairement courts : une liste longue gonflerait artificiellement
#: la mesure de fuite en retirant les mots que question et source partagent le plus.
VIDES = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "do", "for", "from", "how",
    "i", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "which", "who", "why", "with", "over", "into", "not", "but", "if",
    "has", "have", "had", "can", "could", "should", "would", "my", "me", "he", "she",
    "they", "them", "their", "there", "here", "than", "then", "so", "such", "also",
}
_MOT = re.compile(r"[a-z0-9]+")


def mots(texte: str) -> set[str]:
    return {m for m in _MOT.findall((texte or "").lower()) if m not in VIDES and len(m) > 2}


def fuite(question: str, source: str) -> float:
    """Part des mots de contenu de la question qui apparaissent déjà dans la source.

    C'est la grandeur que le banc v3 paie par construction : une question écrite *à partir*
    d'un passage en reprend le vocabulaire, et le retrieval dense retrouve alors le passage
    par les mots plutôt que par le sens. La mesure ne prouve rien à elle seule — elle situe.
    """
    q = mots(question)
    return round(len(q & mots(source)) / len(q), 4) if q else 0.0


def charger() -> list[dict]:
    return [json.loads(l) for l in LOT.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- contrôles


def controle_offsets(items: list[dict]) -> list[str]:
    return mod_offsets.verifier(items)


def controle_pages(items: list[dict]) -> list[str]:
    """La page déclarée est-elle celle du bloc où l'offset tombe ?

    Le champ ``page`` vient du chunk ; l'offset vise le texte canonique, dont les blocs
    portent leur propre ``page_idx``. Les deux chemins sont indépendants — c'est ce qui rend
    le contrôle utile. Une citation qui donne la bonne phrase à la mauvaise page est une
    citation fausse pour un lecteur qui ouvre le PDF.
    """
    registre = json.loads((ROOT / "rag" / "ingestion" / "registry-v1.json").read_text(encoding="utf-8"))
    dossiers = {e["document_id"]: e["folder"] for e in registre.get("documents", [])}
    erreurs, cache = [], {}
    for item in items:
        for appui in item.get("appuis") or []:
            offsets = appui.get("offsets")
            if not offsets:
                continue
            doc = appui["document_id"]
            if doc not in cache:
                chemin = INGESTED / dossiers.get(doc, "") / "blocks.jsonl"
                if not chemin.exists():
                    erreurs.append(f"{item['id']}/{appui['ancre_id']} : blocs absents")
                    cache[doc] = None
                    continue
                blocs = [json.loads(x) for x in chemin.open(encoding="utf-8") if x.strip()]
                position, plages = 0, []
                for b in blocs:
                    t = b.get("text") or ""
                    plages.append((position, position + len(t), b.get("page_idx")))
                    position += len(t) + 2                       # SEPARATEUR = "\n\n"
                cache[doc] = plages
            plages = cache[doc]
            if plages is None:
                continue
            pages = {p for debut, fin, p in plages
                     if debut < offsets["fin"] and fin > offsets["debut"] and p is not None}
            attendues = {p + 1 for p in pages}
            if appui.get("page") not in attendues:
                erreurs.append(f"{item['id']}/{appui['ancre_id']} : page déclarée "
                               f"{appui.get('page')}, blocs couverts en page(s) {sorted(attendues)}")
    return erreurs


def controle_gold(items: list[dict]) -> list[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("hv1", HERE / "valider.py")
    valider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(valider)
    erreurs = []
    for item in items:
        for k in (1, 2, 3):
            retenus = valider.gold_a_la_version(item, k)
            if retenus:
                erreurs.append(f"{item['id']} : {len(retenus)} fait(s) compteraient en v{k}")
        if item.get("statut") not in {"idee", "candidat", "pilote"}:
            erreurs.append(f"{item['id']} : statut {item['statut']!r} — un lot non relu ne peut "
                           "pas prétendre davantage")
        if item.get("relecture_humaine") is not None:
            erreurs.append(f"{item['id']} : porte une relecture humaine qui n'a pas eu lieu")
    return erreurs


def controle_diversite(items: list[dict]) -> tuple[list[str], dict]:
    couvertes = {s for i in items for s in i.get("situations") or []}
    documents = {a["document_id"] for i in items for a in i.get("appuis") or []}
    resume = {
        "items": len(items),
        "familles": sorted({i["famille"] for i in items}),
        "domaines": sorted({i["domaine"] for i in items}),
        "documents_ancres": len(documents),
        "situations_couvertes": sorted(couvertes),
        "repondable": {v: sum(1 for i in items if i["repondable"] == v)
                       for v in ("oui", "partiel", "non")},
        "difficultes": {v: sum(1 for i in items if i.get("difficulte") == v)
                        for v in ("facile", "moyenne", "difficile")},
    }
    erreurs = []
    manquantes = SITUATIONS - couvertes
    if manquantes:
        erreurs.append(f"situations exigées non couvertes : {sorted(manquantes)}")
    if resume["repondable"]["non"] < 1:
        erreurs.append("aucun cas d'abstention")
    if not any(len({a['document_id'] for a in i['appuis']}) > 1 for i in items):
        erreurs.append("aucun item ne demande plusieurs documents")
    # Ce qui doit être vérifié n'est pas que **tout** item soit multi-passage — un item peut
    # légitimement tenir dans un passage — mais que **la promesse soit tenue** : un item qui
    # annonce deux passages ou deux documents doit réellement les demander. Une première
    # version de ce contrôle imposait le multi-passage partout ; elle signalait un item sain
    # et n'aurait pas vu un item qui promet et ne tient pas.
    chunks = [a.get("chunk_id_origine") for i in items for a in i.get("appuis") or []]
    resume["chunks_distincts"] = len(set(chunks))
    return erreurs, resume


def exigence_reelle(item: dict) -> dict:
    """Ce que l'item demande **vraiment** : les passages de ses faits obligatoires.

    Un fait ``souhaitable`` peut venir d'où il veut : il ne conditionne aucun score. La
    promesse d'un item se mesure donc sur ses seuls faits ``obligatoire``.
    """
    ancres = {a["ancre_id"]: a for a in item.get("appuis") or []}
    obligatoires = [f for f in item.get("faits_attendus") or []
                    if f.get("exigence") == "obligatoire"]
    references = sorted({aid for f in obligatoires for aid in f.get("ancres") or []})
    utilisees = [ancres[a] for a in references if a in ancres]
    return {
        "faits_obligatoires": len(obligatoires),
        "ancres": references,
        "chunks": sorted({a.get("chunk_id_origine") for a in utilisees}),
        "documents": sorted({a["document_id"] for a in utilisees}),
        "toutes_ancres_chunks": len({a.get("chunk_id_origine") for a in ancres.values()}),
    }


def controle_multi_passage(items: list[dict]) -> tuple[list[str], dict]:
    """Objection 3 — **la promesse est-elle tenue ?**

    Ce contrôle a existé dans une forme qui ne pouvait pas échouer : il comptait *toutes*
    les ancres de l'item, y compris celles de faits ``souhaitable``. `C01` annonce deux
    passages du même document, ses trois faits obligatoires tiennent dans **un seul chunk
    servi** sur 436 caractères contigus, et son unique second passage n'appuie qu'un fait
    facultatif : l'ancien contrôle le déclarait conforme. Un item mono-passage déguisé en
    multi-passage ment sur ce qu'un score mesure — il donnera le même chiffre à un système
    incapable d'assembler deux passages.

    La règle : un item qui **annonce** plusieurs passages ou plusieurs documents doit les
    exiger pour ses faits **obligatoires**. Rien n'oblige un item à être multi-passage ;
    tout oblige un item à ne pas le promettre en vain.
    """
    erreurs, detail, multi = [], {}, 0
    for item in items:
        e = exigence_reelle(item)
        situations = set(item.get("situations") or [])
        detail[item["id"]] = e
        if not e["faits_obligatoires"]:
            continue  # item d'abstention : aucune promesse de passage à tenir
        multi += len(e["chunks"]) > 1
        if "deux_passages_meme_document" in situations and len(e["chunks"]) < 2:
            erreurs.append(
                f"{item['id']} annonce deux passages du même document, mais ses "
                f"{e['faits_obligatoires']} faits obligatoires tiennent dans un seul "
                f"passage servi ({e['chunks'][0] if e['chunks'] else '—'}) : la promesse "
                f"n'est portée que par des faits souhaitables")
        if "plusieurs_documents" in situations and len(e["documents"]) < 2:
            erreurs.append(f"{item['id']} annonce plusieurs documents mais ses faits "
                           f"obligatoires n'en ancrent que {len(e['documents'])}")
        if (item.get("citation_minimale") or {}).get("documents_distincts", 0) > len(e["documents"]):
            erreurs.append(f"{item['id']} exige {item['citation_minimale']['documents_distincts']} "
                           f"documents distincts en citation mais ses faits obligatoires "
                           f"n'en ancrent que {len(e['documents'])}")
        if len(e["ancres"]) < 2 and "deux_passages_meme_document" not in situations \
                and "plusieurs_documents" not in situations and e["faits_obligatoires"] > 1:
            detail[item["id"]]["mono_ancre"] = True
    resume = {"items_multi_passage": multi, "par_item": detail}
    # Un lot dont toutes les questions tiendraient dans un passage ne testerait qu'un cas.
    if multi < 3:
        erreurs.append(f"seuls {multi} items demandent plus d'un passage servi pour leurs faits "
                       "obligatoires : le lot ne couvre pas le cas dominant du diagnostic "
                       "(49 questions perdent un or déjà récupéré, souvent réparti)")
    return erreurs, resume


def controle_fuite(items: list[dict]) -> tuple[list[str], dict]:
    """Objection 1 : les questions recopient-elles le vocabulaire de leurs sources ?

    Mesurée sur la cohorte **et** sur le banc v3, avec la même fonction **et la même base** —
    le texte du chunk servi. v3 écrit ses questions *à partir* d'un chunk : c'est le majorant
    naturel, et la cohorte doit s'en écarter vers le bas pour que « question écrite à neuf »
    veuille dire quelque chose. Elle ne s'en écarte pas : voir le commentaire ci-dessous.
    """
    index = ChunkIndex.load(verbose=False)
    valeurs = []
    detail = {}
    for item in items:
        # **Base commune, et c'est le correctif.** La version précédente comparait la cohorte
        # à ses *extraits ancrés* (66 mots médians) et v3 au *texte entier* de ses chunks d'or
        # (130 mots) : un vocabulaire source deux fois plus grand fait mécaniquement monter la
        # fuite, donc le contrôle validait par construction. Il annonçait 0,300 contre 0,333 ;
        # sur la base commune la cohorte sort à 0,500, et le contrôle échoue.
        #
        # La bonne base est le **chunk servi** : c'est lui que le dense plonge et que la
        # production montre, jamais l'extrait que le rédacteur a découpé. Trouvé par la revue
        # contradictoire du 7 septembre 2026 (rapport E), permutation p = 0,002.
        chunks = {a.get("chunk_id_origine") for a in item.get("appuis") or []}
        source = " ".join((index.chunks.get(c) or {}).get("text") or "" for c in chunks if c)
        if not source:
            continue
        v = fuite(item["question"], source)
        detail[item["id"]] = v
        valeurs.append(v)

    v3 = [json.loads(l) for l in (BENCHMARK / "questions-v3.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    reference = []
    for q in v3:
        if q.get("kind") == "negative":
            continue
        textes = " ".join((index.chunks.get(c) or {}).get("text") or "" for c in q.get("gold_chunks") or [])
        if textes:
            reference.append(fuite(q["question"], textes))

    resume = {
        "cohorte1": {"n": len(valeurs), "mediane": round(statistics.median(valeurs), 4),
                     "min": min(valeurs), "max": max(valeurs), "par_item": detail},
        "banc_v3": {"n": len(reference), "mediane": round(statistics.median(reference), 4),
                    "min": round(min(reference), 4), "max": round(max(reference), 4)},
    }
    erreurs = []
    if resume["cohorte1"]["mediane"] >= resume["banc_v3"]["mediane"]:
        erreurs.append("la fuite lexicale médiane de la cohorte atteint ou dépasse celle de v3 : "
                       "les questions recopient leurs sources autant que le banc qu'on veut quitter")
    return erreurs, resume


def controle_nouveaute(items: list[dict]) -> tuple[list[str], dict]:
    """Objection 2 : une question du banc v3 est-elle reprise, même reformulée ?"""
    v3 = [json.loads(l) for l in (BENCHMARK / "questions-v3.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    pires = {}
    erreurs = []
    for item in items:
        q = mots(item["question"])
        meilleur, cle = 0.0, None
        for autre in v3:
            a = mots(autre["question"])
            if not a or not q:
                continue
            j = len(q & a) / len(q | a)
            if j > meilleur:
                meilleur, cle = j, autre["qid"]
        pires[item["id"]] = {"jaccard_max": round(meilleur, 4), "contre": cle}
        if meilleur > 0.5:
            erreurs.append(f"{item['id']} : trop proche de v3/{cle} (Jaccard {meilleur:.2f})")
    return erreurs, pires


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    items = charger()
    rapport, echecs = {}, 0

    for nom, fonction in (("offsets", controle_offsets), ("pages", controle_pages),
                          ("gold", controle_gold)):
        erreurs = fonction(items)
        rapport[nom] = {"erreurs": erreurs}
        echecs += len(erreurs)
        print(f"{nom:<11} {'OK' if not erreurs else str(len(erreurs)) + ' ÉCHEC(S)'}")
        for e in erreurs[:6]:
            print(f"    {e}")

    erreurs, resume = controle_diversite(items)
    rapport["diversite"] = {"erreurs": erreurs, **resume}
    echecs += len(erreurs)
    print(f"diversite   {'OK' if not erreurs else str(len(erreurs)) + ' ÉCHEC(S)'}")
    print(f"    {resume['items']} items · {len(resume['familles'])} familles · "
          f"{len(resume['domaines'])} domaines · {resume['documents_ancres']} documents · "
          f"{resume['chunks_distincts']} chunks distincts")
    print(f"    répondable {resume['repondable']} · difficulté {resume['difficultes']}")
    print(f"    situations couvertes : {len(resume['situations_couvertes'])}/10")
    for e in erreurs:
        print(f"    {e}")

    erreurs, resume = controle_multi_passage(items)
    rapport["multi_passage"] = {"erreurs": erreurs, **resume}
    echecs += len(erreurs)
    print(f"multi-pass. {'OK' if not erreurs else str(len(erreurs)) + ' ÉCHEC(S)'}")
    print(f"    {resume['items_multi_passage']} items exigent plus d'un passage servi "
          f"pour leurs faits obligatoires")
    for e in erreurs:
        print(f"    {e}")

    erreurs, resume = controle_fuite(items)
    rapport["fuite"] = {"erreurs": erreurs, **resume}
    echecs += len(erreurs)
    print(f"fuite       {'OK' if not erreurs else 'ÉCHEC'}")
    print(f"    cohorte 1 : médiane {resume['cohorte1']['mediane']:.3f} "
          f"[{resume['cohorte1']['min']:.3f} ; {resume['cohorte1']['max']:.3f}] sur {resume['cohorte1']['n']}")
    print(f"    banc v3   : médiane {resume['banc_v3']['mediane']:.3f} "
          f"[{resume['banc_v3']['min']:.3f} ; {resume['banc_v3']['max']:.3f}] sur {resume['banc_v3']['n']}")
    for e in erreurs:
        print(f"    {e}")

    erreurs, pires = controle_nouveaute(items)
    rapport["nouveaute"] = {"erreurs": erreurs, "par_item": pires}
    echecs += len(erreurs)
    pire = max(pires.values(), key=lambda x: x["jaccard_max"])
    print(f"nouveaute   {'OK' if not erreurs else 'ÉCHEC'}")
    print(f"    similarité maximale à une question v3 : {pire['jaccard_max']:.3f} "
          f"(contre v3/{pire['contre']}) — seuil 0,50")
    for e in erreurs:
        print(f"    {e}")

    print(f"\n{echecs} échec(s) au total.")
    if args.json:
        args.json.write_text(json.dumps(rapport, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  écrit  {args.json}")
    sys.exit(1 if echecs else 0)


if __name__ == "__main__":
    main()
