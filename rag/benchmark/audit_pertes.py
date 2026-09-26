"""Où le chemin servi perd ses questions — taxonomie ordonnée, population fixe, zéro appel.

**Ce que ce module est.** Le rejeu, sur le corpus courant, de l'anatomie des ratés que le
chantier « rappel du pool » avait établie le 5 septembre 2026 à la signature
``10390927db`` (319 documents, 22 190 chunks). Le corpus est depuis à ``5530cba145``
(418 documents, 26 120 chunks) : +99 documents, +3 930 passages, purement additifs — les
identifiants de chunks existants ne bougent pas, donc l'or des bancs reste valide.

**Ce qu'il n'est pas.** Un pré-enregistrement. La taxonomie ci-dessous est *reprise* d'un
rapport publié, pas écrite à l'aveugle ; ce qui est neuf, c'est l'état du corpus sur lequel
elle est mesurée. Un chiffre qui s'écarte de son précédent est une information sur ce
qu'ont fait 99 documents de plus, pas sur la validité de la taxonomie.

**La taxonomie est un arbre ordonné, pas une liste.** Le premier test qui se déclenche
donne la classe, ce qui garantit qu'elles ne se recouvrent pas et que les comptes se
somment à 155. L'ordre suit le chemin de la donnée, de la fin vers le début :

    SERVI                l'or est dans les cinq passages montrés au générateur
    EVICTION_DOCUMENT    l'or est dans les cinq premiers du pool, mais per_document=2 l'écarte
    RANG_6_10            l'or est au rang 6-10 : nDCG@10 le voit, la production non
    RANG_11_50           l'or est dans le pool, hors des dix premiers
    GRANULARITE          l'or n'est pas dans le pool, mais **son document y est**
    DECOUVERTE           ni l'or ni son document ne sont dans le pool
    OR_HORS_CORPUS       le chunk d'or n'existe plus — compté, jamais retiré (population fixe)

Les quatre premières classes sont des pannes de **classement ou d'assemblage** ; les deux
suivantes, des pannes de **génération de candidats** ; la dernière est une dette de banc.

**Population fixe, non négociable.** 155 questions (25 v1 + 130 v3 positives). Une question
devenue immesurable reste au dénominateur et vaut zéro : retirer les cinq plus dures rend
un Δ apparent de +0,0200, le double du seuil tenu pour décisif (``docs/BASE-SAINE``).

**Aucun appel LLM, aucune écriture.** Tout se lit dans
``.cache/router-retrievals-<signature>.json`` — les classements que Qdrant a réellement
rendus, écrits par ``calibrate_router.py`` —, et le pool servi est reconstruit par
``pipeline.build_context``, la fonction de production, jamais une copie.

    .venv/bin/python rag/benchmark/audit_pertes.py
    .venv/bin/python rag/benchmark/audit_pertes.py --signature 10390927db   # l'état d'avant
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import metrics  # noqa: E402
import pipeline  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from eval_pool_recall import build_frame  # noqa: E402

#: Profondeurs rapportées. 50 est le pool de production (``pipeline.POOL``) ; au-delà, il
#: faudrait une passe dense neuve, et c'est un élargissement de pool — fermé par la règle
#: §14 du TODO, et mesuré NO-GO par la Phase A.
PROFONDEURS = (1, 3, 5, 10, 20, 30, 50)

CLASSES = ("SERVI", "EVICTION_DOCUMENT", "RANG_6_10", "RANG_11_50",
           "GRANULARITE", "DECOUVERTE", "OR_HORS_CORPUS")

#: La production d'abord — c'est elle la référence de tous les écarts —, puis les variantes
#: du plafond par document à taille de contexte constante, puis l'agrandissement du
#: contexte. Grille déclarée ici, pas ajustée après avoir vu un chiffre.
GRILLE_ASSEMBLAGE = ((5, 2), (5, 1), (5, 3), (5, 0), (8, 2), (10, 2), (10, 3), (10, 0))


def rang_du_premier_or(pool: list[str], ors: set[str]) -> int | None:
    """Rang 1-indexé du premier chunk d'or dans une liste, ``None`` s'il n'y est pas."""
    for rang, chunk in enumerate(pool, start=1):
        if chunk in ors:
            return rang
    return None


def passages_servis(row: dict) -> list[str]:
    """Les cinq passages réellement montrés — par ``pipeline.build_context``, pas une copie.

    Le plafond ``per_document=2`` est la raison d'être de cette classe : un pool dont les
    cinq premiers rangs viennent tous du même document ne montre que deux d'entre eux, et
    trois places vont à des documents moins bien classés. Une question peut donc avoir son
    or au rang 3 et ne pas le voir servi.
    """
    classe = [{"chunk_id": c, "document_id": d}
              for c, d in zip(row["pool_chunks"], row["pool_documents"])]
    return [p["chunk_id"] for p in pipeline.build_context(classe)]


def classer(row: dict) -> tuple[str, dict]:
    """L'arbre ordonné. Le premier test qui se déclenche donne la classe."""
    ors = row["gold_chunks"]
    detail = {"rang": None, "n_or": len(ors), "n_pool": row["n_pool"]}
    if not ors:
        return "OR_HORS_CORPUS", detail
    servis = passages_servis(row)
    detail["rang"] = rang_du_premier_or(row["pool_chunks"], ors)
    detail["servis"] = servis
    if ors & set(servis):
        return "SERVI", detail
    if detail["rang"] is None:
        return ("GRANULARITE" if row["doc_hit"] else "DECOUVERTE"), detail
    if detail["rang"] <= len(servis) or detail["rang"] <= 5:
        return "EVICTION_DOCUMENT", detail
    if detail["rang"] <= 10:
        return "RANG_6_10", detail
    return "RANG_11_50", detail


def rappel_par_profondeur(cadre: dict, champ: str) -> dict[int, float]:
    """Part des questions dont **au moins un** or est dans les ``k`` premiers."""
    out = {}
    for k in PROFONDEURS:
        atteintes = sum(1 for row in cadre.values()
                        if row["gold_chunks"] and (row["gold_chunks"] & set(row[champ][:k])))
        out[k] = round(atteintes / len(cadre), 4)
    return out


def complementarite(cadre: dict, pools: dict) -> dict:
    """Ce que le lexical ajoute au dense, et réciproquement — au même budget.

    La question n'est pas « BM25 classe-t-il bien », à quoi l'arbitrage de la fusion a
    répondu NO-GO, mais « BM25 **trouve**-t-il des ors que le dense ne trouve pas ». Les
    deux se comportent en sens inverse et il faut les mesurer séparément.
    """
    dense_seul, bm25_seul, ni_l_un_ni_l_autre, les_deux = [], [], [], []
    for cle, row in cadre.items():
        ors = row["gold_chunks"]
        if not ors:
            continue
        d = bool(ors & set(row["pool_chunks"]))
        b = bool(ors & {c for c, _ in pools[cle]["bm25"]})
        (les_deux if d and b else dense_seul if d else bm25_seul if b else ni_l_un_ni_l_autre).append(cle)
    total = len(cadre)
    return {
        "dense_seul": len(dense_seul), "bm25_seul": len(bm25_seul),
        "les_deux": len(les_deux), "aucun": len(ni_l_un_ni_l_autre),
        "rappel_dense": round((len(dense_seul) + len(les_deux)) / total, 4),
        "rappel_bm25": round((len(bm25_seul) + len(les_deux)) / total, 4),
        "rappel_union_oracle": round((len(dense_seul) + len(bm25_seul) + len(les_deux)) / total, 4),
        "apport_bm25": len(bm25_seul), "apport_bm25_cles": sorted(bm25_seul),
        "apport_dense": len(dense_seul),
    }


def assemblage(cadre: dict, grille) -> dict:
    """Ce que coûtent ``passages=5`` et ``per_document=2`` — à pool strictement inchangé.

    C'est le seul levier de ce dossier qui ne touche ni au découpage, ni au plongement, ni
    au classement : le pool dense est identique, seule change la règle qui en tire les
    passages montrés. Un gain ici n'est donc pas une réorganisation du retrieval au sens
    de la règle §14 — rien n'est reclassé, rien n'est élargi.

    La grandeur mesurée est **la présence de l'or dans les passages servis**, et pas un
    nDCG. C'est le critère que le §14 impose désormais : *où* le gain se trouve, avant
    *combien* il vaut. Le taux de change est mesuré — la présence de l'or dans le contexte
    vaut +0,462 de couverture sur la population générale (§11), +0,958 sur les questions où
    elle est la pièce manquante (§15) — mais il ne s'applique que là où la présence bouge.

    **Les deux sens sont comptés.** Relever le plafond par document peut aussi *retirer*
    un or : les rangs de tête d'un document dominant reprennent les places que le plafond
    libérait pour d'autres documents. Un bilan qui ne compterait que les entrants serait
    faux, et c'est exactement l'erreur que la Phase A avait payée.
    """
    reference = None
    lignes = {}
    for passages, per_document in grille:
        servis, tous_les_ors, documents, tailles = {}, {}, [], []
        for cle, row in cadre.items():
            classe = [{"chunk_id": c, "document_id": d}
                      for c, d in zip(row["pool_chunks"], row["pool_documents"])]
            retenus = pipeline.build_context(classe, passages=passages, per_document=per_document)
            choisis = {p["chunk_id"] for p in retenus}
            servis[cle] = bool(row["gold_chunks"] and (row["gold_chunks"] & choisis))
            tous_les_ors[cle] = bool(row["gold_chunks"] and row["gold_chunks"] <= choisis)
            documents.append(len({p["document_id"] for p in retenus}))
            tailles.append(len(retenus))
        if reference is None:
            reference = servis
        entrent = sorted(c for c in servis if servis[c] and not reference[c])
        sortent = sorted(c for c in servis if reference[c] and not servis[c])
        base = [1.0 if reference[c] else 0.0 for c in cadre]
        variante = [1.0 if servis[c] else 0.0 for c in cadre]
        delta = metrics.paired_delta(base, variante)
        lignes[f"passages={passages} per_document={per_document}"] = {
            "or_servi": sum(variante), "part": round(sum(variante) / len(cadre), 4),
            "entrent": len(entrent), "sortent": len(sortent), "net": len(entrent) - len(sortent),
            "entrent_cles": entrent, "sortent_cles": sortent,
            "delta": delta.get("delta"), "ci95": delta.get("ci95"),
            "significant": delta.get("significant"),
            "passages_moyens": round(statistics.mean(tailles), 2),
            "documents_distincts_moyens": round(statistics.mean(documents), 2),
            # Le prix possible du plafond levé : un contexte qui vient d'un seul document.
            # C'est la raison d'être du plafond, et la seule façon de la peser est de la
            # compter — les questions `multi` sont celles qui la paieraient en premier.
            "contextes_mono_document": sum(1 for d in documents if d == 1),
            "tous_les_ors_servis": sum(tous_les_ors.values()),
            "entrent_par_famille": dict(collections.Counter(cadre[c]["kind"] for c in entrent)),
            "sortent_par_famille": dict(collections.Counter(cadre[c]["kind"] for c in sortent)),
        }
    return lignes


def controle_ligne_de_base(signature: str) -> dict | None:
    """Les chiffres publiés de la ligne de base, pour que le rejeu se compare à eux."""
    chemin = HERE / f"results-dense-controle-{signature}.json"
    if not chemin.exists():
        return None
    return json.loads(chemin.read_text(encoding="utf-8")).get("summary")


def plafond_oracle(cadre: dict, brut: dict) -> dict:
    """La condition de réouverture de l'hybride, recalculée — et sa grandeur exacte.

    `docs/STRATEGIE.md` §6.4 rouvre la fusion dense + lexical si, et seulement si, le
    **plafond d'oracle recalculé dépasse +0,037**. Ce nombre a une définition précise et il
    serait facile de le comparer à autre chose : c'est un **nDCG@10 poolé**, l'écart entre
    un routeur *parfait* — qui choisirait par question le meilleur des deux chemins — et le
    dense seul (`docs/TODO.md` §8 : +0,032 puis +0,037 après les titres du registre).

    Un rappel de pool à profondeur 50 n'est **pas** cette grandeur, et les deux ne se
    ressemblent que par leur ordre de grandeur. On calcule donc ici le nDCG@10, avec la
    fonction du dépôt et l'or pondéré, exactement comme la ligne de base.

    Le contrôle est publié à côté : le dense recalculé doit reproduire les chiffres de
    `results-dense-controle-<signature>.json`. Sans cela, aucun plafond n'est croyable.
    """
    def rangs(entree, champ):
        return [{"chunk_id": c, "document_id": d} for c, d, _ in entree[champ]]

    chemins = ("dense", "bm25", "rrf")
    scores = {chemin: {} for chemin in chemins}
    for cle, row in cadre.items():
        poids = row["item"].get("gold_poids") or None
        for chemin in chemins:
            scores[chemin][cle] = metrics.ndcg(rangs(brut[cle], chemin), row["gold_chunks"],
                                               row["gold_documents"], poids=poids)
    cles = list(cadre)

    def moyenne(valeurs, filtre=None):
        retenues = [valeurs[c] for c in cles if filtre is None or filtre(c)]
        return round(statistics.mean(retenues), 4) if retenues else None

    out = {"ndcg10": {chemin: {"pooled": moyenne(scores[chemin]),
                               "v1": moyenne(scores[chemin], lambda c: c.startswith("v1/")),
                               "v3": moyenne(scores[chemin], lambda c: c.startswith("v3/"))}
                      for chemin in chemins}}
    base = [scores["dense"][c] for c in cles]
    for nom, membres in (("dense_vs_bm25", ("dense", "bm25")),
                         ("dense_vs_rrf", ("dense", "rrf")),
                         ("dense_vs_bm25_vs_rrf", chemins)):
        oracle = [max(scores[m][c] for m in membres) for c in cles]
        delta = metrics.paired_delta(base, oracle)
        gagnantes = [c for i, c in enumerate(cles) if oracle[i] > base[i] + 1e-9]
        plafond = statistics.mean(oracle) - statistics.mean(base)
        # Le seuil de réouverture est un point, la mesure un intervalle. Dire « franchi »
        # sans dire si l'intervalle contient le seuil serait une fausse précision : c'est
        # exactement la manœuvre que le dossier s'interdit depuis l'arbitrage de la fusion.
        borne = delta.get("ci95") or (None, None)
        out[nom] = {"plafond": round(plafond, 4), "ci95": delta.get("ci95"),
                    "questions_ameliorees": len(gagnantes),
                    "cles_ameliorees": sorted(gagnantes),
                    "familles_ameliorees": dict(collections.Counter(
                        cadre[c]["kind"] for c in gagnantes)),
                    "point_au_dessus_de_0_037": plafond > 0.037,
                    "seuil_dans_l_intervalle": bool(borne[0] is not None
                                                    and borne[0] <= 0.037 <= borne[1])}
    return out


def ventilation(cadre: dict, classes: dict, cle_strate) -> dict:
    """Comptes par strate — **diagnostic exploratoire**, jamais une preuve de décision.

    Aucune de ces strates n'est pré-enregistrée. Elles nomment où regarder ; elles ne
    valident rien, et elles ne peuvent pas rouvrir une avenue fermée.
    """
    out: dict = collections.defaultdict(lambda: collections.Counter())
    for cle, row in cadre.items():
        out[cle_strate(row)][classes[cle][0]] += 1
    return {strate: dict(compte) for strate, compte in sorted(out.items())}


def audit(signature: str) -> dict:
    index = ChunkIndex.load(verbose=False)
    cadre = build_frame(index, signature)
    chemin = HERE / ".cache" / f"router-retrievals-{signature}.json"
    brut = json.loads(chemin.read_text(encoding="utf-8"))
    pools = {cle: {"bm25": [(c, d) for c, d, _ in entree["bm25"]]} for cle, entree in brut.items()}

    classes = {cle: classer(row) for cle, row in cadre.items()}
    comptes = collections.Counter(classe for classe, _ in classes.values())
    assert sum(comptes.values()) == len(cadre) == 155, "la population fixe est 155"

    exemples = {classe: [cle for cle, (c, _) in classes.items() if c == classe][:8] for classe in CLASSES}
    perdues = {cle: {"classe": classe, "famille": cadre[cle]["kind"],
                     "rang": detail["rang"], "n_or": detail["n_or"], "n_pool": detail["n_pool"],
                     "documents_or": sorted(cadre[cle]["gold_documents"])}
               for cle, (classe, detail) in classes.items() if classe != "SERVI"}

    rangs_servis = [detail["rang"] for classe, detail in classes.values()
                    if classe == "SERVI" and detail["rang"]]
    return {
        "signature": signature,
        "population": len(cadre),
        # L'index est toujours celui du corpus **courant** ; seuls les classements viennent
        # de la signature demandée. C'est licite pour rejouer un état antérieur — les
        # ajouts de documents ont été purement additifs, donc les identifiants d'or de
        # l'époque existent toujours — mais l'écrire évite de lire ces chunks comme ceux
        # de la signature du titre.
        "index_charge": {"signature": corpus_overlay.signature(),
                         "chunks": len(index.chunks),
                         "documents": len({c["document_id"] for c in index.chunks.values()})},
        "classements_de": signature,
        "classes": {classe: comptes.get(classe, 0) for classe in CLASSES},
        "exemples": exemples,
        "rappel_dense_par_profondeur": rappel_par_profondeur(cadre, "pool_chunks"),
        "rappel_des_cinq_servis": round(comptes["SERVI"] / len(cadre), 4),
        "rang_median_quand_servi": statistics.median(rangs_servis) if rangs_servis else None,
        "complementarite": complementarite(cadre, pools),
        "assemblage": assemblage(cadre, GRILLE_ASSEMBLAGE),
        "plafond_oracle": plafond_oracle(cadre, brut),
        "controle_ligne_de_base": controle_ligne_de_base(signature),
        "par_famille": ventilation(cadre, classes, lambda row: row["kind"]),
        "par_nombre_d_ors": ventilation(cadre, classes,
                                        lambda row: "0" if not row["gold_chunks"]
                                        else "1" if len(row["gold_chunks"]) == 1 else ">=2"),
        "par_banc": ventilation(cadre, classes, lambda row: row["bench"]),
        "perdues": perdues,
    }


def imprimer(rapport: dict) -> None:
    total = rapport["population"]
    index = rapport["index_charge"]
    print(f"\nClassements de {rapport['classements_de']} · index chargé {index['signature']} "
          f"({index['documents']} documents, {index['chunks']} passages)"
          f"{'  — REJEU D’UN ÉTAT ANTÉRIEUR' if rapport['classements_de'] != index['signature'] else ''}"
          f"\nPopulation fixe : {total} questions.\n")
    print("  classe                part      n     cumul")
    cumul = 0
    for classe in CLASSES:
        n = rapport["classes"][classe]
        cumul += n
        print(f"  {classe:<20} {n / total:6.1%}  {n:4}     {cumul:4}")
    print()
    servi = rapport["classes"]["SERVI"]
    pool = servi + sum(rapport["classes"][c] for c in ("EVICTION_DOCUMENT", "RANG_6_10", "RANG_11_50"))
    print(f"  or dans le pool dense@50 : {pool}/{total} ({pool / total:.1%})")
    print(f"  or dans les 5 servis     : {servi}/{total} ({servi / total:.1%})")
    print(f"  perdu entre les deux     : {pool - servi} questions — assemblage, pas récupération")
    print()
    print("  rappel dense par profondeur :")
    for k, v in rapport["rappel_dense_par_profondeur"].items():
        print(f"    @{k:<3} {v:.3f}")
    print()
    c = rapport["complementarite"]
    print(f"  dense {c['rappel_dense']:.3f} · BM25 {c['rappel_bm25']:.3f} · "
          f"union oracle {c['rappel_union_oracle']:.3f}")
    print(f"    ors que seul BM25 trouve : {c['apport_bm25']}  {c['apport_bm25_cles']}")
    print(f"    ors que seul le dense trouve : {c['apport_dense']}")
    print(f"    ors qu'aucun des deux ne trouve : {c['aucun']}")
    print("\n  assemblage du contexte — or présent dans les passages servis, pool inchangé :")
    print("   configuration              or servi   part   ent  sort   net   Δ [IC95]"
          "              psg  docs  mono  tous")
    for nom, ligne in rapport["assemblage"].items():
        ic = (f"{ligne['delta']:+.3f} [{ligne['ci95'][0]:+.3f};{ligne['ci95'][1]:+.3f}]"
              + (" *" if ligne.get("significant") else "  ")) if ligne.get("ci95") else " " * 24
        print(f"   {nom:<26} {ligne['or_servi']:4.0f}  {ligne['part']:6.1%}  "
              f"{ligne['entrent']:3}  {ligne['sortent']:4}  {ligne['net']:+5}   {ic:<24}"
              f" {ligne['passages_moyens']:4.1f} {ligne['documents_distincts_moyens']:5.2f}"
              f" {ligne['contextes_mono_document']:5} {ligne['tous_les_ors_servis']:5}")
    print("   psg = passages servis · docs = documents distincts · mono = contextes à un seul "
          "document · tous = questions dont TOUS les ors sont servis")
    po = rapport["plafond_oracle"]
    print("\n  nDCG@10 recalculé depuis le cache des classements servis :")
    for chemin, v in po["ndcg10"].items():
        print(f"    {chemin:<6} pooled {v['pooled']:.4f}   v1 {v['v1']:.4f}   v3 {v['v3']:.4f}")
    ref = rapport.get("controle_ligne_de_base")
    if ref:
        d = po["ndcg10"]["dense"]
        accord = all(abs(d[k] - ref[k]["nDCG@10"]) < 5e-4 for k in ("pooled", "v1", "v3"))
        print(f"    CONTRÔLE ligne de base publiée : pooled {ref['pooled']['nDCG@10']:.4f} "
              f"v1 {ref['v1']['nDCG@10']:.4f} v3 {ref['v3']['nDCG@10']:.4f}  "
              f"→ {'reproduite' if accord else 'ÉCART — rien de ce qui suit n’est croyable'}")
    print("\n  plafond d'oracle — la condition de réouverture de l'hybride est > +0,037 :")
    for nom in ("dense_vs_bm25", "dense_vs_rrf", "dense_vs_bm25_vs_rrf"):
        v = po[nom]
        ic = f"[{v['ci95'][0]:+.4f} ; {v['ci95'][1]:+.4f}]" if v.get("ci95") else ""
        verdict = ("point sous le seuil" if not v["point_au_dessus_de_0_037"]
                   else "point au-dessus, MAIS le seuil est DANS l'intervalle"
                   if v["seuil_dans_l_intervalle"] else "au-dessus, seuil hors intervalle")
        print(f"    {nom:<22} {v['plafond']:+.4f} {ic:<24} "
              f"{v['questions_ameliorees']:3} questions   {verdict}")
        print(f"      {'':<22} familles : {v['familles_ameliorees']}")
    print("\n  ventilation par famille (exploratoire, ne valide rien) :")
    for famille, compte in rapport["par_famille"].items():
        n = sum(compte.values())
        servis = compte.get("SERVI", 0)
        print(f"    {famille:<10} {servis:3}/{n:<3} servis   " +
              "  ".join(f"{k}={v}" for k, v in sorted(compte.items()) if k != "SERVI"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--signature", default=corpus_overlay.signature())
    parser.add_argument("--json", type=Path, help="écrire le rapport complet")
    args = parser.parse_args()
    rapport = audit(args.signature)
    imprimer(rapport)
    chemin = args.json or HERE / f"results-audit-pertes-{args.signature}.json"
    chemin.write_text(json.dumps(rapport, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n  écrit  {chemin.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
