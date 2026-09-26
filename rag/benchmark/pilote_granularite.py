"""Le pilote de granularité — un chunk d'or trop long noie-t-il le passage qui répond ?

**L'hypothèse, et elle n'a jamais été mesurée.** Les 20 questions `GRANULARITE` ont leur
**document** dans le pool dense mais pas leur **passage**. Le §20 octies a trouvé un gradient
de longueur monotone — médiane du chunk d'or 2 880 caractères quand l'or est hors pool, 2 017
quand il est servi, 1 733 dans le corpus. C'est une **association**, sur 24 chunks d'or, et
rien n'en fait une cause. L'hypothèse à tuer ou à ouvrir : le passage qui répond est noyé dans
un chunk trop long, dont le plongement est dominé par le reste.

**Ce que ce module est.** Quelques centaines de plongements, pas trois heures de GPU. Il
découpe les 24 chunks d'or à leurs **frontières de blocs**, plonge chaque fragment avec la
recette **exacte** de la production, et regarde si l'un d'eux serait entré dans le pool.

**CORRECTION DU 8 SEPTEMBRE 2026, APPORTÉE CONTRE LE PREMIER RÉSULTAT DE CE MODULE.** Le
maximum sur tous les sous-intervalles suppose un découpeur qui **connaît la question**. Un vrai
chunker coupe **avant** de la voir et ne produit qu'une **partition**. ``partitions`` simule le
chunker glouton sur une grille de cibles déclarée, et le résultat tombe de **10/16 à 6/16**
(cible 500 c.) — l'oracle gonflait le franchissement de deux tiers. **C'est la partition qui
décide, pas le maximum** ; le maximum reste publié comme ce qu'il est, une borne inatteignable.

**Ce qu'il n'est pas.** Un pré-enregistrement : il ne décide aucune promotion, il oriente un
investissement (le ré-embarquement vaut 2 h 51 par bras et **change la signature du corpus**,
donc engage le gel). Les trois issues ci-dessous sont néanmoins écrites et commitées **avant
la première mesure**.

Les trois issues, écrites avant de regarder le moindre score
------------------------------------------------------------
- **A — aucun fragment ne bat son chunk parent.** L'hypothèse de dilution est **réfutée**, la
  granularité meurt comme levier de rappel, et un chantier de plusieurs heures est économisé
  avec l'ouverture du gel. On l'écrit et on arrête ce fil.
- **B — des fragments battent leur parent, aucun ne franchit le seuil du pool.** Le mécanisme
  existe et ne suffit pas. On **chiffre l'écart au seuil**, distribution complète — c'est lui
  qui dira si un découpage plus fin *global* aurait une chance. Rien de plus.
- **C — un nombre significatif franchit le seuil.** Le chantier de re-découpage est justifié.
  **On ne le lance pas.** On chiffre son coût et on remonte l'arbitrage.

La méthode, et pourquoi elle donne une borne et non un échantillon
------------------------------------------------------------------
Un chunk d'or est reconstruit exactement à partir de ses blocs dans **21 cas sur 24**
(``src/parsing/document_text.provenance_du_chunk``, granularité ``exacte``). Le module
n'énumère pas quelques fragments : il énumère **tous les segments contigus de blocs** du
chunk — les k(k+1)/2 sous-intervalles, chunk parent compris. Le meilleur score obtenu est
donc le **maximum sur tout découpage respectant la grille des blocs**, pas le score d'une
découpe particulière qu'on aurait choisie. C'est ce qui rend l'issue A concluante : si le
maximum ne bat pas le parent, aucun découpage plus fin ne le battra.

Les pièges, tous nommés
-----------------------
1. **La recette de plongement.** La production plonge ``Document: <titre>\\nPath: <chemin>\\n\\n
   <texte>`` (``rechunk_corpus.retrieval_text``, ``avec_page=False``), et le **titre a deux
   sources** — 317 documents figés, 101 recalculés à la livraison (``titres_plonges``). La
   requête, elle, porte un prompt **différent** (``Instruct: …\\nQuery:``,
   ``quant_rag.encode_query``). Plonger un fragment sans cette enveloppe comparerait deux
   espaces. Contrôle exigé avant toute conclusion : ``embed_candidat.py --check``, qui
   ré-embarque 40 chunks servis et refuse sous un cosinus de 0,99 — **passé le 8 septembre
   2026 : min 0,99964, médiane 0,99999**.
2. **Le seuil n'est pas « le 50ᵉ du cache ».** ``quant_rag._select`` (l. 728) écarte
   **inconditionnellement** tout candidat ``content_type == "heading"``, sans repêchage : le
   pool dense mis en cache fait 29 à 50 éléments et son dernier score est **supérieur** au vrai
   seuil d'entrée. Le seuil décisif est le **50ᵉ score brut**, obtenu par ``dense_matrix``
   (recherche exacte, sans verrou Qdrant, reproduit Qdrant à 153/155 top-50). **Et
   ``Matrix.search`` filtre lui aussi les ``heading`` par défaut** (``skip_headings=True``,
   l. 264) : le pilote a fait cette confusion une fois et l'a corrigée le même jour. Il passe
   désormais ``skip_headings=False`` et vérifie que la liste fait exactement 50 entrées. Les
   **trois** seuils sont publiés — brut, filtré, dernier du cache ; **c'est le brut qui
   décide**.
3. **Un fragment `heading` n'entrerait jamais.** Les segments dont **tous** les blocs sont des
   titres sont comptés à part et exclus du décompte décisif.
4. **L'artefact de longueur.** Un fragment court a mécaniquement un cosinus plus élevé face à
   une requête courte. Sans contrôle, on prendrait cet artefact pour un gain de rappel. **Le
   contrôle négatif est donc un sabotage** : la même procédure est appliquée au chunk d'or
   d'une **autre** question (permutation déterministe, graine 20260908), qui ne contient pas le
   matériau qui répond. Si les fragments sabotés franchissent le seuil aussi souvent, la mesure
   ne dit rien du rappel. **Ce contrôle n'était pas au mandat ; sans lui le pilote n'est pas
   lisible.**
5. **Franchir un seuil n'est pas entrer.** Un fragment n'entre qu'en chassant un candidat.
   Le **rang** qu'il occuperait dans le top-50 brut est rapporté, pas seulement le
   franchissement.

Aucun appel LLM. Aucune écriture en production. Le gel n'est pas levé, la signature ne bouge
pas : rien n'est ré-embarqué dans la collection, tout est calculé en mémoire.

    .venv/bin/python rag/benchmark/pilote_granularite.py
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rag" / "ingestion"))

import numpy as np  # noqa: E402

import corpus_overlay  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from rechunk_corpus import retrieval_text, titres_plonges  # noqa: E402 — un seul domicile
from src.parsing.document_text import (GRANULARITE_EXACTE, provenance_du_chunk,  # noqa: E402
                                       sha256_texte, texte_canonique)

SIGNATURE = corpus_overlay.signature()
GRAINE = 20260908
PROFONDEUR = quant_rag.POOL          # 50, la profondeur brute interrogée par la production
BATCH = 8                            # comme apply_delivery / reembed_titles / embed_candidat
#: Échantillon du contrôle de sabotage — toutes les questions, pas un sous-ensemble.
REGISTRE = ROOT / "rag" / "ingestion" / "registry-v1.json"
INGERES = ROOT / "data" / "processed" / "ingested"


def questions_granularite() -> list[tuple[str, dict]]:
    audit = json.loads((HERE / f"results-audit-pertes-{SIGNATURE}.json").read_text(encoding="utf-8"))
    cles = [c for c, v in audit["perdues"].items() if v["classe"] == "GRANULARITE"]
    bancs = {}
    for chemin in (HERE / "questions-v1.jsonl", HERE / "questions-v3.jsonl"):
        banc = "v1" if chemin.name.endswith("v1.jsonl") else "v3"
        for ligne in chemin.read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                item = json.loads(ligne)
                ors = item.get("gold_chunks")
                if ors is None and item.get("target_chunk"):
                    ors = [item["target_chunk"]]
                bancs[f"{banc}/{item['qid']}"] = {**item, "gold_chunks": ors or []}
    return [(c, bancs[c]) for c in sorted(cles)]


def dossiers() -> dict[str, str]:
    registre = json.loads(REGISTRE.read_text(encoding="utf-8"))
    return {e["document_id"]: e["folder"] for e in registre.get("documents", [])}


def blocs_du_document(dossier: str):
    base = INGERES / dossier
    blocs = [json.loads(l) for l in (base / "blocks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    bruts = {}
    for l in (base / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            row = json.loads(l)
            bruts[row["chunk_id"]] = row
    return blocs, bruts


def segments(chunk_brut: dict, blocs: list[dict], texte: str, spans: dict, sha: str) -> dict:
    """Tous les segments contigus de blocs du chunk — le maximum sur tout découpage.

    Rend ``{"segments": [(debut_bloc, fin_bloc, texte, titres_seuls)], "granularite": …}``.
    """
    provenance = provenance_du_chunk(chunk_brut, spans, sha)
    intervalles = list(provenance.intervalles)
    types = {b["block_id"]: (b.get("block_type") or b.get("content_type") or "") for b in blocs}
    morceaux = [(iv.block_id, texte[iv.debut:iv.fin]) for iv in intervalles]
    out = []
    for i in range(len(morceaux)):
        for j in range(i, len(morceaux)):
            ids = [m[0] for m in morceaux[i:j + 1]]
            corps = "\n\n".join(m[1] for m in morceaux[i:j + 1]).strip()
            if not corps:
                continue
            titres_seuls = all(str(types.get(b, "")).lower() in ("title", "heading") for b in ids)
            out.append({"i": i, "j": j, "n_blocs": j - i + 1, "texte": corps,
                        "titre_seul": titres_seuls, "entier": (i == 0 and j == len(morceaux) - 1)})
    return {"segments": out, "granularite": provenance.granularite, "n_blocs": len(morceaux)}


def plonger(textes: list[str]) -> np.ndarray:
    modele = quant_rag.embedder()
    return np.asarray(modele.encode(textes, normalize_embeddings=True, show_progress_bar=False,
                                    batch_size=BATCH), dtype=np.float32)


def pilote() -> dict:
    from dense_matrix import Matrix

    index = ChunkIndex.load(verbose=False)
    titres, _ = titres_plonges()
    dossier_de = dossiers()
    matrice = Matrix.load()
    cache = json.loads((HERE / ".cache" / f"router-retrievals-{SIGNATURE}.json").read_text(encoding="utf-8"))
    questions = questions_granularite()

    # sabotage : le chunk d'or d'une AUTRE question, permutation déterministe
    alea = random.Random(GRAINE)
    ordre = list(range(len(questions)))
    alea.shuffle(ordre)
    decalage = {i: ordre[(ordre.index(i) + 1) % len(ordre)] for i in range(len(questions))}

    lignes, ignores = [], []
    for position, (cle, item) in enumerate(questions):
        requete = pipeline.query_of(item)
        vecteur = np.asarray(quant_rag.encode_query(requete), dtype=np.float32)
        # ``skip_headings=False`` : le top-50 **brut** de Qdrant, avant que ``_select`` n'en
        # retire les ``heading``. C'est le seul seuil qui décide de l'entrée dans le pool ; le
        # défaut de ``Matrix.search`` est ``True`` (l. 264) et rendrait le seuil *filtré*, plus
        # haut. La confusion a été faite une fois dans ce module, et corrigée le même jour.
        brut = matrice.search(vecteur, pool=PROFONDEUR, skip_headings=False)
        scores_bruts = [float(r["score"]) for r in brut]
        assert len(scores_bruts) == PROFONDEUR, (cle, len(scores_bruts))
        seuil_brut = scores_bruts[-1]
        seuil_filtre = float(matrice.search(vecteur, pool=PROFONDEUR)[-1]["score"])
        dense_cache = cache[cle]["dense"]
        seuil_cache = float(dense_cache[-1][2])

        for role, source_cle in (("or", cle), ("sabotage", questions[decalage[position]][0])):
            source_item = dict(questions)[source_cle] if role == "sabotage" else item
            for chunk_id in source_item["gold_chunks"]:
                servi = index.get(chunk_id)
                if not servi:
                    ignores.append({"cle": cle, "role": role, "chunk": chunk_id, "cause": "hors index"})
                    continue
                document = servi["document_id"]
                dossier = dossier_de.get(document)
                if not dossier or not (INGERES / dossier / "blocks.jsonl").exists():
                    ignores.append({"cle": cle, "role": role, "chunk": chunk_id, "cause": "sans blocs"})
                    continue
                blocs, bruts = blocs_du_document(dossier)
                chunk_brut = bruts.get(chunk_id)
                if not chunk_brut:
                    ignores.append({"cle": cle, "role": role, "chunk": chunk_id, "cause": "absent du brut"})
                    continue
                texte, spans = texte_canonique(blocs)
                decoupe = segments(chunk_brut, blocs, texte, spans, sha256_texte(texte))
                if decoupe["n_blocs"] < 2:
                    ignores.append({"cle": cle, "role": role, "chunk": chunk_id,
                                    "cause": "un seul bloc", "granularite": decoupe["granularite"]})
                    continue
                chemin = chunk_brut.get("title_path") or (servi.get("title_path"))
                page = chunk_brut.get("page_start") or servi.get("page_start")
                enveloppes = [retrieval_text(titres.get(document, ""), chemin, page, s["texte"],
                                             avec_page=False) for s in decoupe["segments"]]
                vecteurs = plonger(enveloppes)
                cosinus = vecteurs @ vecteur
                # le parent : l'enveloppe du texte SERVI, la seule qui reproduise le vecteur stocké
                parent = plonger([retrieval_text(titres.get(document, ""), chemin, page,
                                                 servi["text"], avec_page=False)])[0] @ vecteur
                for s, c in zip(decoupe["segments"], cosinus):
                    lignes.append({
                        "cle": cle, "role": role, "chunk": chunk_id, "n_blocs": s["n_blocs"],
                        "i": s["i"], "j": s["j"], "entier": s["entier"], "titre_seul": s["titre_seul"],
                        "longueur": len(s["texte"]), "cosinus": float(c),
                        "parent": float(parent), "seuil_brut": seuil_brut, "seuil_filtre": seuil_filtre, "seuil_cache": seuil_cache,
                        "granularite": decoupe["granularite"],
                        "rang": 1 + sum(1 for v in scores_bruts if v > float(c)),
                    })
        print(f"  {position + 1:2d}/{len(questions)}  {cle:10s} "
              f"{sum(1 for l in lignes if l['cle'] == cle)} segments", end="\r", flush=True)
    print()
    return {"signature": SIGNATURE, "lignes": lignes, "ignores": ignores,
            "matrice": getattr(matrice, "label", None), "profondeur": PROFONDEUR,
            "questions": [c for c, _ in questions]}


#: Plancher de longueur de la **sélection de production** — ``quant_rag.MIN_CHARACTERS``.
#: Un fragment plus court entrerait dans le pool (qui est interrogé à ``min_characters=0``)
#: et ne serait **jamais servi** : ``_select`` l'écarterait. Le décompte qui compte pour le
#: produit est donc celui des fragments **servables**, et il est publié à part.
MIN_SERVABLE = quant_rag.MIN_CHARACTERS


#: Tailles de découpe simulées par ``partitions``. Grille déclarée, toutes publiées.
CIBLES = (500, 750, 1000, 1500, 2000)


def partitions(donnees: dict, longueur_minimale: int = 0) -> dict:
    """La correction de l'oracle — **et c'est elle qui décide, pas le maximum**.

    Le maximum sur tous les sous-intervalles suppose un découpeur qui **connaît la question** :
    il choisit, parmi les k(k+1)/2 segments, celui qui répond. Un vrai chunker coupe **avant**
    de voir la question, et ne produit qu'une **partition** — des morceaux contigus, disjoints,
    couvrants. Cette fonction simule le chunker glouton : il avance bloc par bloc et ferme dès
    que la cible de longueur est dépassée. Une question compte comme gagnée si **l'un
    quelconque** de ses morceaux franchit le seuil, ce qui est légitime — dans un index
    re-découpé, tous les morceaux existent.

    Publiée pour toute la grille ``CIBLES``, sans en choisir une après coup.
    """
    par_chunk = collections.defaultdict(dict)
    for l in donnees["lignes"]:
        if l["role"] == "or":
            par_chunk[(l["cle"], l["chunk"])][(l["i"], l["j"])] = l
    out = {}
    for cible in CIBLES:
        franchit, rang5, vues = set(), set(), set()
        for (cle, _), disponibles in par_chunk.items():
            vues.add(cle)
            n = max(j for _, j in disponibles) + 1
            longueur = {i: disponibles[(i, i)]["longueur"] for i in range(n) if (i, i) in disponibles}
            morceaux, debut, cumul = [], 0, 0
            for i in range(n):
                cumul += longueur.get(i, 0)
                if cumul >= cible and (debut, i) in disponibles:
                    morceaux.append((debut, i)); debut, cumul = i + 1, 0
            if debut <= n - 1 and (debut, n - 1) in disponibles:
                morceaux.append((debut, n - 1))
            for m in morceaux:
                l = disponibles.get(m)
                if not l or l["titre_seul"] or l["longueur"] < longueur_minimale:
                    continue
                if l["cosinus"] > l["seuil_brut"]:
                    franchit.add(cle)
                if l["rang"] <= 5:
                    rang5.add(cle)
        out[cible] = {"questions": len(vues), "franchit_le_seuil": len(franchit),
                      "rang_5": len(rang5), "franchissantes": sorted(franchit)}
    return out


def resumer(donnees: dict, longueur_minimale: int = 0) -> dict:
    lignes = donnees["lignes"]
    out = {"signature": donnees["signature"], "appels_llm": 0,
           "longueur_minimale": longueur_minimale,
           "questions": len(donnees["questions"]), "segments": len(lignes),
           "ignores": donnees["ignores"]}
    for role in ("or", "sabotage"):
        util = [l for l in lignes if l["role"] == role and not l["titre_seul"]
                and l["longueur"] >= longueur_minimale]
        titres = [l for l in lignes if l["role"] == role and l["titre_seul"]]
        par_question = collections.defaultdict(list)
        for l in util:
            par_question[l["cle"]].append(l)
        meilleur, bat_parent, franchit_brut, franchit_cache, ecarts, rangs = {}, 0, 0, 0, [], []
        franchit_filtre = 0
        for cle, groupe in par_question.items():
            partiels = [l for l in groupe if not l["entier"]]
            if not partiels:
                continue
            top = max(partiels, key=lambda l: l["cosinus"])
            meilleur[cle] = {"cosinus": round(top["cosinus"], 4), "parent": round(top["parent"], 4),
                             "seuil_brut": round(top["seuil_brut"], 4),
                             "seuil_filtre": round(top["seuil_filtre"], 4),
                             "seuil_cache": round(top["seuil_cache"], 4),
                             "n_blocs": top["n_blocs"], "longueur": top["longueur"],
                             "rang": top["rang"],
                             "bat_parent": top["cosinus"] > top["parent"],
                             "franchit_brut": top["cosinus"] > top["seuil_brut"]}
            bat_parent += int(top["cosinus"] > top["parent"])
            franchit_brut += int(top["cosinus"] > top["seuil_brut"])
            franchit_filtre += int(top["cosinus"] > top["seuil_filtre"])
            franchit_cache += int(top["cosinus"] > top["seuil_cache"])
            ecarts.append(top["cosinus"] - top["seuil_brut"])
            rangs.append(top["rang"])
        out[role] = {
            "questions_mesurees": len(meilleur), "segments_utiles": len(util),
            "segments_titre_seul_exclus": len(titres),
            "bat_son_parent": bat_parent, "franchit_le_seuil_brut": franchit_brut,
            "franchit_le_seuil_filtre": franchit_filtre,
            "franchit_le_seuil_du_cache": franchit_cache,
            "ecart_au_seuil": {
                "mediane": round(statistics.median(ecarts), 4) if ecarts else None,
                "min": round(min(ecarts), 4) if ecarts else None,
                "max": round(max(ecarts), 4) if ecarts else None,
            },
            "rang_median_du_meilleur_fragment": statistics.median(rangs) if rangs else None,
            "detail": meilleur,
        }
    return out


def imprimer(r: dict) -> None:
    print(f"\nsignature {r['signature']}   {r['questions']} questions GRANULARITE   "
          f"{r['segments']} segments plongés   appels LLM {r['appels_llm']}")
    for role, titre in (("or", "OR — le chunk qui répond"), ("sabotage", "SABOTAGE — le chunk d'une autre question")):
        v = r[role]
        print(f"\n{titre}")
        print(f"   questions mesurées {v['questions_mesurees']}   segments utiles {v['segments_utiles']}"
              f"   (titres seuls exclus : {v['segments_titre_seul_exclus']})")
        print(f"   le meilleur fragment bat son chunk parent : {v['bat_son_parent']} / {v['questions_mesurees']}")
        print(f"   il franchit le seuil BRUT (50e de Qdrant)  : {v['franchit_le_seuil_brut']} / {v['questions_mesurees']}")
        print(f"   il franchit le seuil filtré (dernier retenu): {v['franchit_le_seuil_filtre']} / {v['questions_mesurees']}")
        print(f"   il franchit le dernier score du cache      : {v['franchit_le_seuil_du_cache']} / {v['questions_mesurees']}")
        e = v["ecart_au_seuil"]
        print(f"   écart au seuil brut — médiane {e['mediane']}   min {e['min']}   max {e['max']}")
        print(f"   rang médian qu'occuperait le meilleur fragment : {v['rang_median_du_meilleur_fragment']}")
        rangs = sorted(d["rang"] for d in v["detail"].values())
        print(f"   rangs : {rangs}")
        print(f"   dont rang <= 5 (fenêtre servie) : {sum(1 for r in rangs if r <= 5)}"
              f"   <= 10 : {sum(1 for r in rangs if r <= 10)}")
    if r["ignores"]:
        causes = collections.Counter(i["cause"] for i in r["ignores"])
        print(f"\nchunks écartés : {dict(causes)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--brut", action="store_true", help="écrit aussi le détail segment par segment")
    a = p.parse_args()
    donnees = pilote()
    resume = resumer(donnees)
    resume["servable"] = resumer(donnees, longueur_minimale=MIN_SERVABLE)
    resume["partitions"] = partitions(donnees, longueur_minimale=MIN_SERVABLE)
    (HERE / f"results-pilote-granularite-{SIGNATURE}.json").write_text(
        json.dumps(resume, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.brut:
        (HERE / ".cache" / f"pilote-granularite-segments-{SIGNATURE}.json").write_text(
            json.dumps(donnees, ensure_ascii=False), encoding="utf-8")
    imprimer(resume)
    print(f"\n————— restreint aux fragments SERVABLES (>= {MIN_SERVABLE} c., quant_rag.MIN_CHARACTERS)")
    imprimer(resume["servable"])
    print("\n————— PARTITIONS — ce qu'un chunker qui ne connaît pas la question obtiendrait")
    o = resume["servable"]["or"]
    print(f"   {'cible':>6s} {'franchit':>9s} {'rang<=5':>8s}   (oracle : "
          f"{o['franchit_le_seuil_brut']} et {sum(1 for d in o['detail'].values() if d['rang'] <= 5)})")
    for cible, v in resume["partitions"].items():
        print(f"   {cible:6d} {v['franchit_le_seuil']:9d} {v['rang_5']:8d}")
    print(f"\nécrit : rag/benchmark/results-pilote-granularite-{SIGNATURE}.json")


if __name__ == "__main__":
    main()
