"""Ce que le serveur sert réellement aux 155 questions du banc — avant et après le contrat.

Le même instrument mesure les deux états, et c'est le point : un « avant » et un « après »
produits par deux scripts différents ne se comparent pas. La coupe est un **paramètre**
(``--coupe brute`` reproduit la troncature historique, ``--coupe sure`` applique
``contrat.couper_passage``), la fenêtre aussi.

Aucun appel LLM. Le chemin exercé est celui de la production : ``quant_rag.search_explained``
avec ses défauts, c'est-à-dire ce que ``mcp_server.search_documents`` appelle.

    .venv/bin/python rag/benchmark/mesure_contrat.py \\
        --fenetre 2500 --coupe brute --sortie rag/benchmark/results-contrat-baseline-5530cba145.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import contrat  # noqa: E402
import corpus_overlay  # noqa: E402
import llm  # noqa: E402  — pour le bloc de dépense, jamais pour un appel
import quant_rag  # noqa: E402

_BLOC = re.compile(r"\$\$")

#: Les 20 questions ``negative`` de v3 n'ont pas d'or et sont exclues des mesures de retrieval
#: du dépôt ; elles restent **posées** au système et le banc de réponse les utilise. Elles sont
#: mesurées ici aussi, mais comptées à part : le contrat porte sur ce qui est servi, et une
#: question sans réponse dans le corpus se voit servir des passages comme les autres.
BANCS = {"v1": HERE / "questions-v1.jsonl", "v3": HERE / "questions-v3.jsonl"}


def questions() -> list[dict]:
    out = []
    for banc, chemin in BANCS.items():
        for ligne in chemin.read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                item = json.loads(ligne)
                item["banc"] = banc
                item["cle"] = f"{banc}/{item['qid']}"
                out.append(item)
    return out


def fautes(rendu: str, source: str) -> list[str]:
    """Ce que la coupe a cassé, jugé sur le texte **rendu**.

    Quatre fautes, et une seule règle : le matériau servi doit être celui du document, entier
    ou proprement interrompu, jamais à moitié. Un bloc ``$$`` laissé ouvert, une formule ``$``
    laissée ouverte, une ligne de tableau interrompue, un mot coupé.
    """
    position = len(rendu)
    trouvees = []
    if len(_BLOC.findall(rendu)) % 2 == 1:
        trouvees.append("bloc")
    if not contrat.math_desequilibree(source) and contrat._dollars_simples(rendu) % 2 == 1:
        trouvees.append("inline")
    if position < len(source) and contrat._dans_un_mot(source, position):
        trouvees.append("mot")
    if position < len(source) and contrat._dans_une_ligne_de_tableau(source, position):
        trouvees.append("tableau")
    return trouvees


def rendre(texte: str, fenetre: int, coupe: str) -> str:
    """Le texte servi pour un passage, selon le mode de coupe demandé."""
    if coupe == "brute":
        return (texte or "")[:fenetre]
    return contrat.couper_passage(texte, fenetre)


def percentile(valeurs: list[float], q: float) -> float:
    ordonnees = sorted(valeurs)
    if not ordonnees:
        return 0.0
    return ordonnees[min(int(q * len(ordonnees)), len(ordonnees) - 1)]


def mesurer(fenetre: int, coupe: str, limite: int | None = None) -> dict:
    items = questions()
    if limite:
        items = items[:limite]
    # Chauffe : le premier appel charge Qwen3-Embedding-0.6B sur MPS et fausserait la latence.
    quant_rag.search_explained("warm up the embedder", limit=1, log=False)

    lignes, latences = [], []
    for item in items:
        debut = time.perf_counter()
        sortie = quant_rag.search_explained(item["question"], limit=5)
        latence = (time.perf_counter() - debut) * 1000
        latences.append(latence)
        passages = []
        for row in sortie["results"]:
            source = row.get("text") or ""
            rendu = rendre(source, fenetre, coupe)
            nu, restants = (source[:fenetre], max(len(source) - fenetre, 0)) if coupe == "brute" \
                else contrat.couper(source, fenetre)
            passages.append({
                "chunk_id": row["chunk_id"], "document_id": row["document_id"],
                "content_type": row.get("content_type"),
                "longueur_source": len(source),
                "longueur_servie": len(rendu),
                "restants": restants,
                "fautes": fautes(nu, source),
                "drapeaux": contrat.drapeaux_qualite(nu, tronque=bool(restants)),
            })
        lignes.append({
            "cle": item["cle"], "banc": item["banc"], "qid": item["qid"],
            "kind": item.get("kind", "known-item"),
            "latency_ms": round(latence, 1),
            "mode": sortie["routing"]["mode"],
            "passages_servis": len(passages),
            "contexte_servi": sum(p["longueur_servie"] for p in passages),
            "passages": passages,
        })

    mesurables = [l for l in lignes if l["kind"] != "negative"]
    negatives = [l for l in lignes if l["kind"] == "negative"]

    def agreger(sous_ensemble: list[dict]) -> dict:
        tous = [p for l in sous_ensemble for p in l["passages"]]
        coupes = [p for p in tous if p["restants"] > 0]
        par_faute: dict[str, int] = {}
        for p in tous:
            for f in p["fautes"]:
                par_faute[f] = par_faute.get(f, 0) + 1
        contextes = [l["contexte_servi"] for l in sous_ensemble]
        return {
            "questions": len(sous_ensemble),
            "passages_servis": len(tous),
            "passages_tronques": len(coupes),
            "passages_avec_faute": sum(1 for p in tous if p["fautes"]),
            "fautes_par_type": par_faute,
            "contexte_servi_moyen": round(sum(contextes) / len(contextes), 1) if contextes else 0,
            "contexte_servi_median": percentile(contextes, 0.5),
            "passage_servi_moyen": round(sum(p["longueur_servie"] for p in tous) / len(tous), 1) if tous else 0,
            "passage_source_moyen": round(sum(p["longueur_source"] for p in tous) / len(tous), 1) if tous else 0,
            "drapeaux": {nom: sum(1 for p in tous if p["drapeaux"][nom])
                         for nom in ("has_math", "math_unbalanced", "has_control_chars",
                                     "has_html_tags", "has_table", "truncated")},
        }

    return {
        "instrument": "mesure_contrat.py",
        "fenetre": fenetre,
        "coupe": coupe,
        "corpus_signature": corpus_overlay.signature(),
        "contrat": contrat.configuration_servie(),
        "config_hash": contrat.config_hash(),
        # Le bloc de traçabilité de dépense, même quand elle est nulle : « 0 € » écrit et daté
        # vaut mieux que « 0 € » supposé parce que le fichier n'en parle pas.
        "depense": llm.tracabilite_run(fenetre),
        "latence_ms": {"p50": round(percentile(latences, 0.5), 1),
                       "p95": round(percentile(latences, 0.95), 1),
                       "moyenne": round(sum(latences) / len(latences), 1) if latences else 0},
        "155_questions": agreger(mesurables),
        "20_negatives": agreger(negatives),
        "par_question": lignes,
    }


def bout_en_bout(limite: int | None = None) -> dict:
    """La vérification la plus forte : le serveur rend-il le texte du document, **verbatim** ?

    On appelle ``mcp_server.search_documents`` — le vrai outil, pas une reconstitution — et on
    vérifie que le texte **entier** de chaque passage récupéré apparaît dans la réponse. Sur le
    corpus ``5530cba145`` le plus long passage fait 9 384 caractères et la fenêtre en vaut
    10 000 : aucun passage ne doit être coupé, donc aucun marqueur de troncature ne doit
    apparaître, et chaque source doit s'y retrouver mot pour mot.

    C'est ce qu'aucune mesure sur ``contrat.couper_passage`` seul ne peut établir : elle
    prouverait que la coupe est sûre, pas qu'elle est **branchée**.
    """
    import mcp_server

    items = questions()
    if limite:
        items = items[:limite]
    quant_rag.search_explained("warm up the embedder", limit=1, log=False)

    verifies = manquants = marques = 0
    echecs = []
    for item in items:
        sortie = quant_rag.search_explained(item["question"], limit=5)
        rendu = mcp_server.search_documents(item["question"], limit=5)
        if "[… tronqué" in rendu:
            marques += 1
        for row in sortie["results"]:
            source = row.get("text") or ""
            verifies += 1
            if source and source not in rendu:
                manquants += 1
                if len(echecs) < 10:
                    echecs.append({"cle": item["cle"], "chunk_id": row["chunk_id"],
                                   "longueur": len(source)})
    return {"passages_verifies": verifies, "passages_non_verbatim": manquants,
            "reponses_portant_un_marqueur": marques, "echecs": echecs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fenetre", type=int, default=None,
                        help="fenêtre servie par passage (défaut : contrat.PASSAGE_CHARACTERS)")
    parser.add_argument("--coupe", choices=("brute", "sure"), default="sure",
                        help="brute = troncature historique ; sure = contrat.couper_passage")
    parser.add_argument("--sortie", type=Path, default=None)
    parser.add_argument("--limite", type=int, default=None, help="ne mesurer que les N premières questions")
    parser.add_argument("--bout-en-bout", action="store_true",
                        help="appeler le vrai mcp_server.search_documents et vérifier le verbatim")
    args = parser.parse_args()

    fenetre = args.fenetre if args.fenetre is not None else contrat.PASSAGE_CHARACTERS
    resultat = mesurer(fenetre, args.coupe, args.limite)
    if args.bout_en_bout:
        resultat["bout_en_bout"] = bout_en_bout(args.limite)
        b = resultat["bout_en_bout"]
        print(f"  bout en bout           : {b['passages_verifies']} passages vérifiés, "
              f"{b['passages_non_verbatim']} non verbatim, "
              f"{b['reponses_portant_un_marqueur']} réponses tronquées")
    resume = resultat["155_questions"]
    print(f"fenêtre {fenetre} · coupe {args.coupe} · signature {resultat['corpus_signature']}")
    print(f"  {resume['questions']} questions, {resume['passages_servis']} passages servis")
    print(f"  passages tronqués      : {resume['passages_tronques']}")
    print(f"  passages avec faute    : {resume['passages_avec_faute']}  {resume['fautes_par_type']}")
    print(f"  contexte servi moyen   : {resume['contexte_servi_moyen']} c. par question")
    print(f"  passage servi moyen    : {resume['passage_servi_moyen']} c. (source {resume['passage_source_moyen']})")
    print(f"  latence p50/p95        : {resultat['latence_ms']['p50']} / {resultat['latence_ms']['p95']} ms")
    if args.sortie:
        args.sortie.write_text(json.dumps(resultat, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  écrit : {args.sortie}")


if __name__ == "__main__":
    main()
