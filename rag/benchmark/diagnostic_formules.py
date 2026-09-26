"""Où se perdent les questions `formula` — et ce que le lot de représentation doit en retenir.

Ce module ne teste aucune hypothèse et n'autorise aucun chantier
-----------------------------------------------------------------
La ligne de base v4 a montré que la famille `formula` échoue d'abord par **récupération** :
sur 24 questions, l'or n'est servi que 6 fois, et conditionnellement à l'or servi le score
remonte de 0,17 à 0,67. Reste à savoir *où* l'or se perd — au rang 6, au rang 40, ou pas du
tout — et si les chunks de formules ont une signature qui les distingue du reste du corpus.

C'est tout ce que fait ce module : un tableau, **zéro appel LLM**, et zéro variante de
retrieval. Il dit au lot de représentation si l'hypothèse « le LaTeX éclaté de MinerU dégrade
le vecteur des chunks de formules » mérite une sonde. **Il ne la teste pas, et il n'autorise
rien** : le §14 du ``docs/TODO.md`` reste fermé, et aucun ré-embarquement n'est fait ici.

Pourquoi un témoin corpus, et pas seulement les chunks d'or
------------------------------------------------------------
« 62 % de LaTeX dans les chunks d'or » ne veut rien dire tout seul : si le corpus entier en
porte autant, le chiffre ne distingue rien. Chaque statistique est donc donnée **à côté de la
même statistique sur l'ensemble du corpus**, et c'est l'écart entre les deux — pas le niveau —
qui est lisible.

    .venv/bin/python rag/benchmark/diagnostic_formules.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import familles_v4  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from dense_matrix import Matrix  # noqa: E402

SIGNATURE = corpus_overlay.signature()
#: Le pool de production. Au-delà, l'or n'est ramené par aucun chemin servi.
POOL = 50
#: Les cinq passages réellement montrés.
SERVIS = 5

#: Régions mathématiques d'un chunk, telles que MinerU les balise. ``$$…$$`` d'abord : sans
#: cette priorité, ``$…$`` couperait chaque bloc affiché en deux et gonflerait le compte.
_MATHS = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$", re.DOTALL)
#: La signature de l'éclatement MinerU : des lettres séparées par des espaces **dans** une
#: commande de police. ``\mathrm { b i d }`` au lieu de ``\mathrm{bid}``.
_LETTRES_ESPACEES = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|text|operatorname)\s*\{[^{}]*"
    r"[A-Za-z]\s+[A-Za-z][^{}]*\}")
_COMMANDE = re.compile(r"\\[a-zA-Z]+")


def part_de_latex(texte: str) -> float:
    """Part des caractères du chunk qui tombent dans une région mathématique."""
    if not texte:
        return 0.0
    return round(sum(len(m.group()) for m in _MATHS.finditer(texte)) / len(texte), 4)


def profil(texte: str) -> dict:
    return {
        "longueur": len(texte or ""),
        "part_latex": part_de_latex(texte),
        "commandes_pour_mille": round(1000 * len(_COMMANDE.findall(texte or ""))
                                      / max(len(texte or ""), 1), 2),
        "lettres_espacees": bool(_LETTRES_ESPACEES.search(texte or "")),
    }


def bande(rang: int | None) -> str:
    """La bande de rang, en langage de production plutôt qu'en nombre."""
    if rang is None:
        return "au-dela_de_50"
    if rang <= SERVIS:
        return "1_a_5_dans_le_pool_servi"
    if rang <= 10:
        return "6_a_10"
    return "11_a_50"


def rang_de_l_or(matrice: Matrix, question: str, ors: set[str]) -> int | None:
    """Rang du premier chunk d'or dans le classement dense, ou ``None`` au-delà du pool.

    C'est le **classement brut**, pas la sélection servie : la sélection retire les en-têtes,
    plafonne à deux passages par document et déduplique. Mesurer le rang brut sépare « le
    plongement ne trouve pas » de « la sélection l'écarte » — deux pannes qui appellent deux
    chantiers différents, et que le §10 avait déjà pris soin de distinguer.
    """
    for rang, ligne in enumerate(matrice.search_text(question, pool=POOL), 1):
        if ligne["chunk_id"] in ors:
            return rang
    return None


def mesurer() -> dict:
    index = ChunkIndex.load(verbose=False)
    matrice = Matrix.load()
    items = familles_v4.lire_jsonl(HERE / "questions-v4-formula.jsonl")

    lignes, bandes = [], Counter()
    for item in items:
        ors = set(item["gold_chunks"])
        rang = rang_de_l_or(matrice, item["question"], ors)
        chunk = index.chunks[item["gold_chunks"][0]]
        ligne = {"qid": item["qid"], "rang_de_l_or": rang, "bande": bande(rang),
                 "leak": item.get("leak"), "atomes_or": item.get("or_atomes"),
                 **profil(chunk["text"])}
        lignes.append(ligne)
        bandes[ligne["bande"]] += 1

    # Le témoin : les mêmes statistiques sur le corpus entier. Sans lui, aucun niveau n'est
    # lisible — seul l'écart l'est.
    tous = [profil(r["text"]) for r in index.chunks.values()]
    ors = [l for l in lignes]

    def resume(profils: list[dict]) -> dict:
        return {
            "n": len(profils),
            "longueur_mediane": round(statistics.median(p["longueur"] for p in profils), 1),
            "part_latex_mediane": round(statistics.median(p["part_latex"] for p in profils), 4),
            "commandes_pour_mille_mediane": round(
                statistics.median(p["commandes_pour_mille"] for p in profils), 2),
            "part_avec_lettres_espacees": round(
                sum(1 for p in profils if p["lettres_espacees"]) / max(len(profils), 1), 4),
        }

    trouves = [l for l in lignes if l["rang_de_l_or"] is not None]
    sortie = {
        "signature": SIGNATURE, "appels_llm": 0, "pool": POOL,
        "n_questions": len(lignes),
        "bandes": dict(bandes),
        "or_dans_le_pool": len(trouves),
        "rang_median_quand_trouve": (round(statistics.median(l["rang_de_l_or"] for l in trouves), 1)
                                     if trouves else None),
        "chunks_d_or": resume(ors),
        "temoin_corpus_entier": resume(tous),
        "detail": lignes,
        "ce_que_ce_tableau_n_autorise_pas": (
            "aucune variante de retrieval, aucun ré-embarquement, aucun chantier. Il dit "
            "seulement si l'hypothèse « le LaTeX éclaté dégrade le vecteur » mérite une sonde."),
    }
    (HERE / f"diagnostic-formules-{SIGNATURE}.json").write_text(
        json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8")
    return sortie


def imprimer(r: dict) -> None:
    print(f"\nsignature {r['signature']} · {r['n_questions']} questions `formula` · "
          f"pool {r['pool']} · appels LLM : {r['appels_llm']}")
    print(f"\n{'bande de rang de l or':32s} {'n':>4s}  part")
    for nom in ("1_a_5_dans_le_pool_servi", "6_a_10", "11_a_50", "au_dela_de_50",
                "au-dela_de_50"):
        if nom in r["bandes"]:
            n = r["bandes"][nom]
            print(f"{nom:32s} {n:4d}  {100 * n / r['n_questions']:5.1f} %")
    print(f"\nor dans le pool de 50 : {r['or_dans_le_pool']}/{r['n_questions']} · "
          f"rang médian quand trouvé : {r['rang_median_quand_trouve']}")
    o, t = r["chunks_d_or"], r["temoin_corpus_entier"]
    print(f"\n{'':38s} {'chunks d or':>12s} {'corpus entier':>14s}")
    for cle, libelle in (("longueur_mediane", "longueur médiane (caractères)"),
                         ("part_latex_mediane", "part de LaTeX (médiane)"),
                         ("commandes_pour_mille_mediane", "commandes LaTeX / 1 000 car."),
                         ("part_avec_lettres_espacees", "part à lettres espacées")):
        print(f"{libelle:38s} {o[cle]:>12} {t[cle]:>14}")
    print(f"\n{r['ce_que_ce_tableau_n_autorise_pas']}")


if __name__ == "__main__":
    imprimer(mesurer())
