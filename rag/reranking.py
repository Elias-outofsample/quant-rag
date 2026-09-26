"""Le reclassement d'un pool déjà récupéré — un seul code, pour le banc et pour la production.

Ce module existe parce que le modèle qui a gagné ne pouvait pas tourner dans le chemin servi.
``quant_rag._rerank`` instancie un ``AutoModelForSequenceClassification`` : c'est le contrat
des cross-encodeurs ``bge-*``. ``Qwen3-Reranker-0.6B`` est un modèle **causal** dont le score
est ``P("yes")`` sur le dernier jeton — il ne pouvait donc pas être servi, seulement mesuré au
banc. Les deux familles vivent ici, et le banc comme la production importent le même code : un
code mesuré et un code servi qui divergent, c'est une mesure qui ne dit rien du produit.

**La règle servie est ``gel1``**, et elle n'est pas le reclassement plein.

    budget 10        seuls les dix premiers candidats du pool sont reclassés ; la queue garde
                     son ordre dense. Mesuré : à budget 10, R@10 est *par construction* celui
                     du dense — on ne peut rien perdre au-delà du rang 10.
    rang 1 protégé   le premier candidat dense n'est jamais déclassé.

Pourquoi cette règle et pas le reclassement plein (`rag/benchmark/PRE-ENREGISTREMENT-RECLASSEMENT-SELECTIF-2026-09-07.md`) :
sur 155 questions, le reclassement plein fait entrer l'or dans les cinq passages servis pour
21 questions et l'en fait sortir pour 7. ``gel1`` en fait entrer 19 et sortir **5**, et ses
cinq pertes sont un **sous-ensemble strict** des sept — elle ne troque pas une perte contre
une autre, elle en retire deux. Le dégât du reclassement est concentré sur une seule position,
et protéger cette position seule le réduit sans coûter les gains.

Coût mesuré, sur ce M4 16 Go, fp16 sur ``mps`` :

    latence          **3,18 s** par requête à chaud, contre **59 ms** au chemin dense —
                     un facteur **54**. Modèle non résident : +2,18 s de chargement.
    mémoire          **2 365 Mo résidents** une fois chargé. La machine dispose réellement de
                     ~5,9 Go (libres + inactives + spéculatives) mais tourne déjà avec 2,4 Go
                     de swap : le modèle ne tue pas le serveur, il le pousse vers le swap —
                     et ce dépôt a déjà mesuré ce que le swap fait à cette latence précise
                     (28,8 s au lieu de 17,6 s, le 3 septembre 2026).

D'où le plancher mémoire : le modèle reste résident tant que la machine a de la marge, et il
est **relâché en fin d'appel** quand elle n'en a plus. La latence devient bimodale sous
pression — 3,18 s à chaud, ~5,4 s après une libération — et c'est écrit plutôt que subi.

**Aucun échec de reclassement ne casse une recherche.** Modèle absent, chargement impossible,
scoring en erreur : on rend l'ordre dense, et la décision de routage le dit.
"""
from __future__ import annotations

import subprocess
import time

#: Le modèle servi par le mode « sélectif ». Le défaut historique de ``quant_rag`` reste
#: ``bge-reranker-base`` et n'est pas touché : il porte le mode ``hybrid``, et il a été
#: mesuré **nuisible** sur un pool dense (−0,076 nDCG@10).
MODELE = "Qwen/Qwen3-Reranker-0.6B"
BUDGET = 10
PROTEGES = 1
MAX_LENGTH = 512
TEXTE_CARACTERES = 2000
LOT = 8
INSTRUCTION = "Given a quantitative-finance research question, judge whether the passage answers it."
#: En dessous de ce seuil de mémoire réellement disponible, le modèle est relâché en fin
#: d'appel. 1,5 Go laisse la place au modèle lui-même (2,4 Go) plus une marge d'inférence.
PLANCHER_MO = 1500

_modele = None
_dernier_id = None


# --------------------------------------------------------------------------- mémoire

def memoire_disponible_mo() -> float | None:
    """Libres + inactives + spéculatives — **pas** « Pages free » seul.

    Sur macOS, les pages inactives sont récupérables : lire « Pages free » comme la mémoire
    disponible fait voir une famine là où il reste des gigaoctets. Cette fonction existe
    parce que l'erreur a été commise en écrivant ce module.
    """
    try:
        sortie = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return None
    taille, total = 16384, 0
    for ligne in sortie.splitlines():
        if "page size of" in ligne:
            taille = int(ligne.split("page size of")[1].split("bytes")[0].strip())
        for cle in ("Pages free", "Pages inactive", "Pages speculative"):
            if ligne.startswith(cle):
                total += int(ligne.split(":")[1].strip().rstrip("."))
    return total * taille / 1_048_576


# --------------------------------------------------------------------------- les deux familles

class CrossEncodeur:
    """``AutoModelForSequenceClassification`` à un logit — la famille ``bge-*``."""

    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=torch.float16 if device == "mps" else torch.float32).to(device).eval()

    def scorer(self, requete: str, textes: list[str]) -> list[float]:
        import torch

        sortie: list[float] = []
        with torch.inference_mode():
            for debut in range(0, len(textes), LOT):
                paires = [[requete, t] for t in textes[debut:debut + LOT]]
                entrees = self.tok(paires, padding=True, truncation=True,
                                   max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
                sortie.extend(self.model(**entrees).logits.view(-1).float().cpu().tolist())
        return sortie


class Causal:
    """``Qwen3-Reranker`` — modèle causal, score = ``P("yes")`` sur le dernier jeton.

    ``logits_to_keep=1`` n'est pas une optimisation facultative : sans lui, ``forward``
    matérialise ``[lot, longueur, vocabulaire]`` — 1,5 Go en fp16 par lot — et le 5 septembre
    2026 cela a mis la machine à 11,1 Go de swap et fait passer la latence de 17 s à 49 s.
    Le paramètre ne change aucun score : ``eval_rerank_latency.py`` exige des classements
    identiques avant et après, et les obtient 8 fois sur 8.
    """

    PREFIXE = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
               "the Query and the Instruct provided. Note that the answer can only be \"yes\" or "
               "\"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIXE = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16 if device == "mps" else torch.float32).to(device).eval()
        self.oui = self.tok.convert_tokens_to_ids("yes")
        self.non = self.tok.convert_tokens_to_ids("no")
        self.prefixe = self.tok.encode(self.PREFIXE, add_special_tokens=False)
        self.suffixe = self.tok.encode(self.SUFFIXE, add_special_tokens=False)

    def scorer(self, requete: str, textes: list[str]) -> list[float]:
        import torch

        sortie: list[float] = []
        budget = MAX_LENGTH + 128 - len(self.prefixe) - len(self.suffixe)
        with torch.inference_mode():
            for debut in range(0, len(textes), LOT):
                corps = [f"<Instruct>: {INSTRUCTION}\n<Query>: {requete}\n<Document>: {t}"
                         for t in textes[debut:debut + LOT]]
                encodes = self.tok(corps, padding=False, truncation=True, max_length=budget,
                                   add_special_tokens=False)["input_ids"]
                sequences = [self.prefixe + ids + self.suffixe for ids in encodes]
                lot = self.tok.pad({"input_ids": sequences}, padding=True,
                                   return_tensors="pt").to(self.device)
                logits = self.model(**lot, logits_to_keep=1).logits[:, -1, :]
                paire = torch.stack([logits[:, self.non], logits[:, self.oui]], dim=1).float()
                sortie.extend(torch.log_softmax(paire, dim=1)[:, 1].exp().cpu().tolist())
        return sortie


def _charger(model_id: str, device: str):
    return Causal(model_id, device) if "Reranker" in model_id and "Qwen" in model_id \
        else CrossEncodeur(model_id, device)


def obtenir(model_id: str, device: str):
    """Le modèle, chargé une fois et gardé tant que la machine a de la marge."""
    global _modele, _dernier_id
    if _modele is None or _dernier_id != model_id:
        liberer()
        _modele = _charger(model_id, device)
        _dernier_id = model_id
    return _modele


def liberer() -> None:
    """Relâche le modèle. ``del`` + ``empty_cache`` rend bien la mémoire (mesuré : +302 Mo)."""
    global _modele, _dernier_id
    if _modele is None:
        return
    _modele = None
    _dernier_id = None
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- la règle

def reordonner(rows: list[dict], scores: dict[str, float], budget: int = BUDGET,
               proteges: int = PROTEGES) -> list[dict]:
    """``gel1`` : les ``proteges`` premiers rangs dense sont figés, la tête est reclassée.

    Le contrat de sortie **n'enlève rien** et ajoute quatre champs : ``score_dense`` et
    ``rang_dense`` (que ``_rerank`` écrasait), ``rerank_score``, ``rang_final``, ``reclasse``.

    Le départage est **explicite** — ``(-score, rang_dense, chunk_id)``. ``quant_rag._rerank``
    s'en remet à la stabilité du tri de Python : c'est déterministe par accident du langage,
    pas par contrat.
    """
    marques = []
    for rang, row in enumerate(rows, 1):
        marques.append({**row, "score_dense": row.get("score_dense", row.get("score")),
                        "rang_dense": row.get("rang_dense", rang)})
    tete, queue = marques[:budget], marques[budget:]
    fige, mobile = tete[:proteges], tete[proteges:]
    classee = sorted(mobile, key=lambda r: (-scores.get(r["chunk_id"], float("-inf")),
                                            r["rang_dense"], r["chunk_id"]))
    sortie = []
    for position, row in enumerate(fige + classee + queue, 1):
        reclasse = position <= len(tete)
        sortie.append({**row, "rang_final": position, "reclasse": reclasse,
                       "rerank_score": scores.get(row["chunk_id"]) if reclasse else None})
    return sortie


def reclasser(requete: str, rows: list[dict], device: str, model_id: str = MODELE,
              budget: int = BUDGET, proteges: int = PROTEGES) -> tuple[list[dict], dict]:
    """Reclasse la tête du pool. **Ne lève jamais** : en cas d'échec, rend l'ordre dense.

    Rend ``(lignes, trace)``. La trace dit ce qui s'est passé — modèle, budget, latence,
    échec éventuel, libération éventuelle — et elle est faite pour entrer au journal de
    décision : un reclassement qu'on ne peut pas auditer après coup ne se diagnostique pas.
    """
    trace = {"modele": model_id, "budget": budget, "proteges": proteges,
             "echec": None, "libere": False}
    debut = time.perf_counter()
    candidats = [r for r in rows[:budget] if r.get("text")]
    if not candidats:
        trace["echec"] = "aucun candidat avec texte"
        return rows, trace
    try:
        scorer = obtenir(model_id, device)
        valeurs = scorer.scorer(requete, [r["text"][:TEXTE_CARACTERES] for r in candidats])
        scores = {r["chunk_id"]: float(v) for r, v in zip(candidats, valeurs)}
    except Exception as erreur:  # noqa: BLE001
        # Un reclassement qui échoue rend l'ordre dense. Il ne rend jamais une erreur :
        # transformer une lenteur locale en panne générale du RAG coûterait bien plus
        # que les quelques questions qu'il fait gagner.
        trace["echec"] = f"{type(erreur).__name__}: {erreur}"[:200]
        liberer()
        return rows, trace
    sortie = reordonner(rows, scores, budget, proteges)
    trace["latence_ms"] = round((time.perf_counter() - debut) * 1000)
    dispo = memoire_disponible_mo()
    trace["memoire_disponible_mo"] = round(dispo) if dispo is not None else None
    if dispo is not None and dispo < PLANCHER_MO:
        liberer()
        trace["libere"] = True
    return sortie, trace
