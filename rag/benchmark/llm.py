"""Client Mistral : le seul point du banc d'essai qui sorte de la machine.

Contraintes réelles du palier gratuit, mesurées le 1er septembre 2026 :

    mistral-medium-latest   50 req/min    25 000 tokens/min   <- juge
    mistral-small-latest    50 req/min    50 000 tokens/min   <- génération
    ministral-8b-latest    188 req/min   625 000 tokens/min
    mistral-large-latest    403 tier_not_allowed

Le plafond de 25 k tokens/min sur *medium* est la vraie contrainte : c'est
~10 appels de jugement par minute. Trois conséquences, toutes traitées ici :

  1. **cache disque obligatoire.** Chaque appel est mis en cache sur le hash de la
     requête complète. Une exécution interrompue reprend là où elle s'est arrêtée,
     et rejouer le banc d'essai après un changement de configuration ne repaie que
     ce qui a changé. Sans ça, le pipeline n'est pas exploitable.
  2. **régulateur de débit local.** On ne découvre pas le plafond en prenant un 429,
     on s'arrête avant. Fenêtre glissante sur les requêtes *et* sur les tokens.
  3. **reprise sur erreur.** Le palier gratuit renvoie régulièrement des 503
     « high load », parfois plusieurs minutes d'affilée. Backoff exponentiel
     (jusqu'à 10 tentatives, plafond 60 s, soit ~5 min de patience par appel)
     avec respect de ``Retry-After``.

     Ce n'est pas suffisant en soi : une indisponibilité plus longue que ça
     existe. C'est pourquoi ``run_benchmark.py`` **n'abandonne pas** sur un appel
     en échec — il marque la ligne et poursuit. Comme tout le reste est en cache,
     relancer le script reprend uniquement les lignes fautives. Un banc d'essai
     de 45 minutes qui meurt sur un 503 à la 40ᵉ n'est pas utilisable.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import httpx
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache" / "llm"
KEY_FILE = HERE / ".mistral-key"
ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

#: Délais séparés par phase, pas un seul délai global. Symptôme observé : sous
#: charge, l'API ferme sa moitié de connexion sans rien envoyer ; le socket reste
#: en CLOSE_WAIT et le client attend le délai *entier* avant d'abandonner. Avec un
#: délai global de 180 s et dix tentatives, un seul appel pouvait bloquer la
#: campagne une demi-heure. Un `read` de 75 s suffit largement pour 450 tokens de
#: complétion : au-delà, la connexion est morte, il faut réessayer, pas patienter.
TIMEOUT = httpx.Timeout(connect=10.0, read=75.0, write=30.0, pool=10.0)
#: Sur une API gratuite, une requête qui pend coûte plus cher qu'une requête qui échoue :
#: 75 s de lecture puis dix tentatives à attente doublante, c'est dix minutes perdues sur
#: un seul appel. Ces valeurs-là font échouer vite, et l'échec est visible (run_benchmark
#: liste les appels perdus). Réglables par l'environnement pour ne rien figer.
GOOGLE_TIMEOUT = httpx.Timeout(connect=10.0, read=float(os.environ.get("QUANT_RAG_LLM_READ", 30)),
                               write=30.0, pool=10.0)
GOOGLE_ATTEMPTS = int(os.environ.get("QUANT_RAG_LLM_ATTEMPTS", 3))
GOOGLE_DEADLINE = float(os.environ.get("QUANT_RAG_LLM_DEADLINE", 40))

#: (requêtes/min, tokens/min). Marge de 10 % pour absorber l'écart entre notre
#: estimation de tokens et la facturation réelle.
LIMITS = {
    "mistral-medium-latest": (45, 22_000),
    "mistral-small-latest": (45, 45_000),
    "ministral-8b-latest": (150, 500_000),
    # Gemini, palier gratuit : plafonds prudents. La latence mesurée (3,5 s pour
    # flash-lite, 20 s pour flash) domine de toute façon le débit.
    "gemini-3.1-flash-lite": (12, 240_000),
    "gemini-3.5-flash": (8, 240_000),
    "gemini-3.6-flash": (8, 240_000),
    "gemini-3.7-flash": (8, 240_000),
    "gemini-3.8-flash": (8, 240_000),
}
DEFAULT_LIMIT = (30, 20_000)

#: Rôles par défaut. Le générateur de réponses et le juge sont volontairement des
#: modèles *différents* : un modèle qui note sa propre production se préfère
#: lui-même. Même famille (contrainte du palier gratuit), tailles différentes —
#: le biais résiduel est mesuré, pas supposé (voir judge.py, items-témoins).
#: Surchargeables par l'environnement — les valeurs par défaut sont l'instrument de
#: référence, et tout écart doit être visible dans le fichier de résultats (``models``).
#: ``pipeline.answer`` et ``judge.grade`` lisent ces noms **au chargement du module** :
#: les changer après import n'aurait aucun effet, d'où le passage par l'environnement.
GENERATOR = os.environ.get("QUANT_RAG_GENERATOR", "mistral-small-latest")   # écrit les réponses évaluées
AUTHOR = os.environ.get("QUANT_RAG_AUTHOR", "mistral-small-latest")        # rédige et blanchit les questions
JUDGE = os.environ.get("QUANT_RAG_JUDGE", "mistral-medium-latest")         # note, plus fort que le générateur

_stats = {"calls": 0, "cached": 0, "retries": 0, "prompt_tokens": 0, "completion_tokens": 0,
          "cout_eur": 0.0, "par_modele": {}}
_lock = threading.Lock()


# ------------------------------------------------------------------- coût réellement engagé

#: Prix publics relevés et datés. Sans ce fichier, ce module compte des jetons et ne sait rien
#: dire du budget : le dépôt n'avait qu'une estimation isolée, « ~0,001 € par appel
#: mistral-small-latest » (``TODOLIST-AMELIORATION.md`` l. 289), ni implémentée ni sourcée.
PRIX_MODELES = HERE / "prix-modeles.json"

#: Plafond de dépense du processus, en euros. ``None`` = pas de plafond.
#:
#: Réglé par la variable d'environnement ``QUANT_RAG_BUDGET_MAX_EUR``, et **c'est délibérément
#: là et non dans un argument de ligne de commande** : le plafond protège alors *tous* les
#: scripts du banc, y compris ceux qui appartiennent à d'autres chantiers et qu'un chantier de
#: contrat de sortie n'a pas à modifier. Un plafond qu'il faut penser à passer à chaque script
#: est un plafond qu'on oublie.
#:
#:     QUANT_RAG_BUDGET_MAX_EUR=2 .venv/bin/python rag/benchmark/run_benchmark.py …
#:
#: Un appelant peut aussi le régler directement (``llm.BUDGET_MAX_EUR = 0.5``).
BUDGET_MAX_EUR: float | None = (float(os.environ["QUANT_RAG_BUDGET_MAX_EUR"])
                                if os.environ.get("QUANT_RAG_BUDGET_MAX_EUR") else None)

_prix: dict = {}


class BudgetDepasse(RuntimeError):
    """Levée **avant** l'appel, jamais après : le cache n'est jamais laissé à moitié écrit."""


def prix() -> dict:
    """Le barème, ou ``{}`` s'il n'a pas été relevé. Un barème absent n'arrête rien."""
    if _prix:
        return _prix
    if not PRIX_MODELES.exists():
        return {}
    try:
        _prix.update(json.loads(PRIX_MODELES.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return {}
    return _prix


def cout_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Le coût d'un appel, en dollars. ``None`` quand le barème ne permet pas de le dire.

    ``None`` et ``0.0`` ne veulent pas dire la même chose et ne doivent jamais être confondus :
    le premier dit « je ne sais pas », le second « c'était gratuit ». Un budget calculé en
    prenant l'un pour l'autre sous-estime la dépense sans le dire.

    Un modèle **présent** au barème mais privé de ``entree`` ou ``sortie`` rend ``None`` lui
    aussi. C'est le cas qui a coûté cher : l'ancien ``banc_v4.chiffrer`` lisait des clés
    absentes avec un défaut à ``0.0`` et publiait un coût nul en annonçant ``tarif_connu``.
    """
    tarif = (prix().get("modeles") or {}).get(model)
    if not tarif or "entree" not in tarif or "sortie" not in tarif:
        return None
    return (prompt_tokens * tarif["entree"] + completion_tokens * tarif["sortie"]) / 1_000_000


def cout_eur(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Le coût d'un appel, en euros, au taux daté du barème. ``None`` si le coût est inconnu."""
    usd = cout_usd(model, prompt_tokens, completion_tokens)
    if usd is None:
        return None
    taux = (prix().get("taux_usd_par_eur") or {}).get("valeur")
    return usd / taux if taux else None


def _comptabiliser(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Un seul endroit où les compteurs bougent, pour les trois chemins d'appel."""
    with _lock:
        _stats["calls"] += 1
        _stats["prompt_tokens"] += prompt_tokens
        _stats["completion_tokens"] += completion_tokens
        cout = cout_eur(model, prompt_tokens, completion_tokens)
        entree = _stats["par_modele"].setdefault(
            model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "cout_eur": 0.0, "tarif_connu": cout is not None})
        entree["calls"] += 1
        entree["prompt_tokens"] += prompt_tokens
        entree["completion_tokens"] += completion_tokens
        if cout is not None:
            entree["cout_eur"] += cout
            _stats["cout_eur"] += cout


def _verifier_budget(model: str, messages: list[dict], max_tokens: int) -> None:
    """Refuse un appel qui ferait dépasser le plafond — **avant** de le faire.

    L'estimation du coût à venir est majorante : on suppose ``max_tokens`` en sortie, ce que
    l'appel ne consommera probablement pas. Un plafond qui laisse passer l'appel de trop parce
    qu'il a estimé au plus juste ne protège rien.
    """
    if BUDGET_MAX_EUR is None:
        return
    a_venir = cout_eur(model, _estimate_tokens(messages, 0), max_tokens) or 0.0
    if _stats["cout_eur"] + a_venir > BUDGET_MAX_EUR:
        raise BudgetDepasse(
            f"plafond de {BUDGET_MAX_EUR:.2f} € atteint : {_stats['cout_eur']:.4f} € engagés, "
            f"l'appel suivant à {model} coûterait au plus {a_venir:.4f} €. "
            "Aucun appel n'a été fait, le cache est intact.")


def stats() -> dict:
    resume = dict(_stats)
    resume["par_modele"] = {m: dict(v) for m, v in _stats["par_modele"].items()}
    resume["cout_eur"] = round(_stats["cout_eur"], 6)
    resume["bareme"] = {"fichier": PRIX_MODELES.name, "present": bool(prix()),
                        "releve_le": prix().get("releve_le")}
    return resume


def tracabilite_run(fenetre: int | None = None) -> dict:
    """Le bloc que **tout** ``results-*.json`` doit embarquer : appels, jetons, coût, modèles.

    Un fichier de résultats qui ne dit pas ce qu'il a coûté ni avec quels modèles il a été
    produit ne peut ni être budgété après coup, ni être comparé à un autre. À poser tel quel à
    la racine du fichier écrit.
    """
    compteurs = stats()
    return {
        "appels": compteurs["calls"],
        "appels_servis_par_le_cache": compteurs["cached"],
        "jetons": {"entree": compteurs["prompt_tokens"], "sortie": compteurs["completion_tokens"]},
        "cout_eur": compteurs["cout_eur"],
        "cout_par_modele": compteurs["par_modele"],
        "bareme": compteurs["bareme"],
        "modeles": {"generateur": GENERATOR, "juge": JUDGE, "auteur": AUTHOR},
        "budget_max_eur": BUDGET_MAX_EUR,
        "fenetre": fenetre,
    }


# ---------------------------------------------------------------------- clé

def api_key() -> str:
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if key:
        return key
    if KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    sys.exit(
        "clé Mistral absente. Deux options :\n"
        "  export MISTRAL_API_KEY=...\n"
        f"  echo '...' > {KEY_FILE}   (ce fichier est gitignoré)"
    )


# ---------------------------------------------------------------------- débit

class _Pacer:
    """Fenêtre glissante d'une minute sur les requêtes et sur les tokens."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.rpm, self.tpm = requests_per_minute, tokens_per_minute
        self.events: deque = deque()  # (timestamp, tokens)
        self.lock = threading.Lock()

    def acquire(self, estimated_tokens: int) -> None:
        # Une requête plus grosse que le quota d'une minute ne pourrait jamais
        # passer : on l'écrête pour qu'elle parte seule dans sa fenêtre, plutôt
        # que de boucler indéfiniment en attendant un budget qui n'arrivera pas.
        estimated_tokens = min(estimated_tokens, self.tpm)
        while True:
            with self.lock:
                now = time.monotonic()
                while self.events and now - self.events[0][0] > 60:
                    self.events.popleft()
                used_tokens = sum(t for _, t in self.events)
                if len(self.events) < self.rpm and used_tokens + estimated_tokens <= self.tpm:
                    self.events.append((now, estimated_tokens))
                    return
                # La fenêtre se libère quand le plus ancien événement en sort.
                wait = 60 - (now - self.events[0][0]) + 0.05 if self.events else 0.05
            time.sleep(max(wait, 0.05))


_pacers: dict[str, _Pacer] = {}


def _pacer(model: str) -> _Pacer:
    with _lock:
        if model not in _pacers:
            _pacers[model] = _Pacer(*LIMITS.get(model, DEFAULT_LIMIT))
        return _pacers[model]


def _estimate_tokens(messages: list[dict], max_tokens: int) -> int:
    # ~3,5 caractères par token sur de l'anglais technique ; on majore.
    body = sum(len(m.get("content", "")) for m in messages)
    return int(body / 3.2) + max_tokens + 32


# ---------------------------------------------------------------------- appel

def _cache_path(payload: dict) -> Path:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return CACHE / f"{hashlib.sha256(blob).hexdigest()[:32]}.json"


# ------------------------------------------------------------------ second fournisseur

#: Google Gemini, second fournisseur. Il n'est **pas** le défaut : le générateur et le juge
#: du banc sont l'instrument de mesure, et en changer rend les chiffres jugés incomparables
#: à tout ce qui est enregistré (README §11, règle 3). Il sert aux usages qui ne sont pas des
#: mesures — extraction de métadonnées, contre-vérification par un modèle indépendant — et
#: à un éventuel nouveau départ de série, assumé et étiqueté comme tel.
GOOGLE_KEY_FILE = HERE / ".google-key"
GOOGLE_ENDPOINT = "https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"


def google_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if key:
        return key
    if GOOGLE_KEY_FILE.exists() and GOOGLE_KEY_FILE.read_text(encoding="utf-8").strip():
        return GOOGLE_KEY_FILE.read_text(encoding="utf-8").strip()
    sys.exit(f"clé Google absente : export GOOGLE_API_KEY=... ou {GOOGLE_KEY_FILE} (gitignoré)")


def _google_complete(messages: list[dict], model: str, temperature: float, max_tokens: int,
                     json_mode: bool, attempts: int) -> tuple[str, dict]:
    """Un appel Gemini, même contrat que le chemin Mistral : texte + usage."""
    prompt = "\n\n".join(m["content"] for m in messages)
    # Les gemini-3.x réfléchissent par défaut, et leurs jetons de réflexion se paient sur
    # `maxOutputTokens` : mesuré, 288 jetons de pensée sur un budget de 300 laissaient
    # 8 jetons de réponse et un JSON tronqué que le juge rendait inexploitable. À budget
    # nul, la même requête sort correcte et coûte 34 jetons au lieu de 311.
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens,
                                    "thinkingConfig": {"thinkingBudget": 0}}}
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    # Plafond mural, indépendant de httpx : mesuré, un serveur qui accepte la connexion
    # sans jamais répondre laisse le délai de *lecture* inactif — la requête pendait plus de
    # quatre minutes avec read=30 s. Un fil de travail borné rend la main, quoi qu'il arrive.
    def _post(url, headers, payload):
        # Pas de ``with`` : sa sortie appelle shutdown(wait=True) et rejoindrait le fil
        # resté pendu, annulant tout le bénéfice du délai. On abandonne le fil, il mourra
        # avec son propre délai de lecture.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(httpx.post, url, headers=headers, json=payload,
                               timeout=GOOGLE_TIMEOUT).result(timeout=GOOGLE_DEADLINE)
        finally:
            pool.shutdown(wait=False)

    delay, last, doubled = 2.0, "", False
    for _ in range(min(attempts, GOOGLE_ATTEMPTS) + 1):
        try:
            response = _post(GOOGLE_ENDPOINT.format(model=model),
                          {"x-goog-api-key": google_key(), "Content-Type": "application/json",
                           # Comme le chemin Mistral : sans ça, une connexion gardée en vie
                           # côté client mais fermée en face (CLOSE_WAIT) fait pendre l'appel
                           # suivant bien au-delà du délai de lecture.
                           "Connection": "close"},
                          payload)
        except (httpx.HTTPError, FuturesTimeout) as error:
            last = f"{type(error).__name__}: {error}"
        else:
            if response.status_code == 200:
                body = response.json()
                candidate = (body.get("candidates") or [{}])[0]
                parts = candidate.get("content", {}).get("parts", [])
                usage = body.get("usageMetadata") or {}
                text = "".join(p.get("text", "") for p in parts)
                # Une réponse coupée n'est pas une réponse : sans ce contrôle, le juge
                # recevait « {\n  "grounded » et le banc rapportait une couverture vide
                # sans que rien n'ait signalé d'erreur.
                # Une réponse coupée par le budget est réessayée une fois avec le double,
                # à l'intérieur de l'appel : la clé de cache reste celle du budget demandé,
                # donc rien de ce qui a déjà abouti n'est invalidé.
                if candidate.get("finishReason") == "MAX_TOKENS" and not doubled:
                    doubled = True
                    payload["generationConfig"]["maxOutputTokens"] = max_tokens * 3
                    continue
                if candidate.get("finishReason") not in (None, "STOP"):
                    raise RuntimeError(
                        f"réponse Google interrompue ({candidate.get('finishReason')}) après "
                        f"{usage.get('candidatesTokenCount', 0)} jetons"
                        + (f", dont {usage['thoughtsTokenCount']} de réflexion"
                           if usage.get("thoughtsTokenCount") else "")
                        + f" — augmenter max_tokens (ici {max_tokens})")
                return text, {
                    "prompt_tokens": usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0)}
            last = f"HTTP {response.status_code}: {response.text[:200]}"
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise RuntimeError(f"appel Google refusé — {last}")
        with _lock:
            _stats["retries"] += 1
        time.sleep(delay + random.uniform(0, delay / 2))
        delay = min(delay * 2, 60.0)
    raise RuntimeError(f"appel Google en échec après {min(attempts, GOOGLE_ATTEMPTS)} tentatives — {last}")


def provider(model: str) -> str:
    return "google" if model.startswith("gemini") else "mistral"


def complete(messages: list[dict], model: str = GENERATOR, temperature: float = 0.0,
             max_tokens: int = 1024, json_mode: bool = False, cache: bool = True,
             attempts: int = 10, seed: int | None = None) -> str:
    """Un appel de complétion, mis en cache, régulé et réessayé.

    ``seed`` n'est pas transmis à l'API : il n'entre que dans la clé de cache, ce
    qui permet d'obtenir plusieurs tirages indépendants d'un même prompt à
    température non nulle (utilisé pour mesurer le bruit du juge).
    """
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    path = _cache_path({**payload, "_seed": seed})
    if cache and path.exists():
        with _lock:
            _stats["cached"] += 1
        return json.loads(path.read_text(encoding="utf-8"))["content"]

    if provider(model) == "google":
        # Le régulateur s'applique aussi ici : sans lui, le banc partait en rafale, prenait
        # des 429 et passait son temps en attente exponentielle — c'est ce qui a fait durer
        # la notation indéfiniment. Espacer coûte moins cher que réessayer.
        _verifier_budget(model, messages, max_tokens)
        _pacer(model).acquire(_estimate_tokens(messages, max_tokens))
        content, usage = _google_complete(messages, model, temperature, max_tokens, json_mode, attempts)
        _comptabiliser(model, usage["prompt_tokens"], usage["completion_tokens"])
        if cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": model, "content": content, "usage": usage},
                                       ensure_ascii=False), encoding="utf-8")
        return content

    _verifier_budget(model, messages, max_tokens)
    headers = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}
    delay = 2.0
    last = ""
    for attempt in range(attempts):
        _pacer(model).acquire(_estimate_tokens(messages, max_tokens))
        try:
            response = httpx.post(ENDPOINT, headers={**headers, "Connection": "close"},
                                  json=payload, timeout=TIMEOUT)
        except httpx.HTTPError as error:
            last = f"{type(error).__name__}: {error}"
            response = None

        if response is not None and response.status_code == 200:
            body = response.json()
            content = body["choices"][0]["message"]["content"] or ""
            usage = body.get("usage") or {}
            _comptabiliser(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            if cache:
                CACHE.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"model": model, "content": content, "usage": usage},
                                           ensure_ascii=False), encoding="utf-8")
            return content

        if response is not None:
            last = f"HTTP {response.status_code}: {response.text[:200]}"
            # 4xx hors 429 : la requête est fautive, réessayer ne sert à rien.
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise RuntimeError(f"appel Mistral refusé — {last}")
            # Quota du *compte* à zéro : réessayer ne sert à rien, et dix tentatives avec
            # un plafond de 60 s font passer un refus immédiat pour un blocage de plusieurs
            # minutes — c'est ce qui a fait croire à un import pendu le 4 septembre 2026.
            if response.status_code == 429 and response.headers.get("x-ratelimit-limit-req-minute") == "0":
                raise RuntimeError(
                    "quota Mistral à zéro sur ce compte (x-ratelimit-limit-req-minute: 0). "
                    "Ce n'est pas une limite passagère et changer de clé n'y change rien : "
                    "le plafond est porté par le compte. Vérifier l'abonnement sur "
                    "console.mistral.ai. La clé elle-même est valide si /v1/models répond 200.")
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = max(delay, float(retry_after))

        with _lock:
            _stats["retries"] += 1
        time.sleep(delay + random.uniform(0, delay / 2))
        delay = min(delay * 2, 60.0)

    raise RuntimeError(f"appel Mistral en échec après {attempts} tentatives — {last}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str):
    text = text.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Dernier recours : le plus grand objet ou tableau bien formé du texte.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def complete_json(messages: list[dict], model: str = GENERATOR, temperature: float = 0.0,
                  max_tokens: int = 1024, cache: bool = True, seed: int | None = None,
                  repairs: int = 2):
    """Complétion en mode JSON, avec relance si la sortie n'est pas analysable.

    Les relances portent une graine différente afin de ne pas retomber sur la
    réponse fautive mise en cache.
    """
    for attempt in range(repairs + 1):
        raw = complete(messages, model=model, temperature=temperature, max_tokens=max_tokens,
                       json_mode=True, cache=cache,
                       seed=seed if attempt == 0 else f"{seed}-repair{attempt}")
        parsed = _extract_json(raw)
        if parsed is not None:
            return parsed
        temperature = max(temperature, 0.3)  # sortir de l'ornière déterministe
    return None


#: Un modèle par fournisseur, le moins cher de chacun. La sonde sert à connaître l'état d'un
#: quota, pas à comparer des modèles : elle ne doit jamais coûter plus d'un appel par
#: fournisseur, et elle ne doit jamais être bloquée par la panne de l'autre.
SONDES = ("mistral-small-latest", "gemini-3.1-flash-lite")


def sonder(modeles=SONDES) -> dict:
    """État de chaque fournisseur, un appel minimal chacun, l'échec de l'un n'arrête pas l'autre.

    Le défaut réparé le 8 septembre 2026 : ce point d'entrée **affichait les six premiers
    caractères de la clé Mistral**, contre la règle du dépôt, et il s'arrêtait au premier
    fournisseur en panne — donc, quota Mistral à zéro, il n'avait jamais pu dire quoi que ce
    soit de Google. Il ne reste ici de la clé que sa présence et sa longueur.
    """
    out = {}
    for model in modeles:
        nom = provider(model)
        started = time.perf_counter()
        try:
            reponse = complete([{"role": "user", "content": "Reply with exactly: OK"}],
                               model=model, max_tokens=8, cache=False, temperature=0.0)
            out[model] = {"fournisseur": nom, "etat": "ouvert", "reponse": reponse[:20],
                          "secondes": round(time.perf_counter() - started, 1)}
        except Exception as erreur:                       # noqa: BLE001 — on rapporte, on ne relaie pas
            out[model] = {"fournisseur": nom, "etat": "fermé",
                          "cause": f"{type(erreur).__name__}: {erreur}"[:300],
                          "secondes": round(time.perf_counter() - started, 1)}
    return out


if __name__ == "__main__":
    import argparse

    analyse = argparse.ArgumentParser(description="Sonde de quota — un appel minimal par fournisseur.")
    analyse.add_argument("--modeles", nargs="*", default=list(SONDES))
    arguments = analyse.parse_args()
    for source, lecture in (("mistral", api_key), ("google", google_key)):
        try:
            longueur = len(lecture())
            print(f"clé {source:8s} : présente ({longueur} caractères)")
        except Exception as erreur:                       # noqa: BLE001
            print(f"clé {source:8s} : absente — {type(erreur).__name__}")
    for model, etat in sonder(arguments.modeles).items():
        marque = "OK " if etat["etat"] == "ouvert" else "HS "
        detail = etat.get("reponse") or etat.get("cause")
        print(f"  {marque}{model:24s} ({etat['fournisseur']:8s}, {etat['secondes']:4.1f} s) -> {detail!r}")
    print("stats :", stats())
