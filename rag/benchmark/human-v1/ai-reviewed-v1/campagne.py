"""Campagne AI-reviewed-v1 — briefings des agents, puis assemblage de la campagne.

**Ce que cette campagne est, et ce qu'elle n'est pas.** Un gold `gold_ai_reviewed` est un fait
validé par une revue contradictoire de modèles, tracée et reproductible. Il peut compter dans
une campagne `multi_agent_ai`, servir à comparer des variantes et détecter une régression. Il
**n'est pas** un fait validé par une personne, et aucun document de cette campagne ne doit le
présenter ainsi. Le validateur rend la confusion impossible : `COMPTE_PAR_MODE` associe un
statut à un mode, et le défaut est le mode humain.

**Pourquoi deux briefings par item.** Un agent à qui l'on montre d'emblée la réponse proposée
la critique ; un agent à qui l'on montre d'abord la question la *construit*, puis la compare.
Les deux ne trouvent pas les mêmes défauts. `briefings/sources/` porte la question et les
extraits sans la proposition ; `briefings/propositions/` porte la proposition. L'agent D lit
le premier, écrit son attente, et n'ouvre le second qu'ensuite.

**Ce que l'indépendance vaut ici, exactement.** Les agents ne voient pas les rapports les uns
des autres — c'est vrai et vérifiable, ils sont lancés avant toute agrégation. Ils sont en
revanche tous de la famille Claude : leurs erreurs sont **corrélées**, et aucune formulation
de ce dossier ne doit laisser croire le contraire. Pour l'agent D, la séquence « question
d'abord, proposition ensuite » est tenue **par instruction dans un même agent**, pas par une
barrière technique.

    .venv/bin/python rag/benchmark/human-v1/ai-reviewed-v1/campagne.py --briefings
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUMAN = HERE.parent
BENCHMARK = HUMAN.parent
ROOT = BENCHMARK.parents[1]
sys.path.insert(0, str(HUMAN))
sys.path.insert(0, str(BENCHMARK))

LOT = HUMAN / "items-cohorte1-v0.jsonl"
METADONNEES = ROOT / "rag" / "metadata" / "documents-metadata-v1.json"
SOURCES = HERE / "briefings" / "sources"
PROPOSITIONS = HERE / "briefings" / "propositions"
RAPPORTS = HERE / "rapports"

VERSION_CAMPAGNE = 1
REVIEW_MODE = "multi_agent_ai"


def fiches_documents() -> dict[str, dict]:
    donnees = json.loads(METADONNEES.read_text(encoding="utf-8"))
    donnees = donnees if isinstance(donnees, list) else donnees.get("documents", [])
    return {d["document_id"]: d for d in donnees}


def reference(fiche: dict) -> str:
    """« A & B — titre (année) ». Deux auteurs se citent tous les deux.

    Défaut trouvé par la revue : la version précédente n'imprimait `auteurs[0]` et n'ajoutait
    « et al. » qu'au-delà de deux — donc « Fama » seul pour Fama & French, « Chen » seul pour
    Chen & Velikov. Une citation qui perd un auteur est fausse, et deux agents l'ont relevée
    indépendamment.
    """
    auteurs = fiche.get("authors") or []
    if not auteurs:
        qui = "auteur inconnu"
    elif len(auteurs) == 1:
        qui = auteurs[0]
    elif len(auteurs) == 2:
        qui = f"{auteurs[0]} & {auteurs[1]}"
    else:
        qui = f"{auteurs[0]} et al."
    return f"{qui} — « {(fiche.get('title') or 'sans titre').strip()} » ({fiche.get('publication_year') or 's.d.'})"


def pages(appui: dict) -> str:
    couvertes = appui.get("pages_couvertes") or ([appui["page"]] if appui.get("page") else [])
    if len(couvertes) > 1:
        return f"p. {min(couvertes)}–{max(couvertes)}"
    return f"p. {couvertes[0]}" if couvertes else "page inconnue"


def charger() -> list[dict]:
    return [json.loads(l) for l in LOT.read_text(encoding="utf-8").splitlines() if l.strip()]


def briefing_sources(item: dict, docs: dict) -> str:
    lignes = [
        f"# {item['id']} — sources et question",
        "",
        "> Ce document ne contient **pas** la réponse proposée. Il porte la question et les "
        "extraits sources sur lesquels elle a été construite.",
        "",
        f"**Corpus** `5530cba145` · **famille** `{item['famille']}` · **domaine** "
        f"`{item['domaine']}` · **répondable** `{item['repondable']}`",
        "",
        "## La question",
        "",
        f"> {item['question']}",
        "",
    ]
    if item.get("couverture_partielle"):
        cp = item["couverture_partielle"]
        lignes += ["## Couverture déclarée par le lot", "",
                   f"- **couvert** : {cp.get('couvert')}",
                   f"- **non couvert** : {cp.get('non_couvert')}",
                   f"- **vérification déclarée** : {cp.get('preuve')}", ""]
    if item.get("absence"):
        a = item["absence"]
        lignes += ["## Absence déclarée par le lot", "",
                   f"- **porte sur** : {a.get('porte')}",
                   f"- **preuve** : {a.get('preuve')}",
                   f"- **force** : {a.get('force_de_la_preuve')}", "",
                   "Recherche sémantique enregistrée :", ""]
        for essai in a.get("recherche_semantique") or []:
            lecture = f" — {essai['lecture']}" if essai.get("lecture") else ""
            lignes.append(f"- `{essai['variante']}` : **{essai['occurrences']}** occurrence(s)"
                          f"{lecture}")
        lignes.append("")
    if item["appuis"]:
        lignes += ["## Les extraits sources", ""]
        for a in item["appuis"]:
            fiche = docs.get(a["document_id"], {})
            o = a.get("offsets") or {}
            lignes += [
                f"### {a['ancre_id']} — {a['role']}",
                "",
                f"- **référence** : {reference(fiche)}",
                f"- **document** : `{a['document_id']}` · {pages(a)}",
                f"- **offsets** dans le texte canonique : `{o.get('debut')}`–`{o.get('fin')}` · "
                f"empreinte `{(o.get('doc_text_sha256') or '')[:16]}…`",
                f"- **type de contenu** : `{a.get('content_type')}`",
                "",
                "Extrait ancré (texte canonique, celui que désignent les offsets) :",
                "",
                "```",
                a["texte"].strip(),
                "```",
                "",
            ]
            if a.get("rendu_servi"):
                lignes += ["Forme **servie** par le corpus (conversion markdown du même tableau) :",
                           "", "```", a["rendu_servi"].strip(), "```", ""]
    return "\n".join(lignes)


def briefing_proposition(item: dict) -> str:
    lignes = [f"# {item['id']} — la réponse proposée, à critiquer", "",
              "> Écrite par un modèle, **non validée**. Chaque fait est un `gold_candidate`.", ""]
    if item["faits_attendus"]:
        lignes += ["## Faits proposés", ""]
        for n, f in enumerate(item["faits_attendus"], 1):
            lignes.append(f"**F{n}** ({f['exigence']}, appuis {', '.join(f['ancres'])})  \n"
                          f"{f['texte']}")
            lignes.append("")
    if item.get("abstention_attendue"):
        lignes += ["## Comportement attendu", "", item["abstention_attendue"], ""]
    if item.get("conditions"):
        lignes += ["## Conditions de validité proposées", ""]
        for c in item["conditions"]:
            lignes.append(f"- ({c['consequence_si_omis']}) {c['texte']}")
        lignes.append("")
    if item.get("pieges"):
        lignes += ["## Pièges nommés par le lot", ""] + [f"- {p}" for p in item["pieges"]] + [""]
    if item.get("risques_ambiguite"):
        lignes += ["## Ambiguïtés déjà identifiées", ""] + \
                  [f"- {r}" for r in item["risques_ambiguite"]] + [""]
    cm = item.get("citation_minimale") or {}
    lignes += ["## Attente de citation", "",
               f"- documents distincts : **{cm.get('documents_distincts')}** — {cm.get('justification')}",
               ""]
    return "\n".join(lignes)


def empreinte(chemin: Path) -> str:
    return hashlib.sha256(chemin.read_bytes()).hexdigest()[:16]


def sha_depot() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True, cwd=ROOT).stdout.strip()




# --------------------------------------------------------------------------- assemblage

DECISIONS = HERE / "decisions.json"
CAMPAGNE = HERE / "items-ai-reviewed-v1.jsonl"
MANIFESTE = HERE / "MANIFESTE.json"


def assembler() -> dict:
    """Applique les décisions d'adjudication aux items, et écrit la campagne.

    **Rien n'est promu par défaut.** Un fait absent du fichier de décisions reste
    `gold_candidate` : le silence de l'adjudicateur n'est pas un accord. C'est l'inverse du
    piège habituel, où l'on promeut tout ce qui n'a pas été contesté.
    """
    if not DECISIONS.exists():
        sys.exit(f"décisions absentes : {DECISIONS}")
    decisions = json.loads(DECISIONS.read_text(encoding="utf-8"))
    items = charger()
    comptes = {"gold_ai_reviewed": 0, "gold_candidate": 0, "gold_retire": 0}
    par_item = {}

    for item in items:
        prises = decisions["items"].get(item["id"], {})
        item["statut"] = "ai_reviewed"
        item["review_mode"] = REVIEW_MODE
        item["valide_par"] = decisions["adjudicateur"]
        item["valide_le"] = decisions["date"]
        item["version_campagne"] = VERSION_CAMPAGNE
        item["gold_version"] = decisions["gold_version"]
        for champ in ("corrections_item", "desaccords_item"):
            if prises.get(champ):
                item[champ] = prises[champ]
        detail = []
        for numero, fait in enumerate(item.get("faits_attendus") or [], 1):
            prise = (prises.get("faits") or {}).get(f"F{numero}")
            if not prise or prise["statut"] == "gold_candidate":
                # Reste candidat : on conserve ce que la revue en a dit, sans le promouvoir.
                if prise:
                    # `verdict_revue` distingue « examiné et refusé » de « non traité ».
                    # Sans lui, un fait réfuté par la source et un fait qu'on n'a pas eu le
                    # temps de regarder auraient le même statut, et la campagne suivante
                    # relancerait le travail déjà fait — ou pire, promouvrait le réfuté.
                    fait["revue_ai"] = {k: prise[k] for k in
                                        ("verdict_revue", "motif", "agents", "desaccords",
                                         "confiance")
                                        if k in prise}
                comptes["gold_candidate"] += 1
                detail.append({"fait": f"F{numero}", "statut": "gold_candidate",
                               "motif": (prise or {}).get("motif", "non adjudiqué")})
                continue
            fait["statut_fait"] = prise["statut"]
            fait["nature"] = prise.get("nature", "explicite")
            if fait["nature"] == "inference":
                fait["premisses"] = prise["premisses"]
                fait["hypotheses"] = prise["hypotheses"]
            if prise["statut"] == "gold_ai_reviewed":
                fait["introduit_en"] = decisions["gold_version"]
                fait["promu_par"] = decisions["adjudicateur"]
                fait["promu_le"] = decisions["date"]
                fait["motif_promotion"] = prise["motif"]
                fait["revue_source"] = prise["revue_source"]
                fait["adjudication"] = prise["adjudication"]
                fait["agents"] = prise["agents"]
                fait["desaccords"] = prise.get("desaccords", [])
                fait["confiance"] = prise.get("confiance")
            elif prise["statut"] == "gold_retire":
                fait["introduit_en"] = fait.get("introduit_en") or 1
                fait["retire_en"] = decisions["gold_version"]
                fait["retire_par"] = decisions["adjudicateur"]
                fait["motif_retrait"] = prise["motif"]
                fait["source_correction"] = prise["revue_source"]
                fait["adjudication"] = prise["adjudication"]
                fait["agents"] = prise["agents"]
            comptes[prise["statut"]] += 1
            detail.append({"fait": f"F{numero}", "statut": prise["statut"],
                           "motif": prise["motif"]})
        par_item[item["id"]] = detail

    CAMPAGNE.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
                        encoding="utf-8")
    empreintes = json.loads((HERE / "briefings" / "empreintes.json").read_text(encoding="utf-8"))
    manifeste = {
        "campagne": "ai-reviewed-v1",
        "review_mode": REVIEW_MODE,
        "avertissement": "Gold validé par une revue contradictoire de modèles. Il compte dans "
                         "une campagne multi_agent_ai et nulle part ailleurs. Ce n'est PAS un "
                         "gold validé par une personne, et aucun résultat tiré de cette "
                         "campagne ne doit être présenté comme évalué par un expert humain.",
        "version_campagne": VERSION_CAMPAGNE,
        "gold_version": decisions["gold_version"],
        "date": decisions["date"],
        "sha_depot": sha_depot(),
        "signature_corpus": "5530cba145",
        "adjudicateur": decisions["adjudicateur"],
        "agents": decisions["agents"],
        "independance": decisions["independance"],
        "regles_de_notation": decisions["regles_de_notation"],
        "briefings": empreintes["briefings"],
        "comptes": comptes,
        "decisions_par_item": par_item,
        "limites": decisions["limites"],
    }
    MANIFESTE.write_text(json.dumps(manifeste, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifeste


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--briefings", action="store_true")
    parser.add_argument("--assembler", action="store_true")
    args = parser.parse_args()
    items = charger()
    docs = fiches_documents()

    if args.briefings:
        SOURCES.mkdir(parents=True, exist_ok=True)
        PROPOSITIONS.mkdir(parents=True, exist_ok=True)
        RAPPORTS.mkdir(parents=True, exist_ok=True)
        index = {}
        for item in items:
            s = SOURCES / f"{item['id']}.md"
            p = PROPOSITIONS / f"{item['id']}.md"
            s.write_text(briefing_sources(item, docs), encoding="utf-8")
            p.write_text(briefing_proposition(item), encoding="utf-8")
            index[item["id"]] = {"sources": empreinte(s), "proposition": empreinte(p),
                                 "faits": len(item["faits_attendus"]),
                                 "ancres": len(item["appuis"])}
            print(f"  {item['id']}  sources {index[item['id']]['sources']}  "
                  f"proposition {index[item['id']]['proposition']}")
        (HERE / "briefings" / "empreintes.json").write_text(
            json.dumps({"sha_depot": sha_depot(), "signature_corpus": "5530cba145",
                        "version_campagne": VERSION_CAMPAGNE, "review_mode": REVIEW_MODE,
                        "briefings": index}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\n  {len(items)} items · briefings et empreintes écrits sous {HERE.name}/briefings/")

    if args.assembler:
        manifeste = assembler()
        print(f"  campagne {manifeste['campagne']} · gold v{manifeste['gold_version']} · "
              f"mode {manifeste['review_mode']}")
        for statut, n in manifeste["comptes"].items():
            print(f"    {statut:<20} {n}")
        print(f"  écrit  {CAMPAGNE.relative_to(ROOT)}")
        print(f"  écrit  {MANIFESTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
