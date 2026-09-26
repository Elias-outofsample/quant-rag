"""Fabrique le paquet de revue humaine de la cohorte 1 — une fiche par item, lisible seule.

**Sens de circulation, et il ne doit pas s'inverser.** Le contenu des items va du JSONL vers
les fiches : `items-cohorte1-v0.jsonl` fait foi, `REVUE-COHORTE1.md` en est une **vue**, et
relancer ce script écrase la vue sans perdre quoi que ce soit. Les décisions vont dans
l'autre sens : le relecteur les écrit dans sa fiche, et **une personne** les reporte ensuite
dans le JSONL. Aucun outil ne fait ce report automatiquement, et c'est délibéré — la seule
chose qui distingue un `gold` d'un `gold_candidate` est qu'un humain l'ait décidé.

**Ce que la fiche montre, et ce qu'elle cache.** Une citation utilisable est *titre, année,
page, extrait*. Les `chunk_id`, `block_ids` et offsets sont des clés **internes** : ils ne
disent rien à un lecteur, ne survivent pas à un re-découpage, et les mettre dans une citation
serait exactement le trou de produit décrit au §4.2 de `docs/STRATEGIE.md`. Ils sont
regroupés dans une **annexe technique** en fin de document, séparée et annoncée comme telle.

**Les tableaux sont montrés dans leur forme lisible.** Le texte ancré d'un tableau est le
HTML canonique — c'est lui qui porte les offsets — mais la fiche affiche la conversion
markdown que le corpus sert. Voir `cohorte1.ancre_de_tableau`.

    .venv/bin/python rag/benchmark/human-v1/paquet_revue.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

LOT = HERE / "items-cohorte1-v0.jsonl"
SORTIE = HERE / "REVUE-COHORTE1.md"
METADONNEES = ROOT / "rag" / "metadata" / "documents-metadata-v1.json"

#: Minutes par item, par famille. Fondé sur la seule mesure dont le dossier dispose —
#: l'estimation de 15 à 20 h pour 30 items du handoff `human-v1`, soit 30 à 40 min en
#: moyenne — et modulé par le nombre de documents à ouvrir. **C'est une estimation, pas une
#: mesure** : la cohorte 1 existe précisément pour la remplacer par un chiffre observé.
MINUTES = {"facile": 20, "moyenne": 35, "difficile": 50}


def fiches_documents() -> dict[str, dict]:
    donnees = json.loads(METADONNEES.read_text(encoding="utf-8"))
    donnees = donnees if isinstance(donnees, list) else donnees.get("documents", [])
    return {d["document_id"]: d for d in donnees}


def citation(fiche: dict, appui: dict) -> str:
    """La forme qu'une citation doit avoir pour un lecteur : titre, année, page, extrait."""
    titre = (fiche.get("title") or "document sans titre").strip()
    annee = fiche.get("publication_year")
    auteurs = fiche.get("authors") or []
    # Deux auteurs se citent tous les deux : « Fama » seul pour Fama & French est une
    # citation fausse. Relevé par la revue contradictoire sur C05 et C08.
    qui = ("" if not auteurs else auteurs[0] if len(auteurs) == 1
           else f"{auteurs[0]} & {auteurs[1]}" if len(auteurs) == 2
           else f"{auteurs[0]} et al.")
    tete = " — ".join(x for x in (qui, f"*{titre}*", str(annee) if annee else "") if x)
    return f"{tete}, p. {appui.get('page')}"


def bloc_extrait(appui: dict) -> str:
    texte = appui.get("rendu_servi") or appui["texte"]
    lignes = ["> " + ligne if ligne.strip() else ">" for ligne in texte.strip().splitlines()]
    sortie = "\n".join(lignes)
    if appui.get("rendu_servi"):
        sortie += ("\n>\n> *(affichage : conversion markdown servie par le corpus ; la citation "
                   "porte sur le tableau source, voir l'annexe technique)*")
    return sortie


def fiche(item: dict, docs: dict[str, dict], numero: int) -> str:
    minutes = MINUTES.get(item.get("difficulte"), 35)
    lignes = [
        f"## Item {numero}/10 — `{item['id']}`",
        "",
        f"| | |",
        f"|---|---|",
        f"| famille | `{item['famille']}` · domaine `{item['domaine']}` |",
        f"| difficulté estimée | {item.get('difficulte')} — **≈ {minutes} min** |",
        f"| ce que l'item éprouve | {', '.join(item.get('situations') or [])} |",
        f"| répondable | **{item['repondable']}** |",
        f"| statut actuel | **{item['statut']}** — proposition écrite par un modèle, non relue |",
        "",
        "### 1. La question, telle qu'elle sera posée au système",
        "",
        f"> {item['question']}",
        "",
        f"*Pourquoi cette question :* {item['intention']}",
        "",
        f"*D'où elle vient :* {item['provenance']['detail']}",
        "",
    ]

    if item["repondable"] != "non":
        lignes += ["### 2. Réponse attendue — **provisoire, à valider ou corriger**", ""]
        obligatoires = [f for f in item["faits_attendus"] if f["exigence"] == "obligatoire"]
        souhaitables = [f for f in item["faits_attendus"] if f["exigence"] == "souhaitable"]
        lignes.append("**Faits indispensables** — leur absence rend la réponse incomplète :")
        lignes.append("")
        for n, f in enumerate(obligatoires, 1):
            lignes.append(f"{n}. {f['texte']}  \n   <sub>appui : {', '.join(f['ancres'])}</sub>")
        lignes.append("")
        if souhaitables:
            lignes.append("**Faits secondaires** — acceptables, jamais exigibles :")
            lignes.append("")
            for n, f in enumerate(souhaitables, 1):
                lignes.append(f"{n}. {f['texte']}  \n   <sub>appui : {', '.join(f['ancres'])}</sub>")
            lignes.append("")
    else:
        lignes += ["### 2. Comportement attendu — **abstention**", "",
                   item.get("abstention_attendue", ""), ""]
        absence = item.get("absence") or {}
        lignes += [f"**Ce que le corpus ne porte pas :** {absence.get('porte')}", "",
                   f"**Comment l'absence a été vérifiée :** {absence.get('preuve')}", "",
                   f"**Force de la preuve :** {absence.get('force_de_la_preuve')}", "",
                   f"**Pourquoi le piège fonctionne :** {absence.get('voisinage_tentant')}", ""]

    if item.get("couverture_partielle"):
        cp = item["couverture_partielle"]
        lignes += ["### 2 bis. Ce que le corpus couvre, et ce qu'il ne couvre pas", "",
                   f"- **couvert** : {cp['couvert']}",
                   f"- **non couvert** : {cp['non_couvert']}",
                   f"- **vérification** : {cp['preuve']}", ""]

    if item["appuis"]:
        lignes += ["### 3. Les extraits sources", ""]
        for appui in item["appuis"]:
            f = docs.get(appui["document_id"], {})
            lignes += [f"**{appui['ancre_id']}** ({appui['role']}) — {citation(f, appui)}", "",
                       bloc_extrait(appui), ""]

    if item.get("conditions"):
        lignes += ["### 4. Conditions de validité — leur omission change la note", ""]
        for c in item["conditions"]:
            effet = {"reponse_fausse": "**rend la réponse FAUSSE**",
                     "reponse_incomplete": "rend la réponse incomplète"}[c["consequence_si_omis"]]
            lignes += [f"- {c['texte']}  \n  <sub>{effet} · appui : {', '.join(c['ancres'])}</sub>"]
        lignes.append("")

    lignes += ["### 5. Ce qui rend une réponse insuffisante, trompeuse ou hors sujet", ""]
    for p in item.get("pieges") or []:
        lignes.append(f"- {p}")
    lignes.append("")

    lignes += ["### 6. Ambiguïtés connues — **le relecteur doit trancher**", ""]
    for r in item.get("risques_ambiguite") or []:
        lignes.append(f"- {r}")
    if not item.get("risques_ambiguite"):
        lignes.append("- *(aucune identifiée par le modèle — ce qui ne veut pas dire qu'il n'y en a pas)*")
    lignes.append("")

    citation_min = item.get("citation_minimale") or {}
    lignes += ["### 7. Attentes de citation", "",
               f"- documents distincts attendus : **{citation_min.get('documents_distincts')}**",
               f"- pourquoi : {citation_min.get('justification')}",
               "- une citation acceptable porte **titre, année, page et extrait**. "
               "Un identifiant interne n'est pas une citation.", ""]

    lignes += [
        "### 8. Décision du relecteur",
        "",
        "Cocher **une** décision principale :",
        "",
        "- [ ] **VALIDER** — la question, les faits et les extraits sont justes. Les faits "
        "passent en `gold`.",
        "- [ ] **CORRIGER** — je réécris ci-dessous ce qui doit changer, puis c'est validé.",
        "- [ ] **AMBIGU** — la question ou le barème ne sont pas décidables en l'état ; à "
        "reformuler avant toute campagne.",
        "- [ ] **GARDER EN CANDIDAT** — plausible, mais je ne me prononce pas ; le fait ne "
        "compte dans aucun score.",
        "- [ ] **RETIRER** — l'item ne doit pas entrer dans la cohorte. Motif obligatoire.",
        "- [ ] **DEMANDER UNE AUTRE SOURCE** — le fait est peut-être juste mais l'extrait ne "
        "le soutient pas.",
        "",
        "Par fait, si la décision diffère de la décision principale :",
        "",
        "| fait | valider | corriger | retirer | commentaire |",
        "|---|:--:|:--:|:--:|---|",
    ]
    for n, f in enumerate(item["faits_attendus"], 1):
        lignes.append(f"| {n}. {f['texte'][:70]}… |  |  |  |  |")
    if not item["faits_attendus"]:
        lignes.append("| *(aucun fait : item d'abstention)* |  |  |  |  |")

    lignes += [
        "",
        "```",
        "relecteur (nom ou identifiant) : ______________________________",
        "date (AAAA-MM-JJ)              : ______________________________",
        f"version de campagne            : {item.get('version_campagne')}",
        "temps réellement passé (min)   : ______________________________",
        "",
        "justification de la décision :",
        "",
        "",
        "désaccords avec la proposition, ou avec un autre relecteur :",
        "",
        "",
        "```",
        "",
        "---",
        "",
    ]
    return "\n".join(lignes)


def annexe(items: list[dict]) -> str:
    lignes = [
        "## Annexe technique — clés internes",
        "",
        "**Rien ici n'est une citation.** Ces identifiants servent au ré-ancrage automatique et "
        "au contrôle des offsets. Ils ne survivent pas à un re-découpage du corpus et ne "
        "doivent jamais apparaître dans une réponse servie à un utilisateur.",
        "",
        "| item | ancre | document | page | chunk d'origine | offsets | empreinte du texte |",
        "|---|---|---|---:|---|---|---|",
    ]
    for item in items:
        for a in item["appuis"]:
            o = a.get("offsets") or {}
            sha = (o.get("doc_text_sha256") or "")[:12]
            lignes.append(f"| `{item['id']}` | {a['ancre_id']} | `{a['document_id']}` | "
                          f"{a.get('page')} | `{a.get('chunk_id_origine')}` | "
                          f"{o.get('debut')}–{o.get('fin')} | `{sha}…` |")
    lignes.append("")
    return "\n".join(lignes)


def entete(items: list[dict]) -> str:
    minutes = sum(MINUTES.get(i.get("difficulte"), 35) for i in items)
    familles = sorted({i["famille"] for i in items})
    return "\n".join([
        "# `human-v1` — cohorte 1, paquet de revue humaine",
        "",
        "> **Ce document contient des propositions écrites par un modèle. Aucune n'a été relue "
        "par une personne.** Rien ici n'est validé, rien n'est publiable, et aucun chiffre ne "
        "doit en sortir avant que les fiches ci-dessous soient remplies et signées.",
        "",
        f"*10 items · {sum(len(i['appuis']) for i in items)} extraits sources · "
        f"{sum(len(i['faits_attendus']) for i in items)} faits proposés, tous en "
        f"`gold_candidate` · corpus `{items[0]['corpus_signature']}` · "
        f"campagne {items[0]['version_campagne']}.*",
        "",
        "## Comment lire et remplir ce paquet",
        "",
        "Chaque fiche est autonome : elle ne demande aucune connaissance du dépôt. Pour chaque "
        "item, il s'agit de répondre à une question simple — **si le système répondait ceci, "
        "aurais-je raison de dire qu'il a bien répondu ?**",
        "",
        "Les six décisions possibles sont les mêmes partout, et elles sont exclusives :",
        "**valider**, **corriger**, **déclarer ambigu**, **garder en candidat**, **retirer**, "
        "**demander une autre source**. Une seule chose est irréversible : valider fait passer "
        "des faits en `gold`, et un `gold` compte dans tous les scores publiés après lui.",
        "",
        "**Ce que la cohorte 1 sert à mesurer**, et qui n'est pas la qualité du RAG : le temps "
        "réel d'annotation, les ambiguïtés que le barème n'a pas prévues, et le niveau de "
        "précision de citation qui est tenable. C'est une **calibration**. Elle ne produit "
        "aucun chiffre publiable, et le champ « temps réellement passé » de chaque fiche est "
        "l'une de ses deux sorties principales.",
        "",
        f"**Temps estimé : ≈ {minutes // 60} h {minutes % 60:02d}** pour les dix "
        f"({minutes} minutes). C'est une **estimation**, dérivée de la seule référence "
        "disponible (15 à 20 h pour 30 items) et non une mesure — la remplacer par un chiffre "
        "observé est l'objet de cette cohorte.",
        "",
        f"**Familles représentées** : {', '.join('`' + f + '`' for f in familles)}.",
        "",
        "---",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lot", type=Path, default=LOT)
    parser.add_argument("--sortie", type=Path, default=SORTIE)
    args = parser.parse_args()
    items = [json.loads(l) for l in args.lot.read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = fiches_documents()
    corps = entete(items) + "".join(fiche(i, docs, n) for n, i in enumerate(items, 1)) + annexe(items)
    args.sortie.write_text(corps, encoding="utf-8")
    print(f"  écrit  {args.sortie.relative_to(ROOT)}  ({len(corps.splitlines())} lignes, "
          f"{len(items)} fiches)")


if __name__ == "__main__":
    main()
