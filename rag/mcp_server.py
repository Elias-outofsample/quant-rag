"""Serveur MCP stdio exposant le corpus Quant RAG à Claude Code.

Aucun port réseau n'est ouvert : Claude Code lance ce process et parle en stdio.
Outils : search_documents, get_passage, list_documents, timeline, corpus_status,
et sur le graphe d'entités (GLiNER2, ``graph_search.py``) : search_graph, expand_entity,
connect_entities. Le graphe complète la recherche dense, il ne la remplace pas : croiser les deux.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.mcpserver import MCPServer

import contrat
import graph_search
import quant_rag

SERVER_VERSION = "1.4.0"
mcp = MCPServer("quant-rag", version=SERVER_VERSION)
# Le serveur déclare sa version au contrat de sortie : elle voyage ensuite dans chaque réponse
# servie et dans chaque ligne de journal, alors qu'elle n'était écrite nulle part jusqu'ici.
contrat.enregistrer_serveur(SERVER_VERSION)


def _comptes() -> dict:
    """Les comptes annoncés à l'appelant, lus sur les artefacts de la signature vivante.

    Ils étaient écrits à la main. Ils ont été faux deux fois en quatorze heures — 256/18 636
    corrigé le 5 septembre, 319/22 190 corrigé le 6 — parce qu'aucune étape de la chaîne
    d'ingestion ne les met à jour. Un compte faux annoncé à l'appelant est pire qu'absent :
    le modèle raisonne dessus (« ce corpus est petit, je vais élargir »).

    **Jamais Qdrant ici.** L'ouvrir à l'import prendrait le verrou du stockage embarqué, et
    ce verrou a déjà coûté une nuit : le 6 septembre, ``apply_delivery`` ET
    ``inspect_delivery`` ont échoué tous les deux parce que ce serveur le tenait. La source
    est le manifeste BM25 de la signature vivante — il n'existe que s'il a été construit
    pour elle, c'est la garde de nommage par signature qui le garantit.

    Quand l'artefact manque, on n'invente rien : le compte vaut ``None`` et la phrase
    annoncée se tait sur ce qu'elle ne sait pas.
    """
    import corpus_overlay

    racine = Path(__file__).resolve().parents[1]
    comptes = {"documents": None, "passages": None, "sans_annee": None}
    manifeste = (racine / "data" / "lexical" /
                 f"bm25-dedup-tables-registry-titles-v1-{corpus_overlay.signature()}.manifest.json")
    if manifeste.exists():
        try:
            origine = json.loads(manifeste.read_text(encoding="utf-8")).get("built_from", {})
            comptes["documents"] = origine.get("documents")
            comptes["passages"] = origine.get("points")
        except (json.JSONDecodeError, OSError):
            pass
    try:
        fiches = json.loads((racine / "rag" / "metadata" /
                             "documents-metadata-v1.json").read_text(encoding="utf-8"))
        fiches = fiches if isinstance(fiches, list) else fiches.get("documents", [])
        retires = corpus_overlay.removed_documents()
        actives = [f for f in fiches if f.get("document_id") not in retires]
        comptes["sans_annee"] = sum(1 for f in actives if not f.get("publication_year"))
        if comptes["documents"] is None:
            comptes["documents"] = len(actives)
    except (json.JSONDecodeError, OSError, TypeError):
        pass
    return comptes


COMPTES = _comptes()


def _avec_comptes(fonction):
    """Formate la docstring avec les comptes réels, avant que ``mcp.tool()`` ne la lise.

    Les décorateurs s'appliquent de bas en haut : placé SOUS ``@mcp.tool()``, celui-ci
    s'exécute en premier et le serveur publie une description déjà à jour. Il rend la
    fonction elle-même, sans emballage : la signature exposée reste intacte.
    """
    documents, passages = COMPTES["documents"], COMPTES["passages"]
    espace = lambda n: f"{n:,}".replace(",", "\u202f")   # séparateur de milliers français
    if documents and passages:
        portee = f"{espace(documents)} documents, {espace(passages)} passages"
    elif documents:
        portee = f"{espace(documents)} documents"
    else:
        portee = "corpus dont la taille n'a pas pu être lue"
    sans = COMPTES["sans_annee"]
    inconnue = (f"{sans} sur {espace(documents)}" if sans is not None and documents
                else "un nombre non lu de documents")
    # Remplacement ciblé, pas ``.format()`` : formater TOUTE la docstring ferait planter
    # le serveur au démarrage dès qu'un exemple y contiendrait une accolade littérale
    # (un objet JSON, une notation ensembliste). Relecture adverse du 6 septembre 2026.
    fonction.__doc__ = (fonction.__doc__.replace("{portee}", portee)
                                        .replace("{annee_inconnue}", inconnue))
    return fonction


def _year_line(row: dict) -> str:
    year = row.get("publication_year")
    return f"{year}" if year else "année inconnue"


def _ligne_contrat(entete: dict) -> str:
    """Une ligne, en tête de chaque réponse servie, qui dit d'où elle sort.

    Sans elle, un passage collé dans une note trois jours plus tard ne peut plus être rattaché
    ni à la décision de routage qui l'a produit, ni à la configuration qui l'a servi. Les mêmes
    champs ouvrent la ligne correspondante de ``rag/logs/router-decisions.jsonl``.
    """
    return (f"trace: request_id={entete['request_id']} · serveur {entete['server_version']} · "
            f"contrat {entete['contrat_version']} · corpus {entete['corpus_signature']} · "
            f"config {entete['config_hash']} · fenêtre {contrat.PASSAGE_CHARACTERS} c.")


def _ligne_ancre(row: dict) -> str:
    """« ancre : » — le seul champ servi avec lequel une citation peut être **vérifiée**.

    Les auteurs, l'année et la page sont vérifiables par un humain qui ouvre le PDF. Rien, dans
    ce qui était servi jusqu'ici, ne permettait de prouver qu'une phrase attribuée au document
    s'y trouve : la page est arrondie au chunk et ne dit pas où chercher. Les offsets dans le
    texte canonique, eux, le disent — et `verify_citation` s'en sert.
    """
    ancre = contrat.ligne_ancre(row)
    return f"    ancre   : {ancre}\n" if ancre else ""


def _ligne_qualite(row: dict) -> str:
    """Ce que le passage servi porte de douteux, quand il porte quelque chose.

    Les défauts sont amont — 323 passages à caractères de contrôle, 13,4 % avec ``<sup>``/
    ``<sub>``, 127 au balisage mathématique impair — et les réparer changerait le texte des
    chunks, donc la signature du corpus, qui est gelée. Les taire serait pire.
    """
    texte = row.get("text") or ""
    ligne = contrat.ligne_qualite(
        contrat.drapeaux_qualite(texte, tronque=len(texte) > contrat.PASSAGE_CHARACTERS))
    return f"    qualité : {ligne}\n" if ligne else ""


@mcp.tool()
@_avec_comptes
def search_documents(query: str, limit: int = 5, document_id: str | None = None,
                     mode: str = "auto", rerank: bool | None = None,
                     year_min: int | None = None, year_max: int | None = None,
                     author: str | None = None, reclassement: str | None = None) -> str:
    """Cherche des passages dans le corpus de recherche quantitative ({portee}).

    Couvre la volatilité et les options, la microstructure et l'exécution, les facteurs et
    l'alpha, la construction de portefeuille et le ML appliqué à la finance (1971–2026).
    Retourne des extraits classés, précédés de la décision de routage. Chaque extrait porte
    une ligne « citer : » — auteurs, année, titre, page — et c'est **la seule** à recopier
    dans une réponse : l'année fait partie de l'information, le corpus mélangeant des travaux
    de 1987 et des preprints de 2026. La ligne « interne : » porte `chunk_id` et
    `document_id`, qui servent à `get_passage` et **ne sont pas des citations** — ils ne
    survivent pas à un re-découpage du corpus et ne disent rien à un lecteur.

    Les pages affichées sont celles d'un lecteur, et une **plage** quand le passage en couvre
    plusieurs : le système ne dispose pas d'un pointeur de phrase, et annoncer une page exacte
    pour un passage à cheval serait une précision qu'il n'a pas.

    Protocole d'usage, mesuré sur 155 questions (benchmark/eval_protocol.py) :
    - Une période explicite (« sources published in 2022 or earlier », « papers before 2010 »,
      « depuis 2020 ») est détectée dans la question et transformée en year_min/year_max, la
      clause retirée du texte : rien à faire, sauf quand la période est implicite (« récent »,
      « d'avant la crise ») — alors la traduire soi-même en year_min/year_max (+0,10 nDCG@10).
    - Question ouverte, conceptuelle : une seule requête en langue naturelle, en anglais,
      qui décrit le problème ; la reformuler pour l'index n'a pas aidé (chantier C).
    - N'utiliser mode="hybrid" que pour une requête faite d'identifiants mémorisés (auteur,
      acronyme, numéro, année : « Fukasawa SVI B2 B3 ») ; sur toute autre question il perd
      (−0,11 nDCG@10 sur 130 questions, −0,19 sur les tableaux). Depuis le 5 septembre 2026
      on sait *pourquoi* : ce chemin empile deux composants mesurés négatifs — la fusion
      (−0,088 pooled, 14 variantes essayées, zéro en GO) et le reranker bge-base sur pool
      dense (−0,076 pooled, mesuré sur ce corpus). Il reste utile pour le cas des identifiants exacts ; ailleurs,
      le choisir est une erreur documentée.
    - Question multi-documents ou exploratoire : compléter par search_graph / expand_entity
      / connect_entities sur les entités nommées de la question, puis lire les passages des
      deux sources ; ne pas fusionner aveuglément (mesuré : la fusion automatique perd).

    Args:
        query: la question, en anglais de préférence (le corpus est anglophone).
        limit: nombre de passages à retourner (défaut 5).
        document_id: restreindre à un seul document.
        mode: "auto" (défaut) = recherche sémantique seule (dense) — depuis le banc de
            150 questions, aucune règle automatique ne fait mieux. "hybrid" ajoute
            BM25+fusion+rerank : à demander explicitement quand la question est faite
            d'identifiants exacts dont l'utilisateur se souvient (Fukasawa, SVI, ITRAXX 2007,
            arXiv 1206.0682), et seulement dans ce cas — sur une question en langue
            naturelle, et sur les tableaux, l'hybride fait moins bien que le dense.
        rerank: laisser vide pour suivre le mode (activé en hybrid, désactivé en dense).
            True/False force. Mesuré : bge-reranker-base aide sur un pool hybride et **nuit**
            sur un pool dense (−0,076 nDCG@10 pooled, −0,147 sur les questions simples).
            Ne pas forcer rerank=True en mode dense.
        year_min: ne garder que les documents publiés à partir de cette année (inclus).
        year_max: ne garder que les documents publiés jusqu'à cette année (inclus).
            Une borne d'année exclut les documents dont l'année est inconnue ({annee_inconnue}).
            Laissées vides, une période explicite dans la question est détectée et appliquée
            (la ligne « période: » de la réponse le dit) ; une borne donnée prime toujours.
        author: ne garder que les documents d'un auteur (sous-chaîne du nom, insensible aux
            accents et à la casse : "lopez de prado", "Gatheral", "Jacquier").
        reclassement: "selectif" reclasse les dix premiers candidats par Qwen3-Reranker-0.6B
            en protégeant le rang 1 dense. Laisser vide (défaut) dans l'immense majorité des
            appels.

            **Ce que ça coûte** : ~3,2 s par requête contre 59 ms sans — un facteur **54** —
            et 2,4 Go de mémoire résidente tant que le modèle est chargé, sur une machine de
            16 Go qui tourne déjà avec 2,4 Go de swap. Ce n'est pas un réglage anodin.

            **Ce que ça rapporte** (155 questions, corpus 5530cba145) : l'or entre dans les
            cinq passages servis pour 19 questions et en sort pour 5. Net +14, et les cinq
            pertes sont un sous-ensemble strict des sept du reclassement plein.

            **Ce que ces 19 et ces 5 valent en RÉPONSE — mesuré le 8 septembre 2026** sur les
            24 questions concernées, jugées dans les deux bras : sur les 5 « pertes », **une
            seule** perd vraiment sa réponse ; les trois autres mesurables avaient une
            couverture nulle **dans les deux bras** — elles n'avaient rien à perdre. Sur les
            19 « gains », **12** gagnent vraiment une réponse. Le rapport réel est **12 pour
            1**, non 19 pour 5, et l'échange net vaut **+11 réponses**.

            **Ce que ça change pour toi** : le risque de dégrader une réponse en l'activant est
            **cinq fois plus petit** que ce que le paragraphe précédent laisse croire. Ce qui
            reste vrai, et qui est désormais la seule raison de ne pas l'activer partout, c'est
            le coût — 3,2 s et 2,4 Go sur une machine de 16 Go. Active-le sans hésiter quand la
            qualité de la réponse compte plus que trois secondes ; ne l'active pas en rafale
            sur une exploration où tu enchaînes dix recherches.

            **Quand le mettre** : quand un humain demande explicitement une recherche plus
            soignée, ou après avoir constaté qu'une première recherche a rendu des passages
            hors sujet sur une question dont on a de bonnes raisons de croire que le corpus
            porte la réponse.

            **Quand ne pas le mettre — et c'est le point important** : ne l'active pas « au
            cas où », ni systématiquement, ni selon une règle que tu te donnerais toi-même
            (longueur de la question, présence de chiffres, famille supposée). Une règle
            d'activation inventée par l'appelant est une **politique de routage non mesurée**,
            et ce dépôt a mesuré six fois que ses politiques de routage perdent. Le gain
            ci-dessus n'est valable que sur un usage à la demande ; il ne dit rien de ce que
            vaudrait une heuristique automatique, qui n'a jamais été évaluée.

            Un échec de reclassement ne casse jamais la recherche : l'ordre dense est rendu.
    """
    out = quant_rag.search_explained(query, limit=limit, document_id=document_id, mode=mode, rerank=rerank,
                                     year_min=year_min, year_max=year_max, author=author,
                                     reclassement=reclassement)
    rows = out["results"]
    header = quant_rag.format_routing(out["routing"]) + "\n" + _ligne_contrat(out["contrat"])
    if not rows:
        return f"{header}\n\nAucun passage trouvé."
    out = [header]
    for i, row in enumerate(rows, 1):
        # La ligne « citer : » est la seule à recopier dans une réponse. `chunk_id` et
        # `document_id` sont des **clés internes** — elles ne survivent pas à un
        # re-découpage du corpus et ne disent rien à un lecteur ; les citer serait donner
        # une référence invérifiable. Les pages affichées sont les pages d'un lecteur, et
        # une **plage** quand le passage en couvre plusieurs (voir
        # `quant_rag.citation_pages` : le système n'a pas de pointeur de phrase).
        out.append(
            f"[{i}] {row['score_kind']}={row['score']:.3f}\n"
            f"    citer   : {row['source']}, {quant_rag.citation_pages(row.get('pages_utilisateur'))}\n"
            f"    section : {row['section'] or '—'}\n"
            f"    interne : chunk_id={row['chunk_id']} · document_id={row['document_id']} · "
            f"{row['content_type']} (clés de travail, ne pas citer)\n"
            + _ligne_ancre(row)
            + _ligne_qualite(row)
            + f"\n{contrat.couper_passage(row['text'], chunk_id=row['chunk_id'])}"
        )
    return "\n\n---\n\n".join(out)


@mcp.tool()
def get_passage(chunk_id: str, max_characters: int | None = None, neighbours: int = 1) -> str:
    """Retourne le passage complet d'un chunk_id, avec ses voisins immédiats, pour citation.

    Args:
        chunk_id: la clé interne rendue par `search_documents` ou `search_graph`.
        max_characters: fenêtre servie. Vide = 15 000, la fenêtre du contrat de sortie. Le
            défaut historique était 6 000 et **tronquait 30,2 % des fenêtres du corpus**
            (médiane 4 520 c., max 13 982) : un outil dont le nom promet le passage entier en
            rendait moins d'un tiers du temps.
        neighbours: nombre de voisins de chaque côté (défaut 1, 0 pour le seul chunk demandé).
            Les voisins ne sont pas filtrés comme les résultats de recherche — un voisin
            en-tête est un intitulé de section, donc du contexte — mais ils sont déclarés dans
            la ligne « voisins : ».
    """
    passage = quant_rag.get_passage(chunk_id, max_characters=max_characters, neighbours=neighbours)
    if not passage:
        return f"chunk_id inconnu: {chunk_id}"
    voisins = ", ".join(f"{v['chunk_id']} ({v['content_type']}, {v['caracteres']} c.)"
                        for v in passage["voisins"]) or "aucun"
    qualite = contrat.ligne_qualite(passage["qualite"])
    ancre = contrat.ligne_ancre(passage)
    return (f"citer : {passage['source']}, "
            f"{quant_rag.citation_pages(passage.get('pages_utilisateur'))}\n"
            f"section: {passage['section'] or '—'}\n"
            f"interne: chunk_id={passage['chunk_id']} (clé de travail, ne pas citer)\n"
            f"voisins: {voisins}\n"
            + (f"ancre  : {ancre}\n" if ancre else "")
            + (f"qualité: {qualite}\n" if qualite else "")
            + f"\n{passage['text']}")


@mcp.tool()
def list_documents(author: str | None = None, year_min: int | None = None, year_max: int | None = None,
                   text: str | None = None, limit: int = 50) -> str:
    """Liste bibliographique des documents du corpus, du plus récent au plus ancien, sans recherche vectorielle.

    Args:
        author: sous-chaîne du nom d'un auteur (insensible aux accents et à la casse).
        year_min: année de publication minimale (inclus).
        year_max: année de publication maximale (inclus).
        text: sous-chaîne du titre.
        limit: nombre maximal de lignes (défaut 50).
    """
    rows = quant_rag.list_documents(author=author, year_min=year_min, year_max=year_max, text=text, limit=limit)
    if not rows:
        return "Aucun document ne correspond."
    lines = [f"{len(rows)} document(s)"]
    for row in rows:
        lines.append(f"- {row['short_ref']} — {row['title']}"
                     + (f" [{row['venue']}]" if row.get("venue") else "")
                     + f" · document_id={row['document_id']}"
                     + (" · métadonnées peu sûres" if row.get("confidence") == "low" else ""))
    return "\n".join(lines)


@mcp.tool()
def timeline(topic: str, year_min: int | None = None, year_max: int | None = None, limit: int = 20) -> str:
    """Chronologie d'un sujet : le meilleur passage de chaque document, groupé par année de publication.

    Utile pour voir comment une idée a évolué (rough volatility 2014 → 2026) ou pour
    distinguer les travaux fondateurs des preprints récents.

    Args:
        topic: le sujet, formulé comme une question ou un intitulé.
        year_min: borne basse (inclus).
        year_max: borne haute (inclus).
        limit: nombre de documents (défaut 20).
    """
    out = quant_rag.timeline(topic, limit=limit, year_min=year_min, year_max=year_max)
    lines = [quant_rag.format_routing(out["routing"])]
    for group in out["years"]:
        lines.append(f"\n== {group['year']}")
        for row in group["results"]:
            # Un **aperçu déclaré**, et le seul de la surface servie : 300 caractères sur un
            # outil qui aligne vingt documents. La garantie « aucune formule coupée » ne s'y
            # applique pas et ne le peut pas — 59,3 % des passages portent du LaTeX, et reculer
            # jusqu'à un point sûr ne rendrait presque rien. Ce qui est garanti : l'aperçu ne se
            # fait jamais passer pour un passage entier, il porte toujours son « … ».
            lines.append(f"- citer   : {row['source']}, "
                         f"{quant_rag.citation_pages(row.get('pages_utilisateur'))}\n"
                         f"  interne : chunk_id={row['chunk_id']} (clé de travail, ne pas citer)\n"
                         f"  {contrat.couper_apercu(row['text'])}")
    return "\n".join(lines)


@mcp.tool()
def verify_citation(document_id: str, quote: str, chunk_id: str | None = None) -> str:
    """Vérifie qu'une citation se trouve **vraiment** dans le document qu'elle nomme.

    À utiliser avant d'affirmer qu'un document dit quelque chose, et **surtout** quand la
    citation vient d'ailleurs que d'un passage servi à l'instant : d'une note prise plus tôt,
    d'un résumé, d'une réponse antérieure. Une citation recopiée de travers, recollée à partir
    de deux endroits ou reformulée reste plausible à la lecture — c'est précisément ce que cet
    outil détecte, et rien d'autre ne le détecte.

    La vérification est **déterministe** : aucune génération, aucun modèle de langue. La
    citation est normalisée (ligatures, `<sup>`/`<sub>`, tirets, guillemets, blancs, casse) puis
    cherchée dans le texte canonique du document, celui dont le `sha256` est servi par la ligne
    « ancre : ».

    Ce que rend l'outil :
    - `trouve` — vrai **seulement** pour une correspondance exacte après normalisation. Une
      citation dont un seul mot diffère rend `false`. Mesuré : 50 extraits réels sur 50
      retrouvés, 0 accepté sur 50 extraits altérés d'un mot.
    - `offsets` — la position `[début, fin]` dans le texte canonique, opposable.
    - `plus_proche` — quand c'est faux : le passage le plus ressemblant, sa ressemblance et le
      nombre de mots qui diffèrent. C'est ce qui dit *où* la citation a dérivé.
    - `source` — `texte canonique`, ou `texte servi` pour les 5 914 tableaux dont le rendu
      Markdown n'est pas une sous-chaîne du document.

    Args:
        document_id: la clé interne du document, servie par la ligne « interne : ».
        quote: le texte cité, tel qu'on s'apprête à l'affirmer. Une phrase ou deux suffisent ;
            au-delà d'un paragraphe, une reformulation invisible fera échouer la vérification
            pour une raison sans rapport avec l'honnêteté de la citation.
        chunk_id: facultatif, restreint la recherche du repli « texte servi » à ce passage.
    """
    import citation as verificateur

    return json.dumps(verificateur.verify_citation(document_id, quote, chunk_id),
                      indent=2, ensure_ascii=False)


@mcp.tool()
def verify_citations(citations: list[dict]) -> str:
    """Vérifie plusieurs citations d'un coup — même contrat que `verify_citation`.

    Args:
        citations: liste d'objets `{"document_id": …, "quote": …, "chunk_id": … (facultatif)}`.
            Vérifier ensemble les citations d'une même réponse coûte à peine plus qu'une seule :
            le texte canonique d'un document n'est lu qu'une fois.
    """
    import citation as verificateur

    resultats = verificateur.verify_citations(citations)
    resume = {"verifiees": len(resultats), "trouvees": sum(1 for r in resultats if r.get("trouve"))}
    return json.dumps({"resume": resume, "resultats": resultats}, indent=2, ensure_ascii=False)


@mcp.tool()
def corpus_status() -> str:
    """Décrit l'index : documents, passages, couverture des métadonnées (années, auteurs), backend et device,
    et l'état du graphe d'entités (nombre d'entités, relations, état du corpus sur lequel il a été bâti)."""
    out = quant_rag.corpus_status()
    try:
        out["graph"] = graph_search.status()
    except RuntimeError as exc:
        out["graph"] = {"error": str(exc)}
    return json.dumps(out, indent=2, ensure_ascii=False)


@mcp.tool()
def search_graph(query: str, entity_type: str | None = None, relation: str | None = None, top_k: int = 10) -> str:
    """Cherche dans le graphe d'entités : les passages qui MENTIONNENT une entité dont le nom contient `query`.

    Complémentaire de search_documents (sémantique) : ici la correspondance est nominale et exacte
    — « Gatheral » retourne les passages où Gatheral est cité, « SVI » ceux où SVI apparaît, même
    quand la question sémantique ne les ferait pas remonter. Les entités ont été extraites par
    GLiNER2 sur chaque passage (personnes, organisations, instruments, concepts de marché,
    mesures, méthodes, jeux de données). Même format de sortie que search_documents : source
    citable, pages, chunk_id (pour get_passage), plus les entités reconnues dans le passage.
    Croiser avec search_documents : le graphe ne connaît que les noms, pas le sens.

    Args:
        query: nom d'entité ou fragment (« Gatheral », « rough volatility », « SVI », « S&P 500 »).
            Casse et accents ignorés ; « Lopez de Prado » trouve « López de Prado ».
        entity_type: restreindre à un type : person, organization, financial_instrument,
            market_concept, measure, method, dataset.
        relation: ne garder que les passages où l'entité est tête ou queue d'une relation de ce
            type : measures, predicts, causes, depends_on, correlates_with, applies_to, uses_method,
            compares_with, is_a, part_of.
        top_k: nombre de passages (défaut 10, au plus 2 par document).
    """
    return graph_search.format_search(graph_search.search_graph(query, entity_type=entity_type, relation=relation, top_k=top_k))


@mcp.tool()
def expand_entity(entity_name: str, relation: str | None = None, entity_type: str | None = None, limit: int = 20) -> str:
    """Entités reliées à `entity_name` dans le graphe, par relation, avec les passages qui l'attestent.

    Répond à « quelles méthodes dépendent de la rough volatility ? », « à quoi SVI est-il comparé ? ».
    Chaque voisin porte la relation et son sens (→ : l'entité est la tête, ← : la queue), le nombre
    de passages qui soutiennent le lien et jusqu'à trois chunk_id à lire avec get_passage. Les
    relations viennent de GLiNER2 (seuil 0,5) et sont clairsemées : un lien attesté par un seul
    passage est une piste, pas un fait ; lire le passage avant de l'affirmer. Sans `relation`, la
    réponse liste aussi les CO-MENTIONS : les entités citées dans les mêmes passages, pondérées
    par leur rareté — la vue la plus utile pour explorer un sujet (auteurs, modèles, mesures
    qui vont avec).

    Args:
        entity_name: nom de l'entité (« rough volatility », « Heston model », « Gatheral »).
            L'entité exacte et ses variantes de nom (« rough volatility models ») sont réunies.
        relation: une seule relation (measures, predicts, causes, depends_on, correlates_with,
            applies_to, uses_method, compares_with, is_a, part_of) ; vide = toutes.
        entity_type: type de l'entité de départ, si le nom est ambigu (person, market_concept…).
        limit: nombre de voisins (défaut 20), classés par nombre de passages qui les attestent.
    """
    return graph_search.format_expand(graph_search.expand_entity(entity_name, relation=relation, entity_type=entity_type, limit=limit))


@mcp.tool()
def connect_entities(entity_a: str, entity_b: str, top_k: int = 10) -> str:
    """Passages et documents qui mentionnent LES DEUX entités (« SVI » et « rough volatility »).

    Pour trouver où deux idées se rencontrent dans le corpus. Liste les documents par nombre de
    passages communs, puis les meilleurs passages avec chunk_id. Aucun passage commun est une
    information en soi : les deux entités ne sont jamais citées ensemble dans un même passage.

    Args:
        entity_a: première entité (nom ou fragment, variantes réunies).
        entity_b: seconde entité.
        top_k: nombre de passages (défaut 10, au plus 2 par document).
    """
    return graph_search.format_connect(graph_search.connect_entities(entity_a, entity_b, top_k=top_k))


if __name__ == "__main__":
    mcp.run(transport="stdio")
