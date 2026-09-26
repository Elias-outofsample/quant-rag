"""La citation servie doit être honnête — et ne rien prétendre que le système ne sache.

Le défaut, mesuré. Les pages du payload sont **0-basées** : elles reprennent le ``page_idx``
des blocs du parse, dont le bloc de titre porte 0. Vérifié sur `doc-2b74a994fcfc1c9c` —
« 1 Introduction » en ``page_idx`` 1, « 6 Conclusion » en 34, chunks déclarant 1 et 34 — et
confirmé indépendamment par extraction du PDF source d'un autre document. Le serveur MCP
affichait « pages 34-35 » et sa documentation disait « cite la source telle qu'affichée » :
la citation envoyait donc le lecteur **une page trop tôt**.

Et un second défaut, qui ne se corrige pas par un « +1 » : ``page_start`` est la page où le
**chunk** commence, pas celle où se trouve la **phrase citée**. Le système n'a pas de pointeur
de phrase au moment de la réponse. La seule sortie honnête est donc une **plage**.

Ce que ces tests gardent :

    la conversion         0-basé interne → page d'un lecteur
    la plage              un passage à cheval sur deux pages s'affiche « p. 35–36 »
    la sémantique         ``page_start`` reste « début du chunk, tel que stocké »
    l'absence de fuite    ni ``chunk_id`` ni ``document_id`` dans une ligne à citer
    la non-régression     le chemin de retrieval n'est pas touché
"""
from __future__ import annotations

import sys
from pathlib import Path

MACOS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MACOS))

import quant_rag  # noqa: E402


# ------------------------------------------------------------------ conversion


def test_la_page_interne_est_zero_basee_et_la_page_citee_ne_l_est_pas():
    """Le cas mesuré : « 6 Conclusion » est stocké en 34 et se lit page 35."""
    assert quant_rag.pages_utilisateur(34, 35) == [35, 36]
    assert quant_rag.pages_utilisateur(0, 0) == [1]


def test_les_pages_couvertes_sont_enumerees_et_pas_bornees():
    """Une liste explicite, pour qu'un appelant qui joint des passages en fasse l'UNION.

    Panne évitée : avec un couple de bornes, joindre deux passages non contigus donnait
    l'enveloppe — et l'enveloppe affirme les pages du trou.
    """
    assert quant_rag.pages_utilisateur(10, 13) == [11, 12, 13, 14]


def test_un_passage_sur_une_seule_page_s_affiche_sans_plage():
    assert quant_rag.citation_pages(quant_rag.pages_utilisateur(34, 34)) == "p. 35"


def test_un_passage_a_cheval_s_affiche_en_plage_honnete():
    """Annoncer « p. 35 » pour un passage qui court sur 35-36 serait une précision que le
    système ne possède pas : il n'a pas de pointeur de phrase."""
    assert quant_rag.citation_pages(quant_rag.pages_utilisateur(34, 35)) == "p. 35–36"
    assert quant_rag.citation_pages(quant_rag.pages_utilisateur(10, 13)) == "p. 11–14"


def test_une_page_de_fin_absente_ne_fabrique_pas_une_plage():
    assert quant_rag.pages_utilisateur(7, None) == [8]
    assert quant_rag.citation_pages(quant_rag.pages_utilisateur(7, None)) == "p. 8"


def test_une_page_inconnue_se_dit_inconnue():
    assert quant_rag.pages_utilisateur(None, None) is None
    assert quant_rag.citation_pages(None) == "page inconnue"
    assert quant_rag.citation_pages([]) == "page inconnue"


def test_une_plage_inversee_ne_peut_pas_sortir():
    """Un ``page_end`` antérieur au début ne doit pas produire « p. 12–8 »."""
    assert quant_rag.pages_utilisateur(11, 7) == [12]


# ---------------------------------------------- une plage ne doit pas SUR-couvrir non plus


def test_une_couverture_discontinue_ne_se_dit_pas_en_plage():
    """Le second défaut, mesuré le 7 septembre 2026, et symétrique du premier.

    ``get_passage`` joint la fratrie d'un chunk **ordonnée par page, pas par contiguïté** :
    ``chunk-5ea0681947ca79cb`` rend du texte des pages 27, 28 et 31. Annoncer « p. 27–31 »
    envoie le lecteur chercher aux pages 29 et 30, où il n'y a rien de ce qu'il a lu.
    """
    assert quant_rag.citation_pages([27, 28, 31]) == "p. 27–28, 31"
    assert quant_rag.citation_pages([31, 27, 28]) == "p. 27–28, 31"
    assert quant_rag.citation_pages([5, 9, 12, 13]) == "p. 5, 9, 12–13"


def test_le_texte_joint_et_les_passages_montres_restent_solidaires():
    """Citer la page d'un passage que ``max_characters`` a coupé est le défaut symétrique."""
    passages = [{"chunk_id": "a", "text": "A" * 100, "page_start": 4, "page_end": 4},
                {"chunk_id": "b", "text": "B" * 100, "page_start": 9, "page_end": 9}]
    texte, montres = quant_rag.joindre_passages(passages, 6000)
    assert len(texte) == 202 and [p["chunk_id"] for p in montres] == ["a", "b"]
    texte, montres = quant_rag.joindre_passages(passages, 50)
    assert [p["chunk_id"] for p in montres] == ["a"], "un passage coupé ne doit pas être cité"
    # Depuis le contrat de sortie, ``joindre_passages`` coupe par ``contrat.couper`` : la
    # fenêtre recule jusqu'à une frontière qui ne casse rien. Demander 103 caractères ne rend
    # donc plus « un seul B » — la coupe recule à 102, puis ``rstrip`` retire le séparateur, et
    # le passage *b* n'apparaît pas du tout. L'invariant en sort **renforcé** : au lieu de citer
    # la page 9 pour un caractère, on ne la cite plus du tout. Un passage est montré entier,
    # coupé à une frontière propre, ou pas montré.
    texte, montres = quant_rag.joindre_passages(passages, 103)
    assert [p["chunk_id"] for p in montres] == ["a"], "un caractère orphelin ne se cite pas"
    # Sur du texte qui a des frontières de mot — c'est-à-dire du vrai texte —, une part réelle
    # du second passage est rendue, et il redevient citable.
    reels = [{"chunk_id": "a", "text": "Le premier passage. " * 5, "page_start": 4, "page_end": 4},
             {"chunk_id": "b", "text": "Le second passage. " * 5, "page_start": 9, "page_end": 9}]
    texte, montres = quant_rag.joindre_passages(reels, 150)
    assert [p["chunk_id"] for p in montres] == ["a", "b"], "une part réelle de *b* se cite"
    assert texte.endswith("passage."), "et elle se termine à une frontière propre"


# ------------------------------------------------------------------ sémantique préservée


def _payload(**surcharge):
    base = {"chunk_id": "chunk-abc", "document_id": "doc-abc", "title": "Un titre",
            "short_ref": "Auteur 2015", "section": "1 Introduction",
            "page_start": 34, "page_end": 35, "content_type": "text", "text": "du texte"}
    base.update(surcharge)
    return base


def test_page_start_garde_sa_semantique_de_debut_de_chunk():
    """Panne évitée : réécrire ``page_start`` casserait tout artefact qui le lit comme stocké."""
    row = quant_rag._payload_row(_payload(), 0.5, "cosine")
    assert row["page_start"] == 34 and row["page_end"] == 35
    assert row["pages_utilisateur"] == [35, 36]


def test_la_ligne_de_citation_ne_contient_aucune_cle_interne():
    """Un `chunk_id` cité est une référence qu'un lecteur ne peut pas suivre, et qui ne
    survit pas à un re-découpage du corpus."""
    row = quant_rag._payload_row(_payload(), 0.5, "cosine")
    ligne = f"{row['source']}, {quant_rag.citation_pages(row['pages_utilisateur'])}"
    assert "chunk-" not in ligne and "doc-" not in ligne
    assert "p. 35–36" in ligne


def test_le_rendu_mcp_separe_ce_qui_se_cite_de_ce_qui_est_interne():
    source = (MACOS / "mcp_server.py").read_text(encoding="utf-8")
    bloc = source.split("def search_documents(")[1].split("@mcp.tool()")[0]
    assert "citer   :" in bloc, "la ligne à recopier doit être nommée"
    assert "interne :" in bloc, "les clés de travail doivent être nommées comme telles"
    assert "citation_pages" in bloc, "les pages affichées doivent passer par la conversion"
    assert "page_start" not in bloc, "aucune page 0-basée ne doit être affichée"


def test_get_passage_affiche_lui_aussi_une_page_de_lecteur():
    source = (MACOS / "mcp_server.py").read_text(encoding="utf-8")
    bloc = source.split("def get_passage(")[1].split("@mcp.tool()")[0]
    assert "citation_pages" in bloc
    assert "ne pas citer" in bloc


# ------------------------------------------- la garde qui aurait dû exister du premier coup


def test_aucun_affichage_servi_ne_montre_une_page_zero_basee():
    """La garde **générique**, et la raison pour laquelle elle l'est.

    La première version de ce fichier ne gardait que ``search_documents`` et ``get_passage``,
    c'est-à-dire exactement les deux fonctions que la correction avait touchées. Elle passait
    au vert pendant que quatre autres affichages servis — ``timeline`` côté MCP,
    ``format_search``, ``format_expand`` et ``format_connect`` côté graphe — continuaient
    d'imprimer ``page_start``/``page_end`` bruts, donc 0-basés. Un test taillé sur le
    correctif ne mesure pas le défaut : il mesure le correctif.
    """
    for module in ("mcp_server.py", "graph_search.py"):
        source = (MACOS / module).read_text(encoding="utf-8")
        for champ in ("page_start", "page_end"):
            for motif in (f"{champ}']}}", f'{champ}"]}}', f"{{{champ}}}"):
                assert motif not in source, (
                    f"{module} interpole {champ} dans une chaîne affichée : c'est une page "
                    "0-basée, elle doit passer par quant_rag.citation_pages")


def test_chaque_affichage_servi_nomme_ce_qui_se_cite_et_ce_qui_est_interne():
    rendus = {
        "mcp_server.py": ("def search_documents(", "def get_passage(", "def timeline("),
        "graph_search.py": ("def format_search(", "def format_connect("),
    }
    for module, fonctions in rendus.items():
        source = (MACOS / module).read_text(encoding="utf-8")
        for fonction in fonctions:
            bloc = source.split(fonction)[1].split("\ndef ")[0]
            assert "citation_pages" in bloc, f"{module}:{fonction} n'affiche pas une page de lecteur"
            assert "citer" in bloc, f"{module}:{fonction} ne nomme pas la ligne à recopier"
            assert "ne pas citer" in bloc, f"{module}:{fonction} ne marque pas ses clés internes"


def test_le_graphe_ne_promet_plus_une_page_citable_qu_il_ne_convertissait_pas():
    """Sa documentation annonçait « source citable, pages » — sans conversion."""
    source = (MACOS / "graph_search.py").read_text(encoding="utf-8")
    entete = source.split('"""')[1]
    assert "0-bas" in entete, "le module doit dire quelle convention ses pages suivaient"
    bloc = source.split("def format_expand(")[1].split("\ndef ")[0]
    assert "pages_utilisateur" in bloc, "les exemples de expand_entity citent une page brute"


# ------------------------------------------------------------------ non-régression


def test_le_correctif_n_ajoute_qu_un_champ_et_ne_touche_pas_au_classement():
    """Le contrat de sortie du retrieval ne doit pas bouger autrement que par un ajout.

    Panne évitée : un « correctif de citation » qui déplacerait un score, un ordre ou un
    filtre serait un changement de retrieval déguisé — et le dossier interdit d'en faire un
    sans mesure sur la population fixe.
    """
    attendus = {"chunk_id", "document_id", "title", "authors", "publication_year", "short_ref",
                "source", "section", "page_start", "page_end", "content_type", "score",
                "score_kind", "text"}
    #: Ajoutés par le contrat de sortie, et **ajoutés seulement** : les offsets du passage dans
    #: le texte canonique de son document, la granularité de cet ancrage, et l'empreinte du
    #: texte. Aucun score, aucun ordre, aucun filtre n'en dépend — ``_select`` et ``route`` ne
    #: les lisent pas, ce que garde le test suivant.
    ancrage = {"doc_text_sha256", "ancrage_granularite", "ancrage_intervalles"}
    row = quant_rag._payload_row(_payload(), 0.5, "cosine")
    assert set(row) == attendus | {"pages_utilisateur"} | ancrage
    assert row["score"] == 0.5 and row["score_kind"] == "cosine"


def test_l_ancrage_ne_touche_ni_au_classement_ni_a_la_selection():
    """Garde textuelle, symétrique de celle des pages : l'ancre est de l'affichage, rien d'autre."""
    source = (MACOS / "quant_rag.py").read_text(encoding="utf-8")
    for fonction in ("def _select(", "def route(", "def _dense(", "def _rerank("):
        if fonction not in source:
            continue
        corps = source.split(fonction)[1].split("\ndef ")[0]
        for champ in ("doc_text_sha256", "ancrage_granularite", "ancrage_intervalles"):
            assert champ not in corps, f"{fonction} ne doit pas dépendre de l'ancrage"


def test_aucune_fonction_de_selection_ou_de_classement_n_est_touchee():
    """Garde textuelle : la conversion de page ne doit apparaître que dans l'affichage."""
    source = (MACOS / "quant_rag.py").read_text(encoding="utf-8")
    for fonction in ("def _select(", "def route(", "def search_explained("):
        if fonction not in source:
            continue
        corps = source.split(fonction)[1].split("\ndef ")[0]
        assert "pages_utilisateur" not in corps, f"{fonction} ne doit pas dépendre de l'affichage"
        assert "citation_pages" not in corps, f"{fonction} ne doit pas dépendre de l'affichage"
