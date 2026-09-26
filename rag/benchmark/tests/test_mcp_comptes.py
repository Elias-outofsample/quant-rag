"""Ce que le serveur MCP annonce du corpus doit être vrai, et le rester sans intervention.

Les comptes de ``search_documents`` étaient des constantes écrites à la main. Ils ont été
faux deux fois en quatorze heures : 256/18 636 corrigé le 5 septembre, 319/22 190 corrigé
le 6, parce qu'aucune étape de l'ingestion ne les met à jour. Un compte faux annoncé à
l'appelant est pire qu'absent — le modèle raisonne dessus.

Ces tests disent trois choses : plus aucun compte n'est écrit en dur, les comptes lus
concordent avec les artefacts de la signature vivante, et l'absence d'artefact ne produit
pas un mensonge mais un silence.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MACOS = Path(__file__).resolve().parents[2]
RACINE = MACOS.parent
sys.path.insert(0, str(MACOS))

import corpus_overlay  # noqa: E402
import mcp_server  # noqa: E402


def test_aucun_compte_ecrit_en_dur_dans_la_docstring_source():
    """Le gabarit ne doit contenir aucun nombre de documents ou de passages."""
    source = (MACOS / "mcp_server.py").read_text(encoding="utf-8")
    gabarit = source.split('def search_documents(')[1].split('"""')[1]
    assert "{portee}" in gabarit and "{annee_inconnue}" in gabarit
    # Le mode de panne réel est précis : un nombre collé au mot « documents » ou
    # « passages ». Un identifiant d'exemple (arXiv 1206.0682) n'en est pas un, et une
    # règle qui l'attraperait serait une règle qu'on désactiverait au premier faux positif.
    compte = re.compile(r"\d[\d   ]*\s+(?:documents?|passages?)")
    suspects = [m for m in compte.findall(gabarit)]
    assert suspects == [], f"comptes réintroduits en dur : {suspects}"


def test_les_comptes_concordent_avec_le_manifeste_de_la_signature_vivante():
    manifeste = (RACINE / "data" / "lexical" /
                 f"bm25-dedup-tables-registry-titles-v1-{corpus_overlay.signature()}"
                 ".manifest.json")
    if not manifeste.exists():          # aucun index construit pour cette signature
        assert mcp_server._comptes()["passages"] is None
        return
    origine = json.loads(manifeste.read_text(encoding="utf-8"))["built_from"]
    comptes = mcp_server._comptes()
    assert comptes["documents"] == origine["documents"]
    assert comptes["passages"] == origine["points"]


def test_la_phrase_publiee_porte_les_comptes_lus():
    comptes = mcp_server._comptes()
    premiere = mcp_server.search_documents.__doc__.splitlines()[0]
    if comptes["documents"]:
        assert str(comptes["documents"]) in premiere
    if comptes["passages"]:
        # séparateur de milliers français dans la phrase publiée
        assert f"{comptes['passages']:,}".replace(",", " ") in premiere


def test_sans_artefact_le_serveur_se_tait_au_lieu_de_mentir(monkeypatch):
    """Une signature sans index ne doit pas produire un compte inventé."""
    monkeypatch.setattr(corpus_overlay, "signature", lambda: "zzzzzzzzzz")
    comptes = mcp_server._comptes()
    assert comptes["passages"] is None
    # les documents restent lisibles depuis les fiches, eux, et ce n'est pas une invention
    assert comptes["documents"] is None or comptes["documents"] > 0


def test_le_nombre_de_documents_sans_annee_exclut_les_documents_retires():
    fiches = json.loads((MACOS / "metadata" / "documents-metadata-v1.json")
                        .read_text(encoding="utf-8"))
    fiches = fiches if isinstance(fiches, list) else fiches.get("documents", [])
    retires = corpus_overlay.removed_documents()
    attendu = sum(1 for f in fiches
                  if f.get("document_id") not in retires and not f.get("publication_year"))
    assert mcp_server._comptes()["sans_annee"] == attendu


def test_une_accolade_litterale_dans_la_docstring_ne_casse_pas_le_serveur():
    """``.format()`` sur toute la docstring était une mine : un exemple JSON ajouté plus
    tard aurait fait planter le serveur au démarrage. Relecture adverse du 6 septembre."""
    def factice():
        pass
    factice.__doc__ = ('Exemple : passe {"chunk_id": "c1"} en argument. '
                       'Corpus : {portee}. Année inconnue : {annee_inconnue}.')
    rendu = mcp_server._avec_comptes(factice).__doc__
    assert '{"chunk_id": "c1"}' in rendu
    assert "{portee}" not in rendu and "{annee_inconnue}" not in rendu
