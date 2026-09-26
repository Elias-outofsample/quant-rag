"""La garde de dérive — « parade B ». Un serveur périmé doit **refuser**, pas répondre.

Le défaut interdit ici a un nom et une adresse : ``../quant-rag-migration-qdrant/qdrant_storage_server/``
pèse 603 Mo et porte une copie complète de la collection servie. Elle est identique au corpus
**aujourd'hui** ; elle divergera au premier document ingéré. Après la fusion de
``migration-qdrant``, ``export QUANT_RAG_QDRANT_URL=http://localhost:6533`` suffit à la servir,
et rien ne le dirait : la recherche répondrait normalement, sur un corpus périmé. C'est la
famille de défaut que le dossier s'interdit — *« un index juste sous un nom qui ment »*.

**La moitié de ces tests exige un refus.** Un test qui vérifie seulement qu'une garde laisse
passer le cas nominal ne prouve rien : c'est le comportement d'une garde absente.

Deux propriétés sont testées séparément, et l'ordre compte :

1. **en mode embarqué, la garde n'existe pas.** C'est la propriété que ``qdrant_backend`` a
   été écrit pour tenir, et elle prime : le chemin par défaut du dépôt ne bouge pas d'un bit,
   même sans manifeste, même avec une collection vide ;
2. **en mode serveur, elle refuse** dans les trois cas où la provenance n'est pas prouvée.

``verdict()`` est pure : elle se teste sans serveur, sans stockage et sans ``qdrant_client``.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

MACOS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MACOS))

import qdrant_backend as qb  # noqa: E402

COLLECTION = "quant_rag_ingested_all_qwen3_06b"
CIBLE = "http://localhost:6533"


@pytest.fixture(autouse=True)
def _sans_variable(monkeypatch):
    """Chaque test part d'un environnement propre : la variable ne fuit pas d'un test à l'autre."""
    monkeypatch.delenv(qb.VARIABLE, raising=False)


class FauxClient:
    """Un client qui ne parle à personne — la garde se teste sans infrastructure.

    Il enregistre sa fermeture : une garde qui refuse en laissant un client ouvert fuirait une
    connexion à chaque refus, et sur le mode serveur un refus n'est pas un cas rare.
    """

    def __init__(self, points: int | None = 26120, **kwargs):
        self.points = points
        self.kwargs = kwargs
        self.ferme = False

    def collection_exists(self, nom):
        return self.points is not None

    def count(self, nom):
        class _C:
            count = self.points
        return _C()

    def close(self):
        self.ferme = True


# --------------------------------------------------------------------- verdict(), fonction pure

def test_sans_manifeste_le_verdict_refuse():
    motif = qb.verdict(26120, None, COLLECTION, CIBLE)
    assert motif is not None
    assert "check_bm25" in motif, "le refus doit nommer le geste qui le lève"


def test_collection_absente_du_serveur_le_verdict_refuse():
    motif = qb.verdict(None, {"points": 26120, "documents": 418}, COLLECTION, CIBLE)
    assert motif is not None
    assert "n'existe pas" in motif


def test_un_ecart_de_compte_refuse_et_donne_les_deux_nombres():
    """Le cas nommé : un serveur en retard d'un document."""
    motif = qb.verdict(26120, {"points": 26185, "documents": 419}, COLLECTION, CIBLE)
    assert motif is not None
    assert "26120" in motif and "26185" in motif, "un refus sans les deux chiffres n'aide personne"
    assert "périmée" in motif


def test_le_compte_juste_laisse_passer():
    assert qb.verdict(26120, {"points": 26120, "documents": 418}, COLLECTION, CIBLE) is None


# ------------------------------------------------------- le défaut du dépôt n'a pas bougé d'un bit

def test_en_embarque_la_garde_ne_se_declenche_jamais(monkeypatch, tmp_path):
    """Même sans manifeste, même sans collection : le mode embarqué est intouché."""
    monkeypatch.setattr(qb, "comptes_attendus", lambda: None)
    monkeypatch.setattr("qdrant_client.QdrantClient", FauxClient)
    handle = qb.ouvrir(tmp_path, collection=COLLECTION)
    assert handle.kwargs == {"path": str(tmp_path)}
    assert not handle.ferme


def test_en_embarque_la_garde_ne_se_declenche_pas_meme_sans_collection_nommee(monkeypatch, tmp_path):
    monkeypatch.setattr(qb, "comptes_attendus", lambda: None)
    monkeypatch.setattr("qdrant_client.QdrantClient", FauxClient)
    assert qb.ouvrir(tmp_path).kwargs == {"path": str(tmp_path)}


# --------------------------------------------------------------- en mode serveur, elle doit refuser

def test_serveur_perime_refuse_et_ferme_le_client(monkeypatch, tmp_path):
    """LE test du chantier : le volume de 603 Mo, servi par accident, ne répond pas."""
    monkeypatch.setenv(qb.VARIABLE, CIBLE)
    monkeypatch.setattr(qb, "comptes_attendus", lambda: {"points": 26185, "documents": 419})
    ouverts = []
    monkeypatch.setattr("qdrant_client.QdrantClient",
                        lambda **kw: ouverts.append(FauxClient(points=26120, **kw)) or ouverts[-1])
    with pytest.raises(RuntimeError) as leve:
        qb.ouvrir(tmp_path, collection=COLLECTION)
    assert "26120" in str(leve.value) and "26185" in str(leve.value)
    assert ouverts[0].ferme, "un refus qui laisse le client ouvert fuit une connexion"


def test_serveur_sans_manifeste_refuse(monkeypatch, tmp_path):
    monkeypatch.setenv(qb.VARIABLE, CIBLE)
    monkeypatch.setattr(qb, "comptes_attendus", lambda: None)
    monkeypatch.setattr("qdrant_client.QdrantClient", FauxClient)
    with pytest.raises(RuntimeError, match="check_bm25"):
        qb.ouvrir(tmp_path, collection=COLLECTION)


def test_serveur_sans_collection_nommee_refuse(monkeypatch, tmp_path):
    """Une garde qu'on désarme en oubliant un argument n'est pas une garde."""
    monkeypatch.setenv(qb.VARIABLE, CIBLE)
    monkeypatch.setattr("qdrant_client.QdrantClient", FauxClient)
    with pytest.raises(RuntimeError, match="n'a pas nommé la collection"):
        qb.ouvrir(tmp_path)


def test_serveur_juste_laisse_passer(monkeypatch, tmp_path):
    monkeypatch.setenv(qb.VARIABLE, CIBLE)
    monkeypatch.setattr(qb, "comptes_attendus", lambda: {"points": 26120, "documents": 418})
    monkeypatch.setattr("qdrant_client.QdrantClient", FauxClient)
    handle = qb.ouvrir(tmp_path, collection=COLLECTION)
    assert handle.kwargs == {"url": CIBLE}
    assert not handle.ferme


def test_garde_false_dispense_explicitement(monkeypatch, tmp_path):
    """``build_index`` s'en dispense parce qu'il détruit la collection à la ligne suivante."""
    monkeypatch.setenv(qb.VARIABLE, CIBLE)
    monkeypatch.setattr(qb, "comptes_attendus", lambda: {"points": 26185, "documents": 419})
    monkeypatch.setattr("qdrant_client.QdrantClient", FauxClient)
    assert qb.ouvrir(tmp_path, garde=False).kwargs == {"url": CIBLE}


def test_une_garde_qui_ne_peut_pas_compter_leve(monkeypatch, tmp_path):
    """Une garde muette vaut une garde absente : l'erreur de comptage remonte."""
    class Muet(FauxClient):
        def count(self, nom):
            raise ConnectionError("connexion refusée")

    monkeypatch.setenv(qb.VARIABLE, CIBLE)
    monkeypatch.setattr(qb, "comptes_attendus", lambda: {"points": 26120, "documents": 418})
    monkeypatch.setattr("qdrant_client.QdrantClient", Muet)
    with pytest.raises(RuntimeError, match="impossible de compter"):
        qb.ouvrir(tmp_path, collection=COLLECTION)


def test_en_mode_serveur_le_refus_est_celui_de_la_garde_jamais_du_stockage(monkeypatch, tmp_path):
    """Les deux refus ne doivent pas se confondre — et l'ancien ne doit pas revenir.

    ``test_qdrant_backend.py`` tient la propriété « un serveur n'exige aucun dossier local ».
    La garde posée par ce chantier passe par le même chemin ; ce test prouve qu'elle ne l'a
    pas réintroduit par la bande : le stockage est absent, et le motif du refus parle du
    corpus, jamais de l'index local.
    """
    monkeypatch.setenv(qb.VARIABLE, CIBLE)
    monkeypatch.setattr(qb, "comptes_attendus", lambda: {"points": 26185, "documents": 419})
    monkeypatch.setattr("qdrant_client.QdrantClient", FauxClient)
    with pytest.raises(RuntimeError) as leve:
        qb.ouvrir(tmp_path / "nexiste-pas", message_si_absent="Index absent", collection=COLLECTION)
    assert "Index absent" not in str(leve.value)
    assert "périmée" in str(leve.value)


# ------------------------------------------------------------------- l'artefact du dépôt, tel quel

def test_le_manifeste_du_depot_dit_ce_que_la_collection_contient():
    """Sans cet artefact, la garde n'a rien à confronter. Il est donc une dépendance."""
    attendu = qb.comptes_attendus()
    assert attendu is not None, ("aucun manifeste BM25 pour la signature courante — "
                                 "lance rag/benchmark/check_bm25.py")
    assert attendu["points"] > 0 and attendu["documents"] > 0
    assert attendu["collection"] == COLLECTION


# ------------------------------------------ aucun site ne peut désarmer la garde par distraction

def test_tout_appelant_de_ouvrir_nomme_sa_collection_ou_se_dispense_par_ecrit():
    """La garde ne doit pas pouvoir s'éteindre par omission.

    Même esprit que la garde AST de ``test_qdrant_backend.py`` : elle attrape un site futur,
    pas seulement les quatre connus. Un appel qui ne nomme ni ``collection`` ni ``garde``
    lèverait en mode serveur — mieux vaut l'attraper ici, à froid, que dans un lot.
    """
    fautifs = []
    for chemin in MACOS.rglob("*.py"):
        if "__pycache__" in chemin.parts or chemin.name == "qdrant_backend.py":
            continue
        if "tests" in chemin.parts:  # les tests appellent ouvrir pour l'éprouver
            continue
        arbre = ast.parse(chemin.read_text(encoding="utf-8"), filename=str(chemin))
        for noeud in ast.walk(arbre):
            if not isinstance(noeud, ast.Call) or getattr(noeud.func, "attr", None) != "ouvrir":
                continue
            mots = {k.arg for k in noeud.keywords}
            if not ({"collection", "garde"} & mots):
                fautifs.append(f"{chemin.relative_to(MACOS)}:{noeud.lineno}")
    assert fautifs == [], ("ces appels à qdrant_backend.ouvrir désarment la garde de dérive "
                           "par omission : " + ", ".join(fautifs))
