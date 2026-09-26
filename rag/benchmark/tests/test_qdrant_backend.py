"""Sans ``QUANT_RAG_QDRANT_URL``, le dépôt ouvre exactement ce qu'il ouvrait hier.

C'est la seule propriété qui compte pour la session A : le point d'injection du chantier B ne
doit rien changer tant que la variable n'est pas définie. Un test qui vérifierait seulement
« la variable marche » laisserait passer le défaut inverse — un backend qui bascule tout seul.

La règle a un domicile unique (``rag/qdrant_backend.py``) parce que quatre modules ouvraient
la même ligne en dur. Une seule copie oubliée servirait un autre corpus sous le même nom : ces
tests vérifient donc aussi qu'il **n'en reste aucune**.

``arguments()`` est une fonction pure : elle ne construit rien, n'importe pas
``qdrant_client``, et n'a besoin ni de serveur ni de stockage. C'est ce qui rend la propriété
ci-dessus prouvable sans infrastructure.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

MACOS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MACOS))

import qdrant_backend as qb  # noqa: E402

#: Deux dispenses, et chacune porte sa raison — une liste de dispenses sans motif finit par
#: tout dispenser.
DISPENSES = {
    # le domicile de la règle : c'est lui qui construit le client
    "qdrant_backend.py",
    # instrument de VÉRIFICATION : il interroge un serveur arbitraire passé en --url, sur un
    # port de test, et ne touche jamais au corpus servi. Le brancher sur qdrant_backend
    # reviendrait à lui faire lire l'environnement du chemin servi — l'inverse de son rôle.
    "verif_exactitude_serveur.py",
    # instrument de MIGRATION : il ouvre DEUX backends nommés explicitement en argument — la
    # source embarquée (--source) et la cible serveur (--url). Les faire passer par
    # qdrant_backend leur ferait lire QUANT_RAG_QDRANT_URL, c'est-à-dire choisir un backend
    # que l'appelant n'a pas demandé : la source cesserait d'être embarquée dès que la
    # variable serait définie, et la copie s'écrirait dans sa propre source.
    "migrer_vers_serveur.py",
    # TÉMOIN M4 : ouvre la cible serveur nommée en --url pour compter AVANT et APRÈS une
    # reconstruction. C'est précisément une garde — elle doit interroger un backend choisi par
    # l'appelant, pas celui que l'environnement désigne.
    "temoin_m4.py",
    # CONTRÔLES OPÉRATIONNELS : ils ouvrent délibérément des clients FAUTIFS — port fermé,
    # hôte inconnu, URL malformée — pour vérifier que chaque panne échoue bruyamment. Les
    # router vers qdrant_backend les empêcherait d'être faux, donc de mesurer quoi que ce soit.
    "controles_operationnels.py",
    # DÉMONSTRATION DE CONCURRENCE : elle ouvre N clients sur une cible nommée en argument,
    # dans des processus distincts, pour opposer le verrou exclusif à la lecture concurrente.
    "demo_concurrence.py",
    # MESURE DE LATENCE (8 septembre 2026, chantier reprise-ingestion) : elle interroge les
    # BRAS de mesure laissés par verif_exactitude_serveur.py --garder — des collections
    # jetables nommées verif_exactitude_* qui ne sont pas le corpus servi. Les router vers
    # qdrant_backend leur ferait lire QUANT_RAG_QDRANT_URL, donc changer de cible selon
    # l'environnement, et armerait la garde de dérive sur des collections dont le compte
    # n'a aucune raison d'égaler celui du corpus : la mesure refuserait de partir.
    # Le côté EMBARQUÉ de cette même mesure, lui, passe bien par quant_rag.client().
    "latence_backends.py",
    # TESTS D'INTÉGRATION DU CHEMIN D'ÉCRITURE (8 septembre 2026) : ils construisent une
    # collection JETABLE dans tmp_path, avec des vecteurs aléatoires, et remplacent 26
    # constantes de chemin pour que rien du corpus servi ne soit touché. Les router vers
    # qdrant_backend leur ferait lire QUANT_RAG_QDRANT_URL — donc, si la variable est
    # définie dans l'environnement, exécuter des tests d'ÉCRITURE contre un vrai serveur.
    # Le module fait d'ailleurs l'inverse et c'est délibéré : il efface la variable
    # (monkeypatch.delenv) avant chaque test.
    "test_apply_delivery.py",
}


@pytest.fixture(autouse=True)
def _sans_variable(monkeypatch):
    """Chaque test part d'un environnement propre : la variable ne fuit pas d'un test à l'autre."""
    monkeypatch.delenv(qb.VARIABLE, raising=False)


# ------------------------------------------------------- le défaut, et il est le contrat


def test_sans_la_variable_on_ouvre_le_stockage_embarque():
    assert qb.arguments("/un/chemin") == {"path": "/un/chemin"}
    assert qb.mode("/un/chemin") == "embarque"


def test_une_variable_vide_vaut_absence(monkeypatch):
    """Une variable exportée à vide est une erreur d'exploitation courante.

    La traiter comme une URL produirait un client qui échoue à la connexion, là où
    l'utilisateur croyait avoir désactivé le serveur.
    """
    for valeur in ("", "   ", "\t\n"):
        monkeypatch.setenv(qb.VARIABLE, valeur)
        assert qb.arguments("/un/chemin") == {"path": "/un/chemin"}


def test_un_objet_Path_devient_une_chaine():
    """``QdrantClient`` attend une chaîne : un Path silencieusement accepté ailleurs casserait ici."""
    assert qb.arguments(Path("/un/chemin")) == {"path": "/un/chemin"}


# ------------------------------------------------------------------- le mode serveur


def test_avec_la_variable_on_ouvre_une_url(monkeypatch):
    monkeypatch.setenv(qb.VARIABLE, "http://localhost:6533")
    assert qb.arguments("/un/chemin") == {"url": "http://localhost:6533"}
    assert qb.mode("/un/chemin") == "serveur"


def test_les_blancs_autour_de_l_url_sont_retires(monkeypatch):
    monkeypatch.setenv(qb.VARIABLE, "  http://localhost:6533\n")
    assert qb.arguments("/x") == {"url": "http://localhost:6533"}


# --------------------------------------------------- le refus historique, et sa portée


def test_le_stockage_absent_est_refuse_en_mode_embarque(tmp_path):
    """``quant_rag.client()`` refusait un index absent : le refus est conservé à l'identique."""
    with pytest.raises(RuntimeError, match="Index absent"):
        qb.ouvrir(tmp_path / "nexiste-pas", message_si_absent="Index absent. Lance d'abord: …")


def test_le_stockage_absent_n_est_PAS_refuse_en_mode_serveur(monkeypatch, tmp_path):
    """Un serveur n'a aucun besoin d'un dossier local.

    Exiger sa présence rendrait la migration impossible sur une machine qui n'a jamais eu de
    collection embarquée — et c'est précisément la situation d'une machine cible.

    ``garde=False`` ajouté le 8 septembre 2026 par le chantier ``reprise-ingestion``, qui a
    posé une garde de dérive dans ``ouvrir`` (« parade B »). **La propriété testée ici est
    inchangée** : ce test porte sur ``message_si_absent``, pas sur la garde, et le faux client
    qu'il injecte est un tuple — il n'a ni ``count`` ni ``close``. Que le refus de la garde et
    celui du stockage absent restent bien distincts est prouvé à part, par
    ``test_garde_derive.py::test_en_mode_serveur_le_refus_est_celui_de_la_garde_jamais_du_stockage``.
    """
    monkeypatch.setenv(qb.VARIABLE, "http://localhost:6533")
    faux = type(sys)("qdrant_client")
    faux.QdrantClient = lambda **kwargs: ("client", kwargs)
    monkeypatch.setitem(sys.modules, "qdrant_client", faux)

    resultat = qb.ouvrir(tmp_path / "nexiste-pas", message_si_absent="Index absent", garde=False)
    assert resultat == ("client", {"url": "http://localhost:6533"})


def test_ouvrir_ne_memoise_pas(monkeypatch, tmp_path):
    """Le cycle de vie appartient à l'appelant : ``build_index`` supprime et recrée, il ne peut
    pas partager la poignée mémoïsée du chemin servi."""
    appels = []
    faux = type(sys)("qdrant_client")
    faux.QdrantClient = lambda **kwargs: appels.append(kwargs) or object()
    monkeypatch.setitem(sys.modules, "qdrant_client", faux)

    a = qb.ouvrir(tmp_path)
    b = qb.ouvrir(tmp_path)
    assert a is not b and len(appels) == 2


# ------------------------------------------- aucune copie de la règle ne doit subsister


def test_aucun_module_de_rag_n_ouvre_encore_un_client_en_dur():
    """Le défaut que ce chantier corrige : quatre copies d'une même ligne.

    Trois sites migrés et un oublié serviraient deux corpus sous le même nom, sans qu'aucun
    test ne le voie. La garde est générique — elle attrape une copie future, pas seulement les
    quatre connues.
    """
    fautifs = []
    for chemin in MACOS.rglob("*.py"):
        if chemin.name in DISPENSES or "__pycache__" in chemin.parts:
            continue
        arbre = ast.parse(chemin.read_text(encoding="utf-8"), filename=str(chemin))
        for noeud in ast.walk(arbre):
            # L'AST, et non un motif textuel : un docstring qui CITE la ligne n'est pas un
            # appel, et un test taillé sur du texte se serait déclenché dessus.
            if not isinstance(noeud, ast.Call):
                continue
            nom = getattr(noeud.func, "id", None) or getattr(noeud.func, "attr", None)
            if nom != "QdrantClient":
                continue
            # Un mot-clé littéral (path=/url=/host=) OU un dépaquetage `**kwargs` : sans le
            # second, `QdrantClient(**args)` passait à travers la garde. Angle mort trouvé en
            # écrivant demo_concurrence.py, corrigé ici plutôt que dispensé.
            explicite = any(k.arg in ("path", "url", "host") for k in noeud.keywords)
            depaquete = any(k.arg is None for k in noeud.keywords)
            if explicite or depaquete:
                fautifs.append(f"{chemin.relative_to(MACOS)}:{noeud.lineno}")
    assert fautifs == [], ("ces modules ouvrent un client sans passer par qdrant_backend : "
                          + ", ".join(fautifs))


def test_les_quatre_sites_historiques_passent_par_le_domicile_unique():
    for relatif in ("quant_rag.py", "build_index.py", "ingestion/recover_vectors.py"):
        texte = (MACOS / relatif).read_text(encoding="utf-8")
        assert "qdrant_backend.ouvrir(" in texte, f"{relatif} n'appelle pas qdrant_backend.ouvrir"
