"""La garde du gel doit refuser — et on doit pouvoir la faire refuser à volonté.

``check_bm25_guard`` a établi l'usage dans ce dépôt : une garde qu'on n'a jamais vue
échouer n'est pas une garde, c'est une décoration. Ces tests la font échouer condition
par condition, et vérifient qu'elle laisse passer quand elle doit laisser passer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

MACOS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MACOS))

import gel_corpus  # noqa: E402


@pytest.fixture
def declaration(tmp_path, monkeypatch):
    chemin = tmp_path / "gel-corpus.json"
    monkeypatch.setattr(gel_corpus, "DECLARATION", chemin)
    return chemin


def _signature(monkeypatch, valeur):
    monkeypatch.setattr(gel_corpus.corpus_overlay, "signature", lambda: valeur)


def test_sans_declaration_rien_n_est_interdit(declaration, monkeypatch):
    _signature(monkeypatch, "aaaaaaaaaa")
    assert gel_corpus.etat()["accorde"] is True
    assert gel_corpus.verifier("un import", sortir=False) is True


def test_gel_pose_puis_signature_inchangee_laisse_passer(declaration, monkeypatch):
    _signature(monkeypatch, "aaaaaaaaaa")
    gel_corpus.geler("chantier découpage")
    assert gel_corpus.etat()["accorde"] is True
    assert gel_corpus.verifier("un import", sortir=False) is True


def test_gel_pose_puis_signature_qui_bouge_refuse(declaration, monkeypatch, capsys):
    """Le cas qui a réellement cassé la mesure : le corpus bouge pendant qu'on mesure."""
    _signature(monkeypatch, "aaaaaaaaaa")
    gel_corpus.geler("chantier découpage")
    _signature(monkeypatch, "bbbbbbbbbb")           # un document est entré
    assert gel_corpus.etat()["accorde"] is False
    assert gel_corpus.verifier("l'import d'un document", sortir=False) is False
    erreur = capsys.readouterr().err
    assert "GEL DU CORPUS ACTIF" in erreur
    assert "chantier découpage" in erreur
    assert "--degeler" in erreur                    # la sortie dit comment lever


def test_la_garde_quitte_vraiment_le_processus(declaration, monkeypatch):
    """``sortir=True`` est le chemin réel du driver : il doit arrêter le processus."""
    _signature(monkeypatch, "aaaaaaaaaa")
    gel_corpus.geler("chantier découpage")
    _signature(monkeypatch, "bbbbbbbbbb")
    with pytest.raises(SystemExit):
        gel_corpus.verifier("l'import d'un document")


def test_degel_rouvre_et_laisse_une_trace(declaration, monkeypatch):
    _signature(monkeypatch, "aaaaaaaaaa")
    gel_corpus.geler("chantier découpage")
    _signature(monkeypatch, "bbbbbbbbbb")
    gel_corpus.degeler("lot arXiv de septembre")
    assert gel_corpus.etat()["accorde"] is True
    historique = json.loads(declaration.read_text(encoding="utf-8"))["historique"]
    assert [entree["acte"] for entree in historique] == ["gel", "dégel"]
    assert historique[-1]["raison"] == "lot arXiv de septembre"


def test_le_driver_appelle_bien_la_garde():
    """Sans cet appel, la règle serait un document sans effet."""
    source = (MACOS / "ingestion" / "batch_driver.py").read_text(encoding="utf-8")
    assert "gel_corpus.verifier(" in source
    # et seulement sur le chemin qui écrit : --dry-run doit rester utilisable sous gel
    avant_garde = source.split("gel_corpus.verifier(")[0]
    assert avant_garde.rstrip().endswith("import gel_corpus")
    assert "if arguments.go:" in avant_garde[-400:]


def test_le_banc_inscrit_le_gel_dans_ses_resultats():
    """Un résultat qui ne dit pas s'il a été pris sous gel est un résultat qu'on ne peut
    pas relire dans six mois."""
    source = (MACOS / "benchmark" / "eval_router.py").read_text(encoding="utf-8")
    assert '"gel": gel_corpus.etat()' in source


def test_un_gel_ne_peut_pas_en_ecraser_un_autre_en_silence(declaration, monkeypatch):
    """Trouvé par relecture adverse : un second gel remplaçait le premier sans un mot,
    et la trace mentait alors sur ce qui était protégé pendant la mesure."""
    _signature(monkeypatch, "aaaaaaaaaa")
    gel_corpus.geler("chantier X, en cours de mesure")
    with pytest.raises(SystemExit) as sortie:
        gel_corpus.geler("chantier Y, lancé par erreur")
    assert "UN GEL EST DÉJÀ ACTIF" in str(sortie.value)
    assert gel_corpus.etat()["chantier"] == "chantier X, en cours de mesure"
    # forcer reste possible, mais il faut le dire
    gel_corpus.geler("chantier Y, assumé", forcer=True)
    assert gel_corpus.etat()["chantier"] == "chantier Y, assumé"


def test_le_degel_garde_le_nom_du_chantier_dans_la_trace(declaration, monkeypatch):
    _signature(monkeypatch, "aaaaaaaaaa")
    gel_corpus.geler("chantier découpage")
    gel_corpus.degeler("lot arXiv")
    derniere = json.loads(declaration.read_text(encoding="utf-8"))["historique"][-1]
    assert derniere["chantier"] == "chantier découpage"
    assert derniere["raison"] == "lot arXiv"


# ------------------------------------------------------------------ la PORTÉE de la garde
# Ajoutés le 8 septembre 2026 par le chantier `reprise-ingestion`. Le comportement testé
# plus haut était juste ; c'est la docstring du module qui décrivait autre chose, et deux
# de ses phrases étaient fausses. Ces trois tests figent la portée réelle, pour qu'un
# commentaire ne puisse plus s'en éloigner sans qu'un test tombe.

def test_le_gel_est_un_detecteur_de_derive_et_non_un_verrou(declaration, monkeypatch):
    """Gel ACTIF et signature INCHANGÉE : ``--go`` passe. C'est l'état du dépôt.

    La docstring annonçait « tant qu'un gel est actif, rien n'entre au corpus ». C'est
    faux : la garde ne se déclenche qu'**après** que la signature a bougé, donc après
    qu'un document est entré. Elle constate la dérive, elle ne l'empêche pas.
    """
    _signature(monkeypatch, "5530cba145")
    gel_corpus.geler("chantier découpage")
    etat = gel_corpus.etat()
    assert etat["actif"] is True
    assert etat["signature_gelee"] == etat["signature_vivante"] == "5530cba145"
    assert etat["accorde"] is True, ("si ceci devient False, le gel est devenu un verrou : "
                                     "c'est un CHANGEMENT DE COMPORTEMENT, pas une correction")
    assert gel_corpus.verifier("l'import d'un document au corpus", sortir=False) is True


def test_seuls_les_chemins_qui_ecrivent_consultent_le_gel():
    """La docstring annonçait « les deux entrées du banc » : elles n'appellent que ``etat()``.

    Liste élargie le 10 septembre 2026 par le lot ``cloture-operationnelle`` : ``apply_delivery``
    et ``retire_document`` écrivaient sous gel sans un mot. Ce test reste un garde-fou de
    documentation — tout nouvel appelant doit apparaître ici **et** dans la docstring du module.
    """
    appelants = sorted(
        chemin.relative_to(MACOS).as_posix()
        for chemin in MACOS.rglob("*.py")
        if "__pycache__" not in chemin.parts
        and chemin.name not in ("gel_corpus.py", "test_gel_corpus.py")
        and "gel_corpus.verifier(" in chemin.read_text(encoding="utf-8"))
    assert appelants == ["ingestion/apply_delivery.py",
                         "ingestion/batch_driver.py",
                         "ingestion/retire_document.py"], (
        "la portée de la garde a changé — mets la docstring de gel_corpus.py d'accord "
        f"avec elle. Appelants trouvés : {appelants}")


# ------------------------------------------------------- les points d'écriture consultent le gel
# Ajoutés le 10 septembre 2026 par le lot `cloture-operationnelle`, décision au dossier
# (`docs/STRATEGIE.md` §12.6 point 4). Le gel RESTE un détecteur de dérive — le transformer
# en verrou dur interdirait de rejouer un lot interrompu. Mais les points d'écriture doivent
# le consulter, ce que sa docstring promettait déjà et que le code ne faisait pas.

def test_apply_delivery_consulte_le_gel_avant_le_verrou_de_collection():
    """Un import refusé ne doit pas avoir pris un verrou pour rien.

    L'ordre compte : si la garde passait APRÈS ``verrou_collection.tenu(...)``, un refus
    laisserait derrière lui un verrou à péremption d'une heure, et le message que
    l'utilisateur lirait serait celui du verrou, pas celui du gel.
    """
    source = (MACOS / "ingestion" / "apply_delivery.py").read_text(encoding="utf-8")
    corps = source.split("def apply(", 1)[1].split("\ndef ", 1)[0]
    assert "gel_corpus.verifier(" in corps, "apply() ne consulte pas le gel"
    assert corps.index("gel_corpus.verifier(") < corps.index("verrou_collection.tenu("), (
        "la garde du gel doit précéder la prise du verrou de collection")


def test_rollback_ne_consulte_jamais_le_gel():
    """Le piège qu'on refuse de poser.

    Un gel non accordé veut dire que la signature a DÉJÀ dérivé. L'annulation est le geste
    qui la remet d'aplomb : la bloquer derrière la garde forcerait à lever le gel pour
    réparer, donc à perdre la trace de ce qu'on protégeait.
    """
    source = (MACOS / "ingestion" / "apply_delivery.py").read_text(encoding="utf-8")
    corps = source.split("\ndef rollback(", 1)[1].split("\ndef ", 1)[0]
    assert "gel_corpus.verifier(" not in corps, (
        "rollback() consulte le gel — c'est un piège : il refuserait le seul geste qui "
        "remet la signature d'aplomb")
    assert "consultera jamais" in corps, "le motif du non-appel doit rester écrit ici"


def test_retire_document_consulte_le_gel_mais_laisse_passer_le_constat():
    """Un constat n'écrit rien : il doit rester lisible sous gel, comme ``--dry-run``."""
    source = (MACOS / "ingestion" / "retire_document.py").read_text(encoding="utf-8")
    assert "gel_corpus.verifier(" in source, "retire_document ne consulte pas le gel"
    avant = source.split("gel_corpus.verifier(")[0]
    assert 'constat["status"] = "CONSTAT"' in avant, (
        "la garde doit venir APRÈS la sortie du constat, sinon --appliquer absent devient "
        "inutilisable sous gel")


def test_les_trois_appelants_refusent_ou_passent_ensemble(declaration, monkeypatch):
    """La garde elle-même, éprouvée dans les deux sens.

    Une mutation qui retire l'appel de ``apply()`` ou de ``retire_document`` ne fait pas
    tomber ce test-ci — c'est le rôle des trois précédents. Celui-ci fige le comportement
    que les trois appelants partagent : refus quand la signature a dérivé, passage sinon.
    """
    _signature(monkeypatch, "e1bdf36e2e")
    gel_corpus.geler("répétition générale")
    assert gel_corpus.verifier("l'import d'un document au corpus", sortir=False) is True
    assert gel_corpus.verifier("le retrait d'un document du corpus", sortir=False) is True

    _signature(monkeypatch, "0000000000")          # un document est entré : la signature dérive
    assert gel_corpus.verifier("l'import d'un document au corpus", sortir=False) is False
    assert gel_corpus.verifier("le retrait d'un document du corpus", sortir=False) is False
    with pytest.raises(SystemExit) as sortie:
        gel_corpus.verifier("l'import d'un document au corpus")
    assert "GEL DU CORPUS ACTIF" in str(sortie.value)


def test_les_deux_ecrivains_du_registre_consultent_desormais_le_gel():
    """La décision a été prise, et voici sa trace.

    Ce test remplace ``test_les_deux_ecrivains_du_registre_ignorent_le_gel``, qui constatait
    l'inverse et prévenait : « s'il tombe un jour, c'est que quelqu'un l'a prise — et le
    rapport doit le dire ». Elle a été prise le 10 septembre 2026 par le lot
    ``cloture-operationnelle``, sur la décision au dossier de ``docs/STRATEGIE.md`` §12.6
    point 4 : *le gel reste un détecteur de dérive, mais les points d'écriture doivent le
    consulter*. Le rapport ``docs/CLOTURE-OPERATIONNELLE-2026-09-10.md`` le dit.

    Ce qui n'a PAS changé, et qu'un lecteur pressé confondrait : le comportement de la garde
    elle-même. ``accorde`` vaut toujours vrai tant que la signature vivante égale la gelée,
    donc rejouer un lot interrompu reste possible. Voir
    ``test_le_gel_est_un_detecteur_de_derive_et_non_un_verrou``.
    """
    for relatif in ("ingestion/apply_delivery.py", "ingestion/retire_document.py"):
        source = (MACOS / relatif).read_text(encoding="utf-8")
        assert "gel_corpus.verifier(" in source, (
            f"{relatif} ne consulte plus le gel — la décision du §12.6 point 4 a été défaite")


# ------------------------------------------------- le comportement, pas seulement la source
# Les tests ci-dessus lisent le code ; ceux-ci l'exécutent. Un test de source survit à un
# appel déplacé dans une branche morte — pas un test de comportement.

@pytest.fixture
def apply_delivery():
    """Importé à l'appel : le module tire `inspect_delivery`, `registry` et `corpus_overlay`."""
    sys.path.insert(0, str(MACOS / "ingestion"))
    import apply_delivery as module
    return module


def test_apply_refuse_en_code_1_sous_un_gel_non_accorde(apply_delivery, declaration,
                                                        monkeypatch, tmp_path):
    """Le refus est un ``SystemExit`` de code 1, et il arrive AVANT toute écriture.

    ``sys.exit(<chaîne>)`` rend 1 et écrit le message sur stderr : c'est le contrat que
    ``verifier()`` a toujours eu, et celui sur lequel le §4.4 du lot D s'appuie.
    """
    _signature(monkeypatch, "e1bdf36e2e")
    gel_corpus.geler("chantier mesuré en cours")
    _signature(monkeypatch, "0000000000")          # un document est entré : la signature dérive

    verrou_pris = []
    monkeypatch.setattr(apply_delivery.verrou_collection, "tenu",
                        lambda nom: verrou_pris.append(nom))

    with pytest.raises(SystemExit) as sortie:
        apply_delivery.apply(tmp_path / "livraison-inexistante", None, True, "essai-gel")
    assert "GEL DU CORPUS ACTIF" in str(sortie.value)
    assert sortie.value.code != 0
    assert verrou_pris == [], ("le verrou de collection a été pris avant le refus — un import "
                              "refusé laisserait un verrou d'une heure derrière lui")


def test_rollback_passe_le_gel_qui_arrete_apply(apply_delivery, declaration, monkeypatch):
    """Sous le gel EXACT qui vient de refuser ``apply``, l'annulation, elle, passe.

    On n'annule rien pour de vrai : ``verrou_collection.tenu`` est remplacé par une sentinelle,
    et la voir remonter prouve que le contrôle a dépassé la garde du gel. C'est le seul point
    à prouver ici — le reste de l'annulation a ses propres tests.
    """
    _signature(monkeypatch, "e1bdf36e2e")
    gel_corpus.geler("chantier mesuré en cours")
    _signature(monkeypatch, "0000000000")
    assert gel_corpus.etat()["accorde"] is False, "le gel doit être non accordé pour ce test"

    class Sentinelle(Exception):
        pass

    def tenu(nom):
        raise Sentinelle(nom)

    monkeypatch.setattr(apply_delivery.verrou_collection, "tenu", tenu)
    with pytest.raises(Sentinelle) as remontee:
        apply_delivery.rollback("livraison-quelconque")
    assert "rollback livraison-quelconque" in str(remontee.value)


def test_le_driver_refuse_toujours_comme_avant(apply_delivery, declaration, monkeypatch):
    """Le troisième appelant n'a pas changé : même garde, même message, même condition.

    ``batch_driver --go`` appelait déjà ``verifier()`` avant ce lot. Ce test fige le fait que
    l'ajout des deux autres appelants ne lui a rien retiré — la garde qu'il invoque est la
    même fonction, avec le même libellé d'action.
    """
    source = (MACOS / "ingestion" / "batch_driver.py").read_text(encoding="utf-8")
    assert 'gel_corpus.verifier("l\'import d\'un document au corpus")' in source
    _signature(monkeypatch, "e1bdf36e2e")
    gel_corpus.geler("chantier mesuré en cours")
    _signature(monkeypatch, "0000000000")
    with pytest.raises(SystemExit) as sortie:
        gel_corpus.verifier("l'import d'un document au corpus")
    assert "GEL DU CORPUS ACTIF" in str(sortie.value)
    assert "chantier mesuré en cours" in str(sortie.value)
