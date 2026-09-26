"""Les règles de décision de `eval_assemblage` sont appliquées par le code, pas par la prose.

Un pré-enregistrement dont les seuils vivent dans un document et non dans une fonction est un
pré-enregistrement qu'on peut relire à son avantage une fois les chiffres connus. Ces tests
fixent les six issues, la formule de la borne économique et le seuil de dégradation stricte
**avant** le premier appel LLM — ils tournent hors ligne, sans corpus et sans réseau.

Le §17 du chantier reranker a consigné une lacune : le cas « contraste acquis, garde échouée »
n'était pas nommé dans son pré-enregistrement et a dû être déclaré après coup. Il est ici, et
il est testé.
"""
from __future__ import annotations

import sys
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(BENCHMARK.parent))

import eval_assemblage as ea  # noqa: E402
from garde_reponse import wilson_haut  # noqa: E402

GELE = {"seuil_degradation_stricte": 2}


def _rapport(delta_e, ic_e, delta_t, ic_t, degradations=0, n_e=17, n_t=56):
    return {"strates": {
        "entrantes": {"n": n_e, "delta": delta_e, "ci95": ic_e, "degradation_stricte": 0},
        "temoins": {"n": n_t, "delta": delta_t, "ci95": ic_t,
                    "degradation_stricte": degradations},
    }}


def test_go_exige_les_trois_gardes_et_un_contraste_acquis():
    v = ea.verdict(_rapport(0.9, [0.4, 1.4], -0.02, [-0.12, 0.08]), GELE)
    assert v["issue"] == "GO"
    assert all(v["gardes"][g] for g in ("instrument", "economique", "degradation_stricte"))


def test_hold_quand_seule_la_borne_d_instrument_echoue():
    """Le cas que le §15 avait oublié de nommer : la perte n'est ni démontrée ni exclue."""
    v = ea.verdict(_rapport(0.9, [0.4, 1.4], -0.05, [-0.20, 0.10]), GELE)
    assert v["issue"] == "HOLD"
    assert v["gardes"]["instrument"] is False
    assert v["gardes"]["economique"] is True      # −0,20 > −0,304 × 0,9 = −0,274


def test_no_go_quand_la_borne_economique_echoue():
    """La perte mesurée mange le gain : ce n'est pas un HOLD, c'est un refus."""
    v = ea.verdict(_rapport(0.5, [0.1, 0.9], -0.10, [-0.30, 0.10]), GELE)
    assert v["issue"] == "NO_GO"
    assert v["gardes"]["economique"] is False     # −0,30 < −0,304 × 0,5 = −0,152


def test_no_go_quand_des_reponses_deviennent_inutilisables():
    """Deux témoins qui tombent de 2 à 0 suffisent, quel que soit le gain."""
    v = ea.verdict(_rapport(1.2, [0.8, 1.6], 0.0, [-0.10, 0.10], degradations=2), GELE)
    assert v["issue"] == "NO_GO"
    assert v["gardes"]["degradation_stricte"] is False
    assert ea.verdict(_rapport(1.2, [0.8, 1.6], 0.0, [-0.10, 0.10], degradations=1),
                      GELE)["issue"] == "GO"


def test_plat_est_defini_par_l_ancre_la_plus_basse_jamais_mesuree():
    """Borne haute sous +0,462 : le contexte doublé annule ce qu'il apporte."""
    v = ea.verdict(_rapport(0.15, [0.02, 0.30], 0.0, [-0.10, 0.10]), GELE)
    assert v["issue"] == "PLAT"
    assert ea.ANCRE_BASSE == 0.462


def test_non_concluant_n_est_jamais_maquille_en_plat():
    v = ea.verdict(_rapport(0.30, [-0.20, 0.90], 0.0, [-0.10, 0.10]), GELE)
    assert v["issue"] == "NON_CONCLUANT"


def test_regression_quand_le_contexte_double_nuit_la_ou_il_apporte_l_or():
    v = ea.verdict(_rapport(-0.40, [-0.90, -0.05], 0.0, [-0.10, 0.10]), GELE)
    assert v["issue"] == "REGRESSION"


def test_la_borne_economique_suit_la_formule_pre_enregistree():
    """−(n_entrantes / n_témoins) × Δ_entrantes, et rien d'autre.

    Le champ publié est arrondi à quatre décimales pour être lisible ; la **comparaison**,
    elle, porte sur la valeur exacte. Le second cas ci-dessous le vérifie : un témoin situé
    entre la valeur arrondie et la valeur exacte doit être tranché par l'exacte.
    """
    v = ea.verdict(_rapport(0.8, [0.4, 1.2], 0.0, [-0.10, 0.10]), GELE)
    assert abs(v["gardes"]["borne_economique"] - (-(17 / 56) * 0.8)) < 5e-5

    exacte = -(17 / 56) * 0.7                       # −0,2125
    juste_au_dessus = ea.verdict(_rapport(0.7, [0.3, 1.1], -0.1, [exacte + 1e-6, 0.1]), GELE)
    juste_en_dessous = ea.verdict(_rapport(0.7, [0.3, 1.1], -0.1, [exacte - 1e-6, 0.1]), GELE)
    assert juste_au_dessus["gardes"]["economique"] is True
    assert juste_en_dessous["gardes"]["economique"] is False


def test_le_seuil_de_degradation_stricte_est_derive_de_wilson_et_non_choisi():
    n = 56
    seuil = next(b for b in range(n + 1) if wilson_haut(b, n) > ea.PLAFOND_DEGRADATION)
    assert seuil == 2
    assert wilson_haut(1, n) <= 0.10 < wilson_haut(2, n)


def test_les_deux_bras_sont_ceux_du_pre_enregistrement():
    assert ea.REFERENCE == {"passages": 5, "per_document": 2}
    assert ea.CANDIDAT == {"passages": 10, "per_document": 3}
    assert ea.BORNE_INSTRUMENT == -0.15
    assert ea.GRAINE == 20260901


def test_un_rapport_incomplet_ne_rend_pas_de_verdict():
    assert ea.verdict({"strates": {}}, GELE)["issue"] == "INCOMPLET"


def test_les_strates_gelees_sont_exhaustives_et_disjointes():
    """Le fichier gelé est le contrat : 17 + 56 + 57 + 0 = 130, sans recouvrement.

    Ce test **sautait** entre le 9 et le 10 septembre 2026 : les strates n'étaient gelées que
    pour ``5530cba145``, et le corpus servi est passé à ``e1bdf36e2e``. Sauter était le bon
    comportement — mieux vaut un test muet qu'un test qui valide un fichier périmé —, mais un
    test qui saute ne garde rien. Les strates ont donc été regelées (``--step strates``, hors
    ligne, zéro appel), et le skip ne subsiste que pour une signature encore inconnue.
    """
    import json

    chemin = BENCHMARK / f"strates-assemblage-{ea.SIGNATURE}.json"
    if not chemin.exists():
        import pytest

        pytest.skip(f"strates non gelées pour {ea.SIGNATURE} — lancer "
                    f"`eval_assemblage.py --step strates` (hors ligne, zéro appel)")
    gele = json.loads(chemin.read_text(encoding="utf-8"))
    assert gele["signature"] == ea.SIGNATURE, "le fichier gelé ne porte pas la signature vivante"
    strates = gele["strates"]
    toutes = [c for cles in strates.values() for c in cles]
    assert len(toutes) == len(set(toutes)) == gele["population"] == 130
    assert gele["tailles"] == {"entrantes": 17, "temoins": 56, "neutres": 57, "sortantes": 0}
    assert all(c.startswith("v3/") for c in toutes), "v1 est exclu faute d'answer_facts"


def test_la_stratification_a_survecu_au_changement_de_corpus():
    """Les 989 chunks corrigés n'ont fait changer de strate à AUCUNE des 130 questions.

    Le fait n'allait pas de soi et il est publiable : `5530cba145` → `e1bdf36e2e` change
    l'**empreinte des pools** (``d87a14f7d14f8907`` → ``5deba2a8292ae34d``, donc les
    classements ont bougé) sans déplacer une seule question entre ``entrantes``, ``temoins``,
    ``neutres`` et ``sortantes``. Vérifier les seuls effectifs n'aurait rien prouvé — trois
    strates de même taille peuvent avoir échangé leurs membres ; c'est l'appartenance qui est
    testée ici. Le test s'efface de lui-même si l'un des deux fichiers disparaît.
    """
    import json

    import pytest

    fichiers = {sig: BENCHMARK / f"strates-assemblage-{sig}.json"
                for sig in ("5530cba145", "e1bdf36e2e")}
    absents = [str(c) for c in fichiers.values() if not c.exists()]
    if absents:
        pytest.skip(f"gel absent : {', '.join(absents)}")
    avant, apres = (json.loads(c.read_text(encoding="utf-8")) for c in fichiers.values())
    assert avant["empreinte_des_pools"] != apres["empreinte_des_pools"], (
        "les pools sont censés avoir bougé — sinon ce test ne prouve rien")
    for nom in ("entrantes", "temoins", "neutres", "sortantes"):
        assert set(avant["strates"][nom]) == set(apres["strates"][nom]), (
            f"la strate {nom} a changé de membres entre les deux corpus")
