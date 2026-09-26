"""Ce qu'un import écrit, une annulation doit le défaire — l'invariant, tenu par un test.

Pourquoi ce fichier existe
---------------------------
Le lot D (10 septembre 2026) a prouvé qu'un import s'annule : la signature du corpus revient,
les points partent, le registre retrouve son empreinte. Mais il a dû **finir l'annulation à la
main**, par un ``git checkout --``, parce que le ``rollback`` ne rend pas l'arborescence à son
état d'avant. Trois faces du même défaut, toutes mesurées ce jour-là :

- **il écrit ce qu'il ne défait pas** : ``rebuild_bm25()`` nomme son index par la signature
  courante et n'efface jamais les précédents. Un import + son annulation laissaient **deux**
  index morts, ≈ 50 Mo chacun. Le lot C fera six cents imports ;
- **il défait mal ce qu'il défait** : l'overlay des tableaux était réécrit ``indent=1`` alors
  que le fichier versionné est **compact sur une ligne** — 5 933 lignes de diff git pour
  **zéro** différence de contenu, et l'annulation ne le remettait pas non plus dans sa forme ;
- **il échoue en laissant des traces** : un ``fail()`` précoce laissait ``incoming/<id>/``.

Ce que ce module vérifie, et comment
-------------------------------------
Il photographie l'arborescence **fichier par fichier, sha256 compris**, importe, annule, et
exige que la photographie soit **la même**. Pas « équivalente », pas « au contenu près » : la
même. Ce qui a le droit de subsister est déclaré dans ``TOLERE``, en clair, avec son motif.

L'assertion sur les sha256 est celle qui compte : c'est elle, et elle seule, qui aurait vu la
réécriture ``indent=1``. Un test qui aurait comparé les fichiers « après ``json.loads`` »
serait passé, et le défaut serait encore là.

Hermétique — et pourquoi ce n'est pas contradictoire avec la propriété visée
-----------------------------------------------------------------------------
Le plan de ce lot demandait de photographier ``git status --porcelain`` du dépôt réel. Ce
module ne le fait pas : il tourne sur un corpus jetable, dans ``tmp_path``, sans réseau, sans
MinerU et sans modèle, avec les vingt-huit constantes détournées par ``test_apply_delivery``.
La raison est qu'un test de la suite ne peut pas importer puis annuler dans le corpus **servi**
à chaque exécution.

La propriété est la même, exprimée sans git : *un fichier qui existait avant a le même sha256
après*. C'est exactement ce que ``git status`` constate, et c'est vérifiable partout. La preuve
sur le dépôt réel, elle, est faite à la main et publiée dans le rapport du lot.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[3]
for _relatif in ("rag", "src", "rag/ingestion", "rag/metadata", "rag/tables",
                 "rag/titles"):
    _chemin = str(RACINE / _relatif)
    if _chemin not in sys.path:
        sys.path.insert(0, _chemin)

import apply_delivery                                          # noqa: E402
import corpus_overlay                                          # noqa: E402
import quant_rag                                               # noqa: E402

# Les fixtures du banc d'essai hermétique : le corpus jetable, ses vingt-huit constantes
# détournées, et la livraison fabriquée. Les réécrire ici les ferait diverger au premier
# changement de l'une ou de l'autre.
from test_apply_delivery import (  # noqa: E402,F401
    arbre, livraison_neuve, Arbre, LIVRAISON)

#: Les répertoires dont l'aller-retour doit être neutre. Ce sont ceux qu'``apply_delivery``
#: écrit — la liste vient du code, pas d'une intuition : ``INCOMING`` (staging), ``INGESTED``
#: (fichiers canoniques), ``VECTOR_DIR`` (.npz), ``quant_rag.LEXICAL_DIR`` (BM25), les trois
#: overlays et le registre.
SURVEILLES = (
    "data/lexical",
    "data/processed/incoming",
    "data/processed/ingested",
    "rag/ingestion/.cache",
    "rag/ingestion/journal",
    "rag/metadata",
    "rag/tables",
    "rag/titles",
)

#: Ce qui a le droit d'apparaître et de rester après un aller-retour complet. **Toute** entrée
#: porte son motif : une liste blanche sans motif finit par absorber les défauts qu'elle
#: devrait signaler.
TOLERE = {
    # Le journal de livraison est la TRACE de l'import et de son annulation. Il porte
    # `state: "rolled-back"`, l'heure, les points retirés. Le supprimer effacerait la seule
    # preuve que l'aller-retour a eu lieu. Voir `_rollback` : il le réécrit exprès.
    f"rag/ingestion/journal/{LIVRAISON}.json",
}


def est_tolere(relatif: str, vivante: str) -> bool:
    """Un fichier apparu a-t-il le droit d'être là ?

    Deux cas, et deux seulement. Le journal de livraison, déclaré dans ``TOLERE``. Et l'index
    BM25 de la signature **vivante** : l'arbre jetable démarre avec ``data/lexical/`` vide,
    alors qu'en production l'index de la signature servie est déjà là avant l'import. Son
    apparition est donc un artefact du banc — mais **seulement pour la signature vivante** :
    tout index d'une autre signature est le défaut que ce module chasse.
    """
    if relatif in TOLERE:
        return True
    return relatif.startswith("data/lexical/") and vivante in relatif

#: Ce qui a le droit de DISPARAÎTRE — et il n'y a qu'une entrée, qui est un artefact du banc,
#: pas du chemin d'écriture.
#:
#: ``test_apply_delivery`` sème ``vectors-<livraison>.npz`` **avant** l'import, pour
#: qu'``embed_delivery`` le trouve déjà là (``embed_delivery``, branche ``if final.exists()``) et n'ait pas à
#: charger le modèle. En production ce fichier n'existe pas avant l'import : c'est l'import qui
#: l'écrit et l'annulation qui l'efface (``:808-809``). Sa disparition est donc **le
#: comportement correct**, vu à travers un banc qui l'a pré-semé.
DISPARITIONS_ATTENDUES = {
    f"rag/ingestion/.cache/vectors-{LIVRAISON}.npz",
}

#: Fichiers réécrits par le chemin d'écriture, dont la forme d'octets doit revenir à
#: l'identique. Nommés un par un parce que chacun a été mesuré divergent ou conforme le
#: 10 septembre 2026, et que le test doit dire lequel casse.
REECRITS = (
    "rag/tables/tables-markdown-v1.json",
    "rag/metadata/documents-metadata-v1.json",
    "rag/ingestion/registry-v1.json",
)


def empreinte(racine: Path) -> dict[str, tuple[int, str]]:
    """{chemin relatif : (taille, sha256)} sur les répertoires surveillés.

    Le sha256 est le point du test. Une comparaison de tailles laisserait passer une
    réécriture de même longueur ; une comparaison ``json.loads`` laisserait passer un
    changement de sérialisation, qui est exactement le défaut qu'on chasse.
    """
    photo: dict[str, tuple[int, str]] = {}
    for relatif in SURVEILLES:
        base = racine / relatif
        if not base.is_dir():
            continue
        for chemin in sorted(base.rglob("*")):
            if chemin.is_file():
                octets = chemin.read_bytes()
                photo[chemin.relative_to(racine).as_posix()] = (
                    len(octets), hashlib.sha256(octets).hexdigest())
    return photo


def index_bm25(racine: Path) -> list[str]:
    """Les index et manifestes BM25 présents, par nom."""
    lexical = racine / "data" / "lexical"
    return sorted(c.name for c in lexical.glob("*.json")) if lexical.is_dir() else []


@pytest.fixture
def arbre_forme_versionnee(arbre):
    """Le corpus jetable, avec la garantie que l'overlay des tableaux est dans sa forme VERSIONNÉE.

    Le fichier réellement commité est **compact, sur une seule ligne** — vérifié le
    10 septembre 2026 : ``git show HEAD:rag/tables/tables-markdown-v1.json | wc -l`` rend
    **0**, et ``rag/tables/appliquer_en_tetes.py`` l'écrit ainsi.

    ``test_apply_delivery`` le semait ``indent=1`` jusqu'au 11 septembre 2026 — une forme que
    le dépôt n'emploie nulle part, et qui rendait le défaut invisible à son propre banc. Sa
    fixture le sème désormais compact ; cette fixture-ci n'est donc plus une conversion mais
    un **garde-fou** : si le banc partagé rechange de forme, c'est ici que ça se voit, et non
    six tests plus loin sous une comparaison d'octets illisible.
    """
    assert arbre.tables.read_text(encoding="utf-8").count("\n") == 0, (
        "le banc partagé sème l'overlay des tableaux dans une forme que le dépôt n'utilise "
        "pas : le fichier versionné est compact (0 saut de ligne)")
    return arbre


def aller_retour(livraison, identifiant: str = LIVRAISON) -> tuple[dict, dict]:
    """Importe puis annule. Rend les deux réponses."""
    entree = apply_delivery.apply(livraison, None, True, identifiant)
    assert entree["status"] == "COMPLETED", entree
    sortie = apply_delivery.rollback(identifiant)
    assert sortie["status"] == "ROLLED_BACK", sortie
    return entree, sortie


# ------------------------------------------------------------------ l'invariant, en entier

def test_l_arborescence_revient_a_l_identique(arbre_forme_versionnee, livraison_neuve):
    """Le test central du lot E. Tout le reste de ce module en isole un morceau.

    Trois assertions, dans cet ordre : rien n'a disparu, rien d'inattendu n'est apparu, et
    ce qui existait des deux côtés a le **même sha256**.
    """
    arbre = arbre_forme_versionnee
    avant = empreinte(arbre.racine)
    signature_avant = corpus_overlay.signature()

    aller_retour(livraison_neuve)

    corpus_overlay.invalidate()
    assert corpus_overlay.signature() == signature_avant, "la signature du corpus n'est pas revenue"

    apres = empreinte(arbre.racine)

    disparus = sorted(set(avant) - set(apres) - DISPARITIONS_ATTENDUES)
    assert disparus == [], f"l'annulation a effacé des fichiers qui existaient avant : {disparus}"

    apparus = sorted(c for c in set(apres) - set(avant)
                     if not est_tolere(c, corpus_overlay.signature()))
    assert apparus == [], (
        "l'annulation laisse des fichiers qui n'existaient pas avant l'import, et qui ne "
        f"sont pas dans la liste blanche déclarée : {apparus}")

    modifies = sorted(c for c in set(avant) & set(apres) if avant[c][1] != apres[c][1])
    assert modifies == [], (
        "des fichiers existaient des deux côtés mais leur contenu d'octets a changé — "
        "c'est le diff que git montrerait : "
        + "; ".join(f"{c} ({avant[c][0]} → {apres[c][0]} octets)" for c in modifies))


# ------------------------------------------------------------------ les trois faces, isolées

def test_aucun_index_bm25_de_signature_morte_ne_subsiste(arbre_forme_versionnee, livraison_neuve):
    """Chaque index présent à la fin porte la signature VIVANTE. Aucun autre.

    Le chiffre qui rend ce test bloquant : un index pèse ≈ 50 Mo, et le lot C fera six cents
    imports. Deux index morts par aller-retour, c'est **≈ 30 Go** sur un disque qui était à
    90 % le 10 septembre 2026.
    """
    arbre = arbre_forme_versionnee
    aller_retour(livraison_neuve)
    corpus_overlay.invalidate()
    vivante = corpus_overlay.signature()

    presents = index_bm25(arbre.racine)
    morts = [n for n in presents if vivante not in n]
    assert morts == [], (
        f"index ou manifestes d'une signature qui n'existe plus : {morts} "
        f"(signature vivante : {vivante})")
    assert any(vivante in n and not n.endswith(".manifest.json") for n in presents), (
        "l'index de la signature vivante devrait exister après l'annulation")


def test_l_overlay_des_tableaux_revient_octet_pour_octet(arbre_forme_versionnee, livraison_neuve):
    """La forme de sérialisation fait partie de l'état, pas de la présentation.

    Mesuré sur le dépôt réel le 10 septembre 2026 : l'overlay commité fait **5 224 037 octets
    et 0 saut de ligne** ; réécrit ``indent=1`` il en fait **5 235 902 et 5 932**. Contenu
    identique — 5 914 entrées, 0 écart au parsing — pour **5 933 lignes de diff** à chaque
    import, sur un fichier de 5 Mo. Un historique que personne ne relit est un historique
    dans lequel personne ne verra jamais rien passer.
    """
    arbre = arbre_forme_versionnee
    avant = arbre.tables.read_bytes()
    aller_retour(livraison_neuve)
    apres = arbre.tables.read_bytes()

    assert json.loads(avant) == json.loads(apres), (
        "le CONTENU de l'overlay a changé — c'est un défaut plus grave que la forme")
    assert avant == apres, (
        f"le contenu est le même mais pas les octets : {len(avant)} → {len(apres)} octets, "
        f"{avant.count(10)} → {apres.count(10)} sauts de ligne. C'est le diff de 5 933 lignes.")


@pytest.mark.parametrize("relatif", REECRITS)
def test_chaque_fichier_reecrit_garde_sa_forme(arbre_forme_versionnee, livraison_neuve, relatif):
    """Les trois fichiers qu'``apply_delivery`` réécrit, un par un.

    Paramétré exprès : quand ce test tombe, son nom dit **lequel** des trois a divergé, sans
    qu'on ait à lire une trace.
    """
    arbre = arbre_forme_versionnee
    chemin = arbre.racine / relatif
    avant = chemin.read_bytes() if chemin.exists() else None
    aller_retour(livraison_neuve)
    apres = chemin.read_bytes() if chemin.exists() else None

    assert (avant is None) == (apres is None), f"{relatif} a été créé ou effacé par l'aller-retour"
    if avant is not None:
        assert avant == apres, (
            f"{relatif} : {len(avant)} → {len(apres)} octets, "
            f"{avant.count(10)} → {apres.count(10)} sauts de ligne")


def test_le_staging_ne_survit_pas_a_un_echec_precoce(arbre_forme_versionnee, livraison_neuve):
    """Un ``fail()`` avant toute écriture ne doit rien laisser dans ``incoming/``.

    Le cas de référence est réel : la garde de titre (``metadata_gate``) refuse un titre de
    provenance ``filename_stem``. Le 10 septembre 2026, ce refus laissait
    ``data/processed/incoming/<id>/`` sur disque — le staging est créé à la ligne 490 et
    n'était effacé qu'à la 687, sans ``finally`` entre les deux. ``batch_driver`` rattrapait,
    mais lui seul : ``apply_delivery`` invoqué à la main ne rattrapait rien.
    """
    arbre = arbre_forme_versionnee
    # On retire le titre manuel semé par le banc : la garde retombe alors sur `filename_stem`
    # et refuse, exactement comme en production.
    arbre.overrides.write_text("{}\n", encoding="utf-8")

    incoming = arbre.racine / "data" / "processed" / "incoming"
    avant = sorted(p.name for p in incoming.iterdir()) if incoming.is_dir() else []

    with pytest.raises(SystemExit) as sortie:
        apply_delivery.apply(livraison_neuve, None, True, "echec-titre")
    assert "REFUS" in str(sortie.value), sortie.value

    apres = sorted(p.name for p in incoming.iterdir()) if incoming.is_dir() else []
    assert apres == avant, (
        f"un refus a laissé du staging derrière lui : {sorted(set(apres) - set(avant))}")


def test_le_npz_partiel_ne_survit_pas_a_une_annulation(arbre_forme_versionnee, livraison_neuve):
    """``vectors-<id>.partial.npz`` est un point de reprise ; après une annulation il ment.

    Il est écrit à chaque bloc d'embarquement (``embed_delivery``, ``np.savez(partial, …)``) et effacé au succès. Un import mort pendant l'embarquement le laisse donc sur disque — c'est voulu,
    c'est ce qui permet la reprise. Mais après une **annulation**, il n'y a plus rien à
    reprendre : le garder ferait qu'un import suivant du même identifiant repartirait de
    vecteurs qui ne correspondent plus à rien.
    """
    import numpy as np

    arbre = arbre_forme_versionnee
    entree = apply_delivery.apply(livraison_neuve, None, True, LIVRAISON)
    assert entree["status"] == "COMPLETED", entree

    # Un partiel déposé APRÈS l'import et AVANT l'annulation : c'est exactement ce qu'un
    # import mort pendant l'embarquement aurait laissé, puis rejoué. L'annulation doit
    # l'emporter avec le reste. Le déposer après l'annulation ne prouverait rien — le second
    # `rollback` rend `NOOP`, le journal étant déjà en `rolled-back`.
    faux = arbre.cache_vecteurs / f"vectors-{LIVRAISON}.partial.npz"
    np.savez(faux, chunk_ids=np.asarray(["chunk-mort"]), vectors=np.zeros((1, 4), dtype=np.float32))
    assert faux.exists()

    sortie = apply_delivery.rollback(LIVRAISON)
    assert sortie["status"] == "ROLLED_BACK", sortie

    assert not faux.exists(), (
        "l'annulation laisse un point de reprise : un import ultérieur du même identifiant "
        "repartirait de vecteurs qui ne correspondent plus au corpus")
    partiels = sorted(p.name for p in arbre.cache_vecteurs.glob("*.partial.npz"))
    assert partiels == [], f"des points de reprise survivent à l'annulation : {partiels}"


# ------------------------------------------------------------------ le refus du verrou

def test_le_verrou_tenu_se_dit_comme_le_reste_du_depot(arbre_forme_versionnee, livraison_neuve,
                                                       monkeypatch, capsys):
    """``VerrouTenu`` n'était rattrapée par aucun appelant : trace Python au lieu de ``REFUS :``.

    Le message lui-même est le mieux écrit du dépôt — il nomme le tenant, son pid, depuis
    quand, et le chemin du fichier à supprimer si le processus est mort. Il arrivait
    simplement sous une trace, là où tout le reste dit ``REFUS :``. Le code de sortie était
    déjà 1 et rien n'était écrit : c'est la lisibilité qui était en cause.

    Ce test prend le verrou pour de vrai, appelle ``main()`` comme la ligne de commande le
    fait, et exige les trois propriétés : ``REFUS :`` en tête, code de sortie 1, et le tenant
    nommé. Une mutation qui retire le ``except`` le fait tomber sur la première.
    """
    import verrou_collection

    arbre = arbre_forme_versionnee
    monkeypatch.setattr(sys, "argv",
                        ["apply_delivery.py", str(livraison_neuve), "--apply", "--no-llm",
                         "--id", "essai-verrou"])
    with verrou_collection.tenu("un autre import, tenu exprès"):
        with pytest.raises(SystemExit) as sortie:
            apply_delivery.main()

    message = str(sortie.value)
    assert message.startswith("REFUS :"), (
        f"le refus du verrou n'est pas dit comme les autres : {message[:120]!r}")
    assert sortie.value.code != 0
    assert "un autre import, tenu exprès" in message, "le refus doit nommer le tenant"
    assert "la collection est déjà en écriture" in message

    # et rien n'a été écrit : le staging n'existe pas
    incoming = arbre.racine / "data" / "processed" / "incoming"
    assert not (incoming / "essai-verrou").exists()


def test_le_refus_du_verrou_est_bien_rattrape_dans_main():
    """Garde-fou de source : le ``except`` ne doit pas disparaître par une refactorisation."""
    source = (Path(apply_delivery.__file__)).read_text(encoding="utf-8")
    corps = source.split("\ndef main(", 1)[1]
    assert "except verrou_collection.VerrouTenu" in corps
    assert "fail(str(refus))" in corps


# ------------------------------------------------------------------ le ménage explicite

def test_l_inventaire_lexical_protege_le_vivant_et_le_gele(arbre_forme_versionnee, monkeypatch,
                                                           tmp_path):
    """Trois catégories, et la distinction entre les deux premières est ce qui rend l'outil sûr.

    Un index se supprime sur une preuve : sa signature n'est **ni** celle du corpus servi,
    **ni** celle que le gel protège. La seconde compte — quand une mesure court sur une
    signature gelée qui n'est plus la vivante, supprimer son index la casserait.
    """
    arbre = arbre_forme_versionnee
    lexical = arbre.racine / "data" / "lexical"
    vivante = corpus_overlay.signature()

    declaration = tmp_path / "gel-corpus.json"
    declaration.write_text(json.dumps({"actif": True, "signature": "aaaaaaaaaa"}), encoding="utf-8")
    monkeypatch.setattr(quant_rag, "ROOT", tmp_path.parent)
    monkeypatch.setattr(quant_rag, "LEXICAL_DIR", lexical)
    (tmp_path.parent / "rag").mkdir(exist_ok=True)
    (tmp_path.parent / "rag" / "gel-corpus.json").write_text(
        json.dumps({"actif": True, "signature": "aaaaaaaaaa"}), encoding="utf-8")

    for nom in (f"bm25-essai-{vivante}.json", f"bm25-essai-{vivante}.manifest.json",
                "bm25-essai-aaaaaaaaaa.json", "bm25-essai-bbbbbbbbbb.json",
                "bm25-essai-bbbbbbbbbb.manifest.json", "bm25_index.json",
                "hybrid-bm25-v1.json"):
        (lexical / nom).write_text("{}", encoding="utf-8")

    inventaire = quant_rag.index_lexicaux()
    proteges = {f["nom"] for f in inventaire["vivants"]}
    morts = {f["nom"] for f in inventaire["morts"]}
    sans = {f["nom"] for f in inventaire["sans_signature"]}

    assert f"bm25-essai-{vivante}.json" in proteges, "la signature vivante doit être protégée"
    assert "bm25-essai-aaaaaaaaaa.json" in proteges, "la signature GELÉE doit être protégée"
    assert morts == {"bm25-essai-bbbbbbbbbb.json", "bm25-essai-bbbbbbbbbb.manifest.json"}
    assert sans == {"bm25_index.json", "hybrid-bm25-v1.json"}, (
        "les formats anciens n'ont pas de signature : ils ne peuvent pas être déclarés morts")


def test_la_purge_ne_supprime_rien_sans_vraiment(arbre_forme_versionnee, monkeypatch, tmp_path):
    """Lecture seule par défaut. Un outil de ménage qui supprime sur une faute de frappe
    n'est pas un outil de ménage."""
    arbre = arbre_forme_versionnee
    lexical = arbre.racine / "data" / "lexical"
    monkeypatch.setattr(quant_rag, "LEXICAL_DIR", lexical)
    monkeypatch.setattr(quant_rag, "ROOT", tmp_path)

    mort = lexical / "bm25-essai-bbbbbbbbbb.json"
    mort.write_text("{}", encoding="utf-8")

    rapport = quant_rag.purger_index_morts()
    assert rapport["applique"] is False
    assert rapport["supprimes"] == []
    assert mort.exists(), "la purge a supprimé sans --vraiment"

    rapport = quant_rag.purger_index_morts(vraiment=True)
    assert rapport["applique"] is True
    assert "bm25-essai-bbbbbbbbbb.json" in rapport["supprimes"]
    assert not mort.exists()


def test_la_purge_ne_touche_jamais_a_l_index_servi(arbre_forme_versionnee, monkeypatch, tmp_path):
    """Le seul défaut qui rendrait cet outil pire que le problème qu'il résout."""
    arbre = arbre_forme_versionnee
    lexical = arbre.racine / "data" / "lexical"
    monkeypatch.setattr(quant_rag, "LEXICAL_DIR", lexical)
    monkeypatch.setattr(quant_rag, "ROOT", tmp_path)

    vivant = lexical / f"bm25-essai-{corpus_overlay.signature()}.json"
    vivant.write_text("{}", encoding="utf-8")
    ancien = lexical / "bm25_index.json"
    ancien.write_text("{}", encoding="utf-8")

    quant_rag.purger_index_morts(vraiment=True)
    assert vivant.exists(), "la purge a supprimé l'index de la signature servie"
    assert ancien.exists(), "la purge a supprimé un format sans signature"


# ------------------------------------------------------- citer une ligne, c'est citer un mensonge

def test_aucun_module_de_rag_ne_cite_un_numero_de_ligne():
    """Un commentaire qui cite « un fichier, deux-points, un numéro » ment dès que le fichier bouge.

    (Les exemples ci-dessous sont écrits en toutes lettres exprès : les écrire dans la forme
    qu'ils dénoncent ferait tomber ce test sur sa propre docstring.)

    Le défaut est apparu **trois fois** dans ce dépôt :

    - ``gel_corpus.py`` renvoyait à la ligne 1298 de ``batch_driver`` pour un appel qui était
      à la 1547 ;
    - ``verrou_collection.py`` renvoyait aux lignes 1319-1324 pour une garde qui était aux
      1619-1628 ;
    - et le lot du 11 septembre 2026 en a **créé six nouvelles** en insérant des fonctions
      dans ``quant_rag`` et ``apply_delivery``. Une citation est même devenue vraie **par
      accident** : celle qui renvoyait à la ligne 267 de ``quant_rag`` désigne bien
      ``def rebuild_bm25()`` depuis le décalage — mais la fonction était à la ligne 203
      avant, donc la citation était fausse quand elle a été écrite.

    La correction n'est donc pas de recompter : c'est de **citer un symbole**. Un nom de
    fonction ou de constante est greppable, il survit à un décalage, et quand il disparaît
    c'est qu'il a été renommé — ce qui se voit.

    Portée : **tout ``rag/``**. Les 43 citations relevées le 11 septembre 2026 ont toutes
    été résolues vers leur symbole englobant — y compris les 8 qui visaient des fichiers de
    ``.venv`` (``qdrant_local``, ``local_collection``, ``distances``, ``persistence``,
    ``qdrant_remote``), lesquelles dérivaient en plus à chaque mise à jour de dépendance.
    """
    import re

    couche = RACINE / "rag"
    motif = re.compile(r"[A-Za-z_0-9/]+\.py:[0-9]+")
    fautes = []
    for chemin in sorted(couche.rglob("*.py")):
        if "__pycache__" in chemin.parts:
            continue
        for numero, ligne in enumerate(
                chemin.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for trouve in motif.findall(ligne):
                fautes.append(f"{chemin.relative_to(couche).as_posix()}:{numero} cite {trouve}")
    assert fautes == [], (
        f"{len(fautes)} citation(s) de numéro de ligne dans rag/ — c'est un mensonge en "
        "sursis. Cite le symbole (fonction, classe, constante) à la place : il est greppable, "
        "il survit à un décalage, et sa disparition se voit.\n  " + "\n  ".join(fautes))
