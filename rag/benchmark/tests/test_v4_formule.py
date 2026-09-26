"""Attaque de la normalisation LaTeX — et garde du scoreur `formula` qui s'appuie dessus.

Ces tests ne confirment pas que ``latex_norme`` marche : ils essaient de le mettre en
défaut, sur le corpus réel, et **ils y arrivent une fois**. Le protocole est celui du
handoff §4 :

  positifs   312 blocs ``$$ … \\tag{n} … $$`` prélevés un par document, ré-écrits
             mécaniquement comme un générateur les écrirait (compacts, sans police, sans
             ``\\left``, ``X_{t}`` au lieu de ``X_{ t }``, enveloppés dans une phrase) ;
             le scoreur doit dire « juste ».
  négatifs   la même formule altérée d'**un seul** symbole — un signe, un indice, un
             exposant, une lettre grecque ; le scoreur doit dire « faux ».
  leurres    les formules des **autres** chunks du même document, et celles d'autres
             documents ; aucune ne doit être trouvée dans une réponse qui ne la contient pas.

Trois défauts trouvés — tous corrigés depuis, et gardés par les tests qui les ont montrés
-------------------------------------------------------------------------------------------
1. **Accolades orphelines.** Au ``NIVEAU_SOUPLE``, effacer une commande de police laissait
   ses accolades dès que l'argument faisait plus d'un atome : ``a + \\mathrm{cov}(x,y)`` se
   normalisait ``a + { c o v } ( x , y )``, quand le générateur écrit ``a + c o v ( x , y )``.
   **60 des 312 formules** échouaient sur la seule ré-écriture « police retirée ». Le cas
   ``^ { \\mathrm{bid} }`` — le seul que le docstring du module citait — passait, et c'est
   ce qui masquait le défaut. La police emporte désormais les accolades de son argument :
   **0,808 -> 1,000**.
2. **``\\operatorname*``.** ``_ATOME`` coupe la commande avant l'étoile : l'entrée
   ``\\operatorname*`` de ``_POLICES`` était du code mort et l'étoile survivait en
   multiplication, confondant ``\\operatorname*{max}(x)`` et ``\\ast{max}(x)``. L'étoile qui
   suit une police est maintenant mangée avec elle.
3. **Accolade non refermée.** ``_fermeture`` rendait ``len - 1`` faute de fermante — un index
   plausible et faux. Sur un passage **tronqué à 2 500 caractères**, la coupe casse des
   paires : le repli pliait un groupe inexistant, perdait du contenu en silence, puis
   débordait de la liste. Trouvé en production, au premier candidat de la fabrique de
   questions. ``_fermeture`` rend ``-1`` et chaque appelant traite l'accolade comme l'atome
   littéral qu'elle est.
"""
from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parent))

import latex_norme as ln  # noqa: E402
import score_formule as sf  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

#: Un bloc affiché du corpus. MinerU balise **tout** en ``$$…$$``, y compris les formules
#: en ligne, et numérote avec ``\tag`` : c'est le repère le plus fiable pour trouver des
#: mathématiques réelles plutôt que des bouts de prose contenant un dollar.
BLOC = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

#: Plancher de l'étude, imposé par le handoff. En dessous, la mesure ne dirait rien.
POPULATION_MINIMALE = 300

#: Un or plus court que cela n'est plus discriminant hors balise (cf. ``score_formule``).
ATOMES_MINIMAUX = sf.ATOMES_SUSPECTS


# ------------------------------------------------------------------ population


@functools.lru_cache(maxsize=1)
def population():
    """Un bloc ``\\tag`` par document : 312 formules, aucune répétée d'un document à l'autre.

    Trois filtres, chacun pour une raison nommée :
      - un seul bloc par document, sinon un article prolifique pèserait pour cinquante ;
      - pas d'environnement multi-ligne — ``latex_norme`` déclare ne pas les évaluer ;
      - au moins ``ATOMES_MINIMAUX`` atomes, sinon le repli « texte entier » du scoreur
        trouverait la formule dans la prose et la mesure serait truquée.
    """
    index = ChunkIndex.load(verbose=False)
    par_document = {}
    for chunk_id, chunk in sorted(index.chunks.items()):
        for trouve in BLOC.finditer(chunk.get("text") or ""):
            corps = trouve.group(1).strip()
            if "\\tag" not in corps:
                continue
            if ln.environnement_multiligne(corps):
                continue
            if len(ln.atomes(corps)) < ATOMES_MINIMAUX:
                continue
            par_document.setdefault(chunk.get("document_id"), []).append((chunk_id, corps))
    etude = [(document, par_document[document][0][0], par_document[document][0][1])
             for document in sorted(par_document)]
    return tuple(etude), par_document


def formules():
    return population()[0]


# ------------------------------------------------------------------ ré-écritures

#: Découpage en jetons **de surface** — un jeton porte ses espaces d'origine, contrairement
#: aux atomes de ``latex_norme``. Les ré-écritures doivent manipuler la graphie, pas la
#: forme canonique : normaliser puis dénormaliser fabriquerait un test circulaire.
JETON = re.compile(r"\\[a-zA-Z]+|\\.|.", re.DOTALL)

POLICES = ("\\mathrm", "\\mathbf", "\\mathbb", "\\mathcal", "\\mathscr", "\\mathsf",
           "\\mathfrak", "\\mathit", "\\mathtt", "\\text", "\\textrm", "\\operatorname",
           "\\pmb", "\\boldsymbol", "\\bm")


def _fermeture(jetons, depart):
    profondeur = 0
    for indice in range(depart, len(jetons)):
        if jetons[indice] == "{":
            profondeur += 1
        elif jetons[indice] == "}":
            profondeur -= 1
            if profondeur == 0:
                return indice
    return len(jetons) - 1


def compacter(source):
    """Tous les espaces retirés — sauf celui qui sépare une commande d'une lettre.

    Le piège n'est pas l'espace, c'est la **frontière de commande** : dans le corpus,
    ``\\sum _ { s \\in t }`` compacté à l'aveugle donne ``\\in t`` -> ``\\int``, et
    ``\\partial F`` -> ``\\partialF``. Ce sont d'autres formules, et le scoreur a raison de
    les refuser. Mesuré : 114 des 312 formules contiennent au moins une de ces frontières.
    Un générateur qui écrit du LaTeX valide ne les efface pas ; ce ré-écrivain non plus.
    """
    sortie = []
    for jeton in JETON.findall(source):
        if jeton.isspace():
            continue
        if sortie and re.fullmatch(r"\\[a-zA-Z]+", sortie[-1]) and re.match(r"[a-zA-Z]", jeton):
            sortie.append(" ")
        sortie.append(jeton)
    return "".join(sortie)


def retirer_police(source):
    """``\\mathrm{bid}`` -> ``bid`` : la commande **et** ses accolades disparaissent.

    C'est exactement la promesse du ``NIVEAU_SOUPLE`` — « ``\\mathbb{E}[X]`` et ``E[X]``
    sont la même formule pour un quant qui la lit » — et c'est la ré-écriture qui met la
    normalisation en défaut.
    """
    jetons = JETON.findall(source)
    sortie, indice = [], 0
    while indice < len(jetons):
        if jetons[indice] in POLICES:
            suivant = indice + 1
            while suivant < len(jetons) and jetons[suivant].isspace():
                suivant += 1
            if suivant < len(jetons) and jetons[suivant] == "{":
                fin = _fermeture(jetons, suivant)
                sortie.extend(jetons[suivant + 1:fin])
                indice = fin + 1
                continue
        sortie.append(jetons[indice])
        indice += 1
    return "".join(sortie)


def retirer_left_right(source):
    return re.sub(r"\\(?:left|right|bigl?|Bigl?|biggl?|Biggl?)(?![a-zA-Z])\s*", "", source)


def retirer_displaystyle(source):
    return re.sub(r"\\(?:displaystyle|limits|nolimits)(?![a-zA-Z])\s*", "", source)


def echanger_variantes(source):
    """Les unifications que ``latex_norme`` **déclare** : ``\\varphi``/``\\phi``, ``\\pmb``/``\\mathbf``…"""
    sortie = source
    for avant, apres in (("\\varphi", "\\phi"), ("\\varepsilon", "\\epsilon"),
                         ("\\pmb", "\\mathbf"), ("\\mathbf", "\\pmb"),
                         ("\\le", "\\leq"), ("\\rightarrow", "\\to")):
        sortie = re.sub(re.escape(avant) + r"(?![a-zA-Z])", apres.replace("\\", "\\\\"), sortie)
    return sortie


def desaccolader(source):
    """``X_{t}`` -> ``X_t`` : les accolades d'un groupe d'un seul jeton sautent."""
    jetons = JETON.findall(source)
    sortie, indice = [], 0
    while indice < len(jetons):
        if jetons[indice] == "{":
            fin = _fermeture(jetons, indice)
            interne = [j for j in jetons[indice + 1:fin] if not j.isspace()]
            if len(interne) == 1 and interne[0] not in "{}":
                sortie.append(interne[0])
                indice = fin + 1
                continue
        sortie.append(jetons[indice])
        indice += 1
    return "".join(sortie)


def retirer_ponctuation_finale(source):
    return re.sub(r"[.,;]\s*$", "", source.strip())


def envelopper(source):
    """La formule au milieu d'une phrase citée, comme une vraie réponse la rendrait."""
    return ("D'après le passage [1], la quantité recherchée s'écrit $$" + source +
            "$$ ce qui donne le prix à la date t.")


def cumuler(source):
    sortie = retirer_police(source)
    sortie = retirer_left_right(sortie)
    sortie = retirer_displaystyle(sortie)
    sortie = desaccolader(sortie)
    sortie = echanger_variantes(sortie)
    sortie = retirer_ponctuation_finale(sortie)
    return compacter(sortie)


#: Les ré-écritures qui doivent **toutes** être absorbées. « police retirée » n'y figure
#: pas : elle a son propre test, en ``xfail``, parce qu'elle échoue.
REECRITURES_TENUES = (
    ("compacte", compacter),
    ("left/right retirés", retirer_left_right),
    ("displaystyle retiré", retirer_displaystyle),
    ("variantes unifiées", echanger_variantes),
    ("accolades d'un atome retirées", desaccolader),
    ("ponctuation finale retirée", retirer_ponctuation_finale),
    ("enveloppée dans une phrase", envelopper),
    ("tout sauf la police", lambda s: envelopper(compacter(desaccolader(
        retirer_displaystyle(retirer_left_right(retirer_ponctuation_finale(s))))))),
)


def taux(reecriture):
    """Part des formules de l'étude que le scoreur retrouve après ``reecriture``."""
    etude = formules()
    justes = [sf.score(reecriture(corps), corps)["juste"] for _, _, corps in etude]
    return sum(justes) / len(justes), [(cid, corps) for (_, cid, corps), ok
                                       in zip(etude, justes) if not ok]


# ------------------------------------------------------------------ altérations

#: Un symbole, et un seul. Les tables évitent les couples que ``latex_norme`` unifie
#: **volontairement** (``\varphi``/``\phi``, ``\le``/``\leq``) : les y mettre mesurerait
#: une décision déclarée, pas un défaut.
ALTERATIONS = (
    ("signe", {"+": "-", "-": "+"}),
    ("exposant", {"2": "3", "3": "2", "1": "4"}),
    ("lettre grecque", {"\\beta": "\\gamma", "\\alpha": "\\lambda", "\\sigma": "\\tau",
                        "\\gamma": "\\beta", "\\lambda": "\\alpha", "\\mu": "\\nu",
                        "\\theta": "\\psi", "\\rho": "\\kappa", "\\phi": "\\chi"}),
    ("indice", {"t": "s", "i": "j", "n": "m", "N": "M", "T": "S", "x": "y"}),
)


def alterations(source):
    """Un mutant par table, altéré **hors** de ``\\tag{…}``.

    Muter le numéro d'équation ne changerait rien — la normalisation efface les ``\\tag`` —
    et gonflerait artificiellement le taux de rejet d'autant de cas vides.
    """
    zones = [(m.start(), m.end()) for m in re.finditer(r"\\tag\s*\{[^{}]*\}", source)]
    jetons = JETON.findall(source)
    positions, curseur = [], 0
    for jeton in jetons:
        positions.append(curseur)
        curseur += len(jeton)

    def dans_un_tag(indice):
        return any(debut <= positions[indice] < fin for debut, fin in zones)

    produits = []
    for nom, table in ALTERATIONS:
        candidats = [i for i, j in enumerate(jetons) if j in table and not dans_un_tag(i)]
        if not candidats:
            continue
        cible = candidats[len(candidats) // 2]   # le milieu : ni le bord gauche, ni le droit
        mutant = "".join(jetons[:cible] + [table[jetons[cible]]] + jetons[cible + 1:])
        produits.append((nom, jetons[cible], table[jetons[cible]], mutant))
    return produits


# ------------------------------------------------------------------ la population


def test_la_population_de_l_etude_est_reelle_et_suffisante():
    """Au moins 300 formules, une par document, tirées du corpus et pas d'un exemple inventé."""
    etude = formules()
    assert len(etude) >= POPULATION_MINIMALE
    assert len({document for document, _, _ in etude}) == len(etude)
    assert all("\\tag" in corps for _, _, corps in etude)
    assert all(len(ln.atomes(corps)) >= ATOMES_MINIMAUX for _, _, corps in etude)


def test_le_corpus_ecrit_bien_ses_mathematiques_caractere_par_caractere():
    """Le fait qui justifie toute la normalisation : MinerU espace les indices et les polices."""
    etude = formules()
    espacees = [c for _, _, c in etude if re.search(r"_ \{ | \^ \{ |\\mathrm \{ ", c)]
    assert len(espacees) > len(etude) // 2


# ------------------------------------------------------------------ (b) positifs


@pytest.mark.parametrize("nom,reecriture", REECRITURES_TENUES, ids=[n for n, _ in REECRITURES_TENUES])
def test_positifs_le_scoreur_absorbe_la_reecriture(nom, reecriture):
    """Un générateur qui réécrit ainsi restitue la **même** formule : le scoreur doit le dire."""
    part, echecs = taux(reecriture)
    assert part >= 0.99, (
        f"{nom} : {part:.4f} — contre-exemples : " +
        " | ".join(f"{cid} or={ln.normaliser(c)[:120]!r} "
                   f"cand={ln.normaliser(reecriture(c))[:120]!r}" for cid, c in echecs[:3]))


def test_positifs_police_retiree():
    """La ré-écriture qui a mis la normalisation en défaut, et qui la garde corrigée.

    Histoire de ce test, parce qu'il est la raison d'être du fichier. Il a d'abord été écrit
    en ``xfail(strict=True)`` : au ``NIVEAU_SOUPLE``, effacer une commande de police laissait
    ses accolades orphelines dès que l'argument faisait plus d'un atome —
    ``a + \\mathrm{cov}(x,y)`` se normalisait ``a + { c o v } ( x , y )`` quand un générateur
    écrit ``a + c o v ( x , y )``. **60 des 312 formules de l'étude échouaient**, toutes par
    ce seul mécanisme. Le cas ``^ { \\mathrm{bid} }`` — le seul que le docstring du module
    citait — passait, et c'est ce qui masquait le défaut. ``_appliquer_polices`` fait
    désormais emporter à la police les accolades de son argument : le taux est passé de
    **0,808 à 1,000**.
    """
    part, echecs = taux(retirer_police)
    assert part >= 0.99, (
        f"police retirée : {part:.4f} — " +
        " | ".join(f"{cid} or={ln.normaliser(c)[:120]!r} "
                   f"cand={ln.normaliser(retirer_police(c))[:120]!r}" for cid, c in echecs[:3]))


def test_aucune_accolade_orpheline_ne_subsiste():
    """La *forme* du défaut, pas seulement son ampleur.

    Le taux ci-dessus pourrait retomber à 0,99 en laissant trois cas pathologiques passer
    inaperçus. Ici on exige que la police effacée ne laisse **jamais** un écart qui ne tienne
    qu'aux accolades : si un tel écart réapparaissait, ce serait la même régression, revenue
    par une autre porte.
    """
    for _, chunk_id, corps in formules():
        atomes_or = ln.atomes(corps)
        atomes_candidat = ln.atomes(retirer_police(corps))
        if atomes_or == atomes_candidat:
            continue
        assert ([a for a in atomes_or if a not in "{}"]
                != [a for a in atomes_candidat if a not in "{}"]), (
            f"{chunk_id} : les deux formes ne diffèrent que par des accolades — "
            f"l'orphelinat des polices est revenu")


def test_le_cas_minimal_de_la_correction_hors_corpus():
    """Quatre lignes qui fixent la règle : la police emporte les accolades de son argument."""
    # Sous un exposant, le groupe extérieur survit — le sens est préservé.
    assert ln.inclus("V _ { i , t } ^ { \\mathrm { b i d } }", "V_{i,t}^{bid}")
    assert ln.normaliser("V _ { i , t } ^ { \\mathrm { b i d } }") == "V _ { i , t } ^ { b i d }"
    # Nue au milieu d'une expression, elle ne laisse plus rien derrière elle.
    assert ln.normaliser("a + \\mathrm { c o v } ( x , y )") == "a + c o v ( x , y )"
    assert ln.inclus("a + \\mathrm { c o v } ( x , y )", "a + \\operatorname{cov}(x, y)")
    # Et le niveau strict, lui, garde la police : c'est ce qui en fait un diagnostic.
    assert ln.normaliser("a + \\mathrm { c o v } ( x , y )",
                         ln.NIVEAU_STRICT) == "a + \\mathrm { c o v } ( x , y )"


def test_operatorname_etoile_est_traite_comme_une_police():
    """L'étoile de ``\\operatorname*`` est mangée avec la commande.

    ``_ATOME`` coupe la commande avant l'étoile : l'entrée ``\\operatorname*`` de ``_POLICES``
    était donc du code mort, et l'étoile survivait **en multiplication**, ce qui confondait
    ``\\operatorname*{max}(x)`` et ``\\ast{max}(x)``. ``_appliquer_polices`` avale maintenant
    l'étoile qui suit une police, aux deux niveaux.
    """
    assert ln.normaliser("\\operatorname* { m a x } ( x )") == "m a x ( x )"
    assert not ln.inclus("\\operatorname* { m a x } ( x )", "\\ast { m a x } ( x )")
    assert ln.normaliser("\\operatorname* { m a x } ( x )",
                         ln.NIVEAU_STRICT) == "\\mathrm { m a x } ( x )"


def test_la_compaction_naive_est_refusee_a_juste_titre():
    """Coller ``\\in`` à ``t`` fabrique ``\\int`` : ce n'est plus la même formule.

    Le scoreur doit refuser — c'est un succès, pas un échec, et ce test empêche qu'on
    « répare » un jour la normalisation en effaçant cette frontière.
    """
    assert not ln.inclus("\\sum _ { s \\in t } x _ s", "\\sum_{s\\int}x_s")
    assert not ln.inclus("\\frac { \\partial F } { \\partial Z }", "\\frac{\\partialF}{\\partialZ}")
    casses = [c for _, _, c in formules() if not ln.inclus(c, re.sub(r"\s+", "", c))]
    assert len(casses) > 100   # 129 sur 312 le 8 septembre 2026


# ------------------------------------------------------------------ (c) négatifs


def test_negatifs_un_seul_symbole_altere_suffit_a_faire_dire_faux():
    """Taux de rejet sur toutes les altérations d'un symbole. Un faux positif est un défaut."""
    tentees, rejetees, faux_positifs = 0, 0, []
    for _, chunk_id, corps in formules():
        for nom, avant, apres, mutant in alterations(corps):
            tentees += 1
            if sf.score(envelopper(mutant), corps)["juste"]:
                faux_positifs.append((chunk_id, nom, avant, apres))
            else:
                rejetees += 1
    assert tentees >= 800
    assert rejetees / tentees >= 0.99, f"faux positifs : {faux_positifs[:5]}"
    assert not faux_positifs, f"{len(faux_positifs)} faux positifs, ex. {faux_positifs[:3]}"


def test_negatifs_chaque_alteration_change_vraiment_la_forme_canonique():
    """Contre-garde du test précédent : un mutant qui se normalise comme l'or ne prouve rien."""
    inertes = []
    for _, chunk_id, corps in formules():
        for nom, avant, apres, mutant in alterations(corps):
            if ln.normaliser(mutant) == ln.normaliser(corps):
                inertes.append((chunk_id, nom, avant, apres))
    assert not inertes, f"altérations sans effet : {inertes[:5]}"


def test_negatifs_les_quatre_familles_d_alteration_sont_toutes_exercees():
    """Sinon le taux de rejet serait celui d'une seule famille, déguisé en mesure générale."""
    vues = {nom for _, _, corps in formules() for nom, _, _, _ in alterations(corps)}
    assert vues == {nom for nom, _ in ALTERATIONS}


def test_negatifs_cas_unitaires_lisibles():
    """Les quatre altérations, en clair, sur une formule qu'on peut lire d'un coup d'œil."""
    formule = "\\sigma _ { t } ^ { 2 } = \\alpha + \\beta \\sigma _ { t - 1 } ^ { 2 }"
    assert sf.score("$$" + formule + "$$", formule)["juste"]
    for mutant in ("\\sigma _ { t } ^ { 2 } = \\alpha - \\beta \\sigma _ { t - 1 } ^ { 2 }",
                   "\\sigma _ { s } ^ { 2 } = \\alpha + \\beta \\sigma _ { t - 1 } ^ { 2 }",
                   "\\sigma _ { t } ^ { 3 } = \\alpha + \\beta \\sigma _ { t - 1 } ^ { 2 }",
                   "\\sigma _ { t } ^ { 2 } = \\alpha + \\gamma \\sigma _ { t - 1 } ^ { 2 }"):
        assert not sf.score("$$" + mutant + "$$", formule)["juste"], mutant


# ------------------------------------------------------------------ (d) leurres


def test_leurres_une_formule_d_un_autre_document_n_est_jamais_trouvee():
    """Contrôle négatif large : 4 leurres étrangers par formule, aucun ne doit être inclus."""
    etude = formules()
    touches, paires = [], 0
    for rang, (document, chunk_id, corps) in enumerate(etude):
        leurres = [etude[(rang + saut) % len(etude)][2] for saut in (1, 7, 31, 97)
                   if etude[(rang + saut) % len(etude)][0] != document]
        paires += len(leurres)
        verdict = sf.score(envelopper(corps), corps, leurres=leurres)
        assert verdict["juste"] or verdict["voie"] is None
        if verdict["leurre_inclus"]:
            touches.append(chunk_id)
    assert paires >= 1000
    assert not touches, touches


def test_leurres_du_meme_document_le_controle_serre():
    """Le vrai contrôle du §4 : les autres formules **du même article**, bien plus proches.

    Les seules collisions du corpus sont des **répétitions** — la même équation dans deux
    chunks sous deux ``\\tag`` différents : 7 documents sur les 279 pourvus d'un leurre
    interne, 8 leurres sur 8 752. ``score_formule`` les écarte, et les compte.
    """
    etude, par_document = population()
    touches, ecartes, testes = [], 0, 0
    for document, chunk_id, corps in etude:
        leurres = [c for identifiant, c in par_document[document] if identifiant != chunk_id]
        if not leurres:
            continue
        testes += 1
        verdict = sf.score(envelopper(corps), corps, leurres=leurres)
        ecartes += verdict["leurres_ecartes"]
        if verdict["leurre_inclus"]:
            touches.append((chunk_id, verdict["leurres_touches"][0][:90]))
    assert testes >= 250
    assert ecartes >= 5, "plus aucune répétition écartée : le filtre ne sert plus à rien"
    assert not touches, touches


def test_leurres_un_leurre_reellement_present_est_signale():
    """Le drapeau doit **savoir** dire vrai, sinon il ne prouve rien quand il dit faux."""
    formule = "\\sigma _ { t } ^ { 2 } = \\alpha + \\beta \\sigma _ { t - 1 } ^ { 2 }"
    leurre = "\\mu _ { t } = \\kappa ( \\theta - X _ { t } ) \\mathrm d t"
    verdict = sf.score("$$" + formule + "$$ et aussi $$" + leurre + "$$", formule,
                       leurres=[leurre])
    assert verdict["juste"] and verdict["leurre_inclus"] is True
    assert verdict["leurres_retenus"] == 1 and verdict["leurres_ecartes"] == 0


def test_leurres_une_repetition_de_l_or_est_ecartee_et_non_comptee():
    """Cas mesuré 7 fois sur 279 documents : `\\tag{A.9a}` et `\\tag{7a}`, même équation."""
    formule = "R _ { i , t } = \\mu _ { i } + \\beta _ { i } \\nu _ { t } \\tag{7a}"
    repetition = "R _ { i , t } = \\mu _ { i } + \\beta _ { i } \\nu _ { t } \\tag{A.9a}"
    verdict = sf.score(envelopper(formule), formule, leurres=[repetition])
    assert verdict["juste"]
    assert verdict["leurre_inclus"] is False
    assert verdict["leurres_ecartes"] == 1 and verdict["leurres_retenus"] == 0


def test_leurres_absents_le_drapeau_vaut_None_et_pas_False():
    """``False`` affirmerait qu'on a cherché. Sans leurre, on n'a rien cherché."""
    assert sf.score("$$x = 1 + y$$", "x = 1 + y")["leurre_inclus"] is None


# ------------------------------------------------------------------ le scoreur


def test_le_tag_du_corpus_est_absorbe_des_deux_cotes():
    """Un or prélevé tel quel dans le corpus porte son numéro d'équation ; pas la réponse."""
    assert sf.score("$$x _ { t } = a + b$$", "x _ { t } = a + b \\tag{2.1}")["juste"]


def test_les_trois_balisages_sont_reconnus():
    """``$$…$$``, ``$…$`` et ``\\[…\\]`` : le générateur choisit, le scoreur ne s'en mêle pas."""
    formule = "\\mathbb { E } [ X _ { t } ] = \\mu + \\beta z"
    for gauche, droite in (("$$", "$$"), ("$", "$"), ("\\[", "\\]")):
        verdict = sf.score("On obtient " + gauche + formule + droite + " au total.", formule)
        assert verdict["juste"] and verdict["balisee"] and verdict["voie"] == "balisee"


def test_le_repli_sur_le_texte_entier_rattrape_une_reponse_non_balisee():
    """Refuser une formule correcte parce qu'elle n'a pas de ``$$`` mesurerait le formatage."""
    formule = "\\mathbb { E } [ X _ { t } ] = \\mu + \\beta z"
    verdict = sf.score("La réponse est E[X_t] = \\mu + \\beta z d'après [1].", formule)
    assert verdict["juste"] and verdict["voie"] == "texte_entier" and not verdict["balisee"]


def test_le_risque_du_repli_est_reel_et_il_est_instrumente():
    """Hors balise, la prose est atomisée lettre à lettre : un or court se cache dans un mot.

    On ne le corrige pas — on le rend visible. ``or_court`` et ``voie`` suffisent à
    recompter, après coup, les « juste » qui reposent sur ce chemin.
    """
    verdict = sf.score("la VaR du portefeuille est élevée", "V a R")
    assert verdict["juste"] and verdict["voie"] == "texte_entier"
    assert verdict["or_court"] is True
    # Un or de taille réelle ne se cache nulle part : c'est ce que mesure la population.
    assert sf.score("le prix est élevé", "\\mathbb { E } [ X _ { t } ] = \\mu + \\beta z")["juste"] is False


def test_le_recouvrement_est_un_diagnostic_et_jamais_une_decision():
    """0,80 sur une formule dont le signe est faux : un seuil ici aurait validé l'erreur."""
    verdict = sf.score("$$a - b = c$$", "a + b = c")
    assert verdict["juste"] is False
    assert verdict["recouvrement"] >= 0.75


def test_le_recouvrement_vaut_un_quand_la_formule_est_bien_la():
    """Invariant vérifié sur les 312 formules : présente ⟹ recouvrement 1,0."""
    manques = [chunk_id for _, chunk_id, corps in formules()
               if sf.score(envelopper(corps), corps)["recouvrement"] < 0.999]
    assert not manques, manques[:5]


def test_juste_strict_diagnostique_la_police_sans_jamais_decider():
    """L'écart souple/strict dit combien de restitutions ne diffèrent que par la police."""
    formule = "\\mathbb { E } [ X ] = \\mu"
    verdict = sf.score("$$E[X] = \\mu$$", formule)
    assert verdict["juste"] is True and verdict["juste_strict"] is False
    identique = sf.score("$$\\mathbb{E}[X] = \\mu$$", formule)
    assert identique["juste"] and identique["juste_strict"]


def test_l_abstention_est_signalee_et_n_est_jamais_juste():
    verdict = sf.score("INSUFFICIENT_EVIDENCE les passages ne donnent pas la formule.",
                       "x _ { t } = a + b z")
    assert verdict["abstenue"] is True and verdict["juste"] is False
    assert sf.score("$$x_t = a + bz$$", "x _ { t } = a + b z")["abstenue"] is False


def test_un_or_vide_ou_absent_ne_vaut_jamais_juste():
    """Sinon toute réponse serait juste sur une question mal formée — le pire des faux positifs."""
    for vide in (None, "", "   ", "\\tag{4}"):
        verdict = sf.score("$$x = 1$$ n'importe quoi", vide)
        assert verdict["juste"] is False and verdict["atomes_or"] == 0
        assert verdict["recouvrement"] == 0.0


def test_une_reponse_vide_ou_absente_ne_fait_pas_tomber_le_scoreur():
    for vide in (None, "", "   "):
        verdict = sf.score(vide, "x _ { t } = a + b z")
        assert verdict["juste"] is False and verdict["balisee"] is False


def test_un_or_multiligne_est_signale_a_la_selection():
    """``latex_norme`` déclare ne pas évaluer ces environnements ; le scoreur le répète."""
    assert sf.score("$$x$$", "\\begin{array}{rl} a & b \\end{array} \\tag{2.7}")["or_multiligne"] == "array"
    assert sf.score("$$x = 1$$", "x = 1 + y")["or_multiligne"] is None


def test_le_scoreur_ne_touche_ni_au_reseau_ni_au_generateur():
    """Garde d'instrument : le module doit rester chargeable et rejouable hors ligne.

    On lit ses ``import`` plutôt que ``sys.modules`` : une garde sur ``sys.modules`` passe
    au vert par accident dès qu'aucun voisin n'a chargé ``llm`` dans la même session, et ne
    dit donc rien. Ici, un ``import`` interdit ajouté demain fait rougir le test.
    """
    source = (BENCH / "score_formule.py").read_text(encoding="utf-8")
    importes = set(re.findall(r"^\s*(?:import|from)\s+([\w.]+)", source, re.MULTILINE))
    assert importes == {"__future__", "re", "sys", "pathlib", "latex_norme"}, importes
    assert sf.JETON_ABSTENTION == "INSUFFICIENT_EVIDENCE"


# ------------------------------------------------------------------ troncature de la fenêtre servie

def test_une_accolade_non_refermee_ne_fait_ni_planter_ni_perdre_de_contenu():
    """Le cas qui a planté la fabrique de questions, en production, au premier candidat.

    ``inclus`` est appliqué à des **passages tronqués à 2 500 caractères** : la coupe casse
    des paires d'accolades. ``_fermeture`` rendait alors ``len - 1`` — un index plausible et
    faux —, si bien que le repli pliait un groupe inexistant, perdait du contenu en silence
    (``a + \\mathrm { c o v`` rendait ``a + c o``) puis débordait de la liste. Elle rend
    maintenant ``-1``, et chaque appelant traite l'accolade comme l'atome littéral qu'elle est.
    """
    for source in ("{", "a { b", "{ { a", "x ^ {", "} }", "{ a } {", "a + \\mathrm { c o v"):
        ln.normaliser(source)                       # ne doit pas lever
    assert ln.normaliser("a { b") == "a { b"        # rien n'est perdu
    assert ln.normaliser("a + \\mathrm { c o v") == "a + { c o v"
    assert ln.normaliser("\\frac { a + b } { c }") == "\\frac { a + b } c"


def test_l_or_est_retrouve_dans_un_passage_tronque_a_la_fenetre_servie():
    """Le bout en bout : chaque formule de l'étude est incluse dans ses 2 500 premiers caractères.

    C'est le verrou d'or servi de ``familles_v4`` : une formule qui échoue ici n'a rien à
    faire dans la famille, et le fait de le vérifier sur 312 cas réels garantit que le verrou
    mesure la fenêtre, pas une faiblesse de la normalisation.
    """
    index = ChunkIndex.load(verbose=False)
    manques = []
    for _, chunk_id, corps in formules():
        servi = index.chunks[chunk_id]["text"][:2500]
        if corps in servi and not ln.inclus(corps, servi):
            manques.append(chunk_id)
    assert not manques, f"formules présentes dans la fenêtre mais non retrouvées : {manques[:5]}"


# ------------------------------------------------------------------ ce que le corpus ne pouvait pas montrer

def test_les_mathematiques_unicode_sont_lues_comme_du_latex():
    """Le trou qu'aucune ré-écriture dérivée du corpus ne pouvait révéler.

    Toutes les ré-écritures de ce fichier partent du LaTeX du corpus : elles ne produisent
    donc **jamais** d'Unicode. Un générateur, lui, répond volontiers en Unicode. Trouvé en
    lisant les réponses réelles de la ligne de base v4 — un des trois échecs à or servi de la
    famille ``formula`` était celui-ci, et non une infidélité du système.
    """
    assert ln.normaliser("σ ^ 2 = α + β") == ln.normaliser("\\sigma ^ 2 = \\alpha + \\beta")
    assert ln.normaliser("x ∈ ℝ") == ln.normaliser("x \\in \\mathbb { R }")
    assert ln.normaliser("∑ _ j x _ j ≤ 1") == ln.normaliser("\\sum _ { j } x _ { j } \\leq 1")
    assert ln.inclus("\\sum _ { j } x _ { j } \\times y", "the sum is ∑_j x_j × y here")
    # La frontière de commande tient : « ∑x » doit donner deux atomes, pas « \sumx ».
    assert ln.atomes("∑x") == ["\\sum", "x"]


def test_le_delimiteur_nul_de_left_et_right_ne_laisse_pas_de_point():
    """``\\left.`` est un délimiteur **nul** : le point n'est pas une ponctuation.

    Effacer ``\\left`` en gardant le ``.`` fabriquait un atome que nul générateur n'écrit.
    C'est ce point orphelin qui faisait échouer f001 de la ligne de base — recouvrement 0,984,
    et pourtant « faux ».
    """
    assert ln.normaliser("\\left. \\left[ a \\right] \\right| _ { 0 }") == "[ a ] | _ 0"
    assert ln.inclus("\\sum _ { x } \\left. f ( x ) \\right| _ { 0 }", "$\\sum_x f(x)|_0$")
    # Un point qui suit autre chose qu'une commande de taille reste une ponctuation, et la
    # ponctuation terminale est élaguée comme avant.
    assert ln.normaliser("a = b .") == "a = b"
    assert ln.normaliser("a . b") == "a . b"
