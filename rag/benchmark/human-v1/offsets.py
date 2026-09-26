"""Les offsets des ancres de `human-v1` — la coordonnée qu'une citation doit porter.

**Pourquoi ce module.** `docs/STRATEGIE.md` §4.2 pose que « évaluer les citations » n'est pas
d'abord un trou de mesure mais un trou de **produit** : le système ne sait pas désigner un
passage avec précision, et la localisation qu'il affiche est fausse une fois sur deux là où
on peut la vérifier. Depuis `31cd6fb`, la coordonnée existe — `src/parsing/document_text.py`
définit le **texte canonique** d'un document et l'y ancre au caractère près. Une citation
utilisateur peut donc enfin être ce qu'elle doit être : **document + page + offsets +
extrait source**, où l'extrait est vérifiable *contre* les offsets et non à côté d'eux.

**Ce que le module garantit, et comment.** Un offset ne vaut que relativement à un texte de
référence : si le parse change l'ordre ou le contenu des blocs, l'offset désigne autre chose
sans rien dire. `document_text` répond à cela par le `sha256` du texte canonique, et ce
module l'enregistre **à côté de chaque offset**. `verifier()` recalcule les deux : un offset
dont le `sha256` a bougé est déclaré périmé, jamais réinterprété.

**La normalisation est à longueur préservée, et c'est la seule qui soit permise ici.**
Sur les 28 ancres du lot pilote, 27 se retrouvent telles quelles dans le texte canonique.
La vingt-huitième diffère d'un caractère — `Lesmond et\\xa0 al.` dans la source, une espace
ordinaire dans l'ancre. Une normalisation qui *replierait* les espaces (`\\s+` → `" "`)
réglerait le cas et **détruirait la correspondance de position** : les offsets rendus ne
désigneraient plus le texte réel. On remplace donc chaque séparateur Unicode par une espace
ordinaire, **caractère pour caractère**, ce qui laisse chaque position inchangée. Une ancre
qui ne se retrouve pas même ainsi reçoit `offsets: null` **avec son motif** — jamais un
offset approché.

**Ce que le module ne fait pas.** Il ne remplace pas l'ancrage par le texte, qui reste
l'ancre primaire : un offset est une clé **secondaire exacte**, au même titre que les
`block_ids`, et ses modes de défaillance sont indépendants de ceux du texte. C'est un
contrôle croisé gratuit, pas une redondance.

    .venv/bin/python rag/benchmark/human-v1/offsets.py                 # constat
    .venv/bin/python rag/benchmark/human-v1/offsets.py --ecrire        # enrichit le lot
    .venv/bin/python rag/benchmark/human-v1/offsets.py --verifier      # les offsets tiennent-ils ?
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
ROOT = BENCHMARK.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rag"))

import corpus_overlay  # noqa: E402
from src.parsing.document_text import SEPARATEUR, sha256_texte, texte_canonique  # noqa: E402

LOT = HERE / "items-pilote-v0.jsonl"
INGESTED = ROOT / "data" / "processed" / "ingested"

#: Motifs déclarés quand une ancre ne reçoit pas d'offset. La liste est fermée : un motif
#: libre laisserait passer « pas trouvé », qui n'est pas un diagnostic.
MOTIFS = {
    "document_absent": "le document n'a pas de blocs sur le disque",
    "texte_introuvable": "l'ancre ne se retrouve pas dans le texte canonique, même à espaces normalisées",
    "texte_ambigu": "l'ancre apparaît plusieurs fois : les offsets seuls ne la désignent pas",
}


def normaliser_espaces(texte: str) -> str:
    """Chaque séparateur Unicode devient une espace ordinaire. **La longueur ne change pas.**

    C'est la propriété qui rend la normalisation utilisable pour des offsets : la position
    *i* du texte normalisé est la position *i* du texte réel. Toute normalisation qui
    supprime ou fusionne des caractères la perdrait, et rendrait des offsets qui ont l'air
    justes.
    """
    return "".join(" " if (c == "\t" or unicodedata.category(c) == "Zs") else c for c in texte)


def dossiers_par_document() -> dict[str, str]:
    registre = json.loads(corpus_overlay.REGISTRY.read_text(encoding="utf-8"))
    return {e["document_id"]: e["folder"] for e in registre.get("documents", [])}


class Documents:
    """Textes canoniques, calculés une fois par document (un gros document coûte ~50 ms)."""

    def __init__(self) -> None:
        self._dossiers = dossiers_par_document()
        self._cache: dict[str, tuple[str, str, str] | None] = {}
        self._pages: dict[str, list | None] = {}

    def texte(self, document_id: str) -> tuple[str, str, str] | None:
        """``(texte canonique, texte normalisé, sha256)`` ou ``None`` si les blocs manquent."""
        if document_id in self._cache:
            return self._cache[document_id]
        dossier = self._dossiers.get(document_id)
        chemin = INGESTED / dossier / "blocks.jsonl" if dossier else None
        if chemin is None or not chemin.exists():
            self._cache[document_id] = self._pages[document_id] = None
            return None
        blocs = [json.loads(ligne) for ligne in chemin.open(encoding="utf-8") if ligne.strip()]
        brut, _ = texte_canonique(blocs)
        position, plages = 0, []
        for bloc in blocs:
            corps = bloc.get("text") or ""
            plages.append((position, position + len(corps), bloc.get("page_idx")))
            position += len(corps) + len(SEPARATEUR)
        self._pages[document_id] = plages
        self._cache[document_id] = (brut, normaliser_espaces(brut), sha256_texte(brut))
        return self._cache[document_id]

    def pages(self, document_id: str, debut: int, fin: int) -> list[int]:
        """Les pages **telles qu'un lecteur les compte** que l'intervalle recouvre.

        Le ``page_start`` d'un chunk est faux pour une citation, et pour **deux raisons
        indépendantes** — la seconde ne se corrige pas par un « +1 ».

        **Un.** ``page_idx`` des blocs est **0-basé** — le bloc de titre d'un document porte
        0 — donc la page d'un lecteur vaut ``page_idx + 1``. Mesuré le 7 septembre 2026 sur
        `doc-2b74a994fcfc1c9c` : « 1 Introduction » est en ``page_idx`` 1, « 6 Conclusion » en
        34, et les chunks correspondants déclarent ``page_start`` 1 et 34. Le champ servi est
        donc 0-basé, et une citation qui le reprend envoie le lecteur une page trop tôt.

        **Deux.** ``page_start`` est la page où le **chunk** commence, pas celle où se trouve
        la **phrase citée**. Un chunk qui enjambe une coupure de page les met sur deux pages
        différentes. Mesuré sur les 49 ancres des deux lots : 42 à un écart d'une page,
        **5 à deux pages, 2 à trois**.

        `human-v1` ne propage aucune des deux : la page d'une citation se calcule ici, depuis
        les blocs que l'intervalle recouvre réellement.
        """
        self.texte(document_id)
        plages = self._pages.get(document_id)
        if not plages:
            return []
        return sorted({p + 1 for a, b, p in plages if a < fin and b > debut and p is not None})


def localiser(documents: Documents, document_id: str, ancre: str) -> dict:
    """Les offsets d'une ancre, ou un refus motivé. Jamais un offset approché."""
    trouve = documents.texte(document_id)
    if trouve is None:
        return {"offsets": None, "motif": "document_absent"}
    brut, normalise, sha = trouve
    cible = normaliser_espaces(ancre)
    occurrences = normalise.count(cible)
    if occurrences == 0:
        return {"offsets": None, "motif": "texte_introuvable"}
    debut = normalise.index(cible)
    resultat = {"debut": debut, "fin": debut + len(cible), "doc_text_sha256": sha,
                "occurrences": occurrences,
                # `exact` dit si l'ancre est le texte source au caractère près. Faux ne
                # signifie pas faux offset : il signifie que la transcription a aplati une
                # espace insécable, et que c'est la seule liberté qui a été prise.
                "exact": brut[debut:debut + len(cible)] == ancre,
                "normalisation": "espaces_unicode_longueur_preservee"}
    if occurrences > 1:
        return {"offsets": resultat, "motif": "texte_ambigu"}
    return {"offsets": resultat, "motif": None}


def enrichir(items: list[dict]) -> tuple[list[dict], dict]:
    documents = Documents()
    compte = {"ancres": 0, "avec_offsets": 0, "exactes": 0, "ambigues": 0, "refusees": 0,
              "pages_corrigees": 0}
    refus: list[str] = []
    for item in items:
        for appui in item.get("appuis") or []:
            compte["ancres"] += 1
            trouve = localiser(documents, appui["document_id"], appui["texte"])
            appui["offsets"] = trouve["offsets"]
            if trouve["motif"]:
                appui["offsets_motif"] = trouve["motif"]
                refus.append(f"{item['id']}/{appui['ancre_id']} : {MOTIFS[trouve['motif']]}")
            else:
                appui.pop("offsets_motif", None)
            if trouve["offsets"]:
                compte["avec_offsets"] += 1
                compte["exactes"] += bool(trouve["offsets"]["exact"])
                compte["ambigues"] += trouve["offsets"]["occurrences"] > 1
                # La page d'une citation se calcule ici, depuis les blocs, et non depuis le
                # `page_start` du chunk, qui est 0-basé (voir ``Documents.pages``). Quand les
                # deux diffèrent, l'écart est **conservé dans l'item** plutôt que corrigé en
                # silence : c'est un défaut du chemin servi, pas une coquille de ce lot.
                couvertes = documents.pages(appui["document_id"], trouve["offsets"]["debut"],
                                            trouve["offsets"]["fin"])
                if couvertes:
                    if appui.get("page") is not None and appui["page"] not in couvertes:
                        appui["page_declaree_par_le_chunk"] = appui["page"]
                        compte["pages_corrigees"] += 1
                    appui["page"] = couvertes[0]
                    appui["pages_couvertes"] = couvertes
            else:
                compte["refusees"] += 1
    return items, {"compte": compte, "refus": refus}


def verifier(items: list[dict]) -> list[str]:
    """Les offsets enregistrés désignent-ils encore le texte enregistré ?

    Deux questions distinctes, et il faut les deux. *Le texte de référence a-t-il bougé ?*
    — c'est le `sha256`, et il rend l'offset opposable. *L'offset désigne-t-il bien
    l'extrait ?* — c'est la tranche, et sans elle un `sha256` juste garantirait seulement
    qu'on lit le bon document.
    """
    documents = Documents()
    erreurs = []
    for item in items:
        for appui in item.get("appuis") or []:
            ident = f"{item['id']}/{appui['ancre_id']}"
            offsets = appui.get("offsets")
            if offsets is None:
                if not appui.get("offsets_motif"):
                    erreurs.append(f"{ident} : offsets absents sans motif déclaré")
                elif appui["offsets_motif"] not in MOTIFS:
                    erreurs.append(f"{ident} : motif inconnu ({appui['offsets_motif']!r})")
                continue
            trouve = documents.texte(appui["document_id"])
            if trouve is None:
                erreurs.append(f"{ident} : blocs du document absents, l'offset n'est plus vérifiable")
                continue
            brut, normalise, sha = trouve
            if offsets.get("doc_text_sha256") != sha:
                erreurs.append(f"{ident} : le texte canonique a changé — offset PÉRIMÉ, "
                               "à recalculer, jamais à réinterpréter")
                continue
            tranche = normalise[offsets["debut"]:offsets["fin"]]
            if tranche != normaliser_espaces(appui["texte"]):
                erreurs.append(f"{ident} : l'offset ne désigne pas l'extrait enregistré")
            if offsets.get("exact") and brut[offsets["debut"]:offsets["fin"]] != appui["texte"]:
                erreurs.append(f"{ident} : déclaré exact, mais le texte source diffère")
    return erreurs


def charger(chemin: Path) -> list[dict]:
    return [json.loads(ligne) for ligne in chemin.read_text(encoding="utf-8").splitlines()
            if ligne.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lot", type=Path, default=LOT)
    parser.add_argument("--ecrire", action="store_true", help="enrichir le lot en place")
    parser.add_argument("--verifier", action="store_true", help="contrôler les offsets enregistrés")
    args = parser.parse_args()
    items = charger(args.lot)

    if args.verifier:
        erreurs = verifier(items)
        for erreur in erreurs:
            print(f"  ÉCHEC  {erreur}")
        ancres = sum(len(i.get("appuis") or []) for i in items)
        avec = sum(1 for i in items for a in (i.get("appuis") or []) if a.get("offsets"))
        print(f"\n{ancres} ancres, {avec} avec offsets, {len(erreurs)} erreur(s).")
        sys.exit(1 if erreurs else 0)

    items, rapport = enrichir(items)
    compte = rapport["compte"]
    print(f"ancres            {compte['ancres']}")
    print(f"  avec offsets    {compte['avec_offsets']}")
    print(f"    exactes       {compte['exactes']}  (le texte source au caractère près)")
    print(f"    ambiguës      {compte['ambigues']}  (plusieurs occurrences : les block_ids tranchent)")
    print(f"  refusées        {compte['refusees']}")
    print(f"  pages corrigées {compte['pages_corrigees']}  (le `page_start` du chunk est 0-basé ;\n                  la page citée se calcule depuis les blocs — voir Documents.pages)")
    for ligne in rapport["refus"]:
        print(f"    {ligne}")
    if not args.ecrire:
        print("\nConstat seul — relance avec --ecrire pour enrichir le lot.")
        return
    args.lot.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
                        encoding="utf-8")
    print(f"\n  écrit  {args.lot.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
