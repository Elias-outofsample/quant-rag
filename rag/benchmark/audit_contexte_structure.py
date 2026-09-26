"""Audit du contexte de structure — reconnaissance en **lecture seule**, signature ``5530cba145``.

Ce n'est pas le chantier de représentation : c'est une vérification préalable de la matière
première qu'il compte utiliser. Rien n'est créé, rien n'est ré-embarqué, rien n'est écrit en
production, aucun appel LLM, aucun banc rejoué. Le script lit
``.cache/chunk-index-v5-5530cba145.pkl`` — le corpus **tel qu'il est servi**.

La question, et pourquoi elle se pose maintenant
------------------------------------------------
``docs/STRATEGIE.md`` §3.3 recommande de commencer le chantier par un **préfixe contextuel
déterministe** — titre du document, chemin de section, page — parce que ces champs existent
déjà : ``title_path`` couvre **25 861 / 26 120 chunks (99,0 %)**.

Mais 99,0 % est une **couverture**, pas une **justesse**. Un préfixe se lit d'abord ; s'il
nomme la mauvaise section, il n'ajoute pas du contexte, il en fabrique un faux — et il le
fabrique dans le texte plongé, donc dans le vecteur, donc dans le classement.

Ce que le script mesure, et comment
-----------------------------------
Un chunk est **vérifiable** quand deux conditions tiennent :

1. son champ ``section`` commence par un titre **numéroté** (``2.``, ``3.1``, ``4.2.1``…) ;
2. son **texte** contient au moins une ligne qui est elle-même un titre numéroté.

Sur cette population — et sur elle seule — on compare le numéro que l'étiquette annonce à
celui du premier titre que le passage contient. Quatre issues :

``A``  le chunk **s'ouvre littéralement** sur le titre étiqueté (première ligne non vide) —
       le seul accord franc ;
``A-`` l'étiquette est bien le premier titre du texte, mais le chunk s'ouvre **avant** lui,
       sur de la prose qui appartient à la section précédente. Accord douteux : compté à
       part, jamais avec ``A`` ;
``B``  l'étiquette apparaît plus loin dans le chunk — le passage enjambe la frontière et
       l'étiquette nomme la section où il **finit**, pas celle où il commence ;
``C``  l'étiquette n'apparaît **nulle part** dans le chunk — et on regarde alors si elle est
       *en avance* (elle nomme une section que le passage n'a pas atteinte) ou *en retard*.

La distinction ``A`` / ``A-`` n'était pas dans la première version : elle vient de
l'échantillonnage à la main, où 1 des 4 chunks tirés en ``A`` s'ouvrait sur la prose de la
section précédente. Sans elle, ``A`` aurait surestimé l'accord.

Ce que le script ne mesure pas, et qu'il ne faut pas lui faire dire
------------------------------------------------------------------
- Les titres **non numérotés** (« Abstract », « Appendix A », les ouvrages) sont invisibles au
  détecteur. La population vérifiable est un sous-ensemble, pas un échantillon représentatif.
- Un chunk sans titre dans son texte n'est **pas** un chunk mal étiqueté : c'est une
  continuation, cas parfaitement normal. Il est exclu, pas compté contre.
- ``title_path`` **contient toujours** ``section`` quand celle-ci existe (22 999 / 22 999,
  vérifié) : ce qui est mesuré sur ``section`` vaut pour le chemin qu'un préfixe utiliserait.
- C'est un **diagnostic**. Aucun chiffre d'ici n'entre dans une métrique de décision, et le
  détecteur est une heuristique : ``--echantillon`` existe pour en mesurer la précision à la
  main, parce qu'un compte sans précision mesurée serait un chiffre décoratif.

Le second défaut : le chemin lui-même
-------------------------------------
``--chemin`` mesure une autre incohérence, indépendante de la première : quand ``chapter`` et
``section`` sont tous deux numérotés, leur numéro de premier niveau devrait concorder. Il
arrive qu'il ne concorde pas — ``6.4 … > 7.1 WHAT BROKERS DO`` : l'ancêtre est resté au
chapitre précédent. Cette étape-là lit la collection Qdrant (``title_path`` n'est pas dans le
cache), donc **un seul processus à la fois**.

    .venv/bin/python rag/benchmark/audit_contexte_structure.py --prevalence
    .venv/bin/python rag/benchmark/audit_contexte_structure.py --echantillon C --n 10
    .venv/bin/python rag/benchmark/audit_contexte_structure.py --chemin
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import pickle
import random
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # rag/ : quant_rag, pour --chemin seulement
SIGNATURE = "5530cba145"
INDEX = HERE / ".cache" / f"chunk-index-v5-{SIGNATURE}.pkl"

#: Un titre numéroté : « 2. Model », « 3.1 Data », « 4.2.1. Calibration ». La ligne entière
#: doit être courte — au-delà, c'est une phrase qui commence par un nombre, pas un titre.
TITRE = re.compile(r"^(\d+(?:\.\d+)*)\.?[ \t]+(\S.*)$")
LONGUEUR_TITRE_MAX = 90


def charger() -> dict:
    return pickle.load(INDEX.open("rb"))["chunks"]


def numero_etiquette(section: str) -> str | None:
    """Numéro annoncé par le champ ``section``, ou ``None`` si l'étiquette n'est pas numérotée."""
    if not section or not section.strip():
        return None
    m = TITRE.match(section.strip().splitlines()[0])
    return m.group(1) if m else None


def numeros_du_texte(texte: str) -> list[str]:
    """Numéros des lignes de titre numérotées présentes dans le corps du chunk, dans l'ordre."""
    out = []
    for ligne in (texte or "").splitlines():
        ligne = ligne.strip()
        if 0 < len(ligne) <= LONGUEUR_TITRE_MAX:
            m = TITRE.match(ligne)
            if m:
                out.append(m.group(1))
    return out


def numero_ouverture(texte: str) -> str | None:
    """Numéro du titre sur lequel le chunk s'ouvre — première ligne non vide, ou ``None``."""
    for ligne in (texte or "").splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        m = TITRE.match(ligne) if len(ligne) <= LONGUEUR_TITRE_MAX else None
        return m.group(1) if m else None
    return None


def rang(numero: str) -> tuple[int, ...]:
    return tuple(int(x) for x in numero.split("."))


def classer(chunk: dict) -> tuple[str, dict] | None:
    """Classe un chunk, ou ``None`` s'il n'est pas vérifiable."""
    etiq = numero_etiquette(chunk.get("section", ""))
    if etiq is None:
        return None
    nums = numeros_du_texte(chunk.get("text", ""))
    if not nums:
        return None
    detail = {"etiquette": etiq, "titres": nums}
    if nums[0] == etiq:
        return ("A" if numero_ouverture(chunk.get("text", "")) == etiq else "A-"), detail
    if etiq in nums:
        return "B", detail
    try:
        e, premier, dernier = rang(etiq), rang(nums[0]), rang(nums[-1])
    except ValueError:
        return "C", detail | {"sens": "illisible"}
    if e > dernier:
        return "C", detail | {"sens": "en avance"}
    if e < premier:
        return "C", detail | {"sens": "en retard"}
    return "C", detail | {"sens": "intercalée"}


def step_prevalence(chunks: dict) -> None:
    total = len(chunks)
    c = collections.Counter()
    for chunk in chunks.values():
        verdict = classer(chunk)
        if verdict is None:
            c["non vérifiable"] += 1
            continue
        classe, detail = verdict
        c[classe] += 1
        if classe == "C":
            c["C/" + detail.get("sens", "?")] += 1

    verifiables = c["A"] + c["A-"] + c["B"] + c["C"]

    def pc(n, base):
        return f"{n:>6}  ({100.0 * n / base:5.1f} %)" if base else f"{n:>6}"

    print(f"corpus                                            {total:>6}  (signature {SIGNATURE})")
    print(f"  non vérifiable (étiquette ou texte sans titre)  {pc(c['non vérifiable'], total)}")
    print(f"  POPULATION VÉRIFIABLE                           {pc(verifiables, total)}")
    print()
    print(f"  A  le chunk s'ouvre sur le titre étiqueté       {pc(c['A'], verifiables)}")
    print(f"  A- étiquette = 1er titre, mais ouverture avant  {pc(c['A-'], verifiables)}")
    print(f"  B  étiquette plus loin dans le chunk            {pc(c['B'], verifiables)}")
    print(f"  C  étiquette absente du chunk                   {pc(c['C'], verifiables)}")
    for sens in ("en avance", "en retard", "intercalée", "illisible"):
        n = c["C/" + sens]
        if n:
            print(f"       · {sens:<38} {pc(n, c['C'])} des C")
    print()
    print(f"  le chunk ne s'ouvre PAS sur sa section          "
          f"{pc(c['A-'] + c['B'] + c['C'], verifiables)}")


def step_echantillon(chunks: dict, classe: str, n: int, graine: int = 20260906) -> None:
    retenus = []
    for cid, chunk in sorted(chunks.items()):
        verdict = classer(chunk)
        if verdict and verdict[0] == classe:
            retenus.append((cid, chunk, verdict[1]))
    random.Random(graine).shuffle(retenus)
    print(f"classe {classe} : {len(retenus)} chunks, {min(n, len(retenus))} tirés "
          f"(graine {graine})\n")
    for cid, chunk, detail in retenus[:n]:
        ouverture = "\n".join((chunk.get("text") or "").splitlines()[:3])
        print(f"── {cid}  {chunk.get('document_id')}  p.{chunk.get('page_start')}")
        print(f"   section      : {chunk.get('section')!r}")
        print(f"   titres du texte: {detail['titres'][:6]}"
              f"{'  (' + detail['sens'] + ')' if 'sens' in detail else ''}")
        print(f"   ouverture    : {ouverture[:220]!r}")
        print()


def step_chemin(n_exemples: int = 6) -> None:
    """Cohérence interne du fil d'Ariane. Lit la collection : un seul processus à la fois."""
    import quant_rag  # import tardif : les deux autres étapes n'ont pas besoin de Qdrant

    c = collections.Counter()
    exemples = []
    offset = None
    while True:
        points, offset = quant_rag.client().scroll(
            quant_rag.COLLECTION, limit=4096, offset=offset,
            with_payload=["chapter", "section", "title_path"])
        for point in points:
            payload = point.payload
            n_section = numero_etiquette(payload.get("section") or "")
            n_chapitre = numero_etiquette(payload.get("chapter") or "")
            if n_section is None or n_chapitre is None:
                continue
            c["comparables"] += 1
            if n_section.split(".")[0] == n_chapitre.split(".")[0]:
                c["accord"] += 1
            else:
                c["incoherent"] += 1
                if len(exemples) < n_exemples:
                    exemples.append(payload.get("title_path"))
        if offset is None:
            break

    base = c["comparables"]
    print(f"chapitre ET section numérotés                     {base:>6}")
    print(f"  numéro de chapitre cohérent avec la section     {c['accord']:>6}  "
          f"({100.0 * c['accord'] / base:5.1f} %)" if base else "")
    print(f"  CHEMIN INCOHÉRENT (ancêtre d'un autre chapitre) {c['incoherent']:>6}  "
          f"({100.0 * c['incoherent'] / base:5.1f} %)" if base else "")
    print()
    for chemin in exemples:
        print(f"   {chemin}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prevalence", action="store_true")
    p.add_argument("--echantillon", choices=("A", "A-", "B", "C"))
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--chemin", action="store_true")
    a = p.parse_args()
    if a.prevalence or a.echantillon:
        chunks = charger()
        if a.prevalence:
            step_prevalence(chunks)
        if a.echantillon:
            step_echantillon(chunks, a.echantillon, a.n)
    if a.chemin:
        step_chemin()
    if not a.prevalence and not a.echantillon and not a.chemin:
        p.print_help()


if __name__ == "__main__":
    main()
