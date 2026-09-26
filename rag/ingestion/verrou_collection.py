"""Un verrou applicatif pour l'écriture dans la collection — celui qui manque à la bascule.

Le défaut qu'il ferme
----------------------
``apply_delivery.next_point_id()`` attribue les identifiants par ``max(id) + 1``, calculé
par un scroll complet : **une lecture, puis une écriture**. Entre les deux, rien n'empêche
un second importeur de lire le même maximum. Les deux écriraient alors sur les mêmes
identifiants, et le second effacerait les points du premier — sans erreur, sans trace, avec
un ``status: COMPLETED`` de part et d'autre.

Aujourd'hui c'est impossible, mais pour une raison qui n'est pas du code : le stockage
Qdrant **embarqué** prend un verrou exclusif de fichier, et ``batch_driver.verrou_qdrant_libre()``
refuse même de démarrer quand il est tenu. **Sur un serveur, ce verrou disparaît.** C'est la
condition bloquante nommée au §10.4 de ``docs/FUSION-MIGRATION-QDRANT.md`` et au §5 de
``docs/BASCULE-QDRANT-SERVEUR.md`` : *« un verrou applicatif doit remplacer le verrou de
stockage avant toute bascule »*. Ce module est ce verrou.

Ce qu'il garantit, et ce qu'il ne garantit pas
------------------------------------------------
La prise est **atomique** : ``os.open`` avec ``O_CREAT | O_EXCL`` réussit chez un seul
appelant, quel que soit le nombre de processus. Le second reçoit un refus **explicite**, qui
nomme le tenant — pid, depuis quand, pour quelle livraison —, et n'écrit rien.

Un verrou périmé est repris : un processus tué par ``kill -9`` laisse son fichier derrière
lui, et un verrou qu'on ne peut pas reprendre finit par bloquer le dépôt plus sûrement qu'il
ne le protège. « Périmé » veut dire : **le processus n'existe plus**, ou le verrou est plus
vieux que ``PEREMPTION``. La reprise est bruyante ; elle n'est jamais silencieuse.

**Ce qu'il ne fait pas** : il ne protège pas contre deux machines écrivant dans le même
serveur Qdrant — le fichier est local. Et la reprise d'un verrou périmé porte une course
résiduelle : deux repreneurs simultanés pourraient tous deux conclure que le verrou est mort.
Sur un outil personnel où deux processus au plus s'écrivent, c'est proportionné ; ça ne le
serait pas sur un service partagé, et la limite est écrite ici plutôt que découverte plus tard.

    from verrou_collection import tenu
    with tenu("apply_delivery essai-01"):
        ...                                  # personne d'autre n'écrit pendant ce bloc
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: Hors git : un verrou est un état de machine, pas un fichier du dépôt. Voir ``.gitignore``.
VERROU = HERE / "verrou-collection.json"
#: Au-delà, un verrou dont le processus a disparu est tenu pour périmé. Une heure couvre
#: largement le plus long import mesuré (716 s de parse sur un manuel de 360 pages) sans
#: laisser un verrou mort bloquer une nuit de lot.
PEREMPTION = 3600.0


def _chemin(chemin: Path | None) -> Path:
    """Le chemin du verrou, résolu **à l'appel** et non à la définition.

    ``chemin: Path = VERROU`` en argument par défaut figeait la valeur au moment du ``def`` :
    remplacer ``VERROU`` n'avait alors aucun effet, et les tests hermétiques posaient leur
    verrou dans le dépôt réel en croyant écrire dans leur arbre jetable. Constaté le
    9 septembre 2026 par deux tests qui refusaient de voir le verrou qu'ils venaient d'écrire.
    """
    return Path(chemin) if chemin is not None else VERROU


class VerrouTenu(RuntimeError):
    """Un autre processus écrit dans la collection. Le détail du tenant est dans ``etat``."""

    def __init__(self, message: str, etat: dict):
        super().__init__(message)
        self.etat = etat


def _vivant(pid: int) -> bool:
    """Le processus existe-t-il ? ``signal 0`` ne tue rien, il interroge le noyau."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                 # il existe, il ne nous appartient pas
        return True
    return True


def etat(chemin: Path | None = None) -> dict | None:
    """Ce que le verrou dit, ou ``None`` s'il n'est pas tenu. Lecture seule, sans effet."""
    chemin = _chemin(chemin)
    if not chemin.exists():
        return None
    try:
        contenu = json.loads(chemin.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"corrompu": True, "chemin": str(chemin)}
    contenu["age_secondes"] = round(time.time() - contenu.get("horodatage_epoch", 0), 1)
    contenu["processus_vivant"] = _vivant(int(contenu.get("pid", -1)))
    return contenu


def perime(contenu: dict, peremption: float = PEREMPTION) -> bool:
    return (bool(contenu.get("corrompu"))
            or not contenu.get("processus_vivant")
            or contenu.get("age_secondes", 0) > peremption)


def acquerir(proprietaire: str, chemin: Path | None = None,
             peremption: float = PEREMPTION) -> dict:
    """Prendre le verrou, ou **lever**. Atomique : ``O_CREAT | O_EXCL``.

    Une seule reprise est tentée, et seulement sur un verrou périmé. Boucler serait pire :
    un appelant qui attend indéfiniment un verrou mort ne se distingue pas d'un appelant
    bloqué, et c'est exactement la panne qu'on cherche à rendre visible.
    """
    chemin = _chemin(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    contenu = {
        "proprietaire": proprietaire, "pid": os.getpid(),
        "horodatage": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horodatage_epoch": time.time(),
        "hote": os.uname().nodename,
    }
    for tentative in (1, 2):
        try:
            descripteur = os.open(chemin, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            occupant = etat(chemin)
            if occupant is None:                     # rendu entre-temps
                continue
            if tentative == 1 and perime(occupant, peremption):
                print(f"  ATTENTION verrou périmé repris : {occupant.get('proprietaire')!r}, "
                      f"pid {occupant.get('pid')} "
                      f"{'mort' if not occupant.get('processus_vivant') else 'trop vieux'}, "
                      f"{occupant.get('age_secondes')} s", flush=True)
                chemin.unlink(missing_ok=True)
                continue
            raise VerrouTenu(
                f"la collection est déjà en écriture : {occupant.get('proprietaire')!r} "
                f"(pid {occupant.get('pid')}, depuis {occupant.get('horodatage')}, "
                f"{occupant.get('age_secondes')} s). Un second import calculerait le même "
                f"max(id)+1 et écraserait les points du premier. Attends la fin, ou — si tu "
                f"sais que ce processus est mort — supprime {chemin}.", occupant)
        else:
            with os.fdopen(descripteur, "w", encoding="utf-8") as fichier:
                json.dump(contenu, fichier, ensure_ascii=False, indent=1)
            return contenu
    raise VerrouTenu(f"impossible de prendre {chemin} après deux tentatives", etat(chemin) or {})


def rendre(chemin: Path | None = None) -> None:
    """Rendre le verrou. Idempotent : rendre un verrou déjà rendu n'est pas une erreur."""
    _chemin(chemin).unlink(missing_ok=True)


@contextmanager
def tenu(proprietaire: str, chemin: Path | None = None, peremption: float = PEREMPTION):
    """Le verrou pour la durée d'un bloc, rendu **même en cas d'exception**.

    Sans le ``finally``, un import qui échoue laisserait le verrou derrière lui et le
    suivant devrait attendre la péremption : une garde qui transforme une panne en blocage
    d'une heure est une garde qui coûte plus qu'elle ne protège.
    """
    chemin = _chemin(chemin)
    acquerir(proprietaire, chemin, peremption)
    try:
        yield chemin
    finally:
        rendre(chemin)
