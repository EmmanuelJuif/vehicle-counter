from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Tuple

# ---------------------------------------------------------------------------
# Classes COCO retenues
# ---------------------------------------------------------------------------
COCO_IDS = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
ID_TO_NAME = {v: k for k, v in COCO_IDS.items()}
DEFAULT_CLASSES: Tuple[int, ...] = (2, 3, 5, 7)

# Couleurs BGR (OpenCV) par classe
CLASS_COLORS = {
    "car": (0, 165, 255),         # orange
    "motorcycle": (0, 255, 255),  # jaune
    "truck": (255, 90, 60),       # bleu clair
    "bus": (60, 220, 60),         # vert
    "ghost": (190, 190, 190),     # gris : objet récupéré par le détecteur de mouvement
}


@dataclass
class DetectorConfig:
    """Paramètres du détecteur YOLOv8."""

    weights: str = "yolov8s.pt"
    imgsz: int = 640          # taille d'image imposée
    conf: float = 0.10        # seuil bas : on veut rappeler les petites/motos lointaines
    iou: float = 0.65         # IoU de NMS : élevé => deux motos collées ne fusionnent pas
    max_det: int = 300        # beaucoup d'objets possibles dans une scène CCTV dense
    device: str = "auto"      # auto | cpu | 0 | 1 ...
    half: bool = True         # FP16 (actif uniquement sur GPU)
    classes: Tuple[int, ...] = DEFAULT_CLASSES
    agnostic_nms: bool = False
    verbose: bool = False

    # --- rattrapage des petits objets (motos lointaines / sombres) ---
    tiled: str = "band"       # off | band | full
    tile_size: int = 640
    tile_overlap: float = 0.25
    refine_interval: int = 2      # 1 = chaque frame, 2 = une frame sur deux ...
    band_margin: int = 120        # marge ajoutée autour de la zone pour les tuiles
    zoom_conf: float = 0.08       # seuil encore plus bas sur les tuiles zoomées
    merge_iou: float = 0.65       # IoU pour fusionner tuiles + frame complète
    adaptive: bool = True         # désactive auto le tiling si les FPS tombent
    min_fps: float = 12.0


@dataclass
class TrackerConfig:

    name: str = "bytetrack"           # bytetrack | botsort
    track_high_thresh: float = 0.10   # détections "sûres" (1re association)
    track_low_thresh: float = 0.05    # 2e association : sauve les motos confondues
    new_track_thresh: float = 0.20    # seuil de création d'une nouvelle piste
    track_buffer: int = 60            # frames de grâce avant suppression d'une piste
    match_thresh: float = 0.85        # seuil IoU d'association
    fuse_score: bool = True
    frame_rate: int = 25
    min_hits: int = 2                 # frames min avant de valider une piste
    gmc_method: str = "sparseOptFlow"  # compensation de mouvement caméra
    # --- BoT-SORT uniquement ------------------------------------------------
    with_reid: bool = False           # ReID par apparence (coûteux mais très stable)
    proximity_thresh: float = 0.5     # IoU mini pour comparer deux pistes en ReID
    appearance_thresh: float = 0.8    # similarité d'apparence mini


@dataclass
class ZoneConfig:
    """Counting Zone : bande de 200 px (remplace la ligne de comptage)."""

    height: int = 200
    ratio: float = 0.62          # position du centre de la bande (0 = haut, 1 = bas)
    orientation: str = "horizontal"   # horizontal (axe y) | vertical (axe x)
    y_top: int | None = None     # écrase ratio si fourni
    y_bottom: int | None = None


@dataclass
class MotorcycleConfig:
    """Améliorations ciblées sur la classe moto."""

    boost: bool = True
    temporal_votes: int = 3      # vote de classe sur les N dernières frames
    vote_conf: float = 0.25      # confiance mini pour qu'une frame participe au vote
    min_area: int = 60           # px² : en dessous, on ignore (bruit)
    max_area_ratio: float = 0.55 # au-dessus de 55% de l'image => faux positif
    keep_if_low_conf: bool = True  # conserve la classe moto même si conf < seuil nominal
    low_conf_floor: float = 0.10   # plancher de confiance pour une moto retenue


@dataclass
class FastVehicleConfig:
    """Véhicules rapides franchissant la zone sans être vus dedans."""

    enable: bool = True
    segment_intersection: bool = True   # test d'intersection du segment de mouvement
    extrapolate_on_lost: bool = True    # prolonge la trajectoire à la perte de piste
    max_extrapolation_frames: int = 25
    min_speed: float = 1.0              # px/frame : en dessous, pas d'extrapolation
    max_speed: float = 160.0            # px/frame : au-dessus, aberration
    centerline_weight: float = 0.5      # où tester le franchissement dans la bande


@dataclass
class GhostConfig:
    """Filet de sécurité : soustraction de fond pour les objets jamais détectés.

    La soustraction de fond est bruyante (ombres, arbres, marquages au sol,
    reflets) : sans filtres on observait jusqu'à 25 blobs simultanés. Les
    filtres ci-dessous imposent qu'un blob se comporte comme un **véhicule qui
    traverse** avant de le compter.
    """

    enable: bool = True
    scale: float = 0.5            # facteur de réduction pour l'analyse
    min_area: int = 120           # aire mini (à l'échelle réduite)
    max_area: int = 60000
    min_frames: int = 3           # frames observées avant de pouvoir compter
    min_displacement: float = 45.0
    assoc_dist: float = 70.0
    history: int = 300
    var_threshold: int = 40
    motorcycle_area: int = 2500   # en dessous -> classé moto
    truck_area: int = 12000       # au-dessus -> classé camion/bus
    max_ghosts: int = 200

    # --- filtres anti-bruit (ajoutés après mesure : 25 blobs simultanés) ---
    min_speed: float = 3.0        # px/frame moyens : exclut ce qui stagne
    max_size_var: float = 3.5     # aire_max / aire_min : exclut le scintillement
    min_axis_ratio: float = 0.5   # part du déplacement selon l'axe de la zone
    min_dir_consistency: float = 0.7  # part des pas dans le sens dominant
    min_fill: float = 0.30        # aire du contour / aire de la boîte
    strict: bool = True           # applique les filtres ci-dessus
    trace: bool = False           # relève les caractéristiques de chaque candidat


@dataclass
class OutputConfig:
    display: bool = True
    save_video: str | None = None
    csv: str | None = None
    json: str | None = None
    show_tracks: bool = True
    track_len: int = 30
    window_scale: float = 1.0
    every: int = 1                # traitement d'une frame sur N (1 = temps réel complet)


@dataclass
class Config:
    source: str = "0"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    motorcycle: MotorcycleConfig = field(default_factory=MotorcycleConfig)
    fast: FastVehicleConfig = field(default_factory=FastVehicleConfig)
    ghost: GhostConfig = field(default_factory=GhostConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


def tracker_args(cfg: TrackerConfig):
    """Namespace compatible avec `ultralytics.trackers.byte_tracker.BYTETracker`."""
    from types import SimpleNamespace

    return SimpleNamespace(
        tracker_type=cfg.name,
        track_high_thresh=cfg.track_high_thresh,
        track_low_thresh=cfg.track_low_thresh,
        new_track_thresh=cfg.new_track_thresh,
        track_buffer=cfg.track_buffer,
        match_thresh=cfg.match_thresh,
        fuse_score=cfg.fuse_score,
        gmc_method=cfg.gmc_method,
        # tolérance aux versions récentes d'ultralytics
        min_hits=cfg.min_hits,
        frame_rate=cfg.frame_rate,
    )


def write_tracker_yaml(cfg: TrackerConfig, out_dir: str | None = None) -> str:
    """Rend la config du suiveur en YAML pour ``model.track(tracker=...)``.

    Ultralytics lit les réglages du tracker dans un fichier YAML. On le
    **génère** depuis ``TrackerConfig`` plutôt que de livrer deux fichiers
    statiques : il n'y a ainsi qu'une seule source de vérité, et toutes les
    surcharges CLI (`--track-buffer`, `--with-reid`, ...) s'appliquent.
    """
    import os
    import tempfile

    name = (cfg.name or "bytetrack").lower()
    if name not in ("bytetrack", "botsort"):
        raise ValueError(f"tracker inconnu : {name} (bytetrack | botsort)")

    lines = [
        f"tracker_type: {name}",
        f"track_high_thresh: {cfg.track_high_thresh}",
        f"track_low_thresh: {cfg.track_low_thresh}",
        f"new_track_thresh: {cfg.new_track_thresh}",
        f"track_buffer: {cfg.track_buffer}",
        f"match_thresh: {cfg.match_thresh}",
        f"fuse_score: {bool(cfg.fuse_score)}",
    ]
    if name == "botsort":
        lines += [
            # GMC : compense le mouvement de la caméra (ou le vent sur un mât).
            # Sans ça, une caméra qui bouge décale toutes les prédictions
            # Kalman et provoque des ruptures d'ID.
            f"gmc_method: {cfg.gmc_method}",
            f"proximity_thresh: {cfg.proximity_thresh}",
            f"appearance_thresh: {cfg.appearance_thresh}",
            f"with_reid: {bool(cfg.with_reid)}",
            "model: auto",
        ]
    else:
        lines += [f"gmc_method: {cfg.gmc_method}"]

    d = out_dir or os.path.join(tempfile.gettempdir(), "vehicle_counter_trackers")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
