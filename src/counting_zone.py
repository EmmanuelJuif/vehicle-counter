from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Optional

import numpy as np

from .config import FastVehicleConfig, MotorcycleConfig, TrackerConfig


# ---------------------------------------------------------------------------
# Géométrie de la zone
# ---------------------------------------------------------------------------
class CountingZone:
    """Bande rectangulaire traversant l'image.

    axis="y" -> bande horizontale, franchissement selon y (haut/bas)
    axis="x" -> bande verticale,   franchissement selon x (gauche/droite)
    """

    def __init__(self, top: float, bottom: float, axis: str = "y"):
        self.top = float(min(top, bottom))
        self.bottom = float(max(top, bottom))
        self.axis = axis

    # -- construction -------------------------------------------------------
    @classmethod
    def from_config(cls, cfg, frame_w: int, frame_h: int) -> "CountingZone":
        if cfg.orientation == "vertical":
            size = frame_w
            axis = "x"
        else:
            size = frame_h
            axis = "y"
        if cfg.y_top is not None and cfg.y_bottom is not None:
            top, bottom = float(cfg.y_top), float(cfg.y_bottom)
        else:
            height = float(min(cfg.height, size))
            center = cfg.ratio * size
            top, bottom = center - height / 2.0, center + height / 2.0
        top, bottom = max(0.0, top), min(float(size), bottom)
        return cls(top, bottom, axis)

    # -- requêtes -----------------------------------------------------------
    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> float:
        return (self.top + self.bottom) / 2.0

    def threshold_coord(self, weight: float = 0.5) -> float:
        """Coordonnée de la « ligne virtuelle » à l'intérieur de la bande."""
        return self.top + weight * self.height

    def _proj(self, point) -> float:
        """Projection du point sur l'axe de franchissement."""
        return float(point[1] if self.axis == "y" else point[0])

    def contains(self, point) -> bool:
        c = self._proj(point)
        return self.top <= c <= self.bottom

    def interval(self, box: np.ndarray) -> tuple[float, float]:
        """Intervalle occupé par la boîte sur l'axe de franchissement."""
        b = np.asarray(box, dtype=float)
        if self.axis == "y":
            return float(b[1]), float(b[3])
        return float(b[0]), float(b[2])

    def overlaps_band(self, box: np.ndarray) -> bool:
        lo, hi = self.interval(box)
        return max(lo, self.top) <= min(hi, self.bottom)

    def segment_touches_band(self, p0, p1) -> bool:
        """Le segment de mouvement traverse-t-il (même partiellement) la bande ?"""
        a, b = self._proj(p0), self._proj(p1)
        return min(a, b) <= self.bottom and max(a, b) >= self.top

    def cross_direction(self, p0, p1, weight: float = 0.5) -> int:
        """+1 si franchissement dans le sens croissant (bas / droite),
        -1 dans le sens décroissant (haut / gauche), 0 sinon."""
        a, b = self._proj(p0), self._proj(p1)
        t = self.threshold_coord(weight)
        if a <= t <= b and b != a:
            return 1
        if b <= t <= a and b != a:
            return -1
        return 0

    def rect(self, frame_w: int, frame_h: int):
        if self.axis == "y":
            return 0, int(self.top), frame_w, int(self.bottom)
        return int(self.top), 0, int(self.bottom), frame_h


# ---------------------------------------------------------------------------
# État d'une piste
# ---------------------------------------------------------------------------
@dataclass
class TrackState:
    tid: int
    points: Deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=64))
    frames_seen: int = 0
    last_frame: int = -1
    first_frame: int = -1
    votes: Counter = field(default_factory=Counter)
    conf_votes: dict = field(default_factory=dict)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    inside: bool = False
    entry_side: int = 0
    counted: bool = False
    boxes: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=8))
    last_conf: float = 0.0
    last_area: float = 0.0

    def update(self, point, box, cls: str, conf: float, frame_id: int) -> None:
        if self.frames_seen == 0:
            self.first_frame = frame_id
        if len(self.points):
            prev = np.array(self.points[-1], dtype=float)
            cur = np.array(point, dtype=float)
            step = cur - prev
            if self.frames_seen == 1:
                self.velocity = step
            else:
                self.velocity = 0.55 * self.velocity + 0.45 * step
        self.points.append((float(point[0]), float(point[1])))
        self.boxes.append(np.asarray(box, dtype=float).copy())
        self.votes[cls] += 1
        self.conf_votes[cls] = max(self.conf_votes.get(cls, 0.0), float(conf))
        self.last_conf = float(conf)
        self.last_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
        self.frames_seen += 1
        self.last_frame = frame_id

    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def axis_speed(self, zone: CountingZone) -> float:
        return float(self.velocity[1] if zone.axis == "y" else self.velocity[0])

    def final_class(self, mcfg: MotorcycleConfig) -> str:
        """Vote temporel : stabilise la classe (utile pour les motos hésitantes)."""
        if not self.votes:
            return "car"
        strong = Counter()
        for cls, n in self.votes.items():
            if self.conf_votes.get(cls, 0.0) >= mcfg.vote_conf:
                strong[cls] = n
        pool = strong if strong else self.votes
        best, _ = max(pool.items(), key=lambda kv: (kv[1], self.conf_votes.get(kv[0], 0.0)))
        if mcfg.boost and mcfg.keep_if_low_conf and "motorcycle" in self.votes:
            # Une moto vue une seule fois reste une moto : la classe est « collante ».
            if best != "motorcycle" and self.conf_votes.get("motorcycle", 0.0) >= mcfg.low_conf_floor:
                if self.votes["motorcycle"] >= max(1, self.frames_seen // 4):
                    return "motorcycle"
        return best


# ---------------------------------------------------------------------------
# Compteur
# ---------------------------------------------------------------------------
@dataclass
class CountEvent:
    track_id: int
    cls: str
    direction: str
    frame: int
    method: str      # "yolo" | "extrapolated" | "ghost"
    conf: float = 0.0
    point: tuple[float, float] = (0.0, 0.0)


class ZoneCounter:
    """Compte les franchissements de la Counting Zone, sans doublons."""

    def __init__(
        self,
        zone: CountingZone,
        tracker_cfg: TrackerConfig,
        moto_cfg: MotorcycleConfig,
        fast_cfg: FastVehicleConfig,
        frame_area: int,
    ):
        self.zone = zone
        self.tcfg = tracker_cfg
        self.mcfg = moto_cfg
        self.fcfg = fast_cfg
        self.frame_area = frame_area

        # --- set() : garantit l'unicité des Track IDs comptés ---------------
        self.counted_ids: set[int] = set()
        self.ghost_ids: set[int] = set()

        # --- stabilité des IDs (compare les suiveurs entre eux) -------------
        # all_ids     : tous les IDs distincts créés par le suiveur sur la vidéo.
        #               Plus c'est proche du nombre de véhicules réels, mieux c'est :
        #               un suiveur qui "switch" fabrique plusieurs IDs pour un
        #               seul et même véhicule.
        # id_switches : nombre de fois où un ID tout neuf a dû être rattaché à
        #               une piste perdue (occlusion, croisement) — c'est le
        #               compteur de ruptures d'identité.
        self.all_ids: set[int] = set()
        self.id_switches: int = 0

        self.states: dict[int, TrackState] = {}
        self.stale: dict[int, TrackState] = {}
        self.counts: dict[str, dict[str, int]] = {}
        self.total: dict[str, int] = {"up": 0, "down": 0} if zone.axis == "y" else {"left": 0, "right": 0}
        self.events: list[CountEvent] = []
        self._recent: Deque[tuple[int, float, float, str]] = deque(maxlen=64)
        self.grace = max(8, tracker_cfg.track_buffer // 3)
        self.reid_dist = 150.0

    # -- API ----------------------------------------------------------------
    def reset(self) -> None:
        self.counted_ids.clear()
        self.ghost_ids.clear()
        self.all_ids.clear()
        self.id_switches = 0
        self.states.clear()
        self.stale.clear()
        self.counts.clear()
        self.total = {k: 0 for k in self.total}
        self.events.clear()
        self._recent.clear()

    def _dir_name(self, sign: int) -> str:
        if self.zone.axis == "y":
            return "down" if sign > 0 else "up"
        return "right" if sign > 0 else "left"

    def _register(self, tid: int, cls: str, sign: int, frame_id: int,
                  method: str, conf: float, point) -> CountEvent:
        name = self._dir_name(sign)
        if tid not in self.ghost_ids:
            self.counted_ids.add(tid)
        per = self.counts.setdefault(cls, {k: 0 for k in self.total})
        per[name] = per.get(name, 0) + 1
        self.total[name] = self.total.get(name, 0) + 1
        ev = CountEvent(tid, cls, name, frame_id, method, conf, point)
        self.events.append(ev)
        self._recent.append((frame_id, float(point[0]), float(point[1]), cls))
        return ev

    def _is_duplicate(self, cls: str, point, frame_id: int) -> bool:
       
        for f, x, y, c in self._recent:
            if frame_id - f > self.grace:
                continue
            if abs(x - point[0]) < self.reid_dist and abs(y - point[1]) < self.reid_dist:
                return True
        return False

    # -- entrée principale ---------------------------------------------------
    def update(self, tracks: Iterable[dict], frame_id: int) -> list[CountEvent]:
        """tracks : {id, box, cls, conf}. Retourne les nouveaux comptages."""
        events: list[CountEvent] = []
        seen: set[int] = set()

        for t in tracks:
            tid = int(t["id"])
            seen.add(tid)
            # les pistes issues des tuiles ont un ID synthétique incrémental :
            # elles ne disent rien de la stabilité du suiveur, on les exclut.
            if not t.get("refined"):
                self.all_ids.add(tid)
            st = self.states.get(tid)
            if st is None:
                st = self._adopt(tid, t, frame_id, refined=bool(t.get("refined")))
            self.states[tid] = st
            st.update(t["center"], t["box"], t["cls"], t["conf"], frame_id)
            ev = self._evaluate(st, frame_id, lost=False)
            if ev:
                events.append(ev)

        # pistes disparues -> tentative d'extrapolation puis purge
        for tid in list(self.states.keys()):
            if tid in seen:
                continue
            st = self.states.pop(tid)
            if not st.counted:
                ev = self._evaluate(st, frame_id, lost=True)
                if ev:
                    events.append(ev)
            self.stale[tid] = st
        for tid in [k for k, v in self.stale.items() if frame_id - v.last_frame > self.grace]:
            self.stale.pop(tid, None)

        return events

    def _adopt(self, tid: int, t: dict, frame_id: int, refined: bool = False) -> TrackState:
       
        px, py = t["center"]
        best_old_id, best_d, best = None, float("inf"), None
        for old_id, old in list(self.stale.items()):
            if frame_id - old.last_frame > self.grace:
                continue
            lx, ly = old.points[-1]
            d = max(abs(lx - px), abs(ly - py))
            if d < best_d and d < self.reid_dist:
                best_d = d
                best = old
                best_old_id = old_id
        if best is None:
            return TrackState(tid=tid)
        # un nouvel ID pour un véhicule déjà vu = rupture d'identité
        if not refined:
            self.id_switches += 1
        # transfère l'identité et l'état "déjà compté"
        was_counted = best.counted or best_old_id in self.counted_ids
        best.tid = tid
        if was_counted:
            best.counted = True
            self.counted_ids.discard(best_old_id)
            self.counted_ids.add(tid)
        self.stale.pop(best_old_id, None)
        return best

    # -- décision de comptage -------------------------------------------------
    def _evaluate(self, st: TrackState, frame_id: int, lost: bool) -> Optional[CountEvent]:
        if st.counted or st.tid in self.counted_ids:
            return None
        if len(st.points) == 0:
            return None
        if st.frames_seen < self.tcfg.min_hits and not lost:
            return None

        cls = st.final_class(self.mcfg)
        box = st.boxes[-1]
        if self.mcfg.boost:
            area_ratio = st.last_area / max(1.0, self.frame_area)
            if st.last_area < self.mcfg.min_area or area_ratio > self.mcfg.max_area_ratio:
                return None

        cur = st.points[-1]
        prev = st.points[-2] if len(st.points) >= 2 else cur
        w = self.fcfg.centerline_weight
        sign = 0
        method = "yolo"

        # 1) franchissement observé de la ligne médiane (cas général)
        if self.fcfg.segment_intersection and prev != cur:
            sign = self.zone.cross_direction(prev, cur, w)

        # 2) véhicule rapide : extrapolation de la trajectoire à la perte de piste
        if sign == 0 and lost and self.fcfg.extrapolate_on_lost:
            sign = self._extrapolate(st, w)
            if sign:
                method = "extrapolated"

        # 3) le véhicule est entré dans la bande mais on ne l'a pas vu ressortir
        if sign == 0 and self.zone.overlaps_band(box):
            st.inside = True
            st.entry_side = -1 if self.zone._proj(cur) < self.zone.center else 1
            if lost:
                # perdu dans la bande : on le compte s'il avait une vitesse cohérente
                v_ax = st.axis_speed(self.zone)
                if abs(v_ax) >= self.fcfg.min_speed:
                    sign = 1 if v_ax > 0 else -1
                    method = "extrapolated"
                elif st.entry_side and st.frames_seen >= self.tcfg.min_hits:
                    sign = st.entry_side
                    method = "extrapolated"

        if sign == 0:
            return None
        if self._is_duplicate(cls, cur, frame_id):
            return None

        st.counted = True
        return self._register(st.tid, cls, sign, frame_id, method, st.last_conf, cur)

    def _extrapolate(self, st: TrackState, weight: float) -> int:
        """Prolonge la dernière position connue selon la vitesse estimée."""
        v = st.velocity
        sp = float(np.linalg.norm(v))
        if sp < self.fcfg.min_speed or sp > self.fcfg.max_speed:
            return 0
        p = np.array(st.points[-1], dtype=float)
        for _ in range(self.fcfg.max_extrapolation_frames):
            p = p + v
            prev = p - v
            d = self.zone.cross_direction(prev, p, weight)
            if d:
                return d
        return 0

    # -- comptage « fantôme » (mouvement sans détection) -----------------------
    def register_ghost(self, gid: int, cls: str, sign: int, frame_id: int, point) -> CountEvent:
        self.ghost_ids.add(gid)
        return self._register(gid, cls, sign, frame_id, "ghost", 0.0, point)

    def summary(self) -> dict:
        return {
            "total": dict(self.total),
            "by_class": {k: dict(v) for k, v in self.counts.items()},
            "grand_total": int(sum(self.total.values())),
            "ghosts": len(self.ghost_ids),
            "unique_ids": len(self.counted_ids | self.ghost_ids),
            "ids_created": len(self.all_ids),
            "id_switches": self.id_switches,
        }
