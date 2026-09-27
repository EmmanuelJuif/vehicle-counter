# Comptage de véhicules temps réel — YOLOv8s + ByteTrack + Counting Zone

Système de comptage multi-classes (voiture, moto, camion, bus) sur flux CCTV,
pensé pour deux cas qui cassent les compteurs classiques :

1. **les motos** — petites, sombres, parfois très loin de la caméra ;
2. **les véhicules rapides** — qui traversent la zone sans qu'aucune frame ne
   les montre dedans.

```
vehicle_counter/
├── main.py                     # CLI
├── requirements.txt
├── src/
│   ├── config.py               # hyperparamètres (détecteur / tracker / zone)
│   ├── capture.py              # thread d'acquisition (fichier, webcam, RTSP)
│   ├── detector.py             # YOLOv8s + inférence en tuiles (petits objets)
│   ├── tracker.py              # ByteTrack (+ suiveur IoU de secours)
│   ├── counting_zone.py        # bande de 200 px + machine à états de comptage
│   ├── motion_fallback.py      # filet de sécurité : soustraction de fond
│   ├── geometry.py             # IoU / NMS
│   ├── visualizer.py           # rendu OpenCV
│   └── pipeline.py             # orchestration
└── tools/
    ├── test_zone_logic.py      # 10 tests de la logique de comptage (sans modèle)
    ├── make_test_video.py      # vidéo de trafic synthétique + vérité terrain
    ├── evaluate.py             # comparaison prédiction / vérité terrain
    ├── finetune_motorcycle.py  # fine-tuning YOLOv8s spécialisé motos (VisDrone)
    ├── eval_model.py           # mAP50 / AP50 par classe sur le split VisDrone propre
    ├── make_noise_video.py     # vidéo SANS véhicule : pluie, ombres, scintillement
    ├── probe_mog2.py           # combien de contours MOG2 voit-on ? (diagnostic)
    ├── calibrate_ghost.py      # grid-search des seuils du filet mouvement
    ├── diagnose_ghost.py       # pourquoi tel blob a-t-il été rejeté ?
    ├── rtsp_server.py          # serveur RTSP local (MediaMTX) + publisher ffmpeg
    ├── test_rtsp.py            # banc de test du thread d'acquisition live
    └── test_capture_eof.py     # non-régression : aucune frame perdue en fin de fichier
```

## Installation

```bash
pip install -r requirements.txt
# torch : choisir la roue adaptée à votre CUDA — https://pytorch.org/get-started/locally/
```

## Utilisation

```bash
# flux CCTV RTSP, avec affichage et enregistrement
python main.py --source "rtsp://user:pass@10.0.0.12/stream" --save-video out/annotated.mp4

# fichier vidéo, mode serveur (sans fenêtre) + export CSV/JSON
python main.py --source traffic.mp4 --no-display --csv out/counts.csv --json out/summary.json

# webcam
python main.py --source 0

# mesurer le coût de l'inférence sur votre machine
python main.py --source traffic.mp4 --benchmark
```

Touches : `q` ou `Echap` pour quitter.

## Choix techniques

### 1. Détection — YOLOv8s, imgsz 640, conf 0.10

| Paramètre   | Valeur | Raison                                                                 |
|-------------|--------|------------------------------------------------------------------------|
| `imgsz`     | 640    | compromis vitesse / précision imposé                                    |
| `conf`      | 0.10   | seuil bas : on préfère le rappel, le bruit est filtré ensuite (min_hits, vote de classe) |
| `iou` (NMS) | 0.65   | volontairement **élevé** : deux motos collées ne doivent pas fusionner en une seule boîte |
| `max_det`   | 300    | scènes CCTV denses                                                      |

### 2. Rappel sur les motos — 4 leviers cumulés

* **Inférence en tuiles (`--tiled band`)** : la passe pleine image letterboxe
  1920×1080 → 640×384, une moto de 20 px tombe à ~7 px. En tuile 640×640 prise
  à résolution native, elle garde ses 20 px. C'est le levier le plus efficace
  sur les petits objets. Coût : ~4-5 inférences de plus, exécutées une frame
  sur deux (`--refine-interval 2`), et **désactivées automatiquement** si les
  FPS descendent sous `--min-fps` (12).
* **Seuil abaissé à 0.08 sur les tuiles** (`zoom_conf`).
* **Vote temporel de classe** : la classe d'une piste est décidée à la majorité
  sur toutes ses frames. Une moto détectée « car » une frame sur trois reste
  moto — la classe est *collante* (`MotorcycleConfig.keep_if_low_conf`).
* **Filet de sécurité mouvement** (`src/motion_fallback.py`) : un MOG2 travaille
  sur la bande sous-échantillonnée ; tout blob mobile qui ne correspond à
  aucune piste YOLO est suivi séparément et compté s'il franchit la zone.
  C'est ce qui rattrape une moto avec pilote en vêtements sombres jamais
  détectée. Ces comptages sont marqués `method = "ghost"` dans le CSV pour
  audit. Désactivable avec `--no-ghost`.

### 3. Véhicules rapides

Une ligne de comptage ne fonctionne que si une frame tombe dessus. Trois
mécanismes la remplacent :

* **Counting Zone de 200 px** au lieu d'une ligne : la cible est 200× plus
  large, un saut de 150 px/frame reste à l'intérieur.
* **Test d'intersection de segment** : on ne teste pas « le centre est-il dans
  la zone ? » mais « le segment *position précédente → position courante*
  traverse-t-il la médiane de la bande ? ». Un véhicule vu à y=150 puis y=650
  est compté : le segment traverse la médiane à 400.
* **Extrapolation à la perte de piste** : si une piste disparaît avant
  d'atteindre la médiane, on prolonge sa dernière position selon sa vitesse
  lissée (EMA). Si la trajectoire extrapolée franchit la médiane, le véhicule
  est compté (`method = "extrapolated"`).
* **Anti-doublon** : les IDs comptés sont stockés dans un `set()`, plus une
  garde spatio-temporelle et une ré-identification par proximité, pour que les
  ruptures d'ID ByteTrack (occlusion) ne produisent pas de double comptage.

### 4. Suivi — ByteTrack via `model.track(persist=True)`

Le chemin officiel d'Ultralytics (`model.track(persist=True, tracker="bytetrack.yaml")`)
est utilisé pour tout le suivi principal. Les seuils (haut / bas / nouveau) sont
placés très bas afin que la 2e passe de ByteTrack sauve les motos détectées
faiblement :

| Paramètre              | Valeur | Pourquoi                                              |
|------------------------|--------|-------------------------------------------------------|
| `track_high_thresh`    | 0.10   | seuil bas pour rappeler les motos faibles             |
| `track_low_thresh`     | 0.05   | 2e passe : récupération des détections manquées en 1re passe |
| `new_track_thresh`     | 0.20   | seuil de création d'une nouvelle piste                 |
| `track_buffer`         | 60     | frames de tolérance avant abandon d'une piste          |
| `match_thresh`         | 0.85   | IoU minimal pour l'association                          |
| `fuse_score`           | True   | pondère la distance par le score                        |

Ces valeurs sont volontairement **bien plus basses** que les défauts
Ultralytics (0.25 / 0.10 / 0.25) : à conf 0.10 on veut conserver les pistes
faibles, sinon les motos lointaines sont créées puis abandonnées en boucle.
Elles sont appliquées aux deux backends (`bytetrack` et `botsort`).

Un shim `src/tracker.py` est conservé pour les cas avancés (injection des
détections en tuiles dans un tracker tiers) mais il est contourné par le
pipeline principal à cause d'un comportement de `STrack.activate()` dans
Ultralytics 8.4.x (les pistes créées après la frame 1 ne sont marquées
`is_activated` qu'après ré-activation) qui complique l'usage direct.
`model.track()` est la voie propre.

### 4 bis. Choisir le suiveur : ByteTrack ou BoT-SORT

```bash
python main.py --source rtsp://... --tracker botsort            # BoT-SORT + GMC
python main.py --source rtsp://... --tracker botsort --with-reid  # + ReID apparence
python main.py --source data/eval.mp4 --tracker bytetrack       # défaut
```

Le fichier YAML du suiveur est **généré** au lancement depuis `TrackerConfig`
(`write_tracker_yaml()`), il n'y a donc qu'une seule source de vérité et toutes
les surcharges CLI s'appliquent.

**BoT-SORT** ajoute deux choses par rapport à ByteTrack :

* **GMC** (`--gmc sparseOptFlow`) — compensation du mouvement global. Sans ça,
  une caméra qui bouge (vent sur un mât, PTZ) décale toutes les prédictions
  Kalman et provoque des ruptures d'ID ;
* **ReID** (`--with-reid`) — ré-association par similarité d'apparence, utile
  quand deux véhicules se croisent ou après une occlusion longue.

### Mesure : BoT-SORT ne fait pas de miracle sur notre benchmark

`data/eval.mp4`, `--tiled off` (iso-conditions), **15 véhicules réels** :

| Suiveur                    | compté | IDs créés | ruptures d'ID | FPS  |
|----------------------------|--------|-----------|---------------|------|
| ByteTrack (défaut)         | 16     | **29**    | 27            | 2,49 |
| BoT-SORT + GMC             | 16     | 32        | 31            | 2,56 |
| BoT-SORT sans GMC          | 17     | **29**    | 28            | 2,53 |
| BoT-SORT + ReID            | 16     | 35        | 29            | 2,88 |

Lecture honnête : sur cette vidéo, **BoT-SORT est neutre** (29 IDs, comme
ByteTrack) et même légèrement moins bon avec GMC ou ReID activés. La
fragmentation vient des **détections qui clignotent** (crops collés sur un fond
synthétique → confiance YOLO instable d'une frame à l'autre), pas de
l'algorithme d'association. Les deux atouts de BoT-SORT — caméra mobile et
occlusions réelles — sont absents de ce test synthétique.

**À retenir : la fragmentation ne provoque pas de surcomptage.** Dans tous les
cas `IDs uniques == total compté` : le `set()` du compteur plus sa garde
spatio-temporelle (`_adopt`) absorbent les ruptures d'identité. Changer de
suiveur ne change donc pas le résultat métier ici — c'est sur une **vraie**
caméra (mobile, avec occlusions) que BoT-SORT devrait se justifier.

> Les colonnes « ruptures d'identité » du tableau ci-dessus viennent d'un proxy
> maison, qui **surestime** grossièrement la réalité — voir ci-dessous.

## Métriques MOT standard — MOTA / IDF1

```bash
python tools/eval_mot.py --tracker bytetrack
python tools/eval_mot.py --tracker botsort --json out/mot_botsort.json
```

 Prérequis : une vérité terrain avec **trajectoires** (boîtes + IDs par frame),
 que `make_eval_video.py` émet désormais dans le champ `tracks`. Métriques
 calculées par `motmetrics` — les mêmes que MOTChallenge, donc comparables à
 n'importe quel autre système.

`eval.mp4` (199 frames, 15 identités vérité terrain, imgsz 640, conf 0.10) :

| Métrique | ByteTrack | BoT-SORT | lecture |
|---|---|---|---|
| **MOTA** | 24,23 % | 23,90 % | pénalisé surtout par les 330 faux positifs |
| IoU moyen (MOTP) | 94,83 % | 93,68 % | les boîtes appariées sont bien placées |
| **IDF1** | 64,89 % | **67,04 %** | qualité d'identité : correcte, pas bonne |
| IDP / IDR | 57,1 / 75,1 | 58,8 / 78,1 | |
| précision / rappel | 59,3 / 78,1 | 59,1 / 78,5 | |
| **ruptures d'identité** | **2** | **2** | quasi parfait |
| fragmentations | 21 | 21 | ~1,4 fragment par véhicule |
| objets manqués | 135 | 132 | |
| **faux positifs** | **329** | **334** | ← l'erreur dominante |
| pistes >80 % suivies | 13/15 | 13/15 | |

### Ce que ça apprend

1. **Mon proxy maison était faux.** Il annonçait 27 ruptures d'identité ; la
   vraie mesure (IDSW, définition MOTChallenge) est **2**. Le proxy comptait
   comme « rupture » toute ré-identification spatiale, y compris des fusions
   légitimes. C'est exactement pour ça qu'il fallait une métrique standard.
2. **Le problème n'est pas l'identité, c'est la détection.** 329 fausses
   détections contre 2 ruptures d'ID : MOTA est plombé par le bruit de
   détection, pas par le suivi. Le suiveur fait correctement son travail.
3. **BoT-SORT ≈ ByteTrack** (+2 points d'IDF1, −0,3 de MOTA). Confirmation avec
   une vraie métrique de ce que le proxy laissait déjà supposer.
4. MOTA ~24 % reste **faible** — mais sur cette vidéo synthétique (crops collés
   sur fond artificiel → détections parasites). À refaire sur du réel avant
   d'en tirer une conclusion produit.

## Réglage de la zone

| Option                | Effet                                                     |
|-----------------------|-----------------------------------------------------------|
| `--zone-height 200`   | hauteur de la bande (défaut 200 px)                        |
| `--zone-ratio 0.62`   | position du centre de la bande (0 = haut, 1 = bas)         |
| `--zone-top/--zone-bottom` | coordonnées absolues, écrasent le ratio             |
| `--zone-orientation vertical` | bande verticale (axe x) pour une route horizontale |

Règle pratique : placer la bande perpendiculairement au flux, là où les
véhicules sont le mieux séparés, et lui donner une hauteur ≳ le déplacement
inter-frame du véhicule le plus rapide.

## Sorties

* Fenêtre OpenCV avec boîtes, IDs, trajectoires, bande et panneau de stats ;
* `--save-video` : vidéo annotée ;
* `--csv` : une ligne par franchissement
  (`track_id, class, direction, frame, method, conf`) ;
* `--json` : résumé (totaux par classe et par sens, FPS, méthodes).

## Validation

```bash
python tools/test_zone_logic.py                  # 10 tests de logique (sans YOLO)
python tools/test_tracking_integration.py        # intégration fake-detector + compteur
python tools/test_capture_eof.py                 # aucune frame perdue en fin de fichier
python tools/make_test_video.py --out data/traffic.mp4 --frames 150    # vidéo démo
python tools/make_eval_video.py --out data/eval.mp4 --frames 200      # vidéo d'évaluation (vérité terrain)
python main.py --source data/eval.mp4 --no-display --zone-ratio 0.6667 \
               --json out/eval_summary.json --csv out/eval_counts.csv
python tools/evaluate.py --pred out/eval_summary.json --gt data/eval_gt.json
```

### Mesures obtenues (vidéo `eval.mp4`, 200 frames, 15 véhicules réels)

```
GT  : 15 franchissements   PRÉD: 14   (2.9 FPS sur CPU)
vrais positifs   : 11
faux négatifs    :  4   (véhicules réels non comptés)
faux positifs    :  3   (comptages surnuméraires)
précision : 78.6%    rappel : 73.3%    F1 : 75.9%
```

La justesse de classe sur les paires appariées est plus faible (36 %) parce que
les crops de bus/camion posés sur fond synthétique sont reclassés en "car"
par YOLO : c'est un artefact du test (pas un défaut du compteur). En
vraie scène CCTV la classe est habituellement stable (un bus reste un bus).

Après calibrage du filet mouvement, ce score est **inchangé** (14 / 15) : sur
`eval.mp4` le filet ne déclenche jamais (`récup. mouvement : 0`), car chaque
blob MOG2 est associé à une piste YOLO existante. Le calibrage ne se voit donc
pas ici — il se voit sur les vidéos de bruit pur, où il passe de 5,5 à
0 faux positifs par minute sans perdre de rappel.

## Fine-tuning « motos » (VisDrone)

Les motos lointaines sont le point faible de YOLOv8s pré-entraîné sur COCO.
Le fine-tuning part de **VisDrone** (vues aériennes, objets minuscules) :

```bash
python tools/download_visdrone.py                    # récupère le dataset
python tools/finetune_motorcycle.py --data data/visdrone --weights yolov8s.pt \
       --epochs 60 --imgsz 640 --batch 16 --device 0 \
       --project runs/motorcycle --name moto \
       --eval-video data/eval.mp4 --eval-gt data/eval_gt.json
```

Le script enchaîne : split train/val **disjoint**, évaluation du modèle de
base, entraînement (mosaic 1.0 + copy_paste 0.3 pour sur-représenter les
motos, `freeze 10` contre l'oubli catastrophique), évaluation du modèle
fine-tuné, puis comparaison du *comptage* avant/après sur `data/eval.mp4`.
Rapport : `runs/.../<name>/finetune_report.json`.

> **Piège déjà corrigé** : une version antérieure écrivait `train: images` et
> `val: images` dans le YAML — la validation tournait sur les images
> d'entraînement et tous les scores étaient gonflés. Les deux dossiers doivent
> être disjoints (`train/images` et `val/images`).

### Résultat mesuré (run `ft416`, split VisDrone propre, 50 images, imgsz 416)

| Modèle              | mAP50 | mAP50-95 | car   | **motorcycle** | bus   | truck |
|---------------------|-------|----------|-------|----------------|-------|-------|
| `yolov8s.pt` (COCO) | 0,213 | 0,124    | 0,353 | **0,035**      | 0,274 | 0,190 |
| `ft416/weights/best.pt` | **0,322** | **0,182** | 0,430 | **0,104** | 0,342 | 0,412 |

Le fine-tuning **triple** l'AP50 moto (0,035 → 0,104, +197 %) et gagne +51 %
de mAP50 global. Mais il faut garder la tête froide : 0,104 reste **faible en
absolu**, le rappel moto n'est que de 6,7 %. Ce run est une démonstration
(15 epochs, imgsz 416, 200 images, CPU, early-stop à l'epoch 10). Un vrai gain
demande les 60 epochs / imgsz 640 / GPU de la commande ci-dessus, et surtout
un jeu de données CCTV annoté plutôt que des vues aériennes.

> **Second piège corrigé** : `r.box.ap50` ne contient que les classes
> **présentes** dans le split (4 entrées ici) et les vrais IDs sont dans
> `r.box.ap_class_index` (`[2, 3, 5, 7]`). L'indexer par position faisait
> passer bus/truck pour car/motorcycle et affichait des gains imaginaires.

### Transfert vers la vidéo CCTV : prudence

Le même modèle, passé sur `data/eval.mp4` (vue de caméra, 15 véhicules réels) :

| Modèle        | compté | TP | FP | FN | précision | rappel | F1   |
|---------------|--------|----|----|----|-----------|--------|------|
| `yolov8s.pt`  | 14     | 11 | 3  | 4  | 78,6 %    | 73,3 % | 75,9 % |
| `ft416/best`  | 10     | 9  | 1  | 6  | **90,0 %** | 60,0 % | 72,0 % |

Le fine-tuning sur vues **aériennes** rend le modèle plus précis mais plus
prudent sur une vue **CCTV** : il manque 2 véhicules de plus. C'est un effet de
décalage de domaine, pas une régression du compteur. Conclusion honnête : ce
run prouve que la chaîne de fine-tuning fonctionne, mais les poids `ft416` ne
sont **pas** à utiliser en production sur de la CCTV — il faut fine-tuner sur
des images de caméra annotées (`--from-video` fait du pseudo-étiquetage pour
amorcer ça).

## Calibrage du filet mouvement

Le filet MOG2 (`src/motion_fallback.py`) rattrape ce que YOLO manque, mais il
déclenche aussi sur les ombres, la pluie et le scintillement. Ses seuils ont
été **mesurés**, pas devinés :

```bash
python tools/make_noise_video.py --out data/noise.mp4 --profile severe --seed 7
python tools/probe_mog2.py --source data/noise.mp4     # y a-t-il du bruit à filtrer ?
python tools/calibrate_ghost.py \
       --pos data/eval.mp4 data/traffic.mp4 \
       --gt  data/eval_gt.json data/traffic_gt.json \
       --neg data/noise.mp4 data/noise_7.mp4 data/noise_21.mp4 data/noise_43.mp4
```

Le recalage fait **une seule passe** par vidéo en mode permissif, enregistre
les *caractéristiques* de chaque candidat (`ghost.trace`), puis fait un
grid-search hors ligne sur 24 500 combinaisons de seuils — au lieu de rejouer
la détection pour chaque combinaison.

Résultats (16 vrais positifs, 84 s de bruit pur) :

| Seuils                     | rappel | précision | F1    | faux positifs |
|----------------------------|--------|-----------|-------|---------------|
| aucun filtre               | 64 %   | 64,0 %    | 64,0 % | 5,5 / min    |
| seuils intuitifs (anciens) | 36 %   | 75,0 %    | 48,6 % | 1,8 / min    |
| **seuils mesurés**         | **64 %** | **72,7 %** | **68,1 %** | **3,7 / min** |

Le résultat contre-intuitif : `max_size_var` (stabilité de la taille) était
**nuisible**. L'aire d'un vrai véhicule varie jusqu'à ×50 d'une frame à l'autre
parce que le fond l'absorbe partiellement — exiger une taille stable
supprimait surtout des vrais positifs. Le seul discriminant qui tient est la
**cohérence de direction** (`min_dir_consistency = 0.90`).

Vérification : les 4 vidéos de bruit pur donnent **0 franchissement**.

```bash
python tools/diagnose_ghost.py --source data/noise.mp4   # doit afficher 0
```

## Test sur un flux RTSP réel

Sans caméra ni Internet, on monte un vrai serveur RTSP (MediaMTX) et on y
publie une vidéo en H.264 temps réel avec ffmpeg — exactement comme une caméra
IP :

```bash
python tools/rtsp_server.py --install                    # une fois
python tools/rtsp_server.py --source data/eval.mp4       # -> rtsp://127.0.0.1:8554/cam
python tools/test_rtsp.py   --url rtsp://127.0.0.1:8554/cam
python main.py --source rtsp://127.0.0.1:8554/cam --no-display \
               --json out/rtsp_summary.json --csv out/rtsp_counts.csv
```

Banc d'acquisition (`tools/test_rtsp.py`) — 4 tests, tous verts :

| Test                | Attendu                        | Mesuré                          |
|---------------------|--------------------------------|---------------------------------|
| mode live           | file de taille 1, 1280×720     | OK                              |
| débit               | ~25 FPS reçus                  | 27,3 FPS                        |
| latence bornée      | pas d'accumulation             | médiane 1 ms, max 193–367 ms    |
| reconnexion         | survit à une coupure           | OK (5 frames récupérées)        |

Le « mode live » utilise une file de taille 1 : quand l'inférence est plus
lente que le flux, on traite toujours la frame la plus récente et la latence
ne s'accumule jamais.

Deux bugs réels trouvés par ces tests et corrigés dans `src/capture.py` :

1. **Une lecture en échec tuait le grabber définitivement** — le sentinelle de
   fin était posé à la première erreur, même sur un flux live. Désormais un
   échec est traité comme transitoire : temporisation, puis réouverture
   automatique (`reopen_after`).
2. **La fin de fichier tronquait la vidéo** — le sentinelle vidait la file,
   emportant jusqu'à 32 frames encore en attente (la fin du film, là où sont
   les derniers franchissements). La vidange ne s'applique plus qu'au live ;
   un fichier attend que le consumer ait tout consommé. Non-régression :
   `python tools/test_capture_eof.py` (200/200 frames, même avec un consumer
   plus lent que la lecture).

## CI de non-régression

```bash
python tools/regression.py --update-baseline   # enregistre l'état de référence
python tools/regression.py                     # vérifie — exit 1 si régression
python tools/regression.py --quick             # tests unitaires seuls (~15 s)
```

Les tests unitaires valident la **mécanique**, pas la **performance**. Or le
projet a déjà subi des régressions silencieuses : fin de fichier tronquée (32
frames perdues), 14 ou 16 comptés selon la charge CPU, suiveur « amélioré » qui
créait en fait plus d'IDs. Ce script rejoue les bancs, compare à
`out/regression_baseline.json` et **sort en erreur** dès qu'une métrique recule
au-delà de la tolérance.

| vérification | tolérance de baisse |
|---|---|
| tests unitaires (zone, tracking, fin de fichier) | aucune — doivent être verts |
| rappel moto (banc VisDrone) | 2 points |
| MOTA | 2 points |
| IDF1 | 3 points |
| total compté sur `eval.mp4` | 2 véhicules |

Durée : ~15 s en `--quick` (pré-commit), ~8 min en complet (nightly / release).

Baseline actuelle : rappel moto 10,37 · MOTA 24,23 · IDF1 64,89 · comptage 16.
Le garde-fou a été **vérifié en conditions réelles** : baseline volontairement
gonflée à 30 / 40 / 80 → `exit 1` et « RÉGRESSION DÉTECTÉE » sur les trois
métriques. Baseline restaurée, la passe complète ressort verte.

> **Attention aux échelles** : `bench_motorcycle.py` écrit un ratio (0.237)
> alors qu'`eval_mot.py` écrit déjà un pourcentage (24.23). La CI normalise.

## Banc de test « moto » : à quelle distance perd-on les motos ?

```bash
python tools/bench_motorcycle.py                                  # modèle COCO
python tools/bench_motorcycle.py --tiled band                     # + rattrapage par tuiles
python tools/bench_motorcycle.py --weights runs/detect/runs/motorcycle/ft416/weights/best.pt
```

Mesure le **rappel de détection moto** sur les **405 vraies motos** de VisDrone
val (imgsz 640, conf 0.10), découpé par **taille de boîte** (proxy de la
distance) et par **luminosité moyenne** (proxy des vêtements sombres / nuit).

Cet outil existe parce que l'objectif n°1 du projet — « améliorer la détection
des motos » — **n'était mesuré nulle part** : aucune vérité terrain du projet ne
contient de moto (`eval_gt.json` = bus/camion/voiture).

### Rappel global

| Configuration | rappel moto |
|---|---|
| `yolov8s.pt`, tuiles off | 10,4 % |
| `yolov8s.pt` + **tuiles** | **21,7 %** |
| `ft416/weights/best.pt` | **23,7 %** |

**Les tuiles doublent le rappel moto à elles seules** — et c'est précisément le
levier que l'adaptatif désactivait sur CPU (`FPS < 12`). D'où l'importance
d'OpenVINO ci-dessous.

### Rappel par taille (côté de la boîte vérité terrain, px)

| taille | base | fine-tuné |
|---|---|---|
| < 16 px | 0,0 % | 8,4 % |
| 16-24 px | 1,9 % | 20,2 % |
| 24-32 px | 3,6 % | 14,5 % |
| 32-48 px | 15,5 % | 32,1 % |
| 48-64 px | **38,2 %** | **64,7 %** |
| ≥ 64 px | 26,7 % | 24,4 % |

Lecture : sous **32 px de côté, le système perd l'essentiel des motos**. Au
delà de 48 px il devient utilisable. C'est la spécification à donner au client :
« au-delà de X mètres de la caméra, on ne garantit plus le comptage moto ».

### Rappel par luminosité (niveau de gris moyen du crop, fine-tuné)

| luminosité | rappel |
|---|---|
| 0-50 (très sombre) | 20,8 % |
| 50-80 | 31,3 % |
| 80-110 | 26,1 % |
| 110-145 | 13,6 % |

Contre-intuitif : **la taille domine largement la luminosité**. Les vêtements
sombres coûtent moins cher que la distance.

## Accélération CPU : export OpenVINO

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt').export(format='openvino', imgsz=640)"
python main.py --source rtsp://... --weights yolov8s_openvino_model --min-fps 0
```

Mesuré sur le i5-1250P de développement :

| | torch | **OpenVINO** | gain |
|---|---|---|---|
| inférence seule | 343 ms / 2,9 FPS | **96 ms / 10,4 FPS** | **×3,6** |
| pipeline complet | 2,03 FPS | **8,19 FPS** | **×4,0** |
| pipeline + tuiles | — | 7,74 FPS | tuiles : **−6 % seulement** |

Le point clé : avec OpenVINO, **activer les tuiles ne coûte que ~6 % de FPS**
(le mode `band` limite les tuiles à la zone de comptage) alors qu'elles
**doublent le rappel moto**. C'est le meilleur rapport qualité/prix du projet.

`--min-fps` (défaut 12) est le seuil sous lequel le tiling s'auto-désactive.
Cette valeur venait d'un raisonnement GPU ; sur CPU elle condamne le meilleur
levier du système. `--min-fps 0` = ne jamais désactiver.

## Performance attendue

| Matériel        | Plein cadre | + tuiles (1 frame / 2) |
|-----------------|-------------|------------------------|
| RTX 3060 (FP16) | ~90-120 FPS | ~35-50 FPS             |
| CPU 8 cœurs     | ~8-15 FPS   | ~3-6 FPS               |

Pour tenir le temps réel strict sur un flux 25 FPS, soit on reste en
`--tiled off`, soit on passe sur `--weights yolov8n.pt`, soit on réduit
`--imgsz` à 512 (la consigne impose 640 par défaut).
