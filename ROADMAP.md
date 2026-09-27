# Feuille de route — rendre le système robuste, professionnel, incontournable

## Diagnostic de départ

La **mécanique est saine** : zone de 200 px, test d'intersection, `set()`
d'IDs, extrapolation à la perte de piste, filet MOG2 calibré, reconnexion RTSP
automatique. Ce n'est pas là que ça pèche.

Ce qui manque, mesuré :

| Ce qu'on prétend | Ce qu'on a mesuré |
|---|---|
| « détecte les motos » | **rappel moto 4 % → 7 %**, et **aucune mesure bout en bout** |
| « temps réel » | **2,5 FPS sur CPU** ; les tuiles (levier n°1 moto) sont **auto-désactivées** |
| « compte les véhicules » | **rappel 73 %, précision 79 %**, sur vidéo **synthétique** |
| « identifie la classe » | **36 %** de justesse |
| « IDs stables » | **~2 IDs par véhicule** ; BoT-SORT mesuré **neutre** |

Le constat central : **tout est mesuré sur du synthétique**. Tant qu'il n'y a
pas une vérité terrain CCTV, chaque chiffre ci-dessus est une opinion, et
chaque optimisation est un pari. C'est le goulet d'étranglement numéro un.

---

## Palier 1 — Crédibilité (sans ça, le reste est invisible)

### 1.1 Un jeu de test CCTV annoté, avec des motos
Le seul livrable qui débloque tout le reste. Cible : 20-30 min de flux réel,
300-500 franchissements annotés, **dont des motos** — petites, sombres,
lointaines, de nuit.

> En attendant une vraie caméra : générer un banc synthétique avec de **vrais
> crops de motos** à différentes échelles / luminosités / vitesses. Ce n'est pas
> équivalent à du réel, mais c'est infiniment mieux que rien, et ça mesure enfin
> l'objectif n°1 du projet.

### 1.2 Métriques MOT standard, pas des proxys maison
Remplacer `ids_created` / `id_switches` par les vraies métriques du domaine :

* **MOTA** (précision globale de tracking : FP + FN + ID switches)
* **IDF1** (qualité de l'identité — c'est LA métrique pour « un ID par véhicule »)
* **HOTA** (compromis détection / association)

À brancher via `motmetrics`. Aujourd'hui personne ne peut comparer ce système à
un autre, faute de métrique standard. C'est ça qui le rend « non professionnel ».

### 1.3 CI de non-régression
Chaque commit rejoue le banc et **bloque** si MOTA/IDF1/F1 baissent. Les tests
actuels (10/10 zone, intégration, EOF) valident la mécanique, pas la
performance. Sans ça, toute optimisation risque d'en casser une autre en
silence — et on l'a déjà vu (14 vs 16 comptés selon la charge CPU).

---

## Palier 2 — Performance réelle

### 2.1 OpenVINO / ONNX : réveiller les tuiles *(gain le plus probable à court terme)*
Sur ce i5-1250P, exporter YOLOv8s en OpenVINO donne typiquement **×2 à ×4**.
Le seuil adaptatif est `min_fps = 12` : passer de 2,5 à 8-12 FPS **réactive
l'inférence en tuiles**, c'est-à-dire le seul levier qui rattrape les motos
lointaines. C'est le plus gros gain accessible sans GPU.

### 2.2 Un modèle moto entraîné sur du CCTV, pas sur de l'aérien
Le run VisDrone (`ft416`) a triplé l'AP50 moto (0,035 → 0,104) mais **perd en
comptage CCTV** (F1 75,9 % → 72,0 %) : décalage de domaine. Il faut
fine-tuner sur des images de caméra — `--from-video` fait du pseudo-étiquetage
pour amorcer, puis correction manuelle.

### 2.3 Un classificateur de classe en 2ᵉ étage
36 % de justesse de classe, c'est rédhibitoire pour un outil métier. Une petite
tête de classification sur les crops (car / moto / bus / camion), entraînée sur
quelques milliers de crops, doit porter ça à 85 %+ à coût quasi nul. Le
compteur classe déjà par **vote temporel** ; il ne manque que la qualité du vote.

---

## Palier 3 — Professionnalisation

### 3.1 Caméra calibrée (homographie)
C'est la frontière entre « script de comptage » et « système professionnel ».
Une homographie sol → image permet :

* **comptage par voie** (aujourd'hui : un total global) ;
* **vitesse en km/h** au lieu de px/frame ;
* zone de comptage définie en **mètres**, stable quel que soit le zoom ;
* filtrage géométrique des faux positifs (un blob qui ne roule pas sur la
  chaussée est éliminé).

### 3.2 Persistance, API, supervision
* **SQLite/Postgres** pour les événements (le CSV ne survit pas à un redémarrage) ;
* **API REST** + **dashboard** (comptage par voie, par classe, par heure, courbes) ;
* **multi-caméras** avec un fichier YAML par site ;
* **watchdog** + `/health` + métriques Prometheus : un système de comptage
  qu'on ne surveille pas est un système qui ment silencieusement.

### 3.3 Robustesse aux conditions réelles
Nuit / infrarouge, pluie, ombres portées, saleté sur l'objectif, caméra PTZ
(mouvement global — là BoT-SORT + GMC devient pertinent), occlusions longues
(là le ReID devient pertinent), contre-jour, tunnels.

---

## Palier 4 — Discipline d'ingénierie

* packaging (`pyproject.toml`, Docker, systemd) — aujourd'hui on lance un script ;
* **identifiants RTSP hors de l'URL** (actuellement `rtsp://user:pass@...`, donc
  en clair dans les logs et l'historique shell) ;
* logs structurés (JSON), rotation, niveaux ;
* couverture de tests significative (> 80 %) ;
* conformité : comptage sur voie publique → durée de rétention, anonymisation.

---

## Si on ne fait que deux choses

1. **Le banc de test moto** (1.1) — parce que sans lui, tout le reste est un
   pari, et c'est précisément l'objectif que le projet n'a jamais mesuré.
2. **L'export OpenVINO** (2.1) — parce qu'il réactive les tuiles, donc le
   levier moto, sur la machine qu'on a, sans attendre un GPU.

## Ce qui fera vraiment la différence

Le modèle YOLO n'est pas un avantage concurrentiel : tout le monde l'a. Ce qui
rend un compteur **incontournable**, c'est :

* la **vérité terrain du client** (le système s'étalonne sur *sa* caméra, et
  personne d'autre ne peut le faire à sa place) ;
* la **traçabilité** (chaque comptage est justifiable : frame, méthode,
  confiance, vignette) — déjà en partie en place via `method` dans le CSV ;
* la **stabilité opérationnelle** (ça tourne 24/7/365 sans qu'on y pense).

Autrement dit : la valeur est dans la **boucle de validation et l'exploitation**,
pas dans le détecteur.
