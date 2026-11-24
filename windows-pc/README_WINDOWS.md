# Loto BJ — Station PC Windows 10 / 11 (Réplique Complète Android + GPU OpenCL / CPU Multi-Cœurs / RAM Max)

Ce dossier (`windows-pc/`) contient la version **Windows 10 / 11 Desktop complète** de l'application **Loto BJ**, reproduisant à l'identique les **4 écrans Android** tout en exploitant la puissance matérielle du PC :

## Les 4 Onglets (Parité Complète Android)
1. **1. Résultats LNB** : Bannière **3 Semaines (21j à la date J)**, carte **DERNIER TIRAGE**, boules circulaires et vue tableau complet, téléchargement direct LNB + import CSV/JSON.
2. **2. Moteur & Découverte** : **Strategy Factory** dynamique à l'exécution (**Arithmétique Modulaire**, **Densité de Probabilité**, **Triangulation**, **Synthèse Hybride**) + programmation génétique symbolique accélérée **Cluster Multi-GPU OpenCL (tous les GPUs détectés simultanément) + 100 % cœurs CPU**, **24 signaux mathématiques zéro-fuite** (Base 7, Lucky 9, Mod 10/11, Cube90, Congruence Affine, Extrapolation, Limite Weibull, Proba Z-Score, Triangulation, Grille 9×10), registre **zéro-réutilisation** (aucune méthode déjà testée n'est re-évaluée), rotation dynamique anti-stagnation de **7 familles de théorèmes**, règle stricte **`< 50 %` de 0-boule**, optimisation **3/4/5 bons**, validation **4 niveaux zéro-fuite**, paliers **2J à 7J+**, sauvegarde/transfert **`loto_bj_strategies_backup_v1`**.
3. **3. Sortie IA & Confiances** : Prédiction **uniquement du prochain tirage Bénin (`Africa/Porto-Novo`)**, 5 boules avec anneau & indice de confiance + badge **ÉCART**, barres de comparaison **Top 10 Candidats vs Hasard (5,6%)**, compteur **Jours réussis X/Y**, **Performance par Horaire (`8H..20H`)**, **Récapitulatif Jour par Jour**, backtest étape par étape et **Export/Copie `.txt`**.
4. **4. Labo Scientifique** : Benchmark parallèle multi-cœurs des **13 méthodes** (`Auto-Recherche`, `ML Adaptatif Bayésien`, `Chaud/Froid`, `Dirichlet`, `EMA`, `Markov Stratifié`, `Graphe Co-occurrence`, `Fourier DFT`, `Grille 9×10`, `Croix/Pyramide`, `Lunaire`, `Cube90 audité`, `Horaire Hongrois`) + **10 000+ simulations Monte Carlo** + **Paires Complémentaires**.

---

## Comment lancer sur Windows 10 / 11 (Sans fermeture silencieuse)

### Option 1 — Diagnostic & Lancement direct (Recommandé en premier test)
Double-cliquez sur **`run_windows.bat`** dans `windows-pc/`.
- Détecte automatiquement Java (`PATH`, `JAVA_HOME`, Android Studio `jbr`, Eclipse Temurin, Oracle/Microsoft JDK 17/21+).
- Utilise directement le package pré-compilé **`app\LotoBJ-PC-AllInOne.jar`** (aucune installation Gradle requise).
- Utilise un profil mémoire sûr (`-Xms256m -XX:+UseG1GC -XX:MaxRAMPercentage=65.0`, sans `AlwaysPreTouch`).
- Enregistre les journaux dans `%USERPROFILE%\.LotoBJ_PC\startup.log` et **`crash.log`**, et **met en pause avec le détail exact** en cas d'erreur.

### Option 2 — Lancement silencieux sans fenêtre noire
Double-cliquez sur **`LotoBJ-PC-SansConsole.vbs`**.

### Option 3 — Générer l'exécutable natif `LotoBJ-PC.exe`
Double-cliquez sur **`build_windows_exe.bat`** (nécessite `jpackage.exe` d'un JDK 21+ comme Eclipse Temurin JDK 21).
L'exécutable autonome complet sera créé dans :
`windows-pc\dist\LotoBJ-PC\LotoBJ-PC.exe`

---

## Stockage local & Journaux PC
- Cache tirages & stratégies : `%USERPROFILE%\.LotoBJ_PC\` (`draws_cache.json` et `discovered_formulas.json`).
- Journaux de démarrage/erreur : `%USERPROFILE%\.LotoBJ_PC\startup.log` et `%USERPROFILE%\.LotoBJ_PC\crash.log`.
