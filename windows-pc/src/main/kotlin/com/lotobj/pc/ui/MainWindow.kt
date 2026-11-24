package com.lotobj.pc.ui

import com.lotobj.pc.engine.MathTheoremDomain
import com.lotobj.pc.engine.PcWorkstationCoordinator
import com.lotobj.pc.engine.PcWorkstationUiState
import com.lotobj.pc.model.AutonomousPhase
import com.lotobj.pc.model.BacktestPeriod
import com.lotobj.pc.model.ComputeMode
import com.lotobj.pc.model.PrecisionTargetMode
import com.lotobj.pc.model.PrecisionUiState
import com.lotobj.pc.model.SymbolicFormulaSummary
import com.lotobj.pc.util.BacktestExportHelper
import com.lotobj.pc.util.DesktopClipboard
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.swing.Swing
import java.awt.BorderLayout
import java.awt.Color
import java.awt.Dimension
import java.awt.FlowLayout
import java.awt.Font
import java.awt.GridBagConstraints
import java.awt.GridBagLayout
import java.awt.GridLayout
import java.awt.Insets
import java.awt.event.WindowAdapter
import java.awt.event.WindowEvent
import java.io.File
import java.util.Locale
import javax.swing.BorderFactory
import javax.swing.Box
import javax.swing.BoxLayout
import javax.swing.DefaultComboBoxModel
import javax.swing.JButton
import javax.swing.JCheckBox
import javax.swing.JComboBox
import javax.swing.JFileChooser
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.JOptionPane
import javax.swing.JPanel
import javax.swing.JProgressBar
import javax.swing.JScrollPane
import javax.swing.JSpinner
import javax.swing.JSplitPane
import javax.swing.JTabbedPane
import javax.swing.JTable
import javax.swing.JTextArea
import javax.swing.SpinnerNumberModel
import javax.swing.border.EmptyBorder
import javax.swing.filechooser.FileNameExtensionFilter
import javax.swing.table.DefaultTableModel

class MainWindow(
    private val coordinator: PcWorkstationCoordinator
) : JFrame("Loto BJ — Station PC Double Moteur (5 Boules + Précision 1er/2 N°)") {

    private val uiScope = CoroutineScope(SupervisorJob() + Dispatchers.Swing)
    private var stateCollectJob: Job? = null

    private val mainTabs = JTabbedPane().apply {
        font = Font("Segoe UI", Font.BOLD, 13)
        tabLayoutPolicy = JTabbedPane.SCROLL_TAB_LAYOUT
    }

    // Hardware bar widgets
    private val hwGpuLabel = JLabel("GPU: détection...")
    private val hwCpuLabel = JLabel("CPU: 0%")
    private val hwRamLabel = JLabel("RAM: 0 Mo")
    private val hwSpeedLabel = JLabel("Débit: 0 éval/s")
    private val statusBarLabel = JLabel("Prêt")
    private val nextDrawBannerLabel = JLabel("Prochain tirage : calcul...")

    private val computeModeCombo = JComboBox(ComputeMode.entries.toTypedArray())
    private val popSizeCombo = JComboBox(arrayOf(256, 512, 1024, 2048, 4096))
    private val mcSimsCombo = JComboBox(arrayOf(1500, 5000, 10000, 25000))
    private val periodCombo = JComboBox(BacktestPeriod.entries.toTypedArray())
    private val periodDaysSpinner = JSpinner(SpinnerNumberModel(21, 1, 3650, 1)).apply {
        preferredSize = Dimension(68, 28)
        toolTipText = "Période de travail personnalisée (minimum 1 jour)"
    }

    // Tab 1: Results
    private lateinit var resultsTabPanel: ResultsTabPanel

    // Tab 2: Autonomous Symbolic Engine (claire & épurée)
    private val objectiveOptions = arrayOf("Auto (Fenêtre)", "Palier 1J", "Palier 2J", "Palier 3J", "Palier 4J", "Palier 5J", "Palier 6J", "Palier 7J")
    private val objectiveCombo = JComboBox(objectiveOptions)
    private val startBtn = JButton("Démarrer").apply {
        toolTipText = "Lance une nouvelle recherche GPU+CPU"
    }
    private val suspendBtn = JButton("Mettre en pause").apply {
        toolTipText = "Met en pause en gardant toute la mémoire"
    }
    private val resumeBtn = JButton("Continuer").apply {
        toolTipText = "Reprend la recherche exactement où elle était"
    }
    private val stopBtn = JButton("Stop (Fermer)").apply {
        toolTipText = "Arrête totalement et ferme l'analyse en cours"
    }
    private val restartZeroBtn = JButton("Reprendre à zéro").apply {
        toolTipText = "Réinitialise la session et relance une recherche neuve"
    }
    private val infiniteModeCheckBox = JCheckBox("Continuer jusqu'à l'infini").apply {
        isOpaque = false
        foreground = AndroidM3Colors.Gold
        font = Font("Segoe UI", Font.BOLD, 12)
        toolTipText = "Ne s'arrête pas automatiquement après validation (tourne jusqu'à Pause ou Stop)"
    }
    private val continueBtn = JButton("Mode Infini").apply {
        toolTipText = "Active le mode infini et poursuit sans arrêt"
    }
    private val launchBacktestBtn = JButton("Lancer Backtest & Prédiction")

    // Le Chercheur + Strategy Factory
    private val chercheurRunBtn = JButton("Lancer Le Chercheur & Injecter").apply {
        toolTipText = "Étudie chaque numéro, position (Pos1..5) et tirage complet, distille des règles abstraites et les injecte dans le moteur"
    }
    private val chercheurStatsLabel = JLabel("Le Chercheur : 0 cycles • 0 leçons • 0 injectées • 0 gardées / 0 jetées")
    private val chercheurNoteLabel = JLabel("Prêt à étudier individuellement chaque numéro, position et tirage.")

    private val factoryDomainCombo = JComboBox(
        arrayOf(
            FactoryDomainItem(null, "Toutes Familles (Modulaire + Proba + Triangulation)"),
            FactoryDomainItem(MathTheoremDomain.MODULAR_ARITHMETIC, MathTheoremDomain.MODULAR_ARITHMETIC.label),
            FactoryDomainItem(MathTheoremDomain.PROBABILITY_DENSITY, MathTheoremDomain.PROBABILITY_DENSITY.label),
            FactoryDomainItem(MathTheoremDomain.TRIANGULATION, MathTheoremDomain.TRIANGULATION.label),
            FactoryDomainItem(MathTheoremDomain.HYBRID_SYNTHESIS, MathTheoremDomain.HYBRID_SYNTHESIS.label)
        )
    )
    private val factoryBurstBtn = JButton("Test Rapide Factory")
    private val factorySummaryLabel = JLabel("Factory prête")

    private val phaseBadgeLabel = JLabel("STATUT : PRÊT")
    private val matchedCounterLabel = JLabel("Jours réussis : 0/4")
    private val genStatsLabel = JLabel("Génération : 0 • Expériences : 0")
    private val quickNextPredictionLabel = JLabel("Prochain tirage : lancez une stratégie validée pour voir les 5 boules ici.")
    private val formulaALabel = JTextArea("Formule A : —").apply {
        isEditable = false; isOpaque = false; lineWrap = true; wrapStyleWord = true
        font = Font("Consolas", Font.BOLD, 12); foreground = AndroidM3Colors.Primary
    }
    private val formulaBLabel = JTextArea("Formule B : —").apply {
        isEditable = false; isOpaque = false; lineWrap = true; wrapStyleWord = true
        font = Font("Consolas", Font.BOLD, 12); foreground = AndroidM3Colors.Tertiary
    }
    private val stage1Label = JLabel("1. Découverte : —")
    private val stage2Label = JLabel("2. Glissante : —")
    private val stage3Label = JLabel("3. Confirmation : —")
    private val stage4Label = JLabel("4. Test Scellé : —")
    private val genProgressBar = JProgressBar().apply { isIndeterminate = false }
    private val featureHistogramPanel = JPanel().apply {
        layout = BoxLayout(this, BoxLayout.Y_AXIS)
        background = AndroidM3Colors.SurfaceVariant
    }

    private val savedWinnerCombo = JComboBox<WinnerComboItem>()
    private val copyBackupBtn = JButton("Copier Sauvegarde JSON")
    private val pasteBackupBtn = JButton("Coller Sauvegarde")
    private val exportBackupFileBtn = JButton("Exporter .json")
    private val importBackupFileBtn = JButton("Importer .json")

    private val chercheurTeachingsModel = object : DefaultTableModel(
        arrayOf("Leçon", "Périmètre", "Percées", "Score", "Statut", "Principe Distillé (sans formule figée)"), 0
    ) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val chercheurTeachingsTable = JTable(chercheurTeachingsModel).apply { rowHeight = 24 }

    private val topFormulasModel = object : DefaultTableModel(
        arrayOf("ID", "Gén.", "Statut", "Déc", "Gliss", "≥2/Tir", "0-Boule%", "≥3 / ≥4 / 5", "Moy.Gliss", "Fitness", "Formules A | B"), 0
    ) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val topFormulasTable = JTable(topFormulasModel).apply { rowHeight = 23 }

    private val rejectedFormulasModel = object : DefaultTableModel(
        arrayOf("ID", "Gén.", "Étape Rejet", "Moy. Déc.", "Moy. Val.", "Raison", "Formule"), 0
    ) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val rejectedFormulasTable = JTable(rejectedFormulasModel).apply { rowHeight = 23 }

    private val liveLogsArea = JTextArea().apply {
        isEditable = false
        font = Font("Consolas", Font.PLAIN, 12)
        background = AndroidM3Colors.SurfaceVariant
        foreground = AndroidM3Colors.OnSurface
    }

    // Tab 3: Autonomous Engine Output & Backtest Details
    private val predHeaderLabel = JLabel("Sortie Verrouillée Avant Validation (ou cliquez sur Lancer Backtest)")
    private val predSubBannerLabel = JLabel("Prochain tirage ciblé Bénin • Fenêtre configurée")
    private val approvedBacktestBannerLabel = JLabel("Backtest : aucun verrouillage append-only actif pour l'instant")
    private val outputMatchedBadge = JLabel("0/4 j")
    private val predBallCardsRow = JPanel(GridLayout(1, 5, 10, 0)).apply { isOpaque = false }
    private val confidenceSummaryLabel = JTextArea("Confiance P(≥1 bon) : — • Espérance / 5 : —").apply {
        isEditable = false; isOpaque = false; lineWrap = true; wrapStyleWord = true
        font = Font("Segoe UI", Font.BOLD, 12); foreground = AndroidM3Colors.Tertiary
    }
    private val predFormulaDetailLabel = JTextArea("Formules : —").apply {
        isEditable = false; isOpaque = false; lineWrap = true; wrapStyleWord = true
        font = Font("Consolas", Font.PLAIN, 11); foreground = AndroidM3Colors.Primary
    }
    private val candidateBarsPanel = JPanel().apply {
        layout = BoxLayout(this, BoxLayout.Y_AXIS)
        background = AndroidM3Colors.SurfaceVariant
    }
    private val copyReportBtn = JButton("Copier Rapport Complet")
    private val saveReportBtn = JButton("Enregistrer Rapport .txt")

    private val backtestStepsModel = object : DefaultTableModel(
        arrayOf("#", "Date", "Jeu / H", "Prédit", "Tirage Réel", "Trouvés", "Brier", "Causal"), 0
    ) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val backtestStepsTable = JTable(backtestStepsModel).apply { rowHeight = 23 }

    private val dailySummaryModel = object : DefaultTableModel(
        arrayOf("Date", "Tirages Testés", "Tirages ≥2 bons", "Avant-dernier ≥2", "Meilleur Score", "Jour Qualifié"), 0
    ) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val dailySummaryTable = JTable(dailySummaryModel).apply { rowHeight = 23 }

    private val horairePerfModel = object : DefaultTableModel(
        arrayOf("Horaire", "Tirages", "Moyenne / 5", "≥1 bon", "≥2 bons"), 0
    ) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val horairePerfTable = JTable(horairePerfModel).apply { rowHeight = 23 }

    private val distributionLabel = JLabel("Distribution des scores : —")
    private val detailedReportArea = JTextArea().apply {
        isEditable = false
        font = Font("Consolas", Font.PLAIN, 12)
        background = AndroidM3Colors.SurfaceVariant
        foreground = AndroidM3Colors.OnSurface
    }

    // Tab 4: Scientific Lab
    private lateinit var scientificLabTabPanel: ScientificLabTabPanel

    // Quick Bridge Tab 2 <-> Tab 5 (Moteur 2 Précision)
    private val m2QuickBadgeLabel = JLabel("Moteur 2 (Précision 1er N° & 2 N°) : Prêt — indépendant du moteur 5 boules").apply {
        font = Font("Segoe UI", Font.BOLD, 12)
        foreground = AndroidM3Colors.Primary
    }
    private val openPrecisionTabBtn = JButton("Ouvrir Moteur 2 (1er N° & 2 N°) →").apply {
        background = AndroidM3Colors.Primary
        foreground = AndroidM3Colors.OnPrimary
        font = Font("Segoe UI", Font.BOLD, 12)
        toolTipText = "Basculer vers le 2e moteur autonome (1er Numéro Pos1 et 2 Numéros Exacts)"
    }

    // Tab 5: Standalone Precision Engine (1er Numéro Pos1 & 2 Numéros Exacts)
    private val precModeCombo = JComboBox(PrecisionTargetMode.entries.toTypedArray())
    private val precStartBtn = JButton("Démarrer Précision")
    private val precPauseBtn = JButton("Mettre en pause")
    private val precResumeBtn = JButton("Continuer")
    private val precStopBtn = JButton("Stop (Fermer)")
    private val precRestartBtn = JButton("Reprendre à zéro")
    private val precInfiniteCheck = JCheckBox("Continuer jusqu'à l'infini").apply {
        isOpaque = false
        foreground = AndroidM3Colors.Gold
        font = Font("Segoe UI", Font.BOLD, 12)
    }
    private val precContinueInfiniteBtn = JButton("Mode Infini").apply {
        toolTipText = "Active le mode infini sur le Moteur Précision jusqu'à Pause ou Stop"
    }
    private val precBacktestBtn = JButton("Lancer Backtest & Prédiction")
    private val precCopyReportBtn = JButton("Copier Rapport")
    private val precStatusLabel = JLabel("STATUT : PRÊT")
    private val precStatsLabel = JLabel("Gén. 0 • Exp. 0 • Palier 0j • 0 éval/s")
    private val precModeDescLabel = JLabel(PrecisionTargetMode.FIRST_BALL.description)
    private val precFormulaArea = JTextArea("Formule Précision : —").apply {
        isEditable = false; isOpaque = false; lineWrap = true; wrapStyleWord = true
        font = Font("Consolas", Font.BOLD, 12); foreground = AndroidM3Colors.Primary
    }
    private val precHonestNoteArea = JTextArea("Moteur Précision Indépendant prêt (1er N° & 2 N° exacts).").apply {
        isEditable = false; isOpaque = false; lineWrap = true; wrapStyleWord = true
        font = Font("Segoe UI", Font.PLAIN, 12); foreground = AndroidM3Colors.OnSurfaceVariant
    }
    private val precLiveBallsRow = JPanel(FlowLayout(FlowLayout.LEFT, 14, 4)).apply { isOpaque = false }
    private val precLiveDetailArea = JTextArea("Aucune prédiction Précision verrouillée.").apply {
        isEditable = false; isOpaque = false; lineWrap = true; wrapStyleWord = true
        font = Font("Segoe UI", Font.BOLD, 12); foreground = AndroidM3Colors.Gold
    }
    private val precStage1Lbl = JLabel("1. Découverte : —")
    private val precStage2Lbl = JLabel("2. Glissante : —")
    private val precStage3Lbl = JLabel("3. Confirmation : —")
    private val precStage4Lbl = JLabel("4. Test Scellé : —")
    private val precProgressBar = JProgressBar().apply { isIndeterminate = false }
    private val precStepsModel = object : DefaultTableModel(
        arrayOf("#", "Date", "Jeu / H", "Prédit", "Pos1 / Réel", "Exact", "Brier", "Causal"), 0
    ) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val precStepsTable = JTable(precStepsModel).apply { rowHeight = 23 }
    private val precDailyBlocksModel = object : DefaultTableModel(
        arrayOf("Bloc (6 tirages / 1J)", "Tirages", "Cibles Exactes", "Avant-dernier Exact", "≥1 Bon", "Bloc Qualifié"), 0
    ) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val precDailyBlocksTable = JTable(precDailyBlocksModel).apply { rowHeight = 23 }
    private val precLogsArea = JTextArea().apply {
        isEditable = false
        font = Font("Consolas", Font.PLAIN, 12)
        background = AndroidM3Colors.SurfaceVariant
        foreground = AndroidM3Colors.OnSurface
    }
    private var precisionCollectJob: Job? = null

    private var suppressComboEvents = false
    private var lastFeatureHistSig: String = ""
    private var lastWinnersSig: String = ""
    private var lastTopFormulasSig: String = ""
    private var lastRejectedSig: String = ""
    private var lastTeachingsSig: String = ""
    private var lastPredCardsSig: String = ""
    private var lastCandidateBarsSig: String = ""
    private var lastBtStepsSig: String = ""
    private var lastDailySumSig: String = ""
    private var lastHorairePerfSig: String = ""
    private var lastPrecBallsSig: String = ""
    private var lastPrecStepsSig: String = ""
    private var lastPrecBlocksSig: String = ""

    init {
        defaultCloseOperation = DISPOSE_ON_CLOSE
        minimumSize = Dimension(1200, 780)
        setSize(1400, 900)
        setLocationRelativeTo(null)

        buildLayout()
        wireActions()
        startObserving()

        addWindowListener(object : WindowAdapter() {
            override fun windowClosed(e: WindowEvent?) {
                stateCollectJob?.cancel()
                precisionCollectJob?.cancel()
                uiScope.cancel()
                coordinator.close()
            }
        })
    }

    private fun buildLayout() {
        val root = JPanel(BorderLayout(8, 8)).apply {
            border = EmptyBorder(10, 10, 10, 10)
            background = AndroidM3Colors.Background
        }

        val topContainer = JPanel().apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            background = AndroidM3Colors.Background
        }

        val hwBar = AndroidSwingTheme.makeCard(AndroidM3Colors.SurfaceVariant, AndroidM3Colors.Outline, 12).apply {
            layout = GridLayout(2, 2, 14, 4)
            hwGpuLabel.font = Font("Segoe UI", Font.BOLD, 12)
            hwCpuLabel.font = Font("Segoe UI", Font.BOLD, 12)
            hwCpuLabel.foreground = AndroidM3Colors.OnSurface
            hwRamLabel.font = Font("Segoe UI", Font.BOLD, 12)
            hwRamLabel.foreground = AndroidM3Colors.OnSurface
            hwSpeedLabel.font = Font("Segoe UI", Font.BOLD, 12)
            hwSpeedLabel.foreground = AndroidM3Colors.Gold
            add(hwGpuLabel)
            add(hwCpuLabel)
            add(hwRamLabel)
            add(hwSpeedLabel)
        }

        val configBar = JPanel(GridBagLayout()).apply {
            background = AndroidM3Colors.Background
            border = EmptyBorder(4, 2, 2, 2)
            val gbc = GridBagConstraints().apply {
                insets = Insets(2, 4, 2, 6)
                anchor = GridBagConstraints.WEST
            }
            val row1 = JPanel(FlowLayout(FlowLayout.LEFT, 8, 2)).apply {
                isOpaque = false
                add(JLabel("Puissance :").apply { foreground = AndroidM3Colors.OnSurfaceVariant })
                computeModeCombo.renderer = AndroidSwingTheme.simpleComboRenderer<ComputeMode> { it.label }
                add(computeModeCombo)
                add(JLabel("Pop. GP :").apply { foreground = AndroidM3Colors.OnSurfaceVariant })
                add(popSizeCombo)
                add(JLabel("Monte Carlo :").apply { foreground = AndroidM3Colors.OnSurfaceVariant })
                add(mcSimsCombo)
                add(JLabel("Fenêtre :").apply { foreground = AndroidM3Colors.OnSurfaceVariant })
                periodCombo.renderer = AndroidSwingTheme.simpleComboRenderer<BacktestPeriod> { it.label }
                add(periodCombo)
                add(JLabel("Jours (≥1) :").apply { foreground = AndroidM3Colors.OnSurfaceVariant })
                add(periodDaysSpinner)
            }
            nextDrawBannerLabel.font = Font("Segoe UI", Font.BOLD, 13)
            nextDrawBannerLabel.foreground = AndroidM3Colors.Gold

            gbc.gridx = 0; gbc.gridy = 0; gbc.weightx = 1.0; gbc.fill = GridBagConstraints.HORIZONTAL
            add(row1, gbc)
            gbc.gridy = 1
            add(nextDrawBannerLabel, gbc)
        }

        topContainer.add(hwBar)
        topContainer.add(Box.createVerticalStrut(4))
        topContainer.add(configBar)
        root.add(topContainer, BorderLayout.NORTH)

        resultsTabPanel = ResultsTabPanel(
            onFetchOnline = { coordinator.fetchLatestOnlineResults() },
            onImportArchive = { openImportArchiveChooser() }
        )
        scientificLabTabPanel = ScientificLabTabPanel(
            onRunComparison = { coordinator.runScientificComparison() },
            onPredictWithMethod = { method ->
                coordinator.runClassicalMethodBacktestAndPrediction(method)
                mainTabs.selectedIndex = 2
            },
            onUseSavedModel = { modelId ->
                coordinator.selectSavedWinner(modelId)
                coordinator.launchValidatedBacktestAndPrediction()
                mainTabs.selectedIndex = 2
            }
        )

        mainTabs.addTab("1. Résultats LNB", resultsTabPanel)
        mainTabs.addTab("2. Moteur 1 (5 Boules)", buildEngineTab())
        mainTabs.addTab("3. Sortie IA & Confiances", buildPredictionOutputTab())
        mainTabs.addTab("4. Labo Scientifique", scientificLabTabPanel)
        mainTabs.addTab("5. Moteur 2 : Précision (1er & 2 N°)", buildPrecisionTab())
        root.add(mainTabs, BorderLayout.CENTER)

        val bottomBar = JPanel(BorderLayout()).apply {
            background = AndroidM3Colors.Background
            border = BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(1, 0, 0, 0, AndroidM3Colors.Outline),
                EmptyBorder(6, 8, 2, 8)
            )
            statusBarLabel.foreground = AndroidM3Colors.Primary
            add(statusBarLabel, BorderLayout.WEST)
            val caps = coordinator.uiState.value.hardwareCaps
            add(
                JLabel("Cœurs: ${caps.logicalCores} • JVM Max RAM: ${caps.maxJvmHeapMb} Mo • Physique: ${caps.totalPhysicalRamMb} Mo").apply {
                    foreground = AndroidM3Colors.OnSurfaceVariant
                },
                BorderLayout.EAST
            )
        }
        root.add(bottomBar, BorderLayout.SOUTH)
        contentPane = root
    }

    private fun buildEngineTab(): JPanel {
        val panel = JPanel(BorderLayout(8, 8)).apply {
            border = EmptyBorder(8, 8, 8, 8)
            background = AndroidM3Colors.Background
        }

        val northPanel = JPanel().apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            background = AndroidM3Colors.Background
        }

        // Carte 1 : Contrôles clairs du cycle de vie (sans chevauchement)
        val controlsCard = AndroidSwingTheme.makeCard(AndroidM3Colors.SurfaceVariant, AndroidM3Colors.Outline, 14).apply {
            layout = GridBagLayout()
            val gbc = GridBagConstraints().apply {
                insets = Insets(3, 4, 3, 4)
                anchor = GridBagConstraints.WEST
                fill = GridBagConstraints.HORIZONTAL
                weightx = 1.0
            }

            startBtn.background = AndroidM3Colors.Tertiary
            startBtn.foreground = AndroidM3Colors.OnTertiary
            startBtn.font = Font("Segoe UI", Font.BOLD, 12)
            launchBacktestBtn.background = AndroidM3Colors.Gold
            launchBacktestBtn.foreground = Color.BLACK
            launchBacktestBtn.font = Font("Segoe UI", Font.BOLD, 12)
            stopBtn.background = AndroidM3Colors.ErrorRed
            stopBtn.foreground = Color.WHITE
            continueBtn.background = AndroidM3Colors.Primary
            continueBtn.foreground = AndroidM3Colors.OnPrimary

            val rowButtons = JPanel(FlowLayout(FlowLayout.LEFT, 8, 2)).apply {
                isOpaque = false
                add(JLabel("Cible :").apply { foreground = AndroidM3Colors.OnSurfaceVariant })
                add(objectiveCombo)
                add(startBtn)
                add(suspendBtn)
                add(resumeBtn)
                add(stopBtn)
                add(restartZeroBtn)
                add(infiniteModeCheckBox)
                add(continueBtn)
                add(launchBacktestBtn)
            }
            val helpNote = JLabel("Pause = suspend en gardant la mémoire • Continuer = reprend • Stop = ferme l'analyse • Reprendre à zéro = réinitialise • Infini = cherche sans arrêt auto").apply {
                font = Font("Segoe UI", Font.PLAIN, 11)
                foreground = AndroidM3Colors.OnSurfaceVariant
            }
            val engine2BridgeRow = JPanel(BorderLayout(8, 2)).apply {
                isOpaque = false
                border = EmptyBorder(2, 0, 0, 0)
                add(m2QuickBadgeLabel, BorderLayout.CENTER)
                add(openPrecisionTabBtn, BorderLayout.EAST)
            }

            gbc.gridx = 0; gbc.gridy = 0; add(rowButtons, gbc)
            gbc.gridy = 1; add(helpNote, gbc)
            gbc.gridy = 2; add(engine2BridgeRow, gbc)
        }

        // Carte 2 : État direct du moteur + Prédiction rapide
        val statusCard = AndroidSwingTheme.makeCard(AndroidM3Colors.SurfaceVariant, AndroidM3Colors.Outline, 14).apply {
            layout = GridBagLayout()
            val gbc = GridBagConstraints().apply {
                insets = Insets(3, 6, 3, 6)
                anchor = GridBagConstraints.WEST
                fill = GridBagConstraints.HORIZONTAL
            }
            phaseBadgeLabel.font = Font("Segoe UI", Font.BOLD, 13)
            phaseBadgeLabel.foreground = AndroidM3Colors.Primary
            matchedCounterLabel.font = Font("Segoe UI", Font.BOLD, 13)
            matchedCounterLabel.foreground = AndroidM3Colors.Tertiary
            genStatsLabel.font = Font("Segoe UI", Font.PLAIN, 12)
            genStatsLabel.foreground = AndroidM3Colors.OnSurface
            quickNextPredictionLabel.font = Font("Segoe UI", Font.BOLD, 12)
            quickNextPredictionLabel.foreground = AndroidM3Colors.Gold

            gbc.gridx = 0; gbc.gridy = 0; gbc.gridwidth = 2; gbc.weightx = 0.72; add(phaseBadgeLabel, gbc)
            gbc.gridx = 2; gbc.gridwidth = 1; gbc.weightx = 0.28; add(matchedCounterLabel, gbc)
            gbc.gridx = 0; gbc.gridy = 1; gbc.gridwidth = 3; gbc.weightx = 1.0; add(genStatsLabel, gbc)
            gbc.gridy = 2; add(genProgressBar, gbc)
            gbc.gridy = 3; add(quickNextPredictionLabel, gbc)
            gbc.gridy = 4; add(formulaALabel, gbc)
            gbc.gridy = 5; add(formulaBLabel, gbc)
        }

        // Carte 3 : Le Chercheur + Strategy Factory
        val chercheurCard = AndroidSwingTheme.makeCard(AndroidM3Colors.SurfaceVariant, AndroidM3Colors.Primary, 14).apply {
            layout = GridBagLayout()
            val gbc = GridBagConstraints().apply {
                insets = Insets(2, 6, 2, 6)
                anchor = GridBagConstraints.WEST
                fill = GridBagConstraints.HORIZONTAL
                weightx = 1.0
            }
            chercheurRunBtn.background = AndroidM3Colors.Primary
            chercheurRunBtn.foreground = AndroidM3Colors.OnPrimary
            chercheurRunBtn.font = Font("Segoe UI", Font.BOLD, 12)
            chercheurStatsLabel.font = Font("Segoe UI", Font.BOLD, 12)
            chercheurStatsLabel.foreground = AndroidM3Colors.Tertiary
            chercheurNoteLabel.font = Font("Segoe UI", Font.PLAIN, 11)
            chercheurNoteLabel.foreground = AndroidM3Colors.OnSurfaceVariant

            val topRow = JPanel(FlowLayout(FlowLayout.LEFT, 8, 2)).apply {
                isOpaque = false
                add(chercheurRunBtn)
                add(chercheurStatsLabel)
            }
            val factorySubRow = JPanel(FlowLayout(FlowLayout.LEFT, 8, 2)).apply {
                isOpaque = false
                add(JLabel("Strategy Factory :").apply { foreground = AndroidM3Colors.OnSurfaceVariant })
                add(factoryDomainCombo)
                factoryBurstBtn.background = AndroidM3Colors.SecondaryContainer
                factoryBurstBtn.foreground = AndroidM3Colors.OnSecondaryContainer
                factoryBurstBtn.font = Font("Segoe UI", Font.BOLD, 12)
                add(factoryBurstBtn)
                factorySummaryLabel.foreground = AndroidM3Colors.OnSurfaceVariant
                factorySummaryLabel.font = Font("Segoe UI", Font.PLAIN, 11)
                add(factorySummaryLabel)
            }

            gbc.gridx = 0; gbc.gridy = 0; add(topRow, gbc)
            gbc.gridy = 1; add(chercheurNoteLabel, gbc)
            gbc.gridy = 2; add(factorySubRow, gbc)
        }

        northPanel.add(controlsCard)
        northPanel.add(Box.createVerticalStrut(6))
        northPanel.add(statusCard)
        northPanel.add(Box.createVerticalStrut(6))
        northPanel.add(chercheurCard)
        panel.add(northPanel, BorderLayout.NORTH)

        // Sous-panneaux techniques rangés par onglets propres (zéro surcharge visuelle)
        val fourStagesAndBackupPanel = JPanel(BorderLayout(8, 8)).apply {
            background = AndroidM3Colors.Background
            border = EmptyBorder(6, 6, 6, 6)
            val stagesGrid = JPanel(GridLayout(2, 2, 8, 8)).apply {
                background = AndroidM3Colors.Background
                listOf(stage1Label, stage2Label, stage3Label, stage4Label).forEach { lbl ->
                    lbl.border = BorderFactory.createCompoundBorder(
                        RoundedBorder(AndroidM3Colors.Outline, 12, 1f),
                        EmptyBorder(8, 10, 8, 10)
                    )
                    lbl.isOpaque = true
                    lbl.background = AndroidM3Colors.SurfaceVariant
                    lbl.foreground = AndroidM3Colors.OnSurface
                    add(lbl)
                }
            }
            val backupRow = JPanel(FlowLayout(FlowLayout.LEFT, 8, 6)).apply {
                background = AndroidM3Colors.SurfaceVariant
                border = BorderFactory.createTitledBorder("Stratégies Sauvegardées & Transfert Android <-> PC")
                add(JLabel("Stratégie active :"))
                savedWinnerCombo.preferredSize = Dimension(320, 28)
                add(savedWinnerCombo)
                add(copyBackupBtn)
                add(pasteBackupBtn)
                add(exportBackupFileBtn)
                add(importBackupFileBtn)
            }
            add(stagesGrid, BorderLayout.CENTER)
            add(backupRow, BorderLayout.SOUTH)
        }

        val centerTabs = JTabbedPane().apply {
            addTab("Le Chercheur (Gardées / Jetées)", JScrollPane(chercheurTeachingsTable))
            addTab("Meilleures Formules Symboliques", JScrollPane(topFormulasTable))
            addTab("Formules Rejetées (Anti-Fuite)", JScrollPane(rejectedFormulasTable))
            addTab("Variables Brutes (24 Signaux)", JScrollPane(featureHistogramPanel))
            addTab("4 Étapes & Sauvegardes JSON", fourStagesAndBackupPanel)
        }
        val split = JSplitPane(
            JSplitPane.VERTICAL_SPLIT,
            centerTabs,
            JScrollPane(liveLogsArea).apply { border = BorderFactory.createTitledBorder("Journal Temps Réel GPU / CPU") }
        ).apply {
            resizeWeight = 0.65
        }
        panel.add(split, BorderLayout.CENTER)
        return panel
    }

    private fun buildPredictionOutputTab(): JPanel {
        val panel = JPanel(BorderLayout(8, 8)).apply {
            border = EmptyBorder(8, 8, 8, 8)
            background = AndroidM3Colors.Background
        }

        val northStack = JPanel().apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            background = AndroidM3Colors.Background
        }

        val headerCard = AndroidSwingTheme.makeCard(AndroidM3Colors.SecondaryContainer, AndroidM3Colors.Primary, 16).apply {
            layout = BorderLayout(10, 6)
            val left = JPanel().apply {
                layout = BoxLayout(this, BoxLayout.Y_AXIS)
                isOpaque = false
                predHeaderLabel.font = Font("Segoe UI", Font.BOLD, 15)
                predHeaderLabel.foreground = AndroidM3Colors.OnSecondaryContainer
                predSubBannerLabel.font = Font("Segoe UI", Font.PLAIN, 12)
                predSubBannerLabel.foreground = AndroidM3Colors.PrimaryContainer
                approvedBacktestBannerLabel.font = Font("Segoe UI", Font.BOLD, 12)
                approvedBacktestBannerLabel.foreground = AndroidM3Colors.Gold
                add(predHeaderLabel)
                add(predSubBannerLabel)
                add(Box.createVerticalStrut(3))
                add(approvedBacktestBannerLabel)
            }
            val right = JPanel(FlowLayout(FlowLayout.RIGHT, 8, 2)).apply {
                isOpaque = false
                outputMatchedBadge.font = Font("Segoe UI", Font.BOLD, 13)
                outputMatchedBadge.foreground = AndroidM3Colors.Tertiary
                copyReportBtn.background = AndroidM3Colors.Primary
                copyReportBtn.foreground = AndroidM3Colors.OnPrimary
                copyReportBtn.font = Font("Segoe UI", Font.BOLD, 12)
                add(outputMatchedBadge)
                add(copyReportBtn)
                add(saveReportBtn)
            }
            add(left, BorderLayout.CENTER)
            add(right, BorderLayout.EAST)
        }

        val ballsAndCandidatesRow = JPanel(GridLayout(1, 2, 10, 0)).apply {
            background = AndroidM3Colors.Background
        }

        val ballsWrapper = AndroidSwingTheme.makeCard(AndroidM3Colors.SurfaceVariant, AndroidM3Colors.Outline, 14).apply {
            layout = BorderLayout(6, 6)
            add(JLabel("Numéros Prédits & Indices de Confiance (Base Hasard : 5,6%)").apply {
                font = Font("Segoe UI", Font.BOLD, 13)
                foreground = AndroidM3Colors.OnSurface
            }, BorderLayout.NORTH)
            add(predBallCardsRow, BorderLayout.CENTER)
            val summaryBox = JPanel().apply {
                layout = BoxLayout(this, BoxLayout.Y_AXIS)
                isOpaque = false
                add(confidenceSummaryLabel)
                add(Box.createVerticalStrut(3))
                add(predFormulaDetailLabel)
            }
            add(summaryBox, BorderLayout.SOUTH)
        }

        val topCandidatesCard = AndroidSwingTheme.makeCard(AndroidM3Colors.SurfaceVariant, AndroidM3Colors.Outline, 14).apply {
            layout = BorderLayout(6, 6)
            add(JLabel("Comparaison Top Candidats vs Hasard (5,6%)").apply {
                font = Font("Segoe UI", Font.BOLD, 13)
                foreground = AndroidM3Colors.OnSurface
            }, BorderLayout.NORTH)
            add(JScrollPane(candidateBarsPanel).apply {
                border = BorderFactory.createEmptyBorder()
                preferredSize = Dimension(380, 180)
            }, BorderLayout.CENTER)
        }

        ballsAndCandidatesRow.add(ballsWrapper)
        ballsAndCandidatesRow.add(topCandidatesCard)

        northStack.add(headerCard)
        northStack.add(Box.createVerticalStrut(6))
        northStack.add(ballsAndCandidatesRow)
        panel.add(northStack, BorderLayout.NORTH)

        val detailTabs = JTabbedPane().apply {
            addTab("Backtest Tirage par Tirage (Zéro Fuite)", JScrollPane(backtestStepsTable))
            val horaireAndDist = JPanel(BorderLayout(6, 6)).apply {
                background = AndroidM3Colors.Background
                add(distributionLabel.apply {
                    border = EmptyBorder(6, 8, 6, 8)
                    font = Font("Segoe UI", Font.BOLD, 12)
                    foreground = AndroidM3Colors.Primary
                }, BorderLayout.NORTH)
                add(JScrollPane(horairePerfTable), BorderLayout.CENTER)
            }
            addTab("Performance par Horaire & Distribution", horaireAndDist)
            addTab("Récapitulatif Jour par Jour", JScrollPane(dailySummaryTable))
            addTab("Rapport Détaillé (.txt)", JScrollPane(detailedReportArea))
        }
        panel.add(detailTabs, BorderLayout.CENTER)
        return panel
    }

    private fun buildPrecisionTab(): JPanel {
        val panel = JPanel(BorderLayout(8, 8)).apply {
            border = EmptyBorder(8, 8, 8, 8)
            background = AndroidM3Colors.Background
        }

        val northStack = JPanel().apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            background = AndroidM3Colors.Background
        }

        val controlsCard = AndroidSwingTheme.makeCard(AndroidM3Colors.SurfaceVariant, AndroidM3Colors.Primary, 14).apply {
            layout = GridBagLayout()
            val gbc = GridBagConstraints().apply {
                insets = Insets(2, 5, 2, 5)
                anchor = GridBagConstraints.WEST
                fill = GridBagConstraints.HORIZONTAL
                weightx = 1.0
            }
            precModeCombo.renderer = AndroidSwingTheme.simpleComboRenderer<PrecisionTargetMode> { it.label }
            precStartBtn.background = AndroidM3Colors.Tertiary
            precStartBtn.foreground = AndroidM3Colors.OnTertiary
            precStartBtn.font = Font("Segoe UI", Font.BOLD, 12)
            precStopBtn.background = AndroidM3Colors.ErrorRed
            precStopBtn.foreground = Color.WHITE
            precContinueInfiniteBtn.background = AndroidM3Colors.Primary
            precContinueInfiniteBtn.foreground = AndroidM3Colors.OnPrimary
            precContinueInfiniteBtn.font = Font("Segoe UI", Font.BOLD, 12)
            precBacktestBtn.background = AndroidM3Colors.Gold
            precBacktestBtn.foreground = Color.BLACK
            precBacktestBtn.font = Font("Segoe UI", Font.BOLD, 12)

            val row1 = JPanel(FlowLayout(FlowLayout.LEFT, 8, 2)).apply {
                isOpaque = false
                add(JLabel("Cible Moteur 2 (Indépendant) :").apply { foreground = AndroidM3Colors.OnSurfaceVariant; font = Font("Segoe UI", Font.BOLD, 12) })
                add(precModeCombo)
                precModeDescLabel.foreground = AndroidM3Colors.Gold
                precModeDescLabel.font = Font("Segoe UI", Font.PLAIN, 11)
                add(precModeDescLabel)
            }
            val row2 = JPanel(FlowLayout(FlowLayout.LEFT, 8, 2)).apply {
                isOpaque = false
                add(precStartBtn)
                add(precPauseBtn)
                add(precResumeBtn)
                add(precStopBtn)
                add(precRestartBtn)
                add(precInfiniteCheck)
                add(precContinueInfiniteBtn)
                add(precBacktestBtn)
                add(precCopyReportBtn)
            }
            val helpRow = JLabel("Pause = suspend avec mémoire • Continuer = reprend • Stop = ferme • Reprendre à zéro = reset complet • Infini = boucle GPU+CPU continue").apply {
                font = Font("Segoe UI", Font.PLAIN, 11)
                foreground = AndroidM3Colors.OnSurfaceVariant
            }
            gbc.gridx = 0; gbc.gridy = 0; add(row1, gbc)
            gbc.gridy = 1; add(row2, gbc)
            gbc.gridy = 2; add(helpRow, gbc)
        }

        val statusAndPredCard = AndroidSwingTheme.makeCard(AndroidM3Colors.SurfaceVariant, AndroidM3Colors.Outline, 14).apply {
            layout = GridBagLayout()
            val gbc = GridBagConstraints().apply {
                insets = Insets(2, 6, 2, 6)
                anchor = GridBagConstraints.WEST
                fill = GridBagConstraints.HORIZONTAL
            }
            precStatusLabel.font = Font("Segoe UI", Font.BOLD, 13)
            precStatusLabel.foreground = AndroidM3Colors.Primary
            precStatsLabel.font = Font("Segoe UI", Font.BOLD, 12)
            precStatsLabel.foreground = AndroidM3Colors.Tertiary

            val liveRowWrapper = JPanel(BorderLayout(12, 2)).apply {
                isOpaque = false
                add(precLiveBallsRow, BorderLayout.WEST)
                add(precLiveDetailArea, BorderLayout.CENTER)
            }

            gbc.gridx = 0; gbc.gridy = 0; gbc.weightx = 0.35; add(precStatusLabel, gbc)
            gbc.gridx = 1; gbc.weightx = 0.65; add(precStatsLabel, gbc)
            gbc.gridx = 0; gbc.gridy = 1; gbc.gridwidth = 2; gbc.weightx = 1.0; add(precProgressBar, gbc)
            gbc.gridy = 2; add(precFormulaArea, gbc)
            gbc.gridy = 3; add(precHonestNoteArea, gbc)
            gbc.gridy = 4; add(liveRowWrapper, gbc)
        }

        val stagesRow = JPanel(GridLayout(1, 4, 8, 0)).apply {
            background = AndroidM3Colors.Background
            listOf(precStage1Lbl, precStage2Lbl, precStage3Lbl, precStage4Lbl).forEach { lbl ->
                lbl.border = BorderFactory.createCompoundBorder(
                    RoundedBorder(AndroidM3Colors.Outline, 12, 1f),
                    EmptyBorder(5, 8, 5, 8)
                )
                lbl.isOpaque = true
                lbl.background = AndroidM3Colors.SurfaceVariant
                lbl.foreground = AndroidM3Colors.OnSurface
                add(lbl)
            }
        }

        northStack.add(controlsCard)
        northStack.add(Box.createVerticalStrut(5))
        northStack.add(statusAndPredCard)
        northStack.add(Box.createVerticalStrut(5))
        northStack.add(stagesRow)
        panel.add(northStack, BorderLayout.NORTH)

        val precisionCenterTabs = JTabbedPane().apply {
            addTab("Backtest Tirage par Tirage (Zéro Fuite)", JScrollPane(precStepsTable))
            addTab("Blocs 6 Tirages / 1J", JScrollPane(precDailyBlocksTable))
        }
        val split = JSplitPane(
            JSplitPane.VERTICAL_SPLIT,
            precisionCenterTabs,
            JScrollPane(precLogsArea).apply { border = BorderFactory.createTitledBorder("Journal Moteur 2 Précision (GPU / CPU)") }
        ).apply {
            resizeWeight = 0.68
        }
        panel.add(split, BorderLayout.CENTER)
        return panel
    }

    private fun openImportArchiveChooser() {
        val chooser = JFileChooser().apply {
            dialogTitle = "Importer une archive de tirages LNB (CSV ou JSON)"
            fileFilter = FileNameExtensionFilter("Fichiers CSV / JSON", "csv", "json", "txt")
        }
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            coordinator.importArchiveFile(chooser.selectedFile)
        }
    }

    private fun wireActions() {
        computeModeCombo.addActionListener {
            if (!suppressComboEvents) (computeModeCombo.selectedItem as? ComputeMode)?.let { coordinator.setComputeMode(it) }
        }
        popSizeCombo.addActionListener {
            if (!suppressComboEvents) (popSizeCombo.selectedItem as? Int)?.let { coordinator.setPopulationSize(it) }
        }
        mcSimsCombo.addActionListener {
            if (!suppressComboEvents) (mcSimsCombo.selectedItem as? Int)?.let { coordinator.setMonteCarloSims(it) }
        }
        periodCombo.addActionListener {
            if (!suppressComboEvents) (periodCombo.selectedItem as? BacktestPeriod)?.let { coordinator.setSelectedPeriod(it) }
        }
        periodDaysSpinner.addChangeListener {
            if (!suppressComboEvents) {
                val days = (periodDaysSpinner.value as? Int) ?: 21
                coordinator.setCustomPeriodDays(days)
            }
        }
        objectiveCombo.addActionListener {
            if (!suppressComboEvents) {
                val days = when (objectiveCombo.selectedIndex) {
                    1 -> 1; 2 -> 2; 3 -> 3; 4 -> 4; 5 -> 5; 6 -> 6; 7 -> 7
                    else -> null
                }
                coordinator.selectPreferredStrategyDays(days)
            }
        }
        savedWinnerCombo.addActionListener {
            if (!suppressComboEvents) (savedWinnerCombo.selectedItem as? WinnerComboItem)?.let { coordinator.selectSavedWinner(it.id) }
        }

        startBtn.addActionListener { coordinator.startAutonomousDiscovery(continueBeyond = false) }
        suspendBtn.addActionListener { coordinator.suspendDiscovery() }
        resumeBtn.addActionListener { coordinator.resumeDiscovery() }
        stopBtn.addActionListener { coordinator.stopDiscovery() }
        restartZeroBtn.addActionListener { coordinator.restartDiscoveryFromZero() }
        infiniteModeCheckBox.addActionListener {
            if (!suppressComboEvents) {
                coordinator.setInfiniteMode(infiniteModeCheckBox.isSelected)
            }
        }
        continueBtn.addActionListener { coordinator.startInfiniteMode() }
        chercheurRunBtn.addActionListener { coordinator.runChercheurAndInject() }
        launchBacktestBtn.addActionListener {
            coordinator.launchValidatedBacktestAndPrediction()
            mainTabs.selectedIndex = 2
        }
        factoryDomainCombo.addActionListener {
            if (!suppressComboEvents) {
                (factoryDomainCombo.selectedItem as? FactoryDomainItem)?.let {
                    coordinator.setFactoryDomain(it.domain)
                }
            }
        }
        factoryBurstBtn.addActionListener {
            coordinator.runStrategyFactoryBurst()
        }

        copyBackupBtn.addActionListener {
            val msg = coordinator.copyStrategyBackupToClipboard()
            JOptionPane.showMessageDialog(this, msg, "Sauvegarde Méthodes", JOptionPane.INFORMATION_MESSAGE)
        }
        pasteBackupBtn.addActionListener {
            val clipText = DesktopClipboard.readText()
            val input = JOptionPane.showInputDialog(
                this,
                "Collez ici le bloc JSON de sauvegarde (loto_bj_strategies_backup_v1) :",
                clipText.orEmpty()
            )
            if (!input.isNullOrBlank()) {
                val msg = coordinator.importStrategyBackupText(input)
                JOptionPane.showMessageDialog(this, msg, "Import Sauvegarde", JOptionPane.INFORMATION_MESSAGE)
            }
        }
        exportBackupFileBtn.addActionListener {
            val (json, defaultName) = coordinator.exportStrategyBackupJson()
            val chooser = JFileChooser().apply {
                dialogTitle = "Enregistrer la sauvegarde des méthodes (.json)"
                selectedFile = File(defaultName)
                fileFilter = FileNameExtensionFilter("Fichiers JSON", "json")
            }
            if (chooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
                chooser.selectedFile.writeText(json, Charsets.UTF_8)
                JOptionPane.showMessageDialog(this, "Sauvegarde enregistrée : ${chooser.selectedFile.absolutePath}")
            }
        }
        importBackupFileBtn.addActionListener {
            val chooser = JFileChooser().apply {
                dialogTitle = "Importer un fichier de sauvegarde (.json)"
                fileFilter = FileNameExtensionFilter("Fichiers JSON / TXT", "json", "txt")
            }
            if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                val msg = coordinator.importStrategyBackupText(chooser.selectedFile.readText(Charsets.UTF_8))
                JOptionPane.showMessageDialog(this, msg, "Import Fichier Sauvegarde", JOptionPane.INFORMATION_MESSAGE)
            }
        }

        copyReportBtn.addActionListener {
            val msg = coordinator.copyBacktestReportToClipboard()
            JOptionPane.showMessageDialog(this, msg, "Copie Rapport", JOptionPane.INFORMATION_MESSAGE)
        }
        saveReportBtn.addActionListener {
            val st = coordinator.uiState.value
            val report = st.detailedReportText
            if (report.isBlank()) {
                JOptionPane.showMessageDialog(this, "Lancez d'abord le backtest pour générer le rapport.")
                return@addActionListener
            }
            val defaultName = BacktestExportHelper.buildExportFileName(st.prediction, st.backtestResult)
            val chooser = JFileChooser().apply {
                dialogTitle = "Enregistrer le rapport détaillé (.txt)"
                selectedFile = File(defaultName)
                fileFilter = FileNameExtensionFilter("Fichiers texte (.txt)", "txt")
            }
            if (chooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
                chooser.selectedFile.writeText(report, Charsets.UTF_8)
                JOptionPane.showMessageDialog(this, "Rapport enregistré : ${chooser.selectedFile.absolutePath}")
            }
        }

        // Actions Tab 2 -> Tab 5 Bridge & Tab 5: Moteur 2 Précision Indépendant
        openPrecisionTabBtn.addActionListener { mainTabs.selectedIndex = 4 }
        precModeCombo.addActionListener {
            if (!suppressComboEvents) {
                (precModeCombo.selectedItem as? PrecisionTargetMode)?.let { coordinator.setPrecisionTargetMode(it) }
            }
        }
        precStartBtn.addActionListener { coordinator.startPrecisionDiscovery(continueBeyond = false) }
        precPauseBtn.addActionListener { coordinator.suspendPrecisionDiscovery() }
        precResumeBtn.addActionListener { coordinator.resumePrecisionDiscovery() }
        precStopBtn.addActionListener { coordinator.stopPrecisionDiscovery() }
        precRestartBtn.addActionListener { coordinator.restartPrecisionFromZero() }
        precInfiniteCheck.addActionListener {
            if (!suppressComboEvents) {
                coordinator.setPrecisionInfiniteMode(precInfiniteCheck.isSelected)
            }
        }
        precContinueInfiniteBtn.addActionListener { coordinator.startPrecisionInfiniteMode() }
        precBacktestBtn.addActionListener { coordinator.launchPrecisionBacktestAndPrediction() }
        precCopyReportBtn.addActionListener {
            val msg = coordinator.copyPrecisionReportToClipboard()
            JOptionPane.showMessageDialog(this, msg, "Rapport Moteur 2 Précision", JOptionPane.INFORMATION_MESSAGE)
        }
    }

    private fun startObserving() {
        stateCollectJob = uiScope.launch {
            coordinator.uiState.collectLatest { st ->
                renderState(st)
            }
        }
        precisionCollectJob = uiScope.launch {
            coordinator.precisionUiState.collectLatest { pst ->
                renderPrecisionState(pst)
            }
        }
    }

    private fun renderState(st: PcWorkstationUiState) {
        suppressComboEvents = true
        if (!computeModeCombo.isPopupVisible && computeModeCombo.selectedItem != st.computeMode) {
            computeModeCombo.selectedItem = st.computeMode
        }
        if (!popSizeCombo.isPopupVisible && popSizeCombo.selectedItem != st.populationSize) {
            popSizeCombo.selectedItem = st.populationSize
        }
        if (!mcSimsCombo.isPopupVisible && mcSimsCombo.selectedItem != st.monteCarloSims) {
            mcSimsCombo.selectedItem = st.monteCarloSims
        }
        if (!periodCombo.isPopupVisible && periodCombo.selectedItem != st.selectedPeriod) {
            periodCombo.selectedItem = st.selectedPeriod
        }
        val daysVal = st.customPeriodDays
        if (periodDaysSpinner.isEnabled != (daysVal != null)) {
            periodDaysSpinner.isEnabled = daysVal != null
        }
        if (daysVal != null && !periodDaysSpinner.hasFocus()) {
            val clamped = daysVal.coerceIn(1, 3650)
            if ((periodDaysSpinner.value as? Int) != clamped) {
                periodDaysSpinner.value = clamped
            }
        }
        if (!factoryDomainCombo.isPopupVisible) {
            for (i in 0 until factoryDomainCombo.itemCount) {
                if (factoryDomainCombo.getItemAt(i).domain == st.selectedFactoryDomain) {
                    if (factoryDomainCombo.selectedIndex != i) {
                        factoryDomainCombo.selectedIndex = i
                    }
                    break
                }
            }
        }
        val targetObjectiveIdx = when (st.engineState.preferredStrategyDays) {
            1 -> 1; 2 -> 2; 3 -> 3; 4 -> 4; 5 -> 5; 6 -> 6; 7 -> 7
            else -> 0
        }
        if (!objectiveCombo.isPopupVisible && objectiveCombo.selectedIndex != targetObjectiveIdx) {
            objectiveCombo.selectedIndex = targetObjectiveIdx
        }
        suppressComboEvents = false

        val caps = st.hardwareCaps
        val tel = st.hardwareTelemetry
        val nextGpuText = if (caps.gpuAvailable) {
            val vendorParts = buildList {
                if (caps.nvidiaGpuCount > 0) add("NVIDIA×${caps.nvidiaGpuCount}")
                if (caps.amdGpuCount > 0) add("AMD×${caps.amdGpuCount}")
                if (caps.otherGpuCount > 0) add("Autre×${caps.otherGpuCount}")
            }.joinToString("+").ifBlank { "${caps.gpuCount} GPU" }
            val prefix = if (caps.gpuCount >= 2) "Multi-GPU [$vendorParts]" else "GPU OpenCL [$vendorParts]"
            val errSuffix = if (tel.gpuErrorNote.isNotBlank()) " ⚠" else ""
            "$prefix: ${caps.gpuDeviceName} (${caps.gpuComputeUnits} CU, ${caps.gpuGlobalMemMb} Mo)$errSuffix"
        } else {
            "GPU OpenCL NVIDIA/AMD: absent → Fallback CPU (${caps.logicalCores} threads)"
        }
        if (hwGpuLabel.text != nextGpuText) hwGpuLabel.text = nextGpuText
        val errTip = tel.gpuErrorNote.takeIf { it.isNotBlank() }
        if (hwGpuLabel.toolTipText != errTip) hwGpuLabel.toolTipText = errTip
        hwGpuLabel.foreground = when {
            !caps.gpuAvailable -> AndroidM3Colors.Gold
            tel.gpuErrorNote.isNotBlank() -> AndroidM3Colors.Gold
            else -> AndroidM3Colors.Tertiary
        }
        hwCpuLabel.text = String.format(
            Locale.FRANCE,
            "CPU: %d/%d cœurs • App %.0f%% • Sys %.0f%%",
            tel.activeWorkerThreads, caps.logicalCores, tel.processCpuPercent, tel.systemCpuPercent
        )
        hwRamLabel.text = "RAM: ${tel.jvmUsedRamMb} / ${tel.jvmMaxRamMb} Mo (Libres ${tel.systemFreeRamMb} Mo)"
        hwSpeedLabel.text = String.format(
            Locale.FRANCE,
            "Débit: %.0f éval/s (GPU %.1fms • CPU %.1fms)",
            tel.evaluationsPerSecond, tel.lastGpuBatchMs, tel.lastCpuBatchMs
        )

        val nd = st.nextDrawTarget
        val ndBanner = "Prochain tirage : ${nd.horaire} (${if (nd.isTomorrow) "Demain" else "Aujourd'hui"} ${nd.dayLabel} • ${nd.currentBeninTimeLabel} Bénin)"
        if (nextDrawBannerLabel.text != ndBanner) nextDrawBannerLabel.text = ndBanner
        if (statusBarLabel.text != st.statusBanner) statusBarLabel.text = st.statusBanner

        resultsTabPanel.render(st)
        scientificLabTabPanel.render(st)

        // Engine Tab
        val es = st.engineState
        suppressComboEvents = true
        if (infiniteModeCheckBox.isSelected != es.infiniteMode) {
            infiniteModeCheckBox.isSelected = es.infiniteMode
        }
        suppressComboEvents = false

        val busy = es.isAnalyzing || st.isFactoryRunning
        startBtn.isEnabled = !busy && st.allDraws.size >= 8
        suspendBtn.isEnabled = es.isAnalyzing
        resumeBtn.isEnabled = es.canResume && !st.isFactoryRunning
        stopBtn.isEnabled = es.isAnalyzing || es.isSuspended
        restartZeroBtn.isEnabled = st.allDraws.size >= 8 && !st.isFactoryRunning
        continueBtn.isEnabled = st.allDraws.size >= 8 && !st.isFactoryRunning
        chercheurRunBtn.isEnabled = st.allDraws.size >= 8 && !es.chercheurStatus.isRunning && !st.isFactoryRunning
        launchBacktestBtn.isEnabled = es.canLaunchBacktest && !busy
        factoryBurstBtn.isEnabled = !busy && st.allDraws.size >= 8
        factoryDomainCombo.isEnabled = !st.isFactoryRunning
        st.lastFactoryReport?.let { rep ->
            val fText = "${rep.summaryNote} (${String.format(Locale.FRANCE, "%.0f", rep.evaluationsPerSec)} éval/s)"
            if (factorySummaryLabel.text != fText) factorySummaryLabel.text = fText
        }

        val ch = es.chercheurStatus
        val chStatsText = "Cycles: ${ch.cyclesCompleted} • Leçons: ${ch.teachingsProduced} • Injectées: ${ch.teachingsInjected} • Gardées: ${ch.strategiesKept} • Jetées: ${ch.strategiesDiscarded}"
        if (chercheurStatsLabel.text != chStatsText) chercheurStatsLabel.text = chStatsText
        if (chercheurNoteLabel.text != ch.lastLessonNote) chercheurNoteLabel.text = ch.lastLessonNote
        val teachingsSig = ch.teachings.joinToString("|") { "${it.id}:${it.status}:${it.piercedDrawsCount}:${it.generalizedScore}" }
        if (teachingsSig != lastTeachingsSig) {
            lastTeachingsSig = teachingsSig
            chercheurTeachingsModel.rowCount = 0
            ch.teachings.forEach { t ->
                val stLabel = when (t.status) {
                    "KEPT" -> "GARDÉE"
                    "DISCARDED" -> "JETÉE"
                    else -> "INJECTÉE"
                }
                chercheurTeachingsModel.addRow(
                    arrayOf<Any>(
                        t.title, t.scopeLabel, t.piercedDrawsCount,
                        String.format(Locale.FRANCE, "%.2f", t.generalizedScore),
                        stLabel, t.principleSummary
                    )
                )
            }
        }

        if (genProgressBar.isIndeterminate != busy) genProgressBar.isIndeterminate = busy
        val modeTag = if (es.infiniteMode) "INFINI" else es.phase.name
        val phaseText = "Statut : $modeTag • [${es.activeParadigmName}]"
        if (phaseBadgeLabel.text != phaseText) phaseBadgeLabel.text = phaseText
        val matched = es.matchedDaysCount(st.backtestResult)
        matchedCounterLabel.text = "Jours réussis : $matched/${es.selectedPatternLength}"
        outputMatchedBadge.text = "$matched/${es.selectedPatternLength} j"
        val zPctStr = if (es.generation > 0) "${(es.bestZeroMatchRate * 100f).toInt()}%" else "—"
        val genStatsText = "Gén.${es.generation} • Exp:${es.experimentsCount} • Exclues:${es.excludedMethodsCount} • 0-Boule:$zPctStr • ≥3:${es.bestThreePlusCount} ≥4:${es.bestFourPlusCount} 5:${es.bestFiveMatchCount}"
        if (genStatsLabel.text != genStatsText) genStatsLabel.text = genStatsText
        val fAText = "Formule A (${es.currentSlotSplit}): ${es.currentPrimaryExpr.ifBlank { "—" }}"
        if (formulaALabel.text != fAText) formulaALabel.text = fAText
        val fBText = "Formule B : ${es.currentSecondaryExpr.ifBlank { "—" }}"
        if (formulaBLabel.text != fBText) formulaBLabel.text = fBText

        val quickPred = st.prediction
        val quickPredText = if (quickPred != null && quickPred.predictedDraw.isNotEmpty()) {
            val balls = quickPred.predictedDraw.joinToString(" - ") { it.toString().padStart(2, '0') }
            "PROCHAIN TIRAGE (${quickPred.horaire}) : [ $balls ] • P(≥1 bon) ${String.format(Locale.FRANCE, "%.1f%%", quickPred.calibratedProbAtLeastOne * 100f)}"
        } else if (es.canLaunchBacktest) {
            "Stratégie prête ! Cliquez sur « Lancer Backtest & Prédiction » pour afficher les 5 boules du prochain tirage."
        } else {
            "En attente d'une stratégie validée pour calculer le prochain tirage (${nd.horaire})."
        }
        if (quickNextPredictionLabel.text != quickPredText) quickNextPredictionLabel.text = quickPredText

        val m = es.fourStageMetrics
        if (m != null) {
            val rollZeroPct = (m.rollingZeroMatchRate * 100f).toInt()
            val rollTwoPct = (m.rollingTwoPlusRate * 100f).toInt()
            stage1Label.text = "<html><b>1. Découverte (${m.discoveryDraws})</b><br>Moy ${fmt3(m.discoveryAvgMatches)} • Série ${m.discoveryStreak}j • ≥2: ${m.discoveryTwoPlusCount}</html>"
            stage2Label.text = "<html><b>2. Glissante (${m.rollingDraws})</b><br>Moy ${fmt3(m.rollingAvgMatches)} • ≥2/tir ${rollTwoPct}% • 0-Boule ${rollZeroPct}% • ≥3:${m.rollingThreePlusCount} ≥4:${m.rollingFourPlusCount}</html>"
            stage3Label.text = "<html><b>3. Confirmation (${m.confirmationDraws})</b><br>Moy ${fmt3(m.confirmationAvgMatches)} vs MC ${fmt3(es.randomReference.confirmationMeanAvgMatches)} • p=${fmt4(es.randomReference.pValueAdjusted)}</html>"
            stage4Label.text = "<html><b>4. Test Scellé (${m.sealedDraws})</b><br>Moy ${fmt3(m.sealedAvgMatches)} • ${if (m.sealedCausalVerified) "Causal OK" else "En attente"}</html>"
        }

        val featHistSig = es.discoveredFeaturesHistogram.take(8).joinToString("|") { "${it.first}:${(it.second * 1000).toInt()}" }
        if (featHistSig != lastFeatureHistSig) {
            lastFeatureHistSig = featHistSig
            featureHistogramPanel.removeAll()
            if (es.discoveredFeaturesHistogram.isEmpty()) {
                featureHistogramPanel.add(JLabel("En attente de génération...").apply { foreground = AndroidM3Colors.OnSurfaceVariant })
            } else {
                es.discoveredFeaturesHistogram.take(8).forEach { (feat, share) ->
                    val row = JPanel(BorderLayout(6, 2)).apply {
                        isOpaque = false
                        maximumSize = Dimension(Int.MAX_VALUE, 20)
                        add(JLabel(feat).apply {
                            font = Font("Consolas", Font.BOLD, 11)
                            foreground = AndroidM3Colors.Primary
                            preferredSize = Dimension(52, 16)
                        }, BorderLayout.WEST)
                        val pb = JProgressBar(0, 100).apply {
                            value = (share * 100f).toInt().coerceIn(0, 100)
                            isStringPainted = true
                            string = "${(share * 100f).toInt()}%"
                        }
                        add(pb, BorderLayout.CENTER)
                    }
                    featureHistogramPanel.add(row)
                    featureHistogramPanel.add(Box.createVerticalStrut(2))
                }
            }
            featureHistogramPanel.revalidate()
            featureHistogramPanel.repaint()
        }

        val winners = es.matchingSavedWinners
        val winnersSig = winners.joinToString("|") { "${it.id}:${it.combinedStreak}:${it.generation}" } + "#active=${es.activeWinnerId}"
        if (!savedWinnerCombo.isPopupVisible && winnersSig != lastWinnersSig) {
            lastWinnersSig = winnersSig
            suppressComboEvents = true
            val model = DefaultComboBoxModel<WinnerComboItem>()
            winners.forEach { model.addElement(WinnerComboItem(it)) }
            savedWinnerCombo.model = model
            val selectedIdx = winners.indexOfFirst { it.id == es.activeWinnerId }
            if (selectedIdx >= 0) savedWinnerCombo.selectedIndex = selectedIdx
            suppressComboEvents = false
        }

        val topFormulasSig = es.bestFormulas.joinToString("|") { "${it.id}:${it.generation}:${it.status}:${it.fitnessScore}:${it.rollingAvgMatches}" }
        if (topFormulasSig != lastTopFormulasSig) {
            lastTopFormulasSig = topFormulasSig
            topFormulasModel.rowCount = 0
            es.bestFormulas.forEach { f ->
                topFormulasModel.addRow(
                    arrayOf<Any>(
                        f.id, f.generation, f.status, f.discoveryStreak, f.rollingStreak,
                        "${(f.rollingTwoPlusRate * 100f).toInt()}%",
                        "${(f.rollingZeroMatchRate * 100f).toInt()}%",
                        "${f.rollingThreePlusCount}/${f.rollingFourPlusCount}/${f.rollingFiveMatchCount}",
                        fmt3(f.rollingAvgMatches), String.format(Locale.US, "%.1f", f.fitnessScore),
                        "${f.primaryExprText} | ${f.secondaryExprText}"
                    )
                )
            }
        }

        val rejectedSig = es.rejectedFormulas.joinToString("|") { "${it.id}:${it.generation}:${it.rejectedAtStage}" }
        if (rejectedSig != lastRejectedSig) {
            lastRejectedSig = rejectedSig
            rejectedFormulasModel.rowCount = 0
            es.rejectedFormulas.forEach { r ->
                rejectedFormulasModel.addRow(
                    arrayOf<Any>(
                        r.id, r.generation, r.rejectedAtStage, fmt3(r.trainAvgMatches),
                        fmt3(r.validationAvgMatches), r.reason, r.formulaPreview
                    )
                )
            }
        }

        val joinedLogs = es.liveLogs.joinToString("\n")
        if (liveLogsArea.text != joinedLogs) {
            liveLogsArea.text = joinedLogs
            liveLogsArea.caretPosition = liveLogsArea.document.length
        }

        // Prediction & Output Tab
        val pred = st.prediction
        val bt = st.backtestResult
        if (pred != null && bt != null) {
            val statusStr = if (es.phase == AutonomousPhase.VALIDATED) "VALIDÉ 4 NIVEAUX" else pred.methodName
            predHeaderLabel.text = "Sortie Moteur : $statusStr • Tirage ${pred.horaire} (${if (nd.isTomorrow) "Demain" else "Aujourd'hui"} ${nd.dayLabel})"
            val periodLabelText = st.customPeriodDays?.let { "${it}j (${st.selectedPeriod.label})" } ?: st.selectedPeriod.label
            predSubBannerLabel.text = "Prochain tirage : ${pred.horaire} (${nd.currentBeninTimeLabel} Bénin) • Fenêtre active : $periodLabelText • ${pred.discoveredStrategyName}"
            approvedBacktestBannerLabel.text = es.approvedBacktestNote.ifBlank {
                if (bt.isAppendOnlyApproved) {
                    "Backtest approuvé figé : ${bt.approvedStepsPreservedCount} pas conservés + ${bt.newlyAppendedStepsCount} complétés (${bt.livePredictionsMatchedCount} prédictions live concordantes)"
                } else {
                    "Backtest causal complet (${bt.totalTested} tirages vérifiés)"
                }
            }

            val confs = pred.numberConfidences.ifEmpty {
                pred.predictedDraw.mapIndexed { idx, n ->
                    com.lotobj.pc.model.PredictedNumberConfidence(
                        number = n,
                        marginalProbabilityPercent = 5.56f,
                        relativeConfidencePercent = 90f - idx * 8f,
                        liftVsRandom = 1f,
                        rank = idx + 1,
                        sourceBranch = if (n in pred.methodANumbers) "Formule A" else "Formule B",
                        isGapCandidate = false
                    )
                }
            }
            val predCardsSig = confs.take(5).joinToString("|") { "${it.number}:${(it.marginalProbabilityPercent * 10).toInt()}:${it.sourceBranch}:${it.isGapCandidate}" }
            if (predCardsSig != lastPredCardsSig) {
                lastPredCardsSig = predCardsSig
                predBallCardsRow.removeAll()
                confs.take(5).forEach { c ->
                    val tile = AndroidSwingTheme.makeCard(AndroidM3Colors.SurfaceElevated, AndroidM3Colors.Outline, 12).apply {
                        layout = BoxLayout(this, BoxLayout.Y_AXIS)
                        val style = if (c.sourceBranch.contains("A")) BallStyle.FORMULA_A else BallStyle.FORMULA_B
                        val ringFraction = (c.relativeConfidencePercent / 100f).coerceIn(0.1f, 1f)
                        add(JLabel("#${c.rank}" + if (c.isGapCandidate) " • ÉCART" else "").apply {
                            font = Font("Segoe UI", Font.BOLD, 10)
                            foreground = if (c.isGapCandidate) AndroidM3Colors.Tertiary else AndroidM3Colors.OnSurfaceVariant
                            alignmentX = CENTER_ALIGNMENT
                        })
                        add(Box.createVerticalStrut(4))
                        add(LotoBallBadge(c.number, style, 56, ringFraction).apply { alignmentX = CENTER_ALIGNMENT })
                        add(Box.createVerticalStrut(4))
                        add(JLabel(String.format(Locale.FRANCE, "%.1f%%", c.marginalProbabilityPercent)).apply {
                            font = Font("Segoe UI", Font.BOLD, 14)
                            foreground = if (style == BallStyle.FORMULA_A) AndroidM3Colors.Primary else AndroidM3Colors.Tertiary
                            alignmentX = CENTER_ALIGNMENT
                        })
                        add(JLabel("Idx ${c.relativeConfidencePercent.toInt()}% • ×${String.format(Locale.FRANCE, "%.1f", c.liftVsRandom)}").apply {
                            font = Font("Segoe UI", Font.PLAIN, 10)
                            foreground = AndroidM3Colors.OnSurfaceVariant
                            alignmentX = CENTER_ALIGNMENT
                        })
                        add(JLabel(c.sourceBranch).apply {
                            font = Font("Segoe UI", Font.BOLD, 10)
                            foreground = AndroidM3Colors.Gold
                            alignmentX = CENTER_ALIGNMENT
                        })
                    }
                    predBallCardsRow.add(tile)
                }
                predBallCardsRow.revalidate()
                predBallCardsRow.repaint()
            }

            val pOnePct = (pred.calibratedProbAtLeastOne * 100f).coerceIn(0f, 100f)
            val confSummaryText = String.format(
                Locale.FRANCE,
                "Confiance P(≥1 bon): %.1f%% • Espérance / 5: %.3f • Moy. Backtest: %.3f (%d tir.) • ≥2: %d • ≥3: %d • Jours réussis: %d/%d",
                pOnePct, pred.expectedMatchesEstimate, bt.averageMatches, bt.totalTested,
                bt.drawsWithTwoPlusMatches, bt.drawsWithThreePlusMatches, matched, es.selectedPatternLength
            )
            if (confidenceSummaryLabel.text != confSummaryText) confidenceSummaryLabel.text = confSummaryText
            val predDetailText = "A (${pred.methodANumbers.joinToString("-")}): ${pred.methodALabel}  |  B (${pred.methodBNumbers.joinToString("-")}): ${pred.methodBLabel}"
            if (predFormulaDetailLabel.text != predDetailText) predFormulaDetailLabel.text = predDetailText

            val cands = pred.candidateConfidences.ifEmpty { confs }
            val candBarsSig = cands.take(10).joinToString("|") { "${it.number}:${(it.marginalProbabilityPercent * 10).toInt()}" } + "#${pred.predictedDraw.joinToString("-")}"
            if (candBarsSig != lastCandidateBarsSig) {
                lastCandidateBarsSig = candBarsSig
                candidateBarsPanel.removeAll()
                val maxPct = (cands.maxOfOrNull { it.marginalProbabilityPercent } ?: 15f).coerceAtLeast(10f)
                val predSet = pred.predictedDraw.toSet()
                cands.take(10).forEach { cand ->
                    val isSel = cand.number in predSet
                    val barFrac = cand.marginalProbabilityPercent / maxPct
                    val rndFrac = ((5.0f / 90.0f) * 100f) / maxPct
                    val row = JPanel(BorderLayout(8, 2)).apply {
                        isOpaque = false
                        maximumSize = Dimension(Int.MAX_VALUE, 20)
                        add(JLabel(cand.number.toString().padStart(2, '0')).apply {
                            font = Font("Consolas", Font.BOLD, 12)
                            foreground = if (isSel) AndroidM3Colors.Tertiary else AndroidM3Colors.OnSurfaceVariant
                            preferredSize = Dimension(28, 16)
                        }, BorderLayout.WEST)
                        add(ProbabilityBarCanvas(barFrac, rndFrac, isSel), BorderLayout.CENTER)
                        add(JLabel(String.format(Locale.FRANCE, "%.1f%%", cand.marginalProbabilityPercent)).apply {
                            font = Font("Segoe UI", Font.BOLD, 11)
                            foreground = if (isSel) AndroidM3Colors.Tertiary else AndroidM3Colors.OnSurfaceVariant
                            preferredSize = Dimension(46, 16)
                        }, BorderLayout.EAST)
                    }
                    candidateBarsPanel.add(row)
                    candidateBarsPanel.add(Box.createVerticalStrut(3))
                }
                candidateBarsPanel.revalidate()
                candidateBarsPanel.repaint()
            }

            val btStepsSig = "${bt.totalTested}|${bt.steps.firstOrNull()?.drawDate}|${bt.steps.lastOrNull()?.drawDate}|${bt.steps.lastOrNull()?.predictedNumbers}|${bt.averageMatches}"
            if (btStepsSig != lastBtStepsSig) {
                lastBtStepsSig = btStepsSig
                backtestStepsModel.rowCount = 0
                bt.steps.forEach { step ->
                    backtestStepsModel.addRow(
                        arrayOf<Any>(
                            step.stepIndex,
                            step.drawDate,
                            "${step.game} ${step.horaire}",
                            step.predictedNumbers.joinToString(" ") { it.toString().padStart(2, '0') },
                            step.actualNumbers.joinToString(" ") { it.toString().padStart(2, '0') },
                            "${step.matchCount}/5 (${step.matchedNumbers.joinToString(",")})",
                            String.format(Locale.FRANCE, "%.5f", step.stepBrierScore),
                            if (step.causalVerified) "OUI" else "NON"
                        )
                    )
                }
            }

            val dailySumSig = bt.dailySummaries.joinToString("|") {
                "${it.dayLabel}:${it.drawsTested}:${it.drawsWithTwoPlus}:${it.penultimateTwoPlus}:${it.isQualifyingDay}"
            }
            if (dailySumSig != lastDailySumSig) {
                lastDailySumSig = dailySumSig
                dailySummaryModel.rowCount = 0
                bt.dailySummaries.forEach { d ->
                    dailySummaryModel.addRow(
                        arrayOf<Any>(
                            d.dayLabel,
                            d.drawsTested,
                            d.drawsWithTwoPlus,
                            if (d.penultimateTwoPlus) "OUI (${d.penultimateHitCount}/5)" else "NON (${d.penultimateHitCount}/5)",
                            "${d.bestMatchCount}/5",
                            if (d.isQualifyingDay) "OUI (≥3 + av.-dernier)" else "NON"
                        )
                    )
                }
            }

            val horaireSig = bt.horaireBreakdown.joinToString("|") { "${it.horaire}:${it.testedCount}:${it.twoPlusCount}" }
            if (horaireSig != lastHorairePerfSig) {
                lastHorairePerfSig = horaireSig
                horairePerfModel.rowCount = 0
                bt.horaireBreakdown.forEach { h ->
                    horairePerfModel.addRow(
                        arrayOf<Any>(
                            h.horaire, h.testedCount, String.format(Locale.FRANCE, "%.3f", h.averageMatches),
                            h.onePlusCount, h.twoPlusCount
                        )
                    )
                }
            }
            val distText = "Distribution (0..5 bons) : " +
                bt.matchDistribution.mapIndexed { k, v -> "$k/5 → $v" }.joinToString("  •  ")
            if (distributionLabel.text != distText) distributionLabel.text = distText

            if (detailedReportArea.text != st.detailedReportText) {
                detailedReportArea.text = st.detailedReportText
                detailedReportArea.caretPosition = 0
            }
        }
    }

    private fun renderPrecisionState(pst: PrecisionUiState) {
        suppressComboEvents = true
        if (!precModeCombo.isPopupVisible && precModeCombo.selectedItem != pst.targetMode) {
            precModeCombo.selectedItem = pst.targetMode
        }
        if (precInfiniteCheck.isSelected != pst.infiniteMode) {
            precInfiniteCheck.isSelected = pst.infiniteMode
        }
        suppressComboEvents = false

        val hasData = coordinator.uiState.value.allDraws.size >= 8
        val analyzing = pst.phase == AutonomousPhase.ANALYZING
        precStartBtn.isEnabled = hasData && !analyzing
        precPauseBtn.isEnabled = analyzing
        precResumeBtn.isEnabled = hasData && (pst.phase == AutonomousPhase.SUSPENDED || pst.phase == AutonomousPhase.STOPPED || pst.phase == AutonomousPhase.VALIDATED)
        precStopBtn.isEnabled = analyzing || pst.phase == AutonomousPhase.SUSPENDED
        precRestartBtn.isEnabled = hasData
        precContinueInfiniteBtn.isEnabled = hasData && !analyzing
        precBacktestBtn.isEnabled = hasData || coordinator.uiState.value.allDraws.size >= 4
        precCopyReportBtn.isEnabled = pst.backtestResult != null || pst.livePrediction != null

        if (precModeDescLabel.text != pst.targetMode.description) precModeDescLabel.text = pst.targetMode.description
        val modeTag = if (pst.infiniteMode && analyzing) "INFINI" else pst.phase.name
        val pStatus = "Statut [${pst.targetMode.label}] : $modeTag"
        if (precStatusLabel.text != pStatus) precStatusLabel.text = pStatus
        val pStats = String.format(
            Locale.FRANCE,
            "Gén.%d • Exp:%d • Exclus:%d • Palier:%dJ • Tests Conf:%d • %.0f éval/s",
            pst.generation, pst.experimentsCount, pst.excludedCount, pst.bestValidatedStreak, pst.confirmationTestsCount, pst.evaluationsPerSec
        )
        if (precStatsLabel.text != pStats) precStatsLabel.text = pStats
        if (precProgressBar.isIndeterminate != analyzing) precProgressBar.isIndeterminate = analyzing
        val fText = "Formule Précision : ${pst.activeFormulaDesc.ifBlank { "En attente de recherche..." }}"
        if (precFormulaArea.text != fText) precFormulaArea.text = fText
        if (precHonestNoteArea.text != pst.honestStatusNote) precHonestNoteArea.text = pst.honestStatusNote

        val bridgePred = pst.livePrediction?.let { lp ->
            "Prédit ${lp.horaire}: [${lp.predictedNumbers.joinToString("-") { it.toString().padStart(2, '0') }}]"
        } ?: "Prêt"
        val badgeText = "Moteur 2 (${pst.targetMode.label}) : $modeTag • Gén.${pst.generation} • $bridgePred"
        if (m2QuickBadgeLabel.text != badgeText) m2QuickBadgeLabel.text = badgeText
        m2QuickBadgeLabel.foreground = if (pst.livePrediction != null) AndroidM3Colors.Gold else AndroidM3Colors.Primary

        val d1 = pst.discoveryStats
        val r2 = pst.rollingStats
        val c3 = pst.confirmationStats
        val s4 = pst.sealedStats
        precStage1Lbl.text = if (d1 != null) "<html><b>1. Découverte (${d1.tested})</b><br>Exact:${d1.exactTargetCount} • ≥1:${d1.onePlusCount} • Moy ${fmt3(d1.avgHits)}</html>" else "1. Découverte : —"
        precStage2Lbl.text = if (r2 != null) "<html><b>2. Glissante (${r2.tested})</b><br>Exact:${r2.exactTargetCount} • ≥1:${r2.onePlusCount} • Moy ${fmt3(r2.avgHits)}</html>" else "2. Glissante : —"
        precStage3Lbl.text = if (c3 != null) "<html><b>3. Confirmation (${c3.tested})</b><br>Exact:${c3.exactTargetCount} • ≥1:${c3.onePlusCount} • p_adj=${fmt4(pst.pValueAdjusted)}</html>" else "3. Confirmation : —"
        precStage4Lbl.text = if (s4 != null) "<html><b>4. Test Scellé (${s4.tested})</b><br>Exact:${s4.exactTargetCount} • ${if (s4.causalVerified) "Causal OK" else "Attente"}</html>" else "4. Test Scellé : —"

        val live = pst.livePrediction
        val precBallsSig = live?.let { lp ->
            "${lp.mode.name}|${lp.horaire}|${lp.predictedNumbers.joinToString("-")}|${lp.confidencesPercent.joinToString(",") { "${(it.second * 10).toInt()}" }}|${pst.backtestResult?.totalTested}|${pst.approvedNote}"
        } ?: "NONE"
        if (precBallsSig != lastPrecBallsSig) {
            lastPrecBallsSig = precBallsSig
            precLiveBallsRow.removeAll()
            if (live != null) {
                val baseChance = if (live.mode == PrecisionTargetMode.FIRST_BALL) (100f / 90f) else (500f / 90f)
                live.predictedNumbers.forEachIndexed { idx, num ->
                    val pct = live.confidencesPercent.getOrNull(idx)?.second ?: baseChance
                    val lift = (pct / baseChance).coerceAtLeast(0.1f)
                    val ballPanel = JPanel().apply {
                        layout = BoxLayout(this, BoxLayout.Y_AXIS)
                        isOpaque = false
                        add(LotoBallBadge(num, BallStyle.FORMULA_A, 56, (pct / 25f).coerceIn(0.15f, 1f)).apply { alignmentX = CENTER_ALIGNMENT })
                        add(Box.createVerticalStrut(2))
                        add(JLabel(String.format(Locale.FRANCE, "%.1f%% (×%.1f)", pct, lift)).apply {
                            font = Font("Segoe UI", Font.BOLD, 11)
                            foreground = AndroidM3Colors.Primary
                            alignmentX = CENTER_ALIGNMENT
                        })
                    }
                    precLiveBallsRow.add(ballPanel)
                }
                val bt = pst.backtestResult
                val btInfo = bt?.let {
                    " | Backtest (${it.totalTested} tir.) : Exact=${it.exactTargetCount} • ≥1=${it.onePlusCount} • Blocs ${it.qualifyingBlocksCount} • Série ${it.maxConsecQualifyingBlocks}J • p_adj=${fmt4(it.pValueAdjusted)}"
                }.orEmpty()
                precLiveDetailArea.text = "${live.causalAuditNote}$btInfo\n${pst.approvedNote}"
            } else {
                precLiveDetailArea.text = "Cliquez « Démarrer Précision » puis « Lancer Backtest & Prédiction » pour prédire le prochain tirage (${pst.targetMode.label})."
            }
            precLiveBallsRow.revalidate()
            precLiveBallsRow.repaint()
        }

        val pStepsSig = pst.backtestResult?.let { bt ->
            "${bt.totalTested}|${bt.exactTargetCount}|${bt.steps.firstOrNull()?.drawDate}|${bt.steps.lastOrNull()?.drawDate}|${bt.steps.lastOrNull()?.predictedNumbers}"
        } ?: "NONE"
        if (pStepsSig != lastPrecStepsSig) {
            lastPrecStepsSig = pStepsSig
            precStepsModel.rowCount = 0
            pst.backtestResult?.steps?.forEach { step ->
                precStepsModel.addRow(
                    arrayOf<Any>(
                        step.stepIndex,
                        step.drawDate,
                        "${step.game} ${step.horaire}",
                        step.predictedNumbers.joinToString("-") { it.toString().padStart(2, '0') },
                        "Pos1=${step.actualPos1.toString().padStart(2, '0')} [${step.actualNumbers.joinToString("-") { it.toString().padStart(2, '0') }}]",
                        if (step.exactTargetHit) "EXACT (${step.matchCount})" else "${step.matchCount} (${step.matchedNumbers.joinToString(",")})",
                        String.format(Locale.FRANCE, "%.5f", step.stepBrierScore),
                        if (step.causalVerified) "OUI" else "NON"
                    )
                )
            }
        }

        val pBlocksSig = pst.backtestResult?.dailyBlocks?.joinToString("|") {
            "${it.blockLabel}:${it.drawsTested}:${it.exactTargetHits}:${it.penultimateExact}:${it.isQualifyingBlock}"
        } ?: "NONE"
        if (pBlocksSig != lastPrecBlocksSig) {
            lastPrecBlocksSig = pBlocksSig
            precDailyBlocksModel.rowCount = 0
            pst.backtestResult?.dailyBlocks?.forEach { b ->
                precDailyBlocksModel.addRow(
                    arrayOf<Any>(
                        b.blockLabel,
                        b.drawsTested,
                        b.exactTargetHits,
                        if (b.penultimateExact) "OUI" else "NON",
                        b.onePlusHits,
                        if (b.isQualifyingBlock) "OUI — QUALIFIÉ" else "NON"
                    )
                )
            }
        }

        val joined = pst.liveLogs.joinToString("\n")
        if (precLogsArea.text != joined) {
            precLogsArea.text = joined
            precLogsArea.caretPosition = precLogsArea.document.length
        }
    }

    private fun fmt3(v: Float): String = String.format(Locale.FRANCE, "%.3f", v)
    private fun fmt4(v: Double): String = String.format(Locale.FRANCE, "%.4f", v)

    private data class WinnerComboItem(val summary: SymbolicFormulaSummary) {
        val id: String get() = summary.id
        override fun toString(): String = "Palier ${summary.combinedStreak}J • Gén.${summary.generation} (${summary.slotSplitLabel})"
    }

    private data class FactoryDomainItem(val domain: MathTheoremDomain?, val label: String) {
        override fun toString(): String = label
    }

    companion object {
        fun installLookAndFeel() {
            AndroidSwingTheme.install()
        }
    }
}
