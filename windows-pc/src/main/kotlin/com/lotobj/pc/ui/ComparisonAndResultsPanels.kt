package com.lotobj.pc.ui

import com.lotobj.pc.engine.PcWorkstationUiState
import com.lotobj.pc.model.BacktestPeriod
import com.lotobj.pc.model.LotoDraw
import com.lotobj.pc.model.PredictionMethod
import com.lotobj.pc.model.ScientificReport
import com.lotobj.pc.util.LotoDateUtils
import java.awt.BorderLayout
import java.awt.CardLayout
import java.awt.Dimension
import java.awt.FlowLayout
import java.awt.Font
import java.awt.GridLayout
import java.util.Locale
import javax.swing.BorderFactory
import javax.swing.Box
import javax.swing.BoxLayout
import javax.swing.ButtonGroup
import javax.swing.JButton
import javax.swing.JLabel
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JSplitPane
import javax.swing.JTabbedPane
import javax.swing.JTable
import javax.swing.JTextArea
import javax.swing.JToggleButton
import javax.swing.border.EmptyBorder
import javax.swing.table.DefaultTableModel

/**
 * Onglet 1 : Résultats LNB exact (Bannière fenêtre active + Carte Dernier Tirage + Cartes avec Boules + Vue Tableau).
 */
class ResultsTabPanel(
    onFetchOnline: () -> Unit,
    onImportArchive: () -> Unit
) : JPanel(BorderLayout(10, 10)) {

    val fetchOnlineBtn = JButton("Rafraîchir / Télécharger LNB").apply {
        background = AndroidM3Colors.SecondaryContainer
        foreground = AndroidM3Colors.OnSecondaryContainer
        font = Font("Segoe UI", Font.BOLD, 12)
        addActionListener { onFetchOnline() }
    }
    val importArchiveBtn = JButton("Importer Archive CSV / JSON").apply {
        background = AndroidM3Colors.SurfaceElevated
        foreground = AndroidM3Colors.OnSurface
        font = Font("Segoe UI", Font.BOLD, 12)
        addActionListener { onImportArchive() }
    }
    private val cardsToggleBtn = JToggleButton("Cartes (Android)", true)
    private val tableToggleBtn = JToggleButton("Tableau Complet", false)
    private val titleLabel = JLabel("LNB Résultats (0)").apply {
        font = Font("Segoe UI", Font.BOLD, 17)
        foreground = AndroidM3Colors.OnBackground
    }
    private val activeWindowTitleLabel = JLabel("Historique Ciblé — Fenêtre active").apply {
        font = Font("Segoe UI", Font.BOLD, 13)
        foreground = AndroidM3Colors.OnPrimaryContainer
    }
    private val activeWindowSubLabel = JLabel("0 tirages actifs").apply {
        font = Font("Segoe UI", Font.PLAIN, 12)
        foreground = AndroidM3Colors.OnPrimaryContainer
    }

    private val featuredGameLabel = JLabel("En attente de résultats LNB...").apply {
        font = Font("Segoe UI", Font.BOLD, 18)
        foreground = AndroidM3Colors.OnSecondaryContainer
    }
    private val featuredDateLabel = JLabel("").apply {
        font = Font("Segoe UI", Font.PLAIN, 12)
        foreground = AndroidM3Colors.OnSecondaryContainer
    }
    private val featuredBallsRow = JPanel(FlowLayout(FlowLayout.LEFT, 12, 4)).apply { isOpaque = false }

    private val cardsListContainer = JPanel().apply {
        layout = BoxLayout(this, BoxLayout.Y_AXIS)
        background = AndroidM3Colors.Background
    }
    private val historyTableModel = object : DefaultTableModel(
        arrayOf("Date (Bénin)", "Jeu", "Horaire", "N°1", "N°2", "N°3", "N°4", "N°5"), 0
    ) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val historyTable = JTable(historyTableModel).apply { rowHeight = 25 }
    private val contentCards = JPanel(CardLayout())
    private var lastRenderKey = ""

    init {
        border = EmptyBorder(10, 10, 10, 10)
        background = AndroidM3Colors.Background

        val topStack = JPanel().apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            background = AndroidM3Colors.Background
        }

        val actionBar = JPanel(FlowLayout(FlowLayout.LEFT, 10, 4)).apply {
            background = AndroidM3Colors.Background
            add(titleLabel)
            add(Box.createHorizontalStrut(12))
            add(fetchOnlineBtn)
            add(importArchiveBtn)
            val grp = ButtonGroup()
            grp.add(cardsToggleBtn)
            grp.add(tableToggleBtn)
            add(Box.createHorizontalStrut(12))
            add(cardsToggleBtn)
            add(tableToggleBtn)
        }

        val windowBanner = AndroidSwingTheme.makeCard(
            bg = AndroidM3Colors.PrimaryContainer,
            borderColor = AndroidM3Colors.Primary,
            radius = 14
        ).apply {
            layout = BorderLayout(4, 2)
            val col = JPanel().apply {
                layout = BoxLayout(this, BoxLayout.Y_AXIS)
                isOpaque = false
                add(activeWindowTitleLabel)
                add(activeWindowSubLabel)
            }
            add(col, BorderLayout.CENTER)
            maximumSize = Dimension(Int.MAX_VALUE, 64)
        }

        val featuredCard = AndroidSwingTheme.makeCard(
            bg = AndroidM3Colors.SecondaryContainer,
            borderColor = AndroidM3Colors.Primary,
            radius = 18
        ).apply {
            layout = BorderLayout(8, 8)
            val hdr = JPanel().apply {
                layout = BoxLayout(this, BoxLayout.Y_AXIS)
                isOpaque = false
                add(JLabel("DERNIER TIRAGE").apply {
                    font = Font("Segoe UI", Font.BOLD, 10)
                    foreground = AndroidM3Colors.PrimaryContainer
                })
                add(featuredGameLabel)
            }
            add(hdr, BorderLayout.NORTH)
            add(featuredBallsRow, BorderLayout.CENTER)
            add(featuredDateLabel, BorderLayout.SOUTH)
            maximumSize = Dimension(Int.MAX_VALUE, 145)
        }

        topStack.add(actionBar)
        topStack.add(Box.createVerticalStrut(6))
        topStack.add(windowBanner)
        topStack.add(Box.createVerticalStrut(8))
        topStack.add(featuredCard)
        topStack.add(Box.createVerticalStrut(8))
        add(topStack, BorderLayout.NORTH)

        val cardsScroll = JScrollPane(cardsListContainer).apply {
            verticalScrollBar.unitIncrement = 18
            border = BorderFactory.createEmptyBorder()
        }
        contentCards.add(cardsScroll, "CARDS")
        contentCards.add(JScrollPane(historyTable), "TABLE")
        add(contentCards, BorderLayout.CENTER)

        cardsToggleBtn.addActionListener { (contentCards.layout as CardLayout).show(contentCards, "CARDS") }
        tableToggleBtn.addActionListener { (contentCards.layout as CardLayout).show(contentCards, "TABLE") }
    }

    fun render(st: PcWorkstationUiState) {
        fetchOnlineBtn.isEnabled = !st.isFetchingOnline
        titleLabel.text = "LNB Résultats (${st.allDraws.size})"
        val daysText = when {
            st.selectedPeriod == BacktestPeriod.ALL_HISTORY && st.customPeriodDays == null -> "Historique Complet"
            (st.customPeriodDays ?: st.selectedPeriod.days) == 1 -> "1 Jour (6 derniers tirages)"
            else -> "${st.customPeriodDays ?: st.selectedPeriod.days} Jours à la date J"
        }
        activeWindowTitleLabel.text = "Historique Ciblé — $daysText"
        val w = st.threeWeekWindow
        val activeDraws = w.draws.asReversed().ifEmpty { st.allDraws }
        activeWindowSubLabel.text = "${w.startLabel} → ${w.endLabel} (${w.anchorTimeLabel} Bénin) • ${activeDraws.size} tirages actifs (${st.allDraws.size} total)"

        val renderKey = "${st.allDraws.size}_${st.allDraws.firstOrNull()?.uniqueId}_${activeDraws.size}"
        if (renderKey == lastRenderKey) return
        lastRenderKey = renderKey

        val latest = activeDraws.firstOrNull() ?: st.allDraws.firstOrNull()
        featuredBallsRow.removeAll()
        if (latest != null) {
            featuredGameLabel.text = if (latest.time.isNotBlank()) "${latest.game} ${latest.time}" else latest.game
            featuredDateLabel.text = LotoDateUtils.formatFrenchDayAndDate(latest.timestamp)
            listOf(latest.num1, latest.num2, latest.num3, latest.num4, latest.num5).forEach { n ->
                featuredBallsRow.add(LotoBallBadge(n, BallStyle.FEATURED, 48))
            }
        } else {
            featuredGameLabel.text = "Aucun résultat en cache"
            featuredDateLabel.text = "Cliquez sur « Rafraîchir / Télécharger LNB » ou importez une archive CSV/JSON."
        }
        featuredBallsRow.revalidate()
        featuredBallsRow.repaint()

        cardsListContainer.removeAll()
        activeDraws.drop(1).forEach { draw ->
            cardsListContainer.add(buildDrawCard(draw))
            cardsListContainer.add(Box.createVerticalStrut(6))
        }
        cardsListContainer.revalidate()
        cardsListContainer.repaint()

        historyTableModel.rowCount = 0
        st.allDraws.take(1000).forEach { d ->
            historyTableModel.addRow(arrayOf<Any>(d.date, d.game, d.time, d.num1, d.num2, d.num3, d.num4, d.num5))
        }
    }

    private fun buildDrawCard(d: LotoDraw): JPanel {
        val card = AndroidSwingTheme.makeCard(AndroidM3Colors.SurfaceVariant, AndroidM3Colors.Outline, 14).apply {
            layout = BorderLayout(12, 4)
            maximumSize = Dimension(Int.MAX_VALUE, 84)
        }
        val left = JPanel().apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            isOpaque = false
            add(JLabel("● ${d.game} ${d.time}".trim()).apply {
                font = Font("Segoe UI", Font.BOLD, 13)
                foreground = AndroidM3Colors.Primary
            })
            val balls = JPanel(FlowLayout(FlowLayout.LEFT, 8, 4)).apply { isOpaque = false }
            listOf(d.num1, d.num2, d.num3, d.num4, d.num5).forEach { n ->
                balls.add(LotoBallBadge(n, BallStyle.NORMAL, 34))
            }
            add(balls)
        }
        val right = JLabel(LotoDateUtils.formatFrenchDayAndDate(d.timestamp)).apply {
            font = Font("Segoe UI", Font.PLAIN, 12)
            foreground = AndroidM3Colors.OnSurfaceVariant
        }
        card.add(left, BorderLayout.CENTER)
        card.add(right, BorderLayout.EAST)
        return card
    }
}

/**
 * Onglet 4 : Labo Scientifique clair ("comme de l'eau à boire") :
 * - Verdict global OUI / NON sans mensonge
 * - Modèles IA & Chercheur conservés audités sans triche + bouton "Utiliser ce modèle"
 * - 13 Méthodes statistiques de référence avec colonnes OUI / NON
 * - Détails techniques avancés repliables (Monte Carlo, Split, Cube90, Brier/LogLoss, Paires Complémentaires)
 */
class ScientificLabTabPanel(
    onRunComparison: () -> Unit,
    onPredictWithMethod: (PredictionMethod) -> Unit,
    onUseSavedModel: (String) -> Unit = {}
) : JPanel(BorderLayout(10, 10)) {

    val runComparisonBtn = JButton("Auditer mes Modèles & 13 Méthodes").apply {
        background = AndroidM3Colors.SecondaryContainer
        foreground = AndroidM3Colors.OnSecondaryContainer
        font = Font("Segoe UI", Font.BOLD, 13)
        addActionListener { onRunComparison() }
    }
    private val useSavedModelBtn = JButton("Utiliser le modèle IA sélectionné (Backtest & Prédiction)").apply {
        background = AndroidM3Colors.Tertiary
        foreground = AndroidM3Colors.OnTertiary
        font = Font("Segoe UI", Font.BOLD, 12)
        isEnabled = false
    }
    private val predictWithSelectedBtn = JButton("Prédire avec la méthode statistique sélectionnée").apply {
        background = AndroidM3Colors.Primary
        foreground = AndroidM3Colors.OnPrimary
        font = Font("Segoe UI", Font.BOLD, 12)
        isEnabled = false
    }
    private val toggleTechDetailsBtn = JToggleButton("Afficher Détails Techniques (Monte Carlo, Cube90, Brier, Paires)", false).apply {
        font = Font("Segoe UI", Font.BOLD, 11)
    }

    private val winIndepBtn = JToggleButton("Test Indépendant", true)
    private val winValBtn = JToggleButton("Validation", false)
    private val winDevBtn = JToggleButton("Développement", false)
    private var selectedWindow = 2

    private val verdictBadgeLabel = JLabel("VERDICT : EN ATTENTE D'AUDIT").apply {
        font = Font("Segoe UI", Font.BOLD, 15)
        foreground = AndroidM3Colors.Gold
    }
    private val verdictReasonLabel = JTextArea(
        "Cliquez sur « Auditer mes Modèles & 13 Méthodes » pour vérifier sans triche quels modèles réussissent : " +
            "≥ 3 tirages/jour à ≥ 2 boules exactes dont l'avant-dernier tirage ≥ 2."
    ).apply {
        isEditable = false
        isOpaque = false
        lineWrap = true
        wrapStyleWord = true
        font = Font("Segoe UI", Font.PLAIN, 12)
        foreground = AndroidM3Colors.OnSecondaryContainer
    }
    private val verdictRuleLabel = JLabel("Règle obligatoire : ≥ 3 tirages/jour avec ≥ 2 boules exactes dont l'avant-dernier tirage ≥ 2").apply {
        font = Font("Segoe UI", Font.BOLD, 11)
        foreground = AndroidM3Colors.PrimaryContainer
    }

    private val savedModelsTableModel = object : DefaultTableModel(
        arrayOf(
            "Modèle IA Conservé", "Fenêtre Auditée", "Objectif Validé",
            "Avant-dernier ≥2", "Jours Validés", "Tirages ≥2",
            "Bat le hasard ?", "Zéro triche", "Moy./tir", "Verdict"
        ),
        0
    ) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val savedModelsTable = JTable(savedModelsTableModel).apply { rowHeight = 25 }
    private var displayedSavedModelIds: List<String> = emptyList()

    private val methodsTableModel = object : DefaultTableModel(
        arrayOf(
            "Objectif Validé", "Repère", "Méthode Statistique", "Famille",
            "Tirages", "Jours Validés", "Avant-dernier ≥2", "Tirages ≥2",
            "≥3", "Moy./tir", "Bat le hasard ?"
        ),
        0
    ) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val methodsTable = JTable(methodsTableModel).apply { rowHeight = 24 }
    private var displayedMethods: List<PredictionMethod> = emptyList()

    // Panneau technique repliable
    private val splitSummaryLabel = JLabel("Séparation chronologique : —").apply {
        font = Font("Segoe UI", Font.PLAIN, 12)
        foreground = AndroidM3Colors.OnSurface
    }
    private val mcBaselineLabel = JLabel("Référence Aléatoire Monte Carlo : —").apply {
        font = Font("Segoe UI", Font.BOLD, 12)
        foreground = AndroidM3Colors.OnTertiaryContainer
    }
    private val cube90AuditLabel = JLabel("").apply {
        font = Font("Consolas", Font.PLAIN, 11)
        foreground = AndroidM3Colors.PrimaryContainer
    }
    private val techMetricsTableModel = object : DefaultTableModel(
        arrayOf("Méthode", "Brier", "LogLoss", "ECE", "p-val", "Bonferroni"), 0
    ) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val techMetricsTable = JTable(techMetricsTableModel).apply { rowHeight = 22 }
    private val pairsTableModel = object : DefaultTableModel(
        arrayOf("Méthode A", "Méthode B", "Chevauchement Jaccard", "Corrélation Erreurs", "Tirages Complémentaires"), 0
    ) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val pairsTable = JTable(pairsTableModel).apply { rowHeight = 22 }
    private val techDetailsPanel = JPanel(BorderLayout(8, 8)).apply {
        background = AndroidM3Colors.Background
        isVisible = false
    }
    private var latestReport: ScientificReport? = null

    init {
        border = EmptyBorder(10, 10, 10, 10)
        background = AndroidM3Colors.Background

        val northStack = JPanel().apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            background = AndroidM3Colors.Background
        }

        val actionRow = JPanel(FlowLayout(FlowLayout.LEFT, 10, 4)).apply {
            background = AndroidM3Colors.Background
            add(JLabel("Labo Scientifique (Audit OUI / NON • Zéro Triche)").apply {
                font = Font("Segoe UI", Font.BOLD, 16)
                foreground = AndroidM3Colors.OnBackground
            })
            add(Box.createHorizontalStrut(8))
            add(runComparisonBtn)
            add(useSavedModelBtn)
            add(predictWithSelectedBtn)
            add(toggleTechDetailsBtn)
        }

        val verdictCard = AndroidSwingTheme.makeCard(AndroidM3Colors.SecondaryContainer, AndroidM3Colors.Primary, 14).apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            add(verdictBadgeLabel)
            add(Box.createVerticalStrut(4))
            add(verdictReasonLabel)
            add(Box.createVerticalStrut(4))
            add(verdictRuleLabel)
        }

        val windowTabsRow = JPanel(FlowLayout(FlowLayout.LEFT, 8, 4)).apply {
            background = AndroidM3Colors.Background
            add(JLabel("Fenêtre des 13 méthodes classiques :").apply { foreground = AndroidM3Colors.OnSurfaceVariant })
            val grp = ButtonGroup()
            listOf(winIndepBtn, winValBtn, winDevBtn).forEach { btn ->
                grp.add(btn)
                add(btn)
            }
        }

        northStack.add(actionRow)
        northStack.add(Box.createVerticalStrut(6))
        northStack.add(verdictCard)
        northStack.add(Box.createVerticalStrut(4))
        northStack.add(windowTabsRow)
        add(northStack, BorderLayout.NORTH)

        val splitAndMcRow = JPanel(GridLayout(1, 2, 10, 0)).apply {
            background = AndroidM3Colors.Background
            add(AndroidSwingTheme.makeCard(AndroidM3Colors.SurfaceVariant, AndroidM3Colors.Outline, 12).apply {
                layout = BorderLayout()
                add(splitSummaryLabel, BorderLayout.CENTER)
            })
            add(AndroidSwingTheme.makeCard(AndroidM3Colors.TertiaryContainer, AndroidM3Colors.Tertiary, 12).apply {
                layout = BorderLayout()
                add(mcBaselineLabel, BorderLayout.CENTER)
            })
        }
        val techTop = JPanel(BorderLayout(4, 4)).apply {
            background = AndroidM3Colors.Background
            add(splitAndMcRow, BorderLayout.CENTER)
            add(cube90AuditLabel, BorderLayout.SOUTH)
        }
        val techTabs = JTabbedPane().apply {
            addTab("Métriques Brier / LogLoss / p-val", JScrollPane(techMetricsTable))
            addTab("Paires Complémentaires", JScrollPane(pairsTable))
        }
        techDetailsPanel.add(techTop, BorderLayout.NORTH)
        techDetailsPanel.add(techTabs, BorderLayout.CENTER)
        techDetailsPanel.preferredSize = Dimension(900, 230)

        val mainTablesSplit = JSplitPane(
            JSplitPane.VERTICAL_SPLIT,
            JScrollPane(savedModelsTable).apply {
                border = BorderFactory.createTitledBorder("1. Modèles IA & Chercheur Conservés (Audit OUI / NON sans triche — sélectionnez et cliquez Utiliser)")
                preferredSize = Dimension(900, 170)
            },
            JScrollPane(methodsTable).apply {
                border = BorderFactory.createTitledBorder("2. 13 Méthodes Statistiques (Benchmark de référence OUI / NON)")
            }
        ).apply {
            resizeWeight = 0.36
        }

        val centerWrapper = JPanel(BorderLayout(6, 6)).apply {
            background = AndroidM3Colors.Background
            add(mainTablesSplit, BorderLayout.CENTER)
            add(techDetailsPanel, BorderLayout.SOUTH)
        }
        add(centerWrapper, BorderLayout.CENTER)

        winIndepBtn.addActionListener { selectedWindow = 2; refreshTablesFromReport() }
        winValBtn.addActionListener { selectedWindow = 1; refreshTablesFromReport() }
        winDevBtn.addActionListener { selectedWindow = 0; refreshTablesFromReport() }

        toggleTechDetailsBtn.addActionListener {
            techDetailsPanel.isVisible = toggleTechDetailsBtn.isSelected
            revalidate()
            repaint()
        }

        savedModelsTable.selectionModel.addListSelectionListener {
            useSavedModelBtn.isEnabled = displayedSavedModelIds.isNotEmpty()
        }
        useSavedModelBtn.addActionListener {
            val row = savedModelsTable.selectedRow
            val id = displayedSavedModelIds.getOrNull(row) ?: displayedSavedModelIds.firstOrNull() ?: return@addActionListener
            onUseSavedModel(id)
        }

        methodsTable.selectionModel.addListSelectionListener {
            predictWithSelectedBtn.isEnabled = methodsTable.selectedRow in displayedMethods.indices
        }
        predictWithSelectedBtn.addActionListener {
            val row = methodsTable.selectedRow
            val method = displayedMethods.getOrNull(row) ?: latestReport?.recommendedMethod ?: return@addActionListener
            onPredictWithMethod(method)
        }
    }

    fun render(st: PcWorkstationUiState) {
        runComparisonBtn.isEnabled = !st.isComparing && st.allDraws.size >= 8
        runComparisonBtn.text = if (st.isComparing) {
            "Audit Modèles & 13 Méthodes (${st.monteCarloSims} sim.) en cours..."
        } else {
            "Auditer mes Modèles & 13 Méthodes (${st.allDraws.size} tir. • ${st.monteCarloSims} sim. MC)"
        }

        val r = st.scientificReport
        if (r != latestReport) {
            latestReport = r
            refreshTablesFromReport()
        }
    }

    private fun refreshTablesFromReport() {
        val r = latestReport ?: return
        if (r.anyObjectiveValidated) {
            verdictBadgeLabel.text = "OUI — OBJECTIF VALIDÉ SUR ${r.periodLabel.ifBlank { "LA FENÊTRE ACTIVE" }}"
            verdictBadgeLabel.foreground = AndroidM3Colors.Tertiary
        } else {
            verdictBadgeLabel.text = "NON — AUCUN MODÈLE NE VALIDE L'OBJECTIF SUR ${r.periodLabel.ifBlank { "CETTE FENÊTRE" }}"
            verdictBadgeLabel.foreground = AndroidM3Colors.ErrorRed
        }
        verdictReasonLabel.text = r.recommendationReason
        verdictRuleLabel.text = "Règle obligatoire : ≥ 3 tirages/jour avec ≥ 2 boules exactes dont l'avant-dernier tirage ≥ 2 • Fenêtre : ${r.periodLabel}"
        cube90AuditLabel.text = r.cube90AuditSummary

        // Modèles IA conservés
        savedModelsTableModel.rowCount = 0
        displayedSavedModelIds = r.savedModelAudits.map { it.id }
        if (r.savedModelAudits.isEmpty()) {
            savedModelsTableModel.addRow(
                arrayOf<Any>(
                    "Aucun modèle IA enregistré", "—", "NON", "NON", "0/0", "0/0", "NON", "OUI", "—",
                    "Lancez Moteur 1 pour sauvegarder des modèles"
                )
            )
            useSavedModelBtn.isEnabled = false
        } else {
            r.savedModelAudits.forEach { a ->
                savedModelsTableModel.addRow(
                    arrayOf<Any>(
                        "${a.displayName} (${a.slotSplitLabel})",
                        a.windowName,
                        if (a.objectiveValidated) "OUI" else "NON",
                        if (a.penultimateOkDaysCount > 0) "OUI (${a.penultimateOkDaysCount}/${a.totalDays})" else "NON (0/${a.totalDays})",
                        "${a.qualifyingDaysCount}/${a.totalDays}",
                        "${a.twoPlusDraws}/${a.testedDraws}",
                        if (a.beatsRandom) "OUI" else "NON",
                        if (a.causalVerified) "OUI" else "NON",
                        String.format(Locale.FRANCE, "%.2f", a.averageMatches),
                        a.verdictReason
                    )
                )
            }
            useSavedModelBtn.isEnabled = true
        }

        val s = r.splitInfo
        val periodStr = r.periodLabel.ifBlank { "Fenêtre active" }
        splitSummaryLabel.text = "<html><b>Séparation $periodStr (${s.totalDrawsAvailable} tir.)</b><br>" +
            "Warmup: ${s.warmupDrawsCount} • Dév (${s.devDrawsCount}): ${s.devDateRange}<br>" +
            "Val (${s.valDrawsCount}): ${s.valDateRange} • <b>Test Indép. (${s.indepTestDrawsCount}): ${s.indepTestDateRange}</b></html>"

        winIndepBtn.text = "Test Indép. (${s.indepTestDrawsCount})"
        winValBtn.text = "Validation (${s.valDrawsCount})"
        winDevBtn.text = "Dév. (${s.devDrawsCount})"

        val mc = when (selectedWindow) {
            0 -> r.monteCarloDev
            1 -> r.monteCarloVal
            else -> r.monteCarloIndepTest
        }
        mcBaselineLabel.text = String.format(
            Locale.FRANCE,
            "<html><b>Référence Monte Carlo (%d sim., graine=%d)</b><br>" +
                "Moy: %.3f (IC 95%% [%.3f–%.3f]) • Taux ≥1: %.1f%%<br>" +
                "≥2 bons: %.1f (IC [%d,%d]) • Jours qual.: %.2f (IC [%d,%d])</html>",
            mc.simulationsCount, mc.seed, mc.meanAvgMatches, mc.avgMatchesCiLow, mc.avgMatchesCiHigh,
            mc.meanOnePlusRate * 100f, mc.meanTwoPlusCount, mc.twoPlusCiLow, mc.twoPlusCiHigh,
            mc.meanQualifyingDays, mc.qualifyingDaysCiLow, mc.qualifyingDaysCiHigh
        )

        val entries = when (selectedWindow) {
            0 -> r.devComparisons
            1 -> r.valComparisons
            else -> r.indepTestComparisons
        }
        displayedMethods = entries.map { it.method }
        methodsTableModel.rowCount = 0
        techMetricsTableModel.rowCount = 0
        entries.forEach { e ->
            val mark = when {
                e.objectiveValidated -> "★ Validé"
                e.isRecommendedParsimonious -> "Repère Stat."
                else -> "—"
            }
            methodsTableModel.addRow(
                arrayOf<Any>(
                    if (e.objectiveValidated) "OUI" else "NON",
                    mark,
                    e.method.displayName,
                    e.method.family,
                    e.testedCount,
                    "${e.qualifyingDaysCount}/${e.totalDays}",
                    if (e.penultimateOkDaysCount > 0) "OUI (${e.penultimateOkDaysCount}/${e.totalDays})" else "NON (0/${e.totalDays})",
                    e.twoPlusCount,
                    e.threePlusCount,
                    String.format(Locale.FRANCE, "%.3f", e.averageMatches),
                    if (e.adjustedSignificant) "OUI" else "NON"
                )
            )
            techMetricsTableModel.addRow(
                arrayOf<Any>(
                    e.method.displayName,
                    String.format(Locale.FRANCE, "%.5f", e.brierScore),
                    String.format(Locale.FRANCE, "%.4f", e.bernoulliLogLoss),
                    String.format(Locale.FRANCE, "%.4f", e.calibrationError),
                    String.format(Locale.FRANCE, "%.3f", e.pValueVsRandom),
                    if (e.adjustedSignificant) "OUI" else "Non"
                )
            )
        }
        predictWithSelectedBtn.isEnabled = entries.isNotEmpty()

        pairsTableModel.rowCount = 0
        r.complementaryPairs.forEach { p ->
            pairsTableModel.addRow(
                arrayOf<Any>(
                    p.methodA.shortLabel,
                    p.methodB.shortLabel,
                    String.format(Locale.FRANCE, "%.1f%%", p.jaccardOverlap * 100f),
                    String.format(Locale.FRANCE, "%.2f", p.errorCorrelation),
                    "${p.complementaryHitsCount} tirages"
                )
            )
        }
    }
}
