package com.lotobj.pc.engine

import com.lotobj.pc.model.BacktestPeriod
import com.lotobj.pc.model.ComputeMode
import com.lotobj.pc.model.LotoDraw
import com.lotobj.pc.model.NextDrawTarget
import com.lotobj.pc.model.PrecisionBacktestResult
import com.lotobj.pc.model.PrecisionBacktestStep
import com.lotobj.pc.model.PrecisionDailyBlockSummary
import com.lotobj.pc.model.PrecisionFeature
import com.lotobj.pc.model.PrecisionGenome
import com.lotobj.pc.model.PrecisionLedgerRecord
import com.lotobj.pc.model.PrecisionLivePrediction
import com.lotobj.pc.model.PrecisionPendingPrediction
import com.lotobj.pc.model.PrecisionStageStats
import com.lotobj.pc.model.PrecisionTargetMode
import com.lotobj.pc.model.PrecisionTransform
import com.lotobj.pc.util.LotoDateUtils
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put
import java.util.Locale
import java.util.concurrent.Callable
import java.util.concurrent.ForkJoinPool
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sign
import kotlin.math.sqrt
import kotlin.math.tanh
import kotlin.random.Random

class PrecisionFeatureMatrix(
    val rows: Array<DoubleArray>
) {
    fun get(featIdx: Int, number: Int): Double = rows[featIdx][number]
}

data class PrecisionRawStep(
    val stepIndex: Int,
    val draw: LotoDraw,
    val actualPos1: Int,
    val actualNumbers: List<Int>,
    val actualSet: Set<Int>,
    val actualMask: BooleanArray,
    val strictPastSize: Int,
    val maxPastTs: Long,
    val dayKey: String,
    val horaire: String,
    val game: String,
    val dayEpoch: Long,
    val matrix: PrecisionFeatureMatrix
)

data class PrecisionCausalCache(
    val cacheKey: String,
    val periodLabel: String,
    val dateRangeLabel: String,
    val steps: List<PrecisionRawStep>,
    val discEnd: Int,
    val rollEnd: Int,
    val confEnd: Int
) {
    val discoverySteps: List<PrecisionRawStep> get() = if (discEnd > 0) steps.subList(0, discEnd) else emptyList()
    val rollingSteps: List<PrecisionRawStep> get() = if (rollEnd > discEnd) steps.subList(discEnd, rollEnd) else emptyList()
    val confirmationSteps: List<PrecisionRawStep> get() = if (confEnd > rollEnd) steps.subList(rollEnd, confEnd) else emptyList()
    val sealedSteps: List<PrecisionRawStep> get() = if (steps.size > confEnd) steps.subList(confEnd, steps.size) else emptyList()
}

data class PrecisionCandidateEval(
    val genome: PrecisionGenome,
    val discStats: PrecisionStageStats,
    val rollStats: PrecisionStageStats,
    val confStats: PrecisionStageStats? = null,
    val sealedStats: PrecisionStageStats? = null,
    val stabilityPassed: Boolean,
    val fitnessScore: Double,
    val status: String,
    val pValueRaw: Double = 1.0,
    val pValueAdjusted: Double = 1.0,
    val consolidatedStreak: Int = 0
)

data class PrecisionEvolutionState(
    val mode: PrecisionTargetMode,
    val generation: Int,
    val experimentsCount: Int,
    val confirmationTestsCount: Int,
    val sessionSalt: Long,
    val population: List<PrecisionGenome>,
    val elites: List<PrecisionCandidateEval> = emptyList(),
    val testedConfirmationSigs: MutableSet<String> = LinkedHashSet(),
    val seenSignatures: MutableSet<String> = LinkedHashSet(),
    val validatedCandidate: PrecisionCandidateEval? = null,
    val stagnationCount: Int = 0,
    val bestFitnessOverall: Double = Double.NEGATIVE_INFINITY
)

data class PrecisionGenResult(
    val nextState: PrecisionEvolutionState,
    val bestTrainEval: PrecisionCandidateEval?,
    val winningEval: PrecisionCandidateEval?,
    val auditedEval: PrecisionCandidateEval?,
    val generationNote: String,
    val evaluationsPerSec: Double,
    val gpuBatchMs: Double = 0.0,
    val cpuBatchMs: Double = 0.0,
    val gpuErrorNote: String = ""
)

/**
 * Moteur de Recherche Précision 100 % INDÉPENDANT :
 * - Mode 1 (`FIRST_BALL`) : prédit uniquement le 1er numéro tiré (`draw.num1`).
 * - Mode 2 (`TWO_NUMBERS`) : prédit 2 numéros exacts par tirage.
 * - Strictement causal (`maxPastTs < target.timestamp`), zéro bruit aléatoire (`exploreNoise = 0.0`),
 *   1 Jour = 6 tirages glissants (`00H, 8H, 11H, 14H, 18H, 20H`), validation 4 étapes + registre append-only.
 */
object PrecisionTargetEngine {

    private const val MAX_SEEN_SIGS = 150_000

    fun extractFeatures(strictPast: List<LotoDraw>, targetHoraire: String): PrecisionFeatureMatrix {
        val fCount = PrecisionFeature.entries.size
        val rows = Array(fCount) { DoubleArray(91) }
        val total = strictPast.size.coerceAtLeast(1)

        val pos1Counts = IntArray(91)
        val pos1W5 = IntArray(91)
        val pos1W15 = IntArray(91)
        val pos1Hor = IntArray(91)
        val pos1LastIdx = IntArray(91) { -1 }

        val anyCounts = IntArray(91)
        val anyW5 = IntArray(91)
        val anyW15 = IntArray(91)
        val anyHor = IntArray(91)
        val anyLastIdx = IntArray(91) { -1 }

        val pos1Transition = IntArray(91)
        val coPrevNum1 = IntArray(91)
        val coLastRaw = IntArray(91)

        val lastDraw = strictPast.lastOrNull()
        val prevPos1 = lastDraw?.num1?.takeIf { it in 1..90 } ?: -1
        val lastAnySet = if (lastDraw != null) {
            listOf(lastDraw.num1, lastDraw.num2, lastDraw.num3, lastDraw.num4, lastDraw.num5).filter { it in 1..90 }.toSet()
        } else emptySet()

        val w5Start = (strictPast.size - 5).coerceAtLeast(0)
        val w15Start = (strictPast.size - 15).coerceAtLeast(0)
        var horCount = 0
        var prevP1Transitions = 0
        var lastSetTransitions = 0

        for (i in strictPast.indices) {
            val d = strictPast[i]
            val sameH = LotoDateUtils.normalizeHoraire(d).equals(targetHoraire, ignoreCase = true)
            if (sameH) horCount++

            val p1 = d.num1
            if (p1 in 1..90) {
                pos1Counts[p1]++
                if (i >= w5Start) pos1W5[p1]++
                if (i >= w15Start) pos1W15[p1]++
                if (sameH) pos1Hor[p1]++
                pos1LastIdx[p1] = i
                if (i > 0 && strictPast[i - 1].num1 == prevPos1 && prevPos1 in 1..90) {
                    pos1Transition[p1]++
                    prevP1Transitions++
                }
            }

            val nums = listOf(d.num1, d.num2, d.num3, d.num4, d.num5)
            val prevDraw = if (i > 0) strictPast[i - 1] else null
            val prevHadLastAny = prevDraw != null && listOf(prevDraw.num1, prevDraw.num2, prevDraw.num3, prevDraw.num4, prevDraw.num5).any { it in lastAnySet }
            val prevWasPrevP1 = prevDraw != null && prevDraw.num1 == prevPos1 && prevPos1 in 1..90
            if (prevHadLastAny) lastSetTransitions++

            for (num in nums) {
                if (num !in 1..90) continue
                anyCounts[num]++
                if (i >= w5Start) anyW5[num]++
                if (i >= w15Start) anyW15[num]++
                if (sameH) anyHor[num]++
                anyLastIdx[num] = i
                if (prevWasPrevP1) coPrevNum1[num]++
                if (prevHadLastAny) coLastRaw[num]++
            }
        }

        val horDenom = horCount.coerceAtLeast(1).toDouble()
        val w5Denom = (strictPast.size - w5Start).coerceAtLeast(1).toDouble()
        val w15Denom = (strictPast.size - w15Start).coerceAtLeast(1).toDouble()
        val trDenom = prevP1Transitions.coerceAtLeast(1).toDouble()
        val coLastDenom = lastSetTransitions.coerceAtLeast(1).toDouble()

        for (n in 1..90) {
            rows[PrecisionFeature.X_NORM.ordinal][n] = n / 90.0
            rows[PrecisionFeature.POS1_FREQ_GLOBAL.ordinal][n] = pos1Counts[n].toDouble() / total
            rows[PrecisionFeature.POS1_FREQ_W5.ordinal][n] = pos1W5[n].toDouble() / w5Denom
            rows[PrecisionFeature.POS1_FREQ_W15.ordinal][n] = pos1W15[n].toDouble() / w15Denom
            val p1Gap = if (pos1LastIdx[n] >= 0) (strictPast.size - 1 - pos1LastIdx[n]) else strictPast.size
            rows[PrecisionFeature.POS1_GAP_NORM.ordinal][n] = p1Gap.toDouble() / (total + 1)
            rows[PrecisionFeature.POS1_HORAIRE_FREQ.ordinal][n] = pos1Hor[n].toDouble() / horDenom
            rows[PrecisionFeature.POS1_PREV_TRANSITION.ordinal][n] = pos1Transition[n].toDouble() / trDenom
            rows[PrecisionFeature.POS1_SAME_LAG1.ordinal][n] = if (n == prevPos1) 1.0 else 0.0

            rows[PrecisionFeature.ANY_FREQ_GLOBAL.ordinal][n] = anyCounts[n].toDouble() / (total * 5.0)
            rows[PrecisionFeature.ANY_FREQ_W5.ordinal][n] = anyW5[n].toDouble() / (w5Denom * 5.0)
            rows[PrecisionFeature.ANY_FREQ_W15.ordinal][n] = anyW15[n].toDouble() / (w15Denom * 5.0)
            val aGap = if (anyLastIdx[n] >= 0) (strictPast.size - 1 - anyLastIdx[n]) else strictPast.size
            rows[PrecisionFeature.ANY_GAP_NORM.ordinal][n] = aGap.toDouble() / (total + 1)
            rows[PrecisionFeature.ANY_HORAIRE_FREQ.ordinal][n] = anyHor[n].toDouble() / (horDenom * 5.0)
            rows[PrecisionFeature.CO_PREV_NUM1.ordinal][n] = coPrevNum1[n].toDouble() / (trDenom * 5.0)
            rows[PrecisionFeature.CO_LAST_RAW.ordinal][n] = coLastRaw[n].toDouble() / (coLastDenom * 5.0)

            rows[PrecisionFeature.BASE7_NORM.ordinal][n] = ((n - 1) % 7) / 6.0
            val lucky = ((n - 1) % 9) + 1
            rows[PrecisionFeature.LUCKY_ROOT.ordinal][n] = lucky / 9.0
            rows[PrecisionFeature.AFFINE_MOD90.ordinal][n] = (((7 * n + 9) % 90)) / 89.0
        }
        return PrecisionFeatureMatrix(rows)
    }

    /**
     * Extraction incrémentale O(T) exacte (parité mathématique totale avec extractFeatures)
     * avec contrainte stricte timestamp < cible.
     */
    fun extractPrecisionMatrixSequence(
        chronological: List<LotoDraw>,
        targetIndices: IntArray = IntArray(chronological.size) { it }
    ): Map<Int, PrecisionFeatureMatrix> {
        val nTotal = chronological.size
        if (nTotal == 0 || targetIndices.isEmpty()) return emptyMap()
        val sortedTargets = targetIndices.distinct().filter { it in 0 until nTotal }.sorted()
        if (sortedTargets.isEmpty()) return emptyMap()
        val maxTargetIdx = sortedTargets.last()
        val fCount = PrecisionFeature.entries.size

        val drawPos1 = IntArray(nTotal) { i -> chronological[i].num1 }
        val drawBallRaw = IntArray(nTotal * 5)
        val drawLoMask = LongArray(nTotal)
        val drawHiMask = LongArray(nTotal)
        val horaireIdMap = HashMap<String, Int>(12)
        val horaireIds = IntArray(nTotal) { i ->
            horaireIdMap.getOrPut(LotoDateUtils.normalizeHoraire(chronological[i])) { horaireIdMap.size }
        }
        val numHoraires = horaireIdMap.size.coerceAtLeast(1)

        for (i in 0 until nTotal) {
            val d = chronological[i]
            val off = i * 5
            drawBallRaw[off] = d.num1
            drawBallRaw[off + 1] = d.num2
            drawBallRaw[off + 2] = d.num3
            drawBallRaw[off + 3] = d.num4
            drawBallRaw[off + 4] = d.num5
            var lo = 0L
            var hi = 0L
            for (k in 0 until 5) {
                val v = drawBallRaw[off + k]
                if (v in 1..64) lo = lo or (1L shl (v - 1)) else if (v in 65..90) hi = hi or (1L shl (v - 65))
            }
            drawLoMask[i] = lo
            drawHiMask[i] = hi
        }

        val xNorm = DoubleArray(91) { n -> n / 90.0 }
        val base7Norm = DoubleArray(91) { n -> ((n - 1) % 7) / 6.0 }
        val luckyNorm = DoubleArray(91) { n -> (((n - 1) % 9) + 1) / 9.0 }
        val affineNorm = DoubleArray(91) { n -> (((7 * n + 9) % 90)) / 89.0 }

        val pos1Counts = IntArray(91)
        val pos1W5 = IntArray(91)
        val pos1W15 = IntArray(91)
        val pos1Hor = Array(numHoraires) { IntArray(91) }
        val pos1LastIdx = IntArray(91) { -1 }

        val anyCounts = IntArray(91)
        val anyW5 = IntArray(91)
        val anyW15 = IntArray(91)
        val anyHor = Array(numHoraires) { IntArray(91) }
        val anyLastIdx = IntArray(91) { -1 }
        val horCounts = IntArray(numHoraires)

        val pos1Transition = IntArray(91)
        val coPrevNum1 = IntArray(91)
        val coLastRaw = IntArray(91)

        var committedEnd = 0
        var targetCursor = 0
        val out = HashMap<Int, PrecisionFeatureMatrix>(sortedTargets.size * 2)

        for (t in 0..maxTargetIdx) {
            val targetDraw = chronological[t]
            while (committedEnd < t && chronological[committedEnd].timestamp < targetDraw.timestamp) {
                val idx = committedEnd
                val hId = horaireIds[idx]
                horCounts[hId]++

                val p1 = drawPos1[idx]
                if (p1 in 1..90) {
                    pos1Counts[p1]++
                    pos1W5[p1]++
                    pos1W15[p1]++
                    pos1Hor[hId][p1]++
                    pos1LastIdx[p1] = idx
                }
                val off = idx * 5
                for (k in 0 until 5) {
                    val num = drawBallRaw[off + k]
                    if (num in 1..90) {
                        anyCounts[num]++
                        anyW5[num]++
                        anyW15[num]++
                        anyHor[hId][num]++
                        anyLastIdx[num] = idx
                    }
                }
                val exit5 = idx - 5
                if (exit5 >= 0) {
                    val ep1 = drawPos1[exit5]
                    if (ep1 in 1..90) pos1W5[ep1]--
                    val eOff = exit5 * 5
                    for (k in 0 until 5) {
                        val num = drawBallRaw[eOff + k]
                        if (num in 1..90) anyW5[num]--
                    }
                }
                val exit15 = idx - 15
                if (exit15 >= 0) {
                    val ep1 = drawPos1[exit15]
                    if (ep1 in 1..90) pos1W15[ep1]--
                    val eOff = exit15 * 5
                    for (k in 0 until 5) {
                        val num = drawBallRaw[eOff + k]
                        if (num in 1..90) anyW15[num]--
                    }
                }
                committedEnd++
            }

            if (targetCursor < sortedTargets.size && sortedTargets[targetCursor] == t) {
                targetCursor++
                val pastSize = committedEnd
                val total = pastSize.coerceAtLeast(1)
                val rows = Array(fCount) { DoubleArray(91) }

                val lastIdx = pastSize - 1
                val prevPos1 = if (lastIdx >= 0) drawPos1[lastIdx].takeIf { it in 1..90 } ?: -1 else -1
                val lastLo = if (lastIdx >= 0) drawLoMask[lastIdx] else 0L
                val lastHi = if (lastIdx >= 0) drawHiMask[lastIdx] else 0L

                pos1Transition.fill(0)
                coPrevNum1.fill(0)
                coLastRaw.fill(0)
                var prevP1Transitions = 0
                var lastSetTransitions = 0

                if (pastSize > 1) {
                    for (i in 1 until pastSize) {
                        val prevWasPrevP1 = prevPos1 in 1..90 && drawPos1[i - 1] == prevPos1
                        val prevHadLastAny = (drawLoMask[i - 1] and lastLo) != 0L || (drawHiMask[i - 1] and lastHi) != 0L
                        if (prevWasPrevP1) {
                            val p1 = drawPos1[i]
                            if (p1 in 1..90) {
                                pos1Transition[p1]++
                                prevP1Transitions++
                            }
                            val off = i * 5
                            for (k in 0 until 5) {
                                val num = drawBallRaw[off + k]
                                if (num in 1..90) coPrevNum1[num]++
                            }
                        }
                        if (prevHadLastAny) {
                            lastSetTransitions++
                            val off = i * 5
                            for (k in 0 until 5) {
                                val num = drawBallRaw[off + k]
                                if (num in 1..90) coLastRaw[num]++
                            }
                        }
                    }
                }

                val tHId = horaireIds[t]
                val horDenom = horCounts[tHId].coerceAtLeast(1).toDouble()
                val w5Denom = minOf(5, pastSize).coerceAtLeast(1).toDouble()
                val w15Denom = minOf(15, pastSize).coerceAtLeast(1).toDouble()
                val trDenom = prevP1Transitions.coerceAtLeast(1).toDouble()
                val coLastDenom = lastSetTransitions.coerceAtLeast(1).toDouble()
                val p1HorRow = pos1Hor[tHId]
                val anyHorRow = anyHor[tHId]

                for (n in 1..90) {
                    rows[PrecisionFeature.X_NORM.ordinal][n] = xNorm[n]
                    rows[PrecisionFeature.POS1_FREQ_GLOBAL.ordinal][n] = pos1Counts[n].toDouble() / total
                    rows[PrecisionFeature.POS1_FREQ_W5.ordinal][n] = pos1W5[n].toDouble() / w5Denom
                    rows[PrecisionFeature.POS1_FREQ_W15.ordinal][n] = pos1W15[n].toDouble() / w15Denom
                    val p1Gap = if (pos1LastIdx[n] >= 0) (pastSize - 1 - pos1LastIdx[n]) else pastSize
                    rows[PrecisionFeature.POS1_GAP_NORM.ordinal][n] = p1Gap.toDouble() / (total + 1)
                    rows[PrecisionFeature.POS1_HORAIRE_FREQ.ordinal][n] = p1HorRow[n].toDouble() / horDenom
                    rows[PrecisionFeature.POS1_PREV_TRANSITION.ordinal][n] = pos1Transition[n].toDouble() / trDenom
                    rows[PrecisionFeature.POS1_SAME_LAG1.ordinal][n] = if (n == prevPos1) 1.0 else 0.0

                    rows[PrecisionFeature.ANY_FREQ_GLOBAL.ordinal][n] = anyCounts[n].toDouble() / (total * 5.0)
                    rows[PrecisionFeature.ANY_FREQ_W5.ordinal][n] = anyW5[n].toDouble() / (w5Denom * 5.0)
                    rows[PrecisionFeature.ANY_FREQ_W15.ordinal][n] = anyW15[n].toDouble() / (w15Denom * 5.0)
                    val aGap = if (anyLastIdx[n] >= 0) (pastSize - 1 - anyLastIdx[n]) else pastSize
                    rows[PrecisionFeature.ANY_GAP_NORM.ordinal][n] = aGap.toDouble() / (total + 1)
                    rows[PrecisionFeature.ANY_HORAIRE_FREQ.ordinal][n] = anyHorRow[n].toDouble() / (horDenom * 5.0)
                    rows[PrecisionFeature.CO_PREV_NUM1.ordinal][n] = coPrevNum1[n].toDouble() / (trDenom * 5.0)
                    rows[PrecisionFeature.CO_LAST_RAW.ordinal][n] = coLastRaw[n].toDouble() / (coLastDenom * 5.0)

                    rows[PrecisionFeature.BASE7_NORM.ordinal][n] = base7Norm[n]
                    rows[PrecisionFeature.LUCKY_ROOT.ordinal][n] = luckyNorm[n]
                    rows[PrecisionFeature.AFFINE_MOD90.ordinal][n] = affineNorm[n]
                }
                out[t] = PrecisionFeatureMatrix(rows)
            }
        }
        return out
    }

    fun buildCausalCache(
        history: List<LotoDraw>,
        period: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
        referenceNowMs: Long = System.currentTimeMillis(),
        customDays: Int? = period.days,
        pool: ForkJoinPool = ForkJoinPool.commonPool()
    ): PrecisionCausalCache {
        val chronological = history
            .distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoDraw> { it.timestamp }.thenBy { it.uniqueId })

        val resolvedDays = if (period == BacktestPeriod.ALL_HISTORY && customDays == null) null else (customDays ?: period.days)?.coerceAtLeast(1)
        val resolvedLabel = when {
            resolvedDays == null -> period.label
            period.days == resolvedDays -> period.label
            else -> "$resolvedDays Jours (Perso)"
        }

        if (chronological.size < 4) {
            return PrecisionCausalCache("empty", resolvedLabel, "Données insuffisantes", emptyList(), 0, 0, 0)
        }

        val range = SymbolicFormula.computePeriodRange(chronological, period, referenceNowMs, resolvedDays)
        val rangeIndices = range.toList()
        val matrixByIdx = extractPrecisionMatrixSequence(chronological, rangeIndices.toIntArray())
        val tasks = rangeIndices.map { t ->
            Callable {
                val targetDraw = chronological[t]
                val pos1 = targetDraw.num1
                val actualNums = SymbolicFormula.extractDistinctBalls(targetDraw).sorted()
                if (pos1 !in 1..90 || actualNums.size < 5) return@Callable null

                val strictPastCount = chronological.subList(0, t)
                    .count { it.timestamp < targetDraw.timestamp && it.uniqueId != targetDraw.uniqueId }
                if (strictPastCount == 0) return@Callable null
                var maxPastTs = Long.MIN_VALUE
                for (pIdx in 0 until t) {
                    val pd = chronological[pIdx]
                    if (pd.timestamp < targetDraw.timestamp && pd.uniqueId != targetDraw.uniqueId && pd.timestamp > maxPastTs) {
                        maxPastTs = pd.timestamp
                    }
                }
                check(maxPastTs < targetDraw.timestamp) {
                    "Violation causale Précision : maxPast=$maxPastTs >= target=${targetDraw.timestamp}"
                }

                val horaire = LotoDateUtils.normalizeHoraire(targetDraw)
                val game = LotoDateUtils.normalizeGame(targetDraw)
                val dayKey = LotoDateUtils.formatDayKey(targetDraw.timestamp)
                val mask = BooleanArray(91)
                for (num in actualNums) if (num in 1..90) mask[num] = true
                val mat = matrixByIdx[t] ?: return@Callable null
                val epoch = runCatching { java.time.LocalDate.parse(dayKey).toEpochDay() }.getOrDefault(0L)

                PrecisionRawStep(
                    stepIndex = t,
                    draw = targetDraw,
                    actualPos1 = pos1,
                    actualNumbers = actualNums,
                    actualSet = actualNums.toSet(),
                    actualMask = mask,
                    strictPastSize = strictPastCount,
                    maxPastTs = maxPastTs,
                    dayKey = dayKey,
                    horaire = horaire,
                    game = game,
                    dayEpoch = epoch,
                    matrix = mat
                )
            }
        }

        val rawList = pool.invokeAll(tasks).mapNotNull { it.get() }
        val n = rawList.size
        val cached = if (resolvedDays != null && n > 0) {
            val totalBlocks = (n + 5) / 6
            rawList.mapIndexed { idx, s ->
                val blockFromEnd = (n - 1 - idx) / 6
                s.copy(stepIndex = idx + 1, dayEpoch = (totalBlocks - 1 - blockFromEnd).toLong())
            }
        } else {
            rawList.mapIndexed { idx, s -> s.copy(stepIndex = idx + 1) }
        }

        val discEnd: Int
        val rollEnd: Int
        val confEnd: Int
        when {
            n >= 20 -> {
                discEnd = ((n * 45) / 100).coerceAtLeast(6)
                rollEnd = (discEnd + ((n * 20) / 100).coerceAtLeast(4)).coerceAtMost(n - 6)
                confEnd = (rollEnd + ((n * 20) / 100).coerceAtLeast(3)).coerceAtMost(n - 3)
            }
            n >= 8 -> {
                discEnd = ((n * 45) / 100).coerceAtLeast(3)
                rollEnd = (discEnd + ((n * 20) / 100).coerceAtLeast(2)).coerceAtMost(n - 3)
                confEnd = (rollEnd + ((n * 20) / 100).coerceAtLeast(2)).coerceAtMost(n - 1)
            }
            n == 6 -> {
                discEnd = 2
                rollEnd = 3
                confEnd = 5
            }
            n >= 5 -> {
                discEnd = ((n * 45) / 100).coerceAtLeast(2)
                rollEnd = (discEnd + 1).coerceAtMost(n - 2)
                confEnd = (rollEnd + 1).coerceAtMost(n - 1)
            }
            else -> {
                discEnd = (n / 2).coerceAtLeast(1)
                rollEnd = (discEnd + 1).coerceAtMost(n)
                confEnd = (rollEnd + 1).coerceAtMost(n)
            }
        }

        val rangeLabel = if (cached.isNotEmpty()) {
            "${LotoDateUtils.formatShortDate(cached.first().draw.timestamp)} → ${LotoDateUtils.formatShortDate(cached.last().draw.timestamp)}"
        } else {
            resolvedLabel
        }
        val cacheKey = "PREC_${period.name}_${resolvedDays ?: "ALL"}_${chronological.size}_${cached.firstOrNull()?.draw?.uniqueId}_${cached.lastOrNull()?.draw?.uniqueId}"
        return PrecisionCausalCache(cacheKey, resolvedLabel, rangeLabel, cached, discEnd, rollEnd, confEnd)
    }

    fun synthesizeCausalDiscoveryGenomes(
        mode: PrecisionTargetMode,
        discoverySteps: List<PrecisionRawStep>,
        generation: Int,
        sessionSalt: Long,
        seenSignatures: Set<String> = emptySet(),
        maxCount: Int = 24
    ): List<PrecisionGenome> {
        if (discoverySteps.isEmpty()) return emptyList()
        val fCount = PrecisionFeature.entries.size
        val sumHit = DoubleArray(fCount)
        val sqHit = DoubleArray(fCount)
        val sumMiss = DoubleArray(fCount)
        val sqMiss = DoubleArray(fCount)
        var nHit = 0
        var nMiss = 0

        for (step in discoverySteps) {
            val mat = step.matrix
            for (n in 1..90) {
                val isTarget = if (mode == PrecisionTargetMode.FIRST_BALL) (n == step.actualPos1) else step.actualMask[n]
                if (isTarget) nHit++ else nMiss++
                for (fi in 0 until fCount) {
                    val v = mat.get(fi, n)
                    if (isTarget) {
                        sumHit[fi] += v
                        sqHit[fi] += v * v
                    } else {
                        sumMiss[fi] += v
                        sqMiss[fi] += v * v
                    }
                }
            }
        }
        if (nHit == 0 || nMiss == 0) return emptyList()

        val zLifts = DoubleArray(fCount) { fi ->
            val mH = sumHit[fi] / nHit
            val mM = sumMiss[fi] / nMiss
            val vH = (sqHit[fi] / nHit - mH * mH).coerceAtLeast(1e-6)
            val vM = (sqMiss[fi] / nMiss - mM * mM).coerceAtLeast(1e-6)
            val z = (mH - mM) / sqrt((vH + vM) * 0.5 + 1e-6)
            if (z.isFinite()) z else 0.0
        }

        val sortedIdx = (0 until fCount).sortedByDescending { abs(zLifts[it]) }
        val baseVec = List(fCount) { fi -> round2(zLifts[fi].coerceIn(-3.0, 3.0)) }

        val out = ArrayList<PrecisionGenome>(maxCount)
        val localSigs = HashSet<String>(maxCount * 2)
        val transforms = PrecisionTransform.entries
        val topKList = listOf(2, 3, 4, 6, fCount)
        val temps = listOf(0.25, 0.45, 0.70, 1.00)
        val boosts = listOf(0.0, 0.25, 0.55)

        var idx = 0
        for (k in topKList) {
            val keepSet = sortedIdx.take(k).toSet()
            val wPrimary = List(fCount) { fi -> if (fi in keepSet) baseVec[fi] else 0.0 }
            val secKeep = sortedIdx.drop(minOf(1, k - 1)).take(minOf(4, fCount)).toSet()
            val wSecondary = List(fCount) { fi -> if (fi in secKeep) round2(baseVec[fi] * 0.85) else 0.0 }
            for (tr in transforms) {
                for (boost in boosts) {
                    if (out.size >= maxCount) break
                    val temp = temps[idx % temps.size]
                    val genome = PrecisionGenome(
                        id = "prec_${mode.name}_causal_g${generation}_${idx++}",
                        mode = mode,
                        generation = generation,
                        weights = wPrimary,
                        secondaryWeights = wSecondary,
                        transform = tr,
                        interactionBoost = boost,
                        samplingTemp = temp,
                        exploreNoise = 0.0
                    )
                    val sig = genome.signatureKey()
                    if (sig !in seenSignatures && localSigs.add(sig)) {
                        out.add(genome)
                    }
                }
                if (out.size >= maxCount) break
            }
            if (out.size >= maxCount) break
        }
        return out
    }

    fun createInitialState(
        mode: PrecisionTargetMode,
        populationSize: Int = 512,
        sessionSalt: Long = 1L,
        preExcludedSignatures: Set<String> = emptySet()
    ): PrecisionEvolutionState {
        val targetSize = populationSize.coerceIn(32, 4096)
        val seen = LinkedHashSet<String>(maxOf(1024, preExcludedSignatures.size + targetSize))
        seen.addAll(preExcludedSignatures)
        val pop = generateRandomGenomes(mode, targetSize, 1, sessionSalt, seen)
        return PrecisionEvolutionState(
            mode = mode,
            generation = 0,
            experimentsCount = 0,
            confirmationTestsCount = 0,
            sessionSalt = sessionSalt,
            population = pop,
            seenSignatures = seen
        )
    }

    private fun generateRandomGenomes(
        mode: PrecisionTargetMode,
        count: Int,
        generation: Int,
        sessionSalt: Long,
        seen: Set<String>
    ): List<PrecisionGenome> {
        val fCount = PrecisionFeature.entries.size
        val rng = Random(sessionSalt * 104_729L + generation * 7919L + mode.ordinal * 31L)
        val out = ArrayList<PrecisionGenome>(count)
        val localSigs = HashSet<String>(count * 2)
        val transforms = PrecisionTransform.entries

        for (idx in 0 until count) {
            var attempt = 0
            var g: PrecisionGenome
            do {
                val activeK = rng.nextInt(2, minOf(8, fCount))
                val chosenFeats = (0 until fCount).shuffled(rng).take(activeK).toSet()
                val w1 = List(fCount) { fi ->
                    if (fi in chosenFeats) round2((rng.nextDouble() * 4.0) - 2.0) else 0.0
                }
                val secChosen = (0 until fCount).shuffled(rng).take(activeK.coerceAtMost(5)).toSet()
                val w2 = List(fCount) { fi ->
                    if (fi in secChosen) round2((rng.nextDouble() * 3.0) - 1.5) else 0.0
                }
                g = PrecisionGenome(
                    id = "prec_${mode.name}_g${generation}_$idx",
                    mode = mode,
                    generation = generation,
                    weights = w1,
                    secondaryWeights = w2,
                    transform = transforms[rng.nextInt(transforms.size)],
                    interactionBoost = round2(rng.nextDouble() * 0.8),
                    samplingTemp = round2(0.25 + rng.nextDouble() * 0.85),
                    exploreNoise = 0.0
                )
                val sig = g.signatureKey()
                if ((sig !in seen && localSigs.add(sig)) || attempt >= 6) break
                attempt++
            } while (true)
            out.add(g)
        }
        return out
    }

    private fun scoreStepNumbers(
        genome: PrecisionGenome,
        mat: PrecisionFeatureMatrix,
        outScores1: DoubleArray,
        outScores2: DoubleArray
    ) {
        val fCount = PrecisionFeature.entries.size
        val w1 = genome.weights
        val w2 = genome.secondaryWeights
        val boost = genome.interactionBoost
        val p1Row = PrecisionFeature.POS1_FREQ_W15.ordinal
        val trRow = PrecisionFeature.POS1_PREV_TRANSITION.ordinal
        val coRow = PrecisionFeature.CO_LAST_RAW.ordinal

        for (n in 1..90) {
            var s1 = 0.0
            var s2 = 0.0
            for (fi in 0 until fCount) {
                val v = mat.get(fi, n)
                s1 += w1[fi] * v
                s2 += w2[fi] * v
            }
            if (boost != 0.0) {
                val synergy = if (genome.mode == PrecisionTargetMode.FIRST_BALL) {
                    mat.get(p1Row, n) * mat.get(trRow, n)
                } else {
                    mat.get(coRow, n) * mat.get(PrecisionFeature.ANY_FREQ_W15.ordinal, n)
                }
                s1 += boost * synergy * 4.0
            }
            outScores1[n] = applyTransform(genome.transform, s1)
            outScores2[n] = applyTransform(genome.transform, s2)
        }
    }

    private fun applyTransform(tr: PrecisionTransform, x: Double): Double {
        val c = x.coerceIn(-12.0, 12.0)
        return when (tr) {
            PrecisionTransform.LINEAR -> c
            PrecisionTransform.TANH -> tanh(c)
            PrecisionTransform.SIGMOID -> 1.0 / (1.0 + exp(-c))
            PrecisionTransform.SQRT_SIGNED -> c.sign * sqrt(abs(c))
        }
    }

    private fun softmaxProbs(scores: DoubleArray, temp: Double, targetSum: Double, out: DoubleArray) {
        val invT = 1.0 / temp.coerceIn(0.15, 2.0)
        var maxS = Double.NEGATIVE_INFINITY
        for (n in 1..90) if (scores[n].isFinite() && scores[n] > maxS) maxS = scores[n]
        if (!maxS.isFinite()) maxS = 0.0
        var sum = 0.0
        for (n in 1..90) {
            val e = exp(((scores[n] - maxS) * invT).coerceIn(-15.0, 15.0))
            out[n] = e
            sum += e
        }
        val safeSum = if (sum > 0.0 && sum.isFinite()) sum else 90.0
        for (n in 1..90) {
            out[n] = ((targetSum * out[n]) / safeSum).coerceIn(1e-6, 0.999)
        }
    }

    private fun selectDeterministicNumbers(
        genome: PrecisionGenome,
        mat: PrecisionFeatureMatrix,
        s1: DoubleArray,
        s2: DoubleArray,
        p1: DoubleArray,
        p2: DoubleArray,
        marginal: DoubleArray
    ): List<Int> {
        scoreStepNumbers(genome, mat, s1, s2)
        return if (genome.mode == PrecisionTargetMode.FIRST_BALL) {
            softmaxProbs(s1, genome.samplingTemp, 1.0, p1)
            for (n in 1..90) marginal[n] = p1[n]
            var bestN = 1
            var bestScore = Double.NEGATIVE_INFINITY
            for (n in 1..90) {
                if (s1[n] > bestScore) {
                    bestScore = s1[n]
                    bestN = n
                }
            }
            listOf(bestN)
        } else {
            softmaxProbs(s1, genome.samplingTemp, 1.0, p1)
            softmaxProbs(s2, genome.samplingTemp, 1.0, p2)
            for (n in 1..90) marginal[n] = (p1[n] + p2[n]).coerceIn(1e-6, 0.999)
            var b1 = 1
            var max1 = Double.NEGATIVE_INFINITY
            for (n in 1..90) {
                if (s1[n] > max1) {
                    max1 = s1[n]
                    b1 = n
                }
            }
            var b2 = if (b1 == 1) 2 else 1
            var max2 = Double.NEGATIVE_INFINITY
            for (n in 1..90) {
                if (n == b1) continue
                val blended = 0.65 * s2[n] + 0.35 * s1[n]
                if (blended > max2) {
                    max2 = blended
                    b2 = n
                }
            }
            listOf(b1, b2).sorted()
        }
    }

    fun evaluateGenomeOnStage(
        genome: PrecisionGenome,
        steps: List<PrecisionRawStep>
    ): PrecisionStageStats {
        if (steps.isEmpty()) {
            return PrecisionStageStats(0, 0, 0, 0, 0, 0f, 0, 0.1, emptyList(), emptyList(), true)
        }
        val s1 = DoubleArray(91)
        val s2 = DoubleArray(91)
        val p1 = DoubleArray(91)
        val p2 = DoubleArray(91)
        val marginal = DoubleArray(91)

        var exactCount = 0
        var onePlusCount = 0
        var totalHits = 0
        var zeroCount = 0
        var brierSum = 0.0
        var allCausal = true
        val hitsPerStep = ArrayList<Int>(steps.size)
        val exactHits = ArrayList<Boolean>(steps.size)

        var curEpoch = Long.MIN_VALUE
        var curEpochQual = 0
        var curEpochPrevExact = false
        var curEpochLastExact = false
        val needPen = genome.mode == PrecisionTargetMode.TWO_NUMBERS
        val minQual = if (genome.mode == PrecisionTargetMode.FIRST_BALL) 1 else 3
        var qualDays = 0
        var consecQual = 0
        var maxConsecQual = 0
        var lastQualEpoch = Long.MIN_VALUE

        for (step in steps) {
            if (step.maxPastTs >= step.draw.timestamp) allCausal = false
            val picked = selectDeterministicNumbers(genome, step.matrix, s1, s2, p1, p2, marginal)
            val hits = if (genome.mode == PrecisionTargetMode.FIRST_BALL) {
                if (picked.firstOrNull() == step.actualPos1) 1 else 0
            } else {
                picked.count { it in step.actualSet }
            }
            val isExact = if (genome.mode == PrecisionTargetMode.FIRST_BALL) {
                hits == 1
            } else {
                hits == 2
            }

            totalHits += hits
            if (hits >= 1) onePlusCount++ else zeroCount++
            if (isExact) exactCount++
            hitsPerStep.add(hits)
            exactHits.add(isExact)

            var stepBrier = 0.0
            for (n in 1..90) {
                val y = if (genome.mode == PrecisionTargetMode.FIRST_BALL) {
                    if (n == step.actualPos1) 1.0 else 0.0
                } else {
                    if (step.actualMask[n]) 1.0 else 0.0
                }
                val d = marginal[n] - y
                stepBrier += d * d
            }
            brierSum += stepBrier / 90.0

            if (step.dayEpoch != curEpoch) {
                if (curEpoch != Long.MIN_VALUE && curEpochQual >= minQual && (!needPen || curEpochPrevExact)) {
                    qualDays++
                    consecQual = if (lastQualEpoch != Long.MIN_VALUE && curEpoch == lastQualEpoch + 1L) consecQual + 1 else 1
                    if (consecQual > maxConsecQual) maxConsecQual = consecQual
                    lastQualEpoch = curEpoch
                } else if (curEpoch != Long.MIN_VALUE) {
                    consecQual = 0
                }
                curEpoch = step.dayEpoch
                curEpochQual = 0
                curEpochPrevExact = false
                curEpochLastExact = false
            }
            curEpochPrevExact = curEpochLastExact
            curEpochLastExact = isExact
            if (isExact) curEpochQual++
        }

        if (curEpoch != Long.MIN_VALUE && curEpochQual >= minQual && (!needPen || curEpochPrevExact)) {
            qualDays++
            consecQual = if (lastQualEpoch != Long.MIN_VALUE && curEpoch == lastQualEpoch + 1L) consecQual + 1 else 1
            if (consecQual > maxConsecQual) maxConsecQual = consecQual
        }

        return PrecisionStageStats(
            tested = steps.size,
            exactTargetCount = exactCount,
            onePlusCount = onePlusCount,
            qualifyingDays = qualDays,
            maxConsecutiveQualifyingDays = maxConsecQual,
            avgHits = totalHits.toFloat() / steps.size,
            zeroMatchCount = zeroCount,
            brierScore = brierSum / steps.size,
            hitsPerStep = hitsPerStep,
            exactHitsPerStep = exactHits,
            causalVerified = allCausal
        )
    }

    private fun computePrecisionBlockGradient(
        steps: List<PrecisionRawStep>,
        hitsPerStep: List<Int>,
        exactHits: List<Boolean>
    ): Pair<Int, Double> {
        val count = minOf(steps.size, exactHits.size, hitsPerStep.size)
        if (count == 0) return 0 to 0.0
        var curEpoch = Long.MIN_VALUE
        var curExactCount = 0
        var curPrevExact = false
        var curLastExact = false
        var curPrevHit = 0
        var curLastHit = 0
        var penExactCount = 0
        var progress = 0.0

        fun flushBlock() {
            if (curEpoch == Long.MIN_VALUE) return
            if (curPrevExact) {
                penExactCount++
                progress += 2.0 + curExactCount * 1.2
            } else if (curPrevHit >= 1) {
                progress += 0.55
            }
            if (curExactCount >= 3 && curPrevExact) {
                progress += 4.5
            } else if (curExactCount >= 2) {
                progress += 1.4
            } else if (curExactCount == 1) {
                progress += 0.4
            }
        }

        for (i in 0 until count) {
            val ep = steps[i].dayEpoch
            if (ep != curEpoch) {
                flushBlock()
                curEpoch = ep
                curExactCount = 0
                curPrevExact = false
                curLastExact = false
                curPrevHit = 0
                curLastHit = 0
            }
            curPrevExact = curLastExact
            curLastExact = exactHits[i]
            curPrevHit = curLastHit
            curLastHit = hitsPerStep[i]
            if (curLastExact) curExactCount++
        }
        flushBlock()
        return penExactCount to progress
    }

    private fun scoreCandidate(
        genome: PrecisionGenome,
        disc: PrecisionStageStats,
        roll: PrecisionStageStats,
        discSteps: List<PrecisionRawStep>,
        rollSteps: List<PrecisionRawStep>
    ): PrecisionCandidateEval {
        val minPerDay = if (genome.mode == PrecisionTargetMode.FIRST_BALL) 1 else 3
        val combinedSteps = discSteps + rollSteps
        val combinedExact = disc.exactHitsPerStep + roll.exactHitsPerStep
        val combinedHits = disc.hitsPerStep + roll.hitsPerStep
        val (consQualDays, consStreak) = computeConsolidatedBlockStats(
            combinedSteps,
            combinedExact,
            minPerDay,
            requirePenultimateExact = genome.mode == PrecisionTargetMode.TWO_NUMBERS
        )
        val (penExactCount, blockProgress) = computePrecisionBlockGradient(combinedSteps, combinedHits, combinedExact)

        val randomExpectedHits = if (genome.mode == PrecisionTargetMode.FIRST_BALL) 1.0 / 90.0 else (2.0 * 5.0) / 90.0
        val overfitDrop = roll.tested >= 4 && roll.avgHits < randomExpectedHits && disc.avgHits > roll.avgHits * 2.0f
        val stabilityPassed = !overfitDrop

        val fitness = (roll.exactTargetCount * 220.0) +
            (disc.exactTargetCount * 95.0) +
            (roll.onePlusCount * 45.0) +
            (disc.onePlusCount * 22.0) +
            (consStreak * 130.0) +
            (consQualDays * 75.0) +
            (penExactCount * 40.0) +
            (blockProgress * 20.0) +
            (roll.avgHits * 160.0) +
            (disc.avgHits * 60.0) -
            (roll.brierScore * 80.0)

        return PrecisionCandidateEval(
            genome = genome,
            discStats = disc,
            rollStats = roll,
            stabilityPassed = stabilityPassed,
            fitnessScore = if (stabilityPassed) fitness else fitness - 200.0,
            status = if (stabilityPassed) "ROLLING_PASSED" else "REJECTED",
            consolidatedStreak = maxOf(consStreak, roll.maxConsecutiveQualifyingDays, disc.maxConsecutiveQualifyingDays)
        )
    }

    private fun buildStatsFromGpuSlice(
        mode: PrecisionTargetMode,
        steps: List<PrecisionRawStep>,
        stepCount: Int,
        gIdx: Int,
        gpuRes: PrecisionGpuStageResult
    ): PrecisionStageStats {
        if (steps.isEmpty()) {
            return PrecisionStageStats(0, 0, 0, 0, 0, 0f, 0, 0.1, emptyList(), emptyList(), true)
        }
        val minQual = if (mode == PrecisionTargetMode.FIRST_BALL) 1 else 3
        val hitsPerStep = ArrayList<Int>(stepCount)
        val exactHits = ArrayList<Boolean>(stepCount)
        var exactCount = 0
        var onePlusCount = 0
        var totalHits = 0
        var zeroCount = 0
        var brierSum = 0.0
        var allCausal = true

        val needPen = mode == PrecisionTargetMode.TWO_NUMBERS
        var curEpoch = Long.MIN_VALUE
        var curEpochQual = 0
        var curEpochPrevExact = false
        var curEpochLastExact = false
        var qualDays = 0
        var consecQual = 0
        var maxConsecQual = 0
        var lastQualEpoch = Long.MIN_VALUE

        val base = gIdx * stepCount
        for (sIdx in 0 until stepCount) {
            val step = steps[sIdx]
            if (step.maxPastTs >= step.draw.timestamp) allCausal = false
            val h = gpuRes.hits[base + sIdx]
            val ex = gpuRes.exactHits[base + sIdx] != 0
            val br = gpuRes.brier[base + sIdx].toDouble()
            hitsPerStep.add(h)
            exactHits.add(ex)
            totalHits += h
            if (h >= 1) onePlusCount++ else zeroCount++
            if (ex) exactCount++
            brierSum += br

            if (step.dayEpoch != curEpoch) {
                if (curEpoch != Long.MIN_VALUE && curEpochQual >= minQual && (!needPen || curEpochPrevExact)) {
                    qualDays++
                    consecQual = if (lastQualEpoch != Long.MIN_VALUE && curEpoch == lastQualEpoch + 1L) consecQual + 1 else 1
                    if (consecQual > maxConsecQual) maxConsecQual = consecQual
                    lastQualEpoch = curEpoch
                } else if (curEpoch != Long.MIN_VALUE) {
                    consecQual = 0
                }
                curEpoch = step.dayEpoch
                curEpochQual = 0
                curEpochPrevExact = false
                curEpochLastExact = false
            }
            curEpochPrevExact = curEpochLastExact
            curEpochLastExact = ex
            if (ex) curEpochQual++
        }
        if (curEpoch != Long.MIN_VALUE && curEpochQual >= minQual && (!needPen || curEpochPrevExact)) {
            qualDays++
            consecQual = if (lastQualEpoch != Long.MIN_VALUE && curEpoch == lastQualEpoch + 1L) consecQual + 1 else 1
            if (consecQual > maxConsecQual) maxConsecQual = consecQual
        }

        return PrecisionStageStats(
            tested = stepCount,
            exactTargetCount = exactCount,
            onePlusCount = onePlusCount,
            qualifyingDays = qualDays,
            maxConsecutiveQualifyingDays = maxConsecQual,
            avgHits = totalHits.toFloat() / stepCount,
            zeroMatchCount = zeroCount,
            brierScore = brierSum / stepCount,
            hitsPerStep = hitsPerStep,
            exactHitsPerStep = exactHits,
            causalVerified = allCausal
        )
    }

    internal fun evaluateStageHeterogeneous(
        stageKey: String,
        mode: PrecisionTargetMode,
        steps: List<PrecisionRawStep>,
        genomes: List<PrecisionGenome>,
        useGpu: Boolean,
        cpuPool: ForkJoinPool,
        gpuAccelerator: GpuOpenClAccelerator?
    ): Triple<List<PrecisionStageStats>, Double, Double> {
        if (steps.isEmpty() || genomes.isEmpty()) {
            val empty = List(genomes.size) {
                PrecisionStageStats(0, 0, 0, 0, 0, 0f, 0, 0.1, emptyList(), emptyList(), true)
            }
            return Triple(empty, 0.0, 0.0)
        }

        val gpuReady = useGpu && gpuAccelerator != null && gpuAccelerator.isPrecisionKernelReady
        if (!gpuReady) {
            val cpuStartNs = System.nanoTime()
            val stats = PcDiscoveryEngine.mapIndexedParallel(cpuPool, genomes.size) { idx ->
                evaluateGenomeOnStage(genomes[idx], steps)
            }
            val cpuMs = ((System.nanoTime() - cpuStartNs) / 1_000_000.0).coerceAtLeast(0.1)
            return Triple(stats, 0.0, cpuMs)
        }

        val gpuCount = if (genomes.size >= 48) {
            ((genomes.size * 3) / 4).coerceIn(16, genomes.size)
        } else {
            genomes.size
        }
        val gpuGenomes = genomes.subList(0, gpuCount)
        val cpuGenomes = if (gpuCount < genomes.size) genomes.subList(gpuCount, genomes.size) else emptyList()

        val cpuFuture = if (cpuGenomes.isNotEmpty()) {
            cpuPool.submit(Callable<Pair<List<PrecisionStageStats>, Double>> {
                val cStart = System.nanoTime()
                val res = ArrayList<PrecisionStageStats>(cpuGenomes.size)
                for (g in cpuGenomes) {
                    res.add(evaluateGenomeOnStage(g, steps))
                }
                res to ((System.nanoTime() - cStart) / 1_000_000.0).coerceAtLeast(0.1)
            })
        } else null

        val gpuStageRes = gpuAccelerator!!.evaluatePrecisionStageDirect(stageKey, steps, gpuGenomes, mode)
        val (cpuTailStats, cpuTailMs) = cpuFuture?.get() ?: (emptyList<PrecisionStageStats>() to 0.0)

        var totalCpuMs = cpuTailMs
        var totalGpuMs = 0.0
        val outStats = ArrayList<PrecisionStageStats>(genomes.size)

        if (gpuStageRes != null) {
            totalGpuMs = gpuStageRes.elapsedMs
            val sCount = steps.size
            val uncovIndices = ArrayList<Int>()
            val tempStats = arrayOfNulls<PrecisionStageStats>(gpuCount)
            for (gIdx in 0 until gpuCount) {
                if (gpuStageRes.isGenomeEvaluated(gIdx)) {
                    tempStats[gIdx] = buildStatsFromGpuSlice(mode, steps, sCount, gIdx, gpuStageRes)
                } else {
                    uncovIndices.add(gIdx)
                }
            }
            if (uncovIndices.isNotEmpty()) {
                val fbStart = System.nanoTime()
                val fbList = PcDiscoveryEngine.mapIndexedParallel(cpuPool, uncovIndices.size) { pos ->
                    evaluateGenomeOnStage(gpuGenomes[uncovIndices[pos]], steps)
                }
                for (pos in uncovIndices.indices) {
                    tempStats[uncovIndices[pos]] = fbList[pos]
                }
                totalCpuMs += (System.nanoTime() - fbStart) / 1_000_000.0
            }
            for (gIdx in 0 until gpuCount) {
                outStats.add(tempStats[gIdx]!!)
            }
        } else {
            val fbStart = System.nanoTime()
            val fbList = PcDiscoveryEngine.mapIndexedParallel(cpuPool, gpuGenomes.size) { idx ->
                evaluateGenomeOnStage(gpuGenomes[idx], steps)
            }
            outStats.addAll(fbList)
            totalCpuMs += (System.nanoTime() - fbStart) / 1_000_000.0
        }

        outStats.addAll(cpuTailStats)
        return Triple(outStats, totalGpuMs, totalCpuMs)
    }

    fun executeEvolutionGeneration(
        state: PrecisionEvolutionState,
        cache: PrecisionCausalCache,
        minObjectiveStreak: Int = 1,
        populationSize: Int = 512,
        computeMode: ComputeMode = ComputeMode.GPU_AND_CPU_MAX,
        cpuPool: ForkJoinPool = ForkJoinPool.commonPool(),
        gpuAccelerator: GpuOpenClAccelerator? = null
    ): PrecisionGenResult {
        val genStartNs = System.nanoTime()
        val nextGen = state.generation + 1
        val targetPop = populationSize.coerceIn(32, 4096)
        val seen = state.seenSignatures
        val cacheDistinctDays = cache.steps.map { it.dayEpoch }.distinct().size.coerceAtLeast(1)
        val requiredStreak = minObjectiveStreak.coerceIn(1, cacheDistinctDays)

        val causalBoost = if ((state.generation == 0 || state.generation % 4 == 0 || state.stagnationCount >= 1) && cache.discoverySteps.isNotEmpty()) {
            val ridgeMax = minOf(24, maxOf(8, targetPop / 6))
            val ridgePop = CausalAiLearner.fitPrecisionRidgeGenomes(
                mode = state.mode,
                discoverySteps = cache.discoverySteps,
                generation = nextGen,
                sessionSalt = state.sessionSalt + nextGen * 19L,
                seenSignatures = seen,
                maxCount = ridgeMax
            )
            val chercheurPop = ChercheurEngine.distillPrecisionTeachings(
                mode = state.mode,
                discoverySteps = cache.discoverySteps,
                cycleIndex = nextGen,
                generation = nextGen,
                seenSignatures = seen
            )
            val zLiftPop = synthesizeCausalDiscoveryGenomes(
                mode = state.mode,
                discoverySteps = cache.discoverySteps,
                generation = nextGen,
                sessionSalt = state.sessionSalt,
                seenSignatures = seen,
                maxCount = minOf(16, maxOf(4, targetPop / 8))
            )
            (ridgePop + chercheurPop + zLiftPop).distinctBy { it.signatureKey() }
        } else emptyList()

        val activePop = buildUniquePopulation(causalBoost + state.population, state.mode, targetPop, nextGen, state.sessionSalt, seen)
        recordSeen(seen, activePop)

        val useGpu = computeMode == ComputeMode.GPU_AND_CPU_MAX && gpuAccelerator != null && gpuAccelerator.isPrecisionKernelReady
        val (discStatsList, discGpuMs, discCpuMs) = evaluateStageHeterogeneous(
            stageKey = "${cache.cacheKey}_DISC",
            mode = state.mode,
            steps = cache.discoverySteps,
            genomes = activePop,
            useGpu = useGpu,
            cpuPool = cpuPool,
            gpuAccelerator = gpuAccelerator
        )
        val (rollStatsList, rollGpuMs, rollCpuMs) = evaluateStageHeterogeneous(
            stageKey = "${cache.cacheKey}_ROLL",
            mode = state.mode,
            steps = cache.rollingSteps,
            genomes = activePop,
            useGpu = useGpu,
            cpuPool = cpuPool,
            gpuAccelerator = gpuAccelerator
        )

        val evals = ArrayList<PrecisionCandidateEval>(activePop.size)
        for (idx in activePop.indices) {
            evals.add(scoreCandidate(activePop[idx], discStatsList[idx], rollStatsList[idx], cache.discoverySteps, cache.rollingSteps))
        }
        val gpuMs = discGpuMs + rollGpuMs
        val cpuMs = discCpuMs + rollCpuMs
        val gpuErr = gpuAccelerator?.lastRuntimeError.orEmpty()

        val ranked = evals.sortedByDescending { it.fitnessScore }
        val newElites = (state.elites + ranked.filter { it.stabilityPassed })
            .distinctBy { it.genome.signatureKey() }
            .sortedByDescending { it.fitnessScore }
            .take(24)

        var confTests = state.confirmationTestsCount
        val testedSigs = LinkedHashSet(state.testedConfirmationSigs)
        var winningEval: PrecisionCandidateEval? = null
        var auditedEval: PrecisionCandidateEval? = null

        val confCandidates = if (cache.confirmationSteps.isNotEmpty()) {
            ranked.filter { cand ->
                cand.stabilityPassed &&
                    cand.genome.signatureKey() !in testedSigs &&
                    (confTests == 0 ||
                        cand.consolidatedStreak >= 1 ||
                        cand.rollStats.exactTargetCount >= 1 ||
                        cand.discStats.onePlusCount + cand.rollStats.onePlusCount >= 1)
            }.take(5)
        } else emptyList()

        val oosCombinedSteps = cache.rollingSteps + cache.confirmationSteps
        val fullStepsSoFar = cache.discoverySteps + cache.rollingSteps + cache.confirmationSteps
        val minPerDay = if (state.mode == PrecisionTargetMode.FIRST_BALL) 1 else 3
        val needPen = state.mode == PrecisionTargetMode.TWO_NUMBERS
        val randExpected = if (state.mode == PrecisionTargetMode.FIRST_BALL) 1.0f / 90f else (2f * 5f / 90f)

        for (cand in confCandidates) {
            val sig = cand.genome.signatureKey()
            if (testedSigs.size >= 5000) testedSigs.iterator().let { if (it.hasNext()) { it.next(); it.remove() } }
            testedSigs.add(sig)
            confTests++

            val confStats = evaluateGenomeOnStage(cand.genome, cache.confirmationSteps)
            val oosStats = evaluateGenomeOnStage(cand.genome, oosCombinedSteps)
            val effTests = PcDiscoveryEngine.effectiveMultipleTestFactor(confTests)
            val pRaw = computePrecisionPValue(state.mode, confStats.hitsPerStep)
            val pAdj = (pRaw * effTests).coerceAtMost(1.0)
            val pOosRaw = computePrecisionPValue(state.mode, oosStats.hitsPerStep)
            val pOosAdj = (pOosRaw * effTests).coerceAtMost(1.0)
            val beatsRandom = confStats.avgHits > randExpected
            val oosBeatsRandom = oosStats.avgHits > randExpected
            val sigOk = (pAdj < 0.05 || pRaw < 0.025) ||
                (cache.confirmationSteps.size < 6 && oosBeatsRandom && (pOosAdj < 0.05 || pOosRaw < 0.025))

            val fullExactSoFar = cand.discStats.exactHitsPerStep + cand.rollStats.exactHitsPerStep + confStats.exactHitsPerStep
            val (_, trainConfStreak) = computeConsolidatedBlockStats(fullStepsSoFar, fullExactSoFar, minPerDay, needPen)

            val confObjectiveMet = if (cacheDistinctDays <= 1) {
                beatsRandom && sigOk &&
                    (trainConfStreak >= 1 || confStats.exactTargetCount >= 1 ||
                        (state.mode == PrecisionTargetMode.TWO_NUMBERS &&
                            (cand.discStats.exactTargetCount + cand.rollStats.exactTargetCount + confStats.exactTargetCount) >= 2 &&
                            confStats.onePlusCount >= 1))
            } else {
                beatsRandom && sigOk &&
                    (confStats.maxConsecutiveQualifyingDays >= requiredStreak || trainConfStreak >= requiredStreak)
            }

            if (confObjectiveMet) {
                val sealedStats = if (cache.sealedSteps.isNotEmpty()) evaluateGenomeOnStage(cand.genome, cache.sealedSteps) else confStats
                val allSteps = fullStepsSoFar + cache.sealedSteps
                val allExact = fullExactSoFar + (if (cache.sealedSteps.isNotEmpty()) sealedStats.exactHitsPerStep else emptyList())
                val (_, fullStreak) = computeConsolidatedBlockStats(allSteps, allExact, minPerDay, needPen)
                val totalExactCount = allExact.count { it }
                val oneDayRuleOk = cacheDistinctDays > 1 ||
                    if (state.mode == PrecisionTargetMode.FIRST_BALL) totalExactCount >= 1 else fullStreak >= 1
                val sealedThresholdOk = if (cacheDistinctDays <= 1 && cache.sealedSteps.size <= 1) {
                    true
                } else {
                    sealedStats.avgHits >= randExpected || sealedStats.onePlusCount >= 1
                }
                val sealedOk = sealedStats.causalVerified &&
                    oneDayRuleOk &&
                    sealedThresholdOk

                if (sealedOk) {
                    val winner = cand.copy(
                        confStats = confStats,
                        sealedStats = sealedStats,
                        status = "VALIDATED",
                        pValueRaw = pRaw,
                        pValueAdjusted = pAdj,
                        consolidatedStreak = maxOf(1, fullStreak, cand.consolidatedStreak)
                    )
                    winningEval = winner
                    auditedEval = winner
                    break
                } else if (auditedEval == null) {
                    auditedEval = cand.copy(confStats = confStats, sealedStats = sealedStats, status = "REJECTED", pValueRaw = pRaw, pValueAdjusted = pAdj)
                }
            } else if (auditedEval == null) {
                auditedEval = cand.copy(confStats = confStats, status = "REJECTED", pValueRaw = pRaw, pValueAdjusted = pAdj)
            }
        }

        val bestFitness = maxOf(state.bestFitnessOverall, ranked.firstOrNull()?.fitnessScore ?: Double.NEGATIVE_INFINITY)
        val improved = (ranked.firstOrNull()?.fitnessScore ?: Double.NEGATIVE_INFINITY) > state.bestFitnessOverall + 0.25
        var nextStag = if (improved) 0 else state.stagnationCount + 1
        val shift = nextStag >= 2
        if (shift) nextStag = 0

        val rng = Random(nextGen * 91_003L + state.sessionSalt * 17L)
        val breedingPool = (newElites + ranked).distinctBy { it.genome.signatureKey() }.sortedByDescending { it.fitnessScore }
        val nextPop = evolveNextGeneration(breedingPool, state.mode, targetPop, nextGen + 1, state.sessionSalt, shift, seen, rng)

        val totalSecs = ((System.nanoTime() - genStartNs) / 1_000_000_000.0).coerceAtLeast(1e-4)
        val totalEval = activePop.size * (cache.discoverySteps.size + cache.rollingSteps.size).coerceAtLeast(1)
        val top = winningEval ?: auditedEval ?: ranked.firstOrNull()
        val note = if (winningEval != null) {
            "Précision VALIDÉ (${state.mode.label}) : ${winningEval.consolidatedStreak}j • p_adj=${String.format(Locale.FRANCE, "%.4f", winningEval.pValueAdjusted)}"
        } else {
            val ex = (top?.discStats?.exactTargetCount ?: 0) + (top?.rollStats?.exactTargetCount ?: 0)
            val o1 = (top?.discStats?.onePlusCount ?: 0) + (top?.rollStats?.onePlusCount ?: 0)
            val tot = (top?.discStats?.tested ?: 0) + (top?.rollStats?.tested ?: 0)
            "Gén.$nextGen [${state.mode.label}] : cibles exactes $ex/$tot • ≥1 boule $o1/$tot • ${seen.size} formules"
        }

        return PrecisionGenResult(
            nextState = PrecisionEvolutionState(
                mode = state.mode,
                generation = nextGen,
                experimentsCount = state.experimentsCount + activePop.size,
                confirmationTestsCount = confTests,
                sessionSalt = state.sessionSalt,
                population = nextPop,
                elites = newElites,
                testedConfirmationSigs = testedSigs,
                seenSignatures = seen,
                validatedCandidate = winningEval ?: state.validatedCandidate,
                stagnationCount = nextStag,
                bestFitnessOverall = bestFitness
            ),
            bestTrainEval = ranked.firstOrNull(),
            winningEval = winningEval,
            auditedEval = auditedEval,
            generationNote = note,
            evaluationsPerSec = totalEval / totalSecs,
            gpuBatchMs = gpuMs,
            cpuBatchMs = cpuMs,
            gpuErrorNote = gpuErr
        )
    }

    private fun evolveNextGeneration(
        pool: List<PrecisionCandidateEval>,
        mode: PrecisionTargetMode,
        targetPop: Int,
        nextGen: Int,
        sessionSalt: Long,
        shift: Boolean,
        seen: Set<String>,
        rng: Random
    ): List<PrecisionGenome> {
        if (pool.isEmpty()) return generateRandomGenomes(mode, targetPop, nextGen, sessionSalt, seen)
        val out = ArrayList<PrecisionGenome>(targetPop)
        val localSigs = HashSet<String>(targetPop * 2)
        val immCount = if (shift) targetPop / 2 else targetPop / 4
        val childTarget = (targetPop - immCount).coerceAtLeast(0)
        val transforms = PrecisionTransform.entries
        val pSize = pool.size.coerceAtLeast(1)

        for (cIdx in 0 until childTarget) {
            val p1 = pool[minOf(rng.nextInt(pSize), rng.nextInt(pSize))].genome
            val p2 = pool[minOf(rng.nextInt(pSize), rng.nextInt(pSize))].genome
            val wCount = p1.weights.size
            for (attempt in 0 until 6) {
                val nw1 = List(wCount) { i ->
                    var v = if (rng.nextBoolean()) p1.weights[i] else p2.weights[i]
                    if (rng.nextDouble() < (if (shift) 0.40 else 0.22)) {
                        v = (v + rng.nextDouble() * 1.2 - 0.6).coerceIn(-3.5, 3.5)
                    }
                    round2(v)
                }
                val nw2 = List(wCount) { i ->
                    var v = if (rng.nextBoolean()) p1.secondaryWeights[i] else p2.secondaryWeights[i]
                    if (rng.nextDouble() < (if (shift) 0.40 else 0.22)) {
                        v = (v + rng.nextDouble() * 1.0 - 0.5).coerceIn(-3.0, 3.0)
                    }
                    round2(v)
                }
                val tr = if (rng.nextDouble() < 0.25) transforms[rng.nextInt(transforms.size)] else p1.transform
                val boost = round2(((p1.interactionBoost + p2.interactionBoost) * 0.5 + (rng.nextDouble() * 0.3 - 0.15)).coerceIn(0.0, 1.5))
                val temp = round2(((p1.samplingTemp + p2.samplingTemp) * 0.5 + (rng.nextDouble() * 0.2 - 0.1)).coerceIn(0.20, 1.4))
                val cand = PrecisionGenome("prec_${mode.name}_g${nextGen}_c$cIdx", mode, nextGen, nw1, nw2, tr, boost, temp, 0.0)
                val sig = cand.signatureKey()
                if ((sig !in seen && localSigs.add(sig)) || attempt == 5) {
                    out.add(cand)
                    break
                }
            }
        }
        val extra = generateRandomGenomes(mode, (targetPop - out.size).coerceAtLeast(0), nextGen, sessionSalt + nextGen * 17L, seen)
        out.addAll(extra)
        return out.take(targetPop)
    }

    private fun buildUniquePopulation(
        seed: List<PrecisionGenome>,
        mode: PrecisionTargetMode,
        targetSize: Int,
        gen: Int,
        salt: Long,
        seen: Set<String>
    ): List<PrecisionGenome> {
        val out = ArrayList<PrecisionGenome>(targetSize)
        val local = HashSet<String>(targetSize * 2)
        for (g in seed) {
            if (out.size >= targetSize) break
            val sig = g.signatureKey()
            if (sig !in seen && local.add(sig)) out.add(g)
        }
        if (out.size < targetSize) {
            out.addAll(generateRandomGenomes(mode, targetSize - out.size, gen, salt + gen, seen))
        }
        return out
    }

    private fun recordSeen(seen: MutableSet<String>, pop: List<PrecisionGenome>) {
        if (seen.size + pop.size > MAX_SEEN_SIGS) {
            val toRem = seen.size + pop.size - (MAX_SEEN_SIGS - 10_000)
            val it = seen.iterator()
            var removed = 0
            while (it.hasNext() && removed < toRem) {
                it.next(); it.remove(); removed++
            }
        }
        for (g in pop) seen.add(g.signatureKey())
    }

    fun computeConsolidatedBlockStats(
        steps: List<PrecisionRawStep>,
        exactFlags: List<Boolean>,
        minQualifyingPerBlock: Int,
        requirePenultimateExact: Boolean = minQualifyingPerBlock >= 3
    ): Pair<Int, Int> {
        val count = minOf(steps.size, exactFlags.size)
        if (count == 0) return 0 to 0
        var curEpoch = Long.MIN_VALUE
        var curQual = 0
        var curPrevExact = false
        var curLastExact = false
        var qualDays = 0
        var consec = 0
        var maxConsec = 0
        var lastQualEpoch = Long.MIN_VALUE
        for (i in 0 until count) {
            val ep = steps[i].dayEpoch
            if (ep != curEpoch) {
                if (curEpoch != Long.MIN_VALUE && curQual >= minQualifyingPerBlock && (!requirePenultimateExact || curPrevExact)) {
                    qualDays++
                    consec = if (lastQualEpoch != Long.MIN_VALUE && curEpoch == lastQualEpoch + 1L) consec + 1 else 1
                    if (consec > maxConsec) maxConsec = consec
                    lastQualEpoch = curEpoch
                } else if (curEpoch != Long.MIN_VALUE) consec = 0
                curEpoch = ep
                curQual = 0
                curPrevExact = false
                curLastExact = false
            }
            curPrevExact = curLastExact
            curLastExact = exactFlags[i]
            if (curLastExact) curQual++
        }
        if (curEpoch != Long.MIN_VALUE && curQual >= minQualifyingPerBlock && (!requirePenultimateExact || curPrevExact)) {
            qualDays++
            consec = if (lastQualEpoch != Long.MIN_VALUE && curEpoch == lastQualEpoch + 1L) consec + 1 else 1
            if (consec > maxConsec) maxConsec = consec
        }
        return qualDays to maxConsec
    }

    fun computePrecisionPValue(mode: PrecisionTargetMode, hitsPerStep: List<Int>): Double {
        if (hitsPerStep.isEmpty() || hitsPerStep.sum() == 0) return 1.0
        val n = hitsPerStep.size.toDouble()
        val mean = hitsPerStep.sum() / n
        val (expected, variance) = if (mode == PrecisionTargetMode.FIRST_BALL) {
            val p = 1.0 / 90.0
            p to (p * (1.0 - p))
        } else {
            val exp = (2.0 * 5.0) / 90.0
            val v = 2.0 * (5.0 / 90.0) * (85.0 / 90.0) * (88.0 / 89.0)
            exp to v
        }
        if (mean <= expected) return 1.0
        val se = sqrt(variance / n).coerceAtLeast(1e-9)
        val z = (mean - expected) / se
        val x = z / sqrt(2.0)
        val t = 1.0 / (1.0 + 0.3275911 * x)
        val poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        return (poly * exp(-x * x)).coerceIn(0.0, 1.0)
    }

    fun buildLedgerKey(mode: PrecisionTargetMode, sig: String, periodKey: String): String =
        "PREC_${mode.name}_${periodKey}__${sig.hashCode().toUInt()}"

    fun mergeAppendOnlyApprovedBacktest(
        genome: PrecisionGenome,
        cache: PrecisionCausalCache,
        existingLedger: PrecisionLedgerRecord?,
        confirmationTestsCount: Int = 1
    ): Pair<PrecisionBacktestResult, PrecisionLedgerRecord> {
        val periodKey = cache.periodLabel
        val sig = genome.signatureKey()
        val frozen = if (existingLedger != null &&
            existingLedger.modeName == genome.mode.name &&
            existingLedger.strategySignature == sig &&
            existingLedger.periodKey == periodKey
        ) {
            deserializeSteps(existingLedger.frozenStepsJson).sortedBy { it.targetTimestamp }
        } else emptyList()

        val pending = if (existingLedger != null && existingLedger.strategySignature == sig) {
            deserializePending(existingLedger.pendingPredictionsJson)
        } else emptyList()

        val frozenById = frozen.associateBy { it.drawId }
        val maxFrozenTs = frozen.maxOfOrNull { it.targetTimestamp } ?: Long.MIN_VALUE
        val combined = ArrayList<PrecisionBacktestStep>(frozen.size + 16).apply { addAll(frozen) }

        val s1 = DoubleArray(91); val s2 = DoubleArray(91)
        val p1 = DoubleArray(91); val p2 = DoubleArray(91); val marginal = DoubleArray(91)
        var appended = 0
        var liveMatched = 0

        for (step in cache.steps) {
            if (frozenById.containsKey(step.draw.uniqueId) || step.draw.timestamp <= maxFrozenTs) continue
            val predicted = selectDeterministicNumbers(genome, step.matrix, s1, s2, p1, p2, marginal)
            val matched = if (genome.mode == PrecisionTargetMode.FIRST_BALL) {
                if (predicted.firstOrNull() == step.actualPos1) listOf(step.actualPos1) else emptyList()
            } else {
                predicted.filter { it in step.actualSet }
            }
            val exactHit = if (genome.mode == PrecisionTargetMode.FIRST_BALL) matched.size == 1 else matched.size == 2

            val matchPending = pending.firstOrNull { it.dayKey == step.dayKey && it.horaire.equals(step.horaire, ignoreCase = true) }
            if (matchPending != null && matchPending.predictedNumbers.sorted() == predicted.sorted()) {
                liveMatched++
            }

            var brier = 0.0
            for (n in 1..90) {
                val y = if (genome.mode == PrecisionTargetMode.FIRST_BALL) {
                    if (n == step.actualPos1) 1.0 else 0.0
                } else {
                    if (step.actualMask[n]) 1.0 else 0.0
                }
                val d = marginal[n] - y
                brier += d * d
            }

            combined.add(
                PrecisionBacktestStep(
                    stepIndex = combined.size + 1,
                    drawId = step.draw.uniqueId,
                    drawDate = LotoDateUtils.formatFrenchDayAndDate(step.draw.timestamp),
                    dayKey = step.dayKey,
                    horaire = step.horaire,
                    game = step.game,
                    strictPastCount = step.strictPastSize,
                    maxPastTs = step.maxPastTs,
                    targetTimestamp = step.draw.timestamp,
                    causalVerified = step.maxPastTs < step.draw.timestamp,
                    predictedNumbers = predicted,
                    actualPos1 = step.actualPos1,
                    actualNumbers = step.actualNumbers,
                    matchedNumbers = matched,
                    matchCount = matched.size,
                    exactTargetHit = exactHit,
                    stepBrierScore = brier / 90.0
                )
            )
            appended++
        }

        val reindexed = combined.mapIndexed { i, st -> st.copy(stepIndex = i + 1) }
        val remainingPending = pending.filter { p ->
            reindexed.none { it.dayKey == p.dayKey && it.horaire.equals(p.horaire, ignoreCase = true) }
        }.takeLast(24)

        val ledger = PrecisionLedgerRecord(
            ledgerKey = buildLedgerKey(genome.mode, sig, periodKey),
            modeName = genome.mode.name,
            strategySignature = sig,
            periodKey = periodKey,
            genomeJson = genomeToJson(genome),
            frozenStepsJson = serializeSteps(reindexed),
            pendingPredictionsJson = serializePending(remainingPending),
            lastApprovedTimestamp = reindexed.maxOfOrNull { it.targetTimestamp } ?: (existingLedger?.lastApprovedTimestamp ?: 0L),
            updatedAt = System.currentTimeMillis()
        )
        val result = aggregateBacktest(
            genome = genome,
            steps = reindexed,
            periodLabel = periodKey,
            confirmationTestsCount = confirmationTestsCount,
            isAppendOnlyApproved = true,
            preservedCount = frozen.size,
            appendedCount = appended,
            liveMatchedCount = liveMatched
        )
        return result to ledger
    }

    fun recordLivePredictionInLedger(
        ledger: PrecisionLedgerRecord,
        prediction: PrecisionLivePrediction
    ): PrecisionLedgerRecord {
        val dayKey = LotoDateUtils.formatDayKey(prediction.targetTimestamp)
        val normH = prediction.horaire.trim().uppercase(Locale.ROOT)
        val list = deserializePending(ledger.pendingPredictionsJson).toMutableList()
        list.removeAll { it.dayKey == dayKey && it.horaire.equals(normH, ignoreCase = true) }
        list.add(PrecisionPendingPrediction(prediction.mode.name, dayKey, normH, prediction.targetTimestamp, prediction.predictedNumbers.sorted()))
        return ledger.copy(pendingPredictionsJson = serializePending(list.takeLast(24)), updatedAt = System.currentTimeMillis())
    }

    fun predictNextUpcoming(
        history: List<LotoDraw>,
        genome: PrecisionGenome,
        nextTarget: NextDrawTarget
    ): PrecisionLivePrediction {
        val strictPast = history
            .distinctBy { it.uniqueId }
            .filter { it.timestamp < nextTarget.targetTimestamp }
            .sortedWith(compareBy<LotoDraw> { it.timestamp }.thenBy { it.uniqueId })

        val normH = nextTarget.horaire.trim().uppercase(Locale.ROOT)
        val mat = extractFeatures(strictPast, normH)
        val s1 = DoubleArray(91); val s2 = DoubleArray(91)
        val p1 = DoubleArray(91); val p2 = DoubleArray(91); val marginal = DoubleArray(91)
        val predicted = selectDeterministicNumbers(genome, mat, s1, s2, p1, p2, marginal)
        val confs = predicted.map { num ->
            num to (marginal[num] * 100.0).toFloat().coerceIn(0.1f, 99.9f)
        }
        return PrecisionLivePrediction(
            mode = genome.mode,
            horaire = normH,
            dayLabel = nextTarget.dayLabel,
            targetTimestamp = nextTarget.targetTimestamp,
            predictedNumbers = predicted,
            confidencesPercent = confs,
            strategyDescription = genome.describeCompact(),
            trainedOnCount = strictPast.size,
            causalAuditNote = "Prédiction ${genome.mode.label} stricte (< $normH) sur ${strictPast.size} tirages passés • 0 dérive"
        )
    }

    fun buildDailyBlocks(
        steps: List<PrecisionBacktestStep>,
        mode: PrecisionTargetMode,
        periodLabel: String = "1 Jour"
    ): List<PrecisionDailyBlockSummary> {
        val n = steps.size
        if (n == 0) return emptyList()
        val use6Blocks = (periodLabel.contains("Jour") && !periodLabel.contains("Tout")) || periodLabel.startsWith("1 Jour")
        val minPerBlock = if (mode == PrecisionTargetMode.FIRST_BALL) 1 else 3
        val needPen = mode == PrecisionTargetMode.TWO_NUMBERS

        return if (use6Blocks) {
            val totalBlocks = (n + 5) / 6
            steps.withIndex().groupBy { (idx, _) ->
                val bFromEnd = (n - 1 - idx) / 6
                totalBlocks - 1 - bFromEnd
            }.entries.sortedBy { it.key }.map { (bIdx, list) ->
                val vals = list.map { it.value }
                val ex = vals.count { it.exactTargetHit }
                val one = vals.count { it.matchCount >= 1 }
                val penExact = if (vals.size >= 2) vals[vals.size - 2].exactTargetHit else false
                val d1 = vals.first().dayKey
                val d2 = vals.last().dayKey
                val lbl = if (d1 == d2) "${vals.first().drawDate} (${vals.size} tir.)" else "$d1 → $d2 (${vals.size} tir.)"
                PrecisionDailyBlockSummary("B${bIdx + 1}_$d2", lbl, vals.size, ex, one, ex >= minPerBlock && (!needPen || penExact), penExact)
            }
        } else {
            steps.groupBy { it.dayKey }.entries.sortedBy { it.key }.map { (k, vals) ->
                val ex = vals.count { it.exactTargetHit }
                val one = vals.count { it.matchCount >= 1 }
                val penExact = if (vals.size >= 2) vals[vals.size - 2].exactTargetHit else false
                PrecisionDailyBlockSummary(k, vals.first().drawDate, vals.size, ex, one, ex >= minPerBlock && (!needPen || penExact), penExact)
            }
        }
    }

    private fun aggregateBacktest(
        genome: PrecisionGenome,
        steps: List<PrecisionBacktestStep>,
        periodLabel: String,
        confirmationTestsCount: Int,
        isAppendOnlyApproved: Boolean,
        preservedCount: Int,
        appendedCount: Int,
        liveMatchedCount: Int
    ): PrecisionBacktestResult {
        val n = steps.size
        if (n == 0) {
            return PrecisionBacktestResult(
                genome.mode, periodLabel, 0, 0, 0, 0, 0, 0f, 1f, 0.1, 1.0, 1.0,
                true, isAppendOnlyApproved, preservedCount, appendedCount, liveMatchedCount,
                genome.signatureKey(), emptyList(), emptyList()
            )
        }
        val blocks = buildDailyBlocks(steps, genome.mode, periodLabel)

        var consec = 0
        var maxConsec = 0
        for (i in blocks.indices) {
            if (blocks[i].isQualifyingBlock) {
                consec++
                if (consec > maxConsec) maxConsec = consec
            } else {
                consec = 0
            }
        }
        val hitsList = steps.map { it.matchCount }
        val pRaw = computePrecisionPValue(genome.mode, hitsList)
        val pAdj = (pRaw * PcDiscoveryEngine.effectiveMultipleTestFactor(confirmationTestsCount)).coerceAtMost(1.0)
        val exactTotal = steps.count { it.exactTargetHit }
        val oneTotal = steps.count { it.matchCount >= 1 }
        return PrecisionBacktestResult(
            mode = genome.mode,
            periodLabel = periodLabel,
            totalTested = n,
            exactTargetCount = exactTotal,
            onePlusCount = oneTotal,
            qualifyingBlocksCount = blocks.count { it.isQualifyingBlock },
            maxConsecQualifyingBlocks = maxConsec,
            avgHits = hitsList.sum().toFloat() / n,
            zeroRate = (n - oneTotal).toFloat() / n,
            averageBrierScore = steps.sumOf { it.stepBrierScore } / n,
            pValueRaw = pRaw,
            pValueAdjusted = pAdj,
            allStepsCausalVerified = steps.all { it.causalVerified },
            isAppendOnlyApproved = isAppendOnlyApproved,
            preservedStepsCount = preservedCount,
            appendedStepsCount = appendedCount,
            liveMatchedCount = liveMatchedCount,
            strategySignature = genome.signatureKey(),
            steps = steps,
            dailyBlocks = blocks
        )
    }

    fun genomeToJson(g: PrecisionGenome): String = buildJsonObject {
        put("id", g.id)
        put("mode", g.mode.name)
        put("gen", g.generation)
        put("weights", buildJsonArray { g.weights.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
        put("secWeights", buildJsonArray { g.secondaryWeights.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
        put("transform", g.transform.name)
        put("boost", g.interactionBoost)
        put("temp", g.samplingTemp)
    }.toString()

    fun genomeFromJson(json: String): PrecisionGenome? = runCatching {
        val o = Json.parseToJsonElement(json).jsonObject
        val mode = PrecisionTargetMode.valueOf(o["mode"]?.jsonPrimitive?.contentOrNull ?: return null)
        val tr = PrecisionTransform.valueOf(o["transform"]?.jsonPrimitive?.contentOrNull ?: "LINEAR")
        PrecisionGenome(
            id = o["id"]?.jsonPrimitive?.contentOrNull ?: "restored",
            mode = mode,
            generation = o["gen"]?.jsonPrimitive?.intOrNull ?: 1,
            weights = o["weights"]?.jsonArray?.mapNotNull { it.jsonPrimitive.doubleOrNull }.orEmpty(),
            secondaryWeights = o["secWeights"]?.jsonArray?.mapNotNull { it.jsonPrimitive.doubleOrNull }.orEmpty(),
            transform = tr,
            interactionBoost = o["boost"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
            samplingTemp = o["temp"]?.jsonPrimitive?.doubleOrNull ?: 0.5,
            exploreNoise = 0.0
        )
    }.getOrNull()

    private fun serializeSteps(steps: List<PrecisionBacktestStep>): String = buildJsonArray {
        for (s in steps) {
            add(buildJsonObject {
                put("stepIndex", s.stepIndex)
                put("drawId", s.drawId)
                put("drawDate", s.drawDate)
                put("dayKey", s.dayKey)
                put("horaire", s.horaire)
                put("game", s.game)
                put("strictPastCount", s.strictPastCount)
                put("maxPastTs", s.maxPastTs)
                put("targetTimestamp", s.targetTimestamp)
                put("causalVerified", s.causalVerified)
                put("predictedNumbers", buildJsonArray { s.predictedNumbers.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
                put("actualPos1", s.actualPos1)
                put("actualNumbers", buildJsonArray { s.actualNumbers.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
                put("matchedNumbers", buildJsonArray { s.matchedNumbers.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
                put("matchCount", s.matchCount)
                put("exactTargetHit", s.exactTargetHit)
                put("stepBrierScore", s.stepBrierScore)
            })
        }
    }.toString()

    private fun deserializeSteps(json: String): List<PrecisionBacktestStep> {
        if (json.isBlank()) return emptyList()
        return runCatching {
            Json.parseToJsonElement(json).jsonArray.mapNotNull { el ->
                val o = el as? JsonObject ?: return@mapNotNull null
                PrecisionBacktestStep(
                    stepIndex = o["stepIndex"]?.jsonPrimitive?.intOrNull ?: 0,
                    drawId = o["drawId"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null,
                    drawDate = o["drawDate"]?.jsonPrimitive?.contentOrNull ?: "",
                    dayKey = o["dayKey"]?.jsonPrimitive?.contentOrNull ?: "",
                    horaire = o["horaire"]?.jsonPrimitive?.contentOrNull ?: "",
                    game = o["game"]?.jsonPrimitive?.contentOrNull ?: "",
                    strictPastCount = o["strictPastCount"]?.jsonPrimitive?.intOrNull ?: 0,
                    maxPastTs = o["maxPastTs"]?.jsonPrimitive?.longOrNull ?: 0L,
                    targetTimestamp = o["targetTimestamp"]?.jsonPrimitive?.longOrNull ?: 0L,
                    causalVerified = o["causalVerified"]?.jsonPrimitive?.booleanOrNull ?: true,
                    predictedNumbers = o["predictedNumbers"]?.jsonArray?.mapNotNull { it.jsonPrimitive.intOrNull }.orEmpty(),
                    actualPos1 = o["actualPos1"]?.jsonPrimitive?.intOrNull ?: 0,
                    actualNumbers = o["actualNumbers"]?.jsonArray?.mapNotNull { it.jsonPrimitive.intOrNull }.orEmpty(),
                    matchedNumbers = o["matchedNumbers"]?.jsonArray?.mapNotNull { it.jsonPrimitive.intOrNull }.orEmpty(),
                    matchCount = o["matchCount"]?.jsonPrimitive?.intOrNull ?: 0,
                    exactTargetHit = o["exactTargetHit"]?.jsonPrimitive?.booleanOrNull ?: false,
                    stepBrierScore = o["stepBrierScore"]?.jsonPrimitive?.doubleOrNull ?: 0.0
                )
            }
        }.getOrDefault(emptyList())
    }

    private fun serializePending(list: List<PrecisionPendingPrediction>): String = buildJsonArray {
        for (p in list) {
            add(buildJsonObject {
                put("modeName", p.modeName)
                put("dayKey", p.dayKey)
                put("horaire", p.horaire)
                put("targetTimestamp", p.targetTimestamp)
                put("predictedNumbers", buildJsonArray { p.predictedNumbers.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
            })
        }
    }.toString()

    private fun deserializePending(json: String): List<PrecisionPendingPrediction> {
        if (json.isBlank()) return emptyList()
        return runCatching {
            Json.parseToJsonElement(json).jsonArray.mapNotNull { el ->
                val o = el as? JsonObject ?: return@mapNotNull null
                PrecisionPendingPrediction(
                    modeName = o["modeName"]?.jsonPrimitive?.contentOrNull ?: "",
                    dayKey = o["dayKey"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null,
                    horaire = o["horaire"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null,
                    targetTimestamp = o["targetTimestamp"]?.jsonPrimitive?.longOrNull ?: 0L,
                    predictedNumbers = o["predictedNumbers"]?.jsonArray?.mapNotNull { it.jsonPrimitive.intOrNull }.orEmpty()
                )
            }
        }.getOrDefault(emptyList())
    }

    private fun round2(v: Double): Double = kotlin.math.round(v * 100.0) / 100.0
}
