package com.example.domain.engine

import com.example.data.local.LotoResultEntity
import com.example.domain.util.LotoDateUtils
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
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
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Primitives brutes et STRICTEMENT causales extraites uniquement du passe (t_train < t_cible).
 * Aucune strategie Loto pre-ecrite (ni Weibull, ni Markov, ni grille 9x10, ni Fourier).
 */
enum class RawFeature(val symbol: String) {
    X_NORM("x"),           // n / 90
    FREQ_GLOBAL("fg"),     // frequence empirique brute globale
    FREQ_W5("f5"),         // frequence brute sur les 5 derniers tirages
    FREQ_W15("f15"),       // frequence brute sur les 15 derniers tirages
    FREQ_W40("f40"),       // frequence brute sur les 40 derniers tirages
    GAP_NORM("gap"),       // ecart brut depuis la derniere apparition / (total + 1)
    PREV_GAP_NORM("pgap"), // avant-dernier ecart brut / (total + 1)
    LAST_HIT("hit1"),      // present dans S(t-1) : 1.0 ou 0.0
    LAG2_HIT("hit2"),      // present dans S(t-2) : 1.0 ou 0.0
    CO_LAST_RAW("co1"),    // co-occurrence brute avec S(t-1)
    HORAIRE_FREQ("fh"),    // frequence brute sur le meme horaire
    SAME_DAY_REP("rep")    // nombre de selections deja emises aujourd'hui (causal)
}

enum class UnaryOp(val symbol: String) {
    NEG("-"),
    ABS("abs"),
    SQRT_ABS("sqrt"),
    LOG1P_ABS("ln1p"),
    EXP_CLIP("exp"),
    SIN_PI("sin"),
    SIGMOID("sigm"),
    SQUARE("sq"),
    INV_SAFE("inv")
}

enum class BinaryOp(val symbol: String) {
    ADD("+"),
    SUB("-"),
    MUL("*"),
    DIV_SAFE("/"),
    MAX("max"),
    MIN("min"),
    POW_SAFE("^"),
    MOD_SAFE("mod")
}

sealed interface ExprNode {
    data class Var(val feature: RawFeature) : ExprNode
    data class Const(val value: Double) : ExprNode
    data class Unary(val op: UnaryOp, val child: ExprNode) : ExprNode
    data class Binary(val op: BinaryOp, val left: ExprNode, val right: ExprNode) : ExprNode
}

/**
 * Genome symbolique decouvert de maniere autonome par programmation genetique + optimisation numerique.
 */
data class SymbolicGenome(
    val id: String,
    val generation: Int,
    val parentIds: List<String>,
    val primaryExpr: ExprNode,
    val secondaryExpr: ExprNode,
    val primarySlots: Int, // 0..5 boules selectionnees depuis primaryExpr, (5 - primarySlots) depuis secondaryExpr
    val samplingTemp: Double,
    val exploreNoise: Double,
    val seed: Long
) {
    private val cachedPrimaryText: String by lazy { SymbolicFormula.prettyPrint(primaryExpr) }
    private val cachedSecondaryText: String by lazy { SymbolicFormula.prettyPrint(secondaryExpr) }
    private val cachedCompactSignature: String by lazy {
        "[${primarySlots}+${5 - primarySlots}] A=($cachedPrimaryText) | B=($cachedSecondaryText)"
    }

    fun describePrimary(): String = cachedPrimaryText
    fun describeSecondary(): String = cachedSecondaryText
    fun describeCompact(): String = cachedCompactSignature
}

class CompiledRpnProgram(
    val opCodes: IntArray,
    val opArgs: IntArray,
    val constants: DoubleArray
) {
    val instructionCount: Int get() = opCodes.size

    fun evaluateCpu(
        matrix: RawFeatureMatrix,
        number: Int,
        sameDayRepNorm: Double,
        scratchStack: DoubleArray = threadLocalStack.get()!!
    ): Double {
        var sp = 0
        val rows = matrix.featureRows
        val repIdx = RawFeature.SAME_DAY_REP.ordinal
        for (pc in opCodes.indices) {
            when (opCodes[pc]) {
                OP_VAR -> {
                    val featIdx = opArgs[pc]
                    scratchStack[sp++] = if (featIdx == repIdx) sameDayRepNorm else rows[featIdx][number]
                }
                OP_CONST -> scratchStack[sp++] = constants[opArgs[pc]]
                OP_UNARY -> {
                    val x = scratchStack[--sp]
                    scratchStack[sp++] = when (opArgs[pc]) {
                        0 -> -x
                        1 -> abs(x)
                        2 -> sqrt(abs(x))
                        3 -> ln(1.0 + abs(x))
                        4 -> exp(x.coerceIn(-4.0, 4.0))
                        5 -> sin(x.coerceIn(-10.0, 10.0) * PI)
                        6 -> 1.0 / (1.0 + exp(-x.coerceIn(-12.0, 12.0)))
                        7 -> {
                            val c = x.coerceIn(-100.0, 100.0)
                            c * c
                        }
                        else -> 1.0 / (1.0 + abs(x))
                    }
                }
                OP_BINARY -> {
                    val b = scratchStack[--sp]
                    val a = scratchStack[--sp]
                    scratchStack[sp++] = when (opArgs[pc]) {
                        0 -> a + b
                        1 -> a - b
                        2 -> a * b
                        3 -> a / (abs(b) + 0.1)
                        4 -> max(a, b)
                        5 -> min(a, b)
                        6 -> {
                            val mag = abs(a).coerceIn(1e-6, 10.0).pow(abs(b).coerceIn(0.2, 2.5))
                            if (a < 0.0) -mag else mag
                        }
                        else -> {
                            val m = abs(b) + 0.5
                            ((a % m) + m) % m
                        }
                    }
                }
            }
        }
        val res = if (sp > 0) scratchStack[0] else 0.0
        return if (res.isFinite()) res.coerceIn(-1e4, 1e4) else 0.0
    }

    companion object {
        const val OP_VAR = 0
        const val OP_CONST = 1
        const val OP_UNARY = 2
        const val OP_BINARY = 3
        const val MAX_STACK_DEPTH = 64

        private val threadLocalStack = object : ThreadLocal<DoubleArray>() {
            override fun initialValue(): DoubleArray = DoubleArray(MAX_STACK_DEPTH)
        }
    }
}

/**
 * Snapshot brut causal pour un tirage cible `t` a partir de `strictPast` uniquement.
 * Stocke les 12 vecteurs bruts pour les 90 numeros afin d'evaluer toute expression symbolique rapidement.
 */
class RawFeatureMatrix(
    val featureRows: Array<DoubleArray> // [RawFeature.entries.size][91]
) {
    fun get(featureIndex: Int, number: Int): Double = featureRows[featureIndex][number]
}

object SymbolicFormula {

    private const val MAX_DEPTH = 4
    private const val MAX_NODES = 21

    fun compileToRpn(node: ExprNode): CompiledRpnProgram {
        val codes = ArrayList<Int>(24)
        val args = ArrayList<Int>(24)
        val consts = ArrayList<Double>(8)

        fun emit(cur: ExprNode) {
            when (cur) {
                is ExprNode.Var -> {
                    codes.add(CompiledRpnProgram.OP_VAR)
                    args.add(cur.feature.ordinal)
                }
                is ExprNode.Const -> {
                    codes.add(CompiledRpnProgram.OP_CONST)
                    args.add(consts.size)
                    consts.add(cur.value)
                }
                is ExprNode.Unary -> {
                    emit(cur.child)
                    codes.add(CompiledRpnProgram.OP_UNARY)
                    args.add(cur.op.ordinal)
                }
                is ExprNode.Binary -> {
                    emit(cur.left)
                    emit(cur.right)
                    codes.add(CompiledRpnProgram.OP_BINARY)
                    args.add(cur.op.ordinal)
                }
            }
        }
        emit(node)
        return CompiledRpnProgram(codes.toIntArray(), args.toIntArray(), consts.toDoubleArray())
    }

    private inline fun scanDistinctBalls(draw: LotoResultEntity, seenStamp: IntArray, stamp: Int, onBall: (Int) -> Unit) {
        val b1 = draw.num1; val b2 = draw.num2; val b3 = draw.num3; val b4 = draw.num4; val b5 = draw.num5
        if (b1 in 1..90 && seenStamp[b1] != stamp) { seenStamp[b1] = stamp; onBall(b1) }
        if (b2 in 1..90 && seenStamp[b2] != stamp) { seenStamp[b2] = stamp; onBall(b2) }
        if (b3 in 1..90 && seenStamp[b3] != stamp) { seenStamp[b3] = stamp; onBall(b3) }
        if (b4 in 1..90 && seenStamp[b4] != stamp) { seenStamp[b4] = stamp; onBall(b4) }
        if (b5 in 1..90 && seenStamp[b5] != stamp) { seenStamp[b5] = stamp; onBall(b5) }
    }

    fun extractRawMatrix(
        strictPast: List<LotoResultEntity>,
        targetHoraire: String
    ): RawFeatureMatrix {
        val featCount = RawFeature.entries.size
        val rows = Array(featCount) { DoubleArray(91) }
        val total = strictPast.size

        val xRow = rows[RawFeature.X_NORM.ordinal]
        for (n in 1..90) {
            xRow[n] = n / 90.0
        }
        if (total == 0) {
            return RawFeatureMatrix(rows)
        }

        val globalCounts = IntArray(91)
        val w5Counts = IntArray(91)
        val w15Counts = IntArray(91)
        val w40Counts = IntArray(91)
        val horaireCounts = IntArray(91)
        var horaireDraws = 0

        val lastSeen = IntArray(91) { -1 }
        val prevSeen = IntArray(91) { -1 }
        val dedupStamp = IntArray(91)

        for (i in 0 until total) {
            val d = strictPast[i]
            val sameH = LotoDateUtils.normalizeHoraire(d) == targetHoraire
            if (sameH) horaireDraws++
            val distFromEnd = total - 1 - i
            scanDistinctBalls(d, dedupStamp, i + 1) { n ->
                globalCounts[n]++
                if (distFromEnd < 5) w5Counts[n]++
                if (distFromEnd < 15) w15Counts[n]++
                if (distFromEnd < 40) w40Counts[n]++
                if (sameH) horaireCounts[n]++
                prevSeen[n] = lastSeen[n]
                lastSeen[n] = i
            }
        }

        val lastMask = BooleanArray(91)
        scanDistinctBalls(strictPast[total - 1], dedupStamp, total + 10) { n -> lastMask[n] = true }
        val lag2Mask = BooleanArray(91)
        if (total >= 2) {
            scanDistinctBalls(strictPast[total - 2], dedupStamp, total + 11) { n -> lag2Mask[n] = true }
        }

        val coCounts = IntArray(91)
        var lastOverlapDraws = 0
        for (i in 0 until total) {
            val d = strictPast[i]
            val b1 = d.num1; val b2 = d.num2; val b3 = d.num3; val b4 = d.num4; val b5 = d.num5
            val hasOverlap =
                (b1 in 1..90 && lastMask[b1]) ||
                (b2 in 1..90 && lastMask[b2]) ||
                (b3 in 1..90 && lastMask[b3]) ||
                (b4 in 1..90 && lastMask[b4]) ||
                (b5 in 1..90 && lastMask[b5])
            if (hasOverlap) {
                lastOverlapDraws++
                scanDistinctBalls(d, dedupStamp, total + 100 + i) { n -> coCounts[n]++ }
            }
        }

        val invTotal = 1.0 / total.toDouble()
        val invW5 = 1.0 / min(5, total).toDouble()
        val invW15 = 1.0 / min(15, total).toDouble()
        val invW40 = 1.0 / min(40, total).toDouble()
        val invH = if (horaireDraws > 0) 1.0 / horaireDraws.toDouble() else 0.0
        val invCo = if (lastOverlapDraws > 0) 1.0 / lastOverlapDraws.toDouble() else 0.0
        val denomGap = total + 1.0

        val rowFreqGlobal = rows[RawFeature.FREQ_GLOBAL.ordinal]
        val rowFreqW5 = rows[RawFeature.FREQ_W5.ordinal]
        val rowFreqW15 = rows[RawFeature.FREQ_W15.ordinal]
        val rowFreqW40 = rows[RawFeature.FREQ_W40.ordinal]
        val rowGap = rows[RawFeature.GAP_NORM.ordinal]
        val rowPrevGap = rows[RawFeature.PREV_GAP_NORM.ordinal]
        val rowLastHit = rows[RawFeature.LAST_HIT.ordinal]
        val rowLag2Hit = rows[RawFeature.LAG2_HIT.ordinal]
        val rowCoLast = rows[RawFeature.CO_LAST_RAW.ordinal]
        val rowHoraire = rows[RawFeature.HORAIRE_FREQ.ordinal]

        for (n in 1..90) {
            rowFreqGlobal[n] = globalCounts[n] * invTotal
            rowFreqW5[n] = w5Counts[n] * invW5
            rowFreqW15[n] = w15Counts[n] * invW15
            rowFreqW40[n] = w40Counts[n] * invW40
            val ls = lastSeen[n]
            val ps = prevSeen[n]
            val gap = if (ls < 0) total else (total - 1 - ls)
            val prevGap = when {
                ls < 0 -> total
                ps < 0 -> ls
                else -> ls - ps
            }
            rowGap[n] = gap / denomGap
            rowPrevGap[n] = prevGap / denomGap
            rowLastHit[n] = if (lastMask[n]) 1.0 else 0.0
            rowLag2Hit[n] = if (lag2Mask[n]) 1.0 else 0.0
            rowCoLast[n] = coCounts[n] * invCo
            rowHoraire[n] = horaireCounts[n] * invH
        }
        return RawFeatureMatrix(rows)
    }

    /**
     * Extraction causale incrementale en O(T) (parite exacte avec extractRawMatrix),
     * avec exclusion stricte `past.timestamp < target.timestamp`.
     */
    fun extractRawMatrixSequence(
        chronological: List<LotoResultEntity>,
        targetIndices: IntArray = IntArray(chronological.size) { it }
    ): Map<Int, RawFeatureMatrix> {
        val nTotal = chronological.size
        if (nTotal == 0 || targetIndices.isEmpty()) return emptyMap()

        val sortedTargets = targetIndices.distinct().filter { it in 0 until nTotal }.sorted()
        if (sortedTargets.isEmpty()) return emptyMap()
        val maxTargetIdx = sortedTargets.last()
        val featCount = RawFeature.entries.size

        val drawBallFlat = IntArray(nTotal * 5)
        val drawBallLen = IntArray(nTotal)
        val drawLoMask = LongArray(nTotal)
        val drawHiMask = LongArray(nTotal)
        val horaireIdMap = HashMap<String, Int>(12)
        val horaireIds = IntArray(nTotal) { i ->
            horaireIdMap.getOrPut(LotoDateUtils.normalizeHoraire(chronological[i])) { horaireIdMap.size }
        }
        val numHoraires = horaireIdMap.size.coerceAtLeast(1)

        val seenTmp = BooleanArray(91)
        for (i in 0 until nTotal) {
            val d = chronological[i]
            var len = 0
            var lo = 0L
            var hi = 0L
            fun addB(v: Int) {
                if (v in 1..90 && !seenTmp[v]) {
                    seenTmp[v] = true
                    drawBallFlat[i * 5 + len] = v
                    len++
                    if (v <= 64) lo = lo or (1L shl (v - 1)) else hi = hi or (1L shl (v - 65))
                }
            }
            addB(d.num1); addB(d.num2); addB(d.num3); addB(d.num4); addB(d.num5)
            for (k in 0 until len) seenTmp[drawBallFlat[i * 5 + k]] = false
            drawBallLen[i] = len
            drawLoMask[i] = lo
            drawHiMask[i] = hi
        }

        val xNorm = DoubleArray(91) { n -> n / 90.0 }
        val globalCounts = IntArray(91)
        val w5Counts = IntArray(91)
        val w15Counts = IntArray(91)
        val w40Counts = IntArray(91)
        val horaireCounts = Array(numHoraires) { IntArray(91) }
        val horaireDraws = IntArray(numHoraires)
        val lastSeen = IntArray(91) { -1 }
        val prevSeen = IntArray(91) { -1 }

        val coCounts = IntArray(91)
        val lastMask = BooleanArray(91)
        val lag2Mask = BooleanArray(91)

        var committedEnd = 0
        var targetCursor = 0
        val out = HashMap<Int, RawFeatureMatrix>(sortedTargets.size * 2)

        for (t in 0..maxTargetIdx) {
            val targetDraw = chronological[t]
            while (committedEnd < t && chronological[committedEnd].timestamp < targetDraw.timestamp) {
                val idx = committedEnd
                val bLen = drawBallLen[idx]
                val baseOff = idx * 5
                val hId = horaireIds[idx]
                horaireDraws[hId]++

                for (k in 0 until bLen) {
                    val n = drawBallFlat[baseOff + k]
                    globalCounts[n]++
                    w5Counts[n]++
                    w15Counts[n]++
                    w40Counts[n]++
                    horaireCounts[hId][n]++
                    prevSeen[n] = lastSeen[n]
                    lastSeen[n] = idx
                }

                val exit5 = idx - 5
                if (exit5 >= 0) {
                    val eLen = drawBallLen[exit5]
                    val eOff = exit5 * 5
                    for (k in 0 until eLen) w5Counts[drawBallFlat[eOff + k]]--
                }
                val exit15 = idx - 15
                if (exit15 >= 0) {
                    val eLen = drawBallLen[exit15]
                    val eOff = exit15 * 5
                    for (k in 0 until eLen) w15Counts[drawBallFlat[eOff + k]]--
                }
                val exit40 = idx - 40
                if (exit40 >= 0) {
                    val eLen = drawBallLen[exit40]
                    val eOff = exit40 * 5
                    for (k in 0 until eLen) w40Counts[drawBallFlat[eOff + k]]--
                }
                committedEnd++
            }

            if (targetCursor < sortedTargets.size && sortedTargets[targetCursor] == t) {
                targetCursor++
                val total = committedEnd
                val rows = Array(featCount) { DoubleArray(91) }
                val xRow = rows[RawFeature.X_NORM.ordinal]
                for (n in 1..90) xRow[n] = xNorm[n]

                if (total == 0) {
                    out[t] = RawFeatureMatrix(rows)
                    continue
                }

                val lastIdx = total - 1
                val lastLen = drawBallLen[lastIdx]
                val lastOff = lastIdx * 5
                val lastLo = drawLoMask[lastIdx]
                val lastHi = drawHiMask[lastIdx]

                lastMask.fill(false)
                for (k in 0 until lastLen) lastMask[drawBallFlat[lastOff + k]] = true

                lag2Mask.fill(false)
                if (total >= 2) {
                    val lag2Idx = total - 2
                    val l2Len = drawBallLen[lag2Idx]
                    val l2Off = lag2Idx * 5
                    for (k in 0 until l2Len) lag2Mask[drawBallFlat[l2Off + k]] = true
                }

                coCounts.fill(0)
                var lastOverlapDraws = 0
                for (i in 0 until total) {
                    if ((drawLoMask[i] and lastLo) != 0L || (drawHiMask[i] and lastHi) != 0L) {
                        lastOverlapDraws++
                        val iLen = drawBallLen[i]
                        val iOff = i * 5
                        for (k in 0 until iLen) coCounts[drawBallFlat[iOff + k]]++
                    }
                }

                val tHId = horaireIds[t]
                val hDraws = horaireDraws[tHId]
                val hRow = horaireCounts[tHId]

                val invTotal = 1.0 / total.toDouble()
                val invW5 = 1.0 / min(5, total).toDouble()
                val invW15 = 1.0 / min(15, total).toDouble()
                val invW40 = 1.0 / min(40, total).toDouble()
                val invH = if (hDraws > 0) 1.0 / hDraws.toDouble() else 0.0
                val invCo = if (lastOverlapDraws > 0) 1.0 / lastOverlapDraws.toDouble() else 0.0
                val denomGap = total + 1.0

                val rowFreqGlobal = rows[RawFeature.FREQ_GLOBAL.ordinal]
                val rowFreqW5 = rows[RawFeature.FREQ_W5.ordinal]
                val rowFreqW15 = rows[RawFeature.FREQ_W15.ordinal]
                val rowFreqW40 = rows[RawFeature.FREQ_W40.ordinal]
                val rowGap = rows[RawFeature.GAP_NORM.ordinal]
                val rowPrevGap = rows[RawFeature.PREV_GAP_NORM.ordinal]
                val rowLastHit = rows[RawFeature.LAST_HIT.ordinal]
                val rowLag2Hit = rows[RawFeature.LAG2_HIT.ordinal]
                val rowCoLast = rows[RawFeature.CO_LAST_RAW.ordinal]
                val rowHoraire = rows[RawFeature.HORAIRE_FREQ.ordinal]

                for (n in 1..90) {
                    rowFreqGlobal[n] = globalCounts[n] * invTotal
                    rowFreqW5[n] = w5Counts[n] * invW5
                    rowFreqW15[n] = w15Counts[n] * invW15
                    rowFreqW40[n] = w40Counts[n] * invW40
                    val ls = lastSeen[n]
                    val ps = prevSeen[n]
                    val gap = if (ls < 0) total else (total - 1 - ls)
                    val prevGap = when {
                        ls < 0 -> total
                        ps < 0 -> ls
                        else -> ls - ps
                    }
                    rowGap[n] = gap / denomGap
                    rowPrevGap[n] = prevGap / denomGap
                    rowLastHit[n] = if (lastMask[n]) 1.0 else 0.0
                    rowLag2Hit[n] = if (lag2Mask[n]) 1.0 else 0.0
                    rowCoLast[n] = coCounts[n] * invCo
                    rowHoraire[n] = hRow[n] * invH
                }
                out[t] = RawFeatureMatrix(rows)
            }
        }
        return out
    }

    fun evaluateForNumber(
        node: ExprNode,
        matrix: RawFeatureMatrix,
        number: Int,
        sameDayRepNorm: Double
    ): Double {
        val v = when (node) {
            is ExprNode.Var -> {
                if (node.feature == RawFeature.SAME_DAY_REP) {
                    sameDayRepNorm
                } else {
                    matrix.get(node.feature.ordinal, number)
                }
            }
            is ExprNode.Const -> node.value
            is ExprNode.Unary -> {
                val x = evaluateForNumber(node.child, matrix, number, sameDayRepNorm)
                when (node.op) {
                    UnaryOp.NEG -> -x
                    UnaryOp.ABS -> abs(x)
                    UnaryOp.SQRT_ABS -> sqrt(abs(x))
                    UnaryOp.LOG1P_ABS -> ln(1.0 + abs(x))
                    UnaryOp.EXP_CLIP -> exp(x.coerceIn(-4.0, 4.0))
                    UnaryOp.SIN_PI -> sin((x.coerceIn(-10.0, 10.0)) * PI)
                    UnaryOp.SIGMOID -> 1.0 / (1.0 + exp(-x.coerceIn(-12.0, 12.0)))
                    UnaryOp.SQUARE -> {
                        val c = x.coerceIn(-100.0, 100.0)
                        c * c
                    }
                    UnaryOp.INV_SAFE -> 1.0 / (1.0 + abs(x))
                }
            }
            is ExprNode.Binary -> {
                val a = evaluateForNumber(node.left, matrix, number, sameDayRepNorm)
                val b = evaluateForNumber(node.right, matrix, number, sameDayRepNorm)
                when (node.op) {
                    BinaryOp.ADD -> a + b
                    BinaryOp.SUB -> a - b
                    BinaryOp.MUL -> a * b
                    BinaryOp.DIV_SAFE -> a / (abs(b) + 0.1)
                    BinaryOp.MAX -> max(a, b)
                    BinaryOp.MIN -> min(a, b)
                    BinaryOp.POW_SAFE -> {
                        val base = abs(a).coerceIn(1e-6, 10.0)
                        val expVal = abs(b).coerceIn(0.2, 2.5)
                        val mag = base.pow(expVal)
                        if (a < 0.0) -mag else mag
                    }
                    BinaryOp.MOD_SAFE -> {
                        val m = abs(b) + 0.5
                        ((a % m) + m) % m
                    }
                }
            }
        }
        return if (v.isFinite()) v.coerceIn(-1e4, 1e4) else 0.0
    }

    fun simplify(node: ExprNode): ExprNode {
        val s = when (node) {
            is ExprNode.Var, is ExprNode.Const -> node
            is ExprNode.Unary -> {
                val child = simplify(node.child)
                if (child is ExprNode.Const) {
                    val v = evalConstantUnary(node.op, child.value)
                    ExprNode.Const(v)
                } else if (node.op == UnaryOp.NEG && child is ExprNode.Unary && child.op == UnaryOp.NEG) {
                    child.child
                } else if (node.op == UnaryOp.ABS && child is ExprNode.Unary && child.op == UnaryOp.ABS) {
                    child
                } else {
                    ExprNode.Unary(node.op, child)
                }
            }
            is ExprNode.Binary -> {
                val l = simplify(node.left)
                val r = simplify(node.right)
                if (l is ExprNode.Const && r is ExprNode.Const) {
                    ExprNode.Const(evalConstantBinary(node.op, l.value, r.value))
                } else if (node.op == BinaryOp.ADD && r is ExprNode.Const && abs(r.value) < 1e-6) {
                    l
                } else if (node.op == BinaryOp.ADD && l is ExprNode.Const && abs(l.value) < 1e-6) {
                    r
                } else if (node.op == BinaryOp.SUB && r is ExprNode.Const && abs(r.value) < 1e-6) {
                    l
                } else if (node.op == BinaryOp.MUL && ((l is ExprNode.Const && abs(l.value) < 1e-6) || (r is ExprNode.Const && abs(r.value) < 1e-6))) {
                    ExprNode.Const(0.0)
                } else if (node.op == BinaryOp.MUL && r is ExprNode.Const && abs(r.value - 1.0) < 1e-6) {
                    l
                } else if (node.op == BinaryOp.MUL && l is ExprNode.Const && abs(l.value - 1.0) < 1e-6) {
                    r
                } else if (node.op == BinaryOp.SUB && l == r) {
                    ExprNode.Const(0.0)
                } else {
                    ExprNode.Binary(node.op, l, r)
                }
            }
        }
        return if (nodeCount(s) > MAX_NODES || depth(s) > MAX_DEPTH) trimToDepth(s, MAX_DEPTH) else s
    }

    fun prettyPrint(node: ExprNode): String {
        return when (node) {
            is ExprNode.Var -> node.feature.symbol
            is ExprNode.Const -> String.format(Locale.US, "%.2f", node.value)
            is ExprNode.Unary -> {
                val inner = prettyPrint(node.child)
                if (node.op == UnaryOp.NEG) "-($inner)" else "${node.op.symbol}($inner)"
            }
            is ExprNode.Binary -> {
                val l = prettyPrint(node.left)
                val r = prettyPrint(node.right)
                when (node.op) {
                    BinaryOp.MAX, BinaryOp.MIN, BinaryOp.MOD_SAFE -> "${node.op.symbol}($l, $r)"
                    else -> "($l ${node.op.symbol} $r)"
                }
            }
        }
    }

    fun nodeCount(node: ExprNode): Int = when (node) {
        is ExprNode.Var, is ExprNode.Const -> 1
        is ExprNode.Unary -> 1 + nodeCount(node.child)
        is ExprNode.Binary -> 1 + nodeCount(node.left) + nodeCount(node.right)
    }

    fun depth(node: ExprNode): Int = when (node) {
        is ExprNode.Var, is ExprNode.Const -> 1
        is ExprNode.Unary -> 1 + depth(node.child)
        is ExprNode.Binary -> 1 + max(depth(node.left), depth(node.right))
    }

    fun usedFeatures(node: ExprNode): Set<RawFeature> {
        val acc = HashSet<RawFeature>()
        fun collect(cur: ExprNode) {
            when (cur) {
                is ExprNode.Var -> acc.add(cur.feature)
                is ExprNode.Const -> Unit
                is ExprNode.Unary -> collect(cur.child)
                is ExprNode.Binary -> {
                    collect(cur.left)
                    collect(cur.right)
                }
            }
        }
        collect(node)
        return acc
    }

    fun randomExpr(maxDepth: Int, rng: Random): ExprNode {
        val expr = buildRandomExpr(maxDepth.coerceIn(1, MAX_DEPTH), rng)
        // Assurer qu'au moins une variable apparait pour eviter une formule constante triviale
        val simplified = simplify(expr)
        return if (usedFeatures(simplified).isEmpty()) {
            val feat = RawFeature.entries[rng.nextInt(RawFeature.entries.size)]
            val c = roundConst(0.2 + rng.nextDouble() * 1.6)
            ExprNode.Binary(BinaryOp.MUL, ExprNode.Const(c), ExprNode.Var(feat))
        } else {
            simplified
        }
    }

    fun crossover(a: ExprNode, b: ExprNode, rng: Random): ExprNode {
        val subB = pickRandomSubtree(b, rng)
        val replaced = replaceRandomSubtree(a, subB, rng)
        val simplified = simplify(replaced)
        return if (usedFeatures(simplified).isEmpty()) simplify(a) else simplified
    }

    fun mutate(node: ExprNode, rng: Random): ExprNode {
        val mutated = mutateStep(node, rng)
        val simplified = simplify(mutated)
        return if (usedFeatures(simplified).isEmpty()) randomExpr(3, rng) else simplified
    }

    // ---- Serialisation JSON propre pour persistance Room ----

    fun exprToJson(node: ExprNode): JsonObject = when (node) {
        is ExprNode.Var -> buildJsonObject {
            put("t", "var")
            put("f", node.feature.name)
        }
        is ExprNode.Const -> buildJsonObject {
            put("t", "const")
            put("v", node.value)
        }
        is ExprNode.Unary -> buildJsonObject {
            put("t", "u")
            put("op", node.op.name)
            put("c", exprToJson(node.child))
        }
        is ExprNode.Binary -> buildJsonObject {
            put("t", "b")
            put("op", node.op.name)
            put("l", exprToJson(node.left))
            put("r", exprToJson(node.right))
        }
    }

    fun exprFromJson(obj: JsonObject): ExprNode {
        return when (obj["t"]?.jsonPrimitive?.contentOrNull) {
            "var" -> {
                val fName = obj["f"]?.jsonPrimitive?.contentOrNull ?: RawFeature.GAP_NORM.name
                val feat = RawFeature.entries.firstOrNull { it.name == fName } ?: RawFeature.GAP_NORM
                ExprNode.Var(feat)
            }
            "const" -> ExprNode.Const(obj["v"]?.jsonPrimitive?.doubleOrNull ?: 1.0)
            "u" -> {
                val opName = obj["op"]?.jsonPrimitive?.contentOrNull ?: UnaryOp.ABS.name
                val op = UnaryOp.entries.firstOrNull { it.name == opName } ?: UnaryOp.ABS
                ExprNode.Unary(op, exprFromJson(obj["c"]!!.jsonObject))
            }
            "b" -> {
                val opName = obj["op"]?.jsonPrimitive?.contentOrNull ?: BinaryOp.ADD.name
                val op = BinaryOp.entries.firstOrNull { it.name == opName } ?: BinaryOp.ADD
                ExprNode.Binary(op, exprFromJson(obj["l"]!!.jsonObject), exprFromJson(obj["r"]!!.jsonObject))
            }
            else -> ExprNode.Var(RawFeature.GAP_NORM)
        }
    }

    fun genomeToJson(genome: SymbolicGenome): String {
        val obj = buildJsonObject {
            put("id", genome.id)
            put("gen", genome.generation)
            put("parents", buildJsonArray { genome.parentIds.forEach { add(it) } })
            put("primary", exprToJson(genome.primaryExpr))
            put("secondary", exprToJson(genome.secondaryExpr))
            put("slots", genome.primarySlots)
            put("temp", genome.samplingTemp)
            put("noise", genome.exploreNoise)
            put("seed", genome.seed)
        }
        return obj.toString()
    }

    fun genomeFromJson(raw: String): SymbolicGenome {
        val obj = Json.parseToJsonElement(raw).jsonObject
        return SymbolicGenome(
            id = obj["id"]?.jsonPrimitive?.contentOrNull ?: "restored",
            generation = obj["gen"]?.jsonPrimitive?.intOrNull ?: 1,
            parentIds = obj["parents"]?.jsonArray?.mapNotNull { it.jsonPrimitive.contentOrNull }.orEmpty(),
            primaryExpr = exprFromJson(obj["primary"]!!.jsonObject),
            secondaryExpr = exprFromJson(obj["secondary"]!!.jsonObject),
            primarySlots = (obj["slots"]?.jsonPrimitive?.intOrNull ?: 3).coerceIn(0, 5),
            samplingTemp = (obj["temp"]?.jsonPrimitive?.doubleOrNull ?: 0.7).coerceIn(0.15, 1.8),
            exploreNoise = (obj["noise"]?.jsonPrimitive?.doubleOrNull ?: 0.15).coerceIn(0.02, 0.45),
            seed = obj["seed"]?.jsonPrimitive?.longOrNull ?: 1L
        )
    }

    fun genomeListToJson(genomes: List<SymbolicGenome>): String {
        val arr = JsonArray(genomes.map { Json.parseToJsonElement(genomeToJson(it)) })
        return arr.toString()
    }

    fun genomeListFromJson(raw: String): List<SymbolicGenome> {
        if (raw.isBlank()) return emptyList()
        return try {
            val arr = Json.parseToJsonElement(raw).jsonArray
            arr.map { genomeFromJson(it.toString()) }
        } catch (_: Exception) {
            emptyList()
        }
    }

    private fun extractDistinct(draw: LotoResultEntity): List<Int> {
        return listOf(draw.num1, draw.num2, draw.num3, draw.num4, draw.num5)
            .filter { it in 1..90 }
            .distinct()
    }

    private fun buildRandomExpr(depthLeft: Int, rng: Random): ExprNode {
        if (depthLeft <= 1 || (depthLeft < MAX_DEPTH && rng.nextDouble() < 0.32)) {
            return if (rng.nextDouble() < 0.74) {
                ExprNode.Var(RawFeature.entries[rng.nextInt(RawFeature.entries.size)])
            } else {
                ExprNode.Const(roundConst((rng.nextDouble() * 4.0) - 2.0))
            }
        }
        return if (rng.nextDouble() < 0.30) {
            val op = UnaryOp.entries[rng.nextInt(UnaryOp.entries.size)]
            ExprNode.Unary(op, buildRandomExpr(depthLeft - 1, rng))
        } else {
            val op = BinaryOp.entries[rng.nextInt(BinaryOp.entries.size)]
            ExprNode.Binary(
                op,
                buildRandomExpr(depthLeft - 1, rng),
                buildRandomExpr(depthLeft - 1, rng)
            )
        }
    }

    private fun collectSubtrees(node: ExprNode, out: MutableList<ExprNode>) {
        out.add(node)
        when (node) {
            is ExprNode.Unary -> collectSubtrees(node.child, out)
            is ExprNode.Binary -> {
                collectSubtrees(node.left, out)
                collectSubtrees(node.right, out)
            }
            else -> Unit
        }
    }

    private fun pickRandomSubtree(node: ExprNode, rng: Random): ExprNode {
        val all = ArrayList<ExprNode>()
        collectSubtrees(node, all)
        return all[rng.nextInt(all.size)]
    }

    private fun replaceRandomSubtree(target: ExprNode, replacement: ExprNode, rng: Random): ExprNode {
        val all = ArrayList<ExprNode>()
        collectSubtrees(target, all)
        val chosenIdx = rng.nextInt(all.size)
        var cursor = 0
        fun walk(cur: ExprNode): ExprNode {
            val myIdx = cursor++
            if (myIdx == chosenIdx) return replacement
            return when (cur) {
                is ExprNode.Var, is ExprNode.Const -> cur
                is ExprNode.Unary -> ExprNode.Unary(cur.op, walk(cur.child))
                is ExprNode.Binary -> ExprNode.Binary(cur.op, walk(cur.left), walk(cur.right))
            }
        }
        return walk(target)
    }

    private fun mutateStep(node: ExprNode, rng: Random): ExprNode {
        val roll = rng.nextDouble()
        if (roll < 0.22) {
            // Remplacer un sous-arbre complet
            val newSub = buildRandomExpr(2, rng)
            return replaceRandomSubtree(node, newSub, rng)
        }
        return when (node) {
            is ExprNode.Const -> ExprNode.Const(roundConst((node.value + (rng.nextDouble() * 1.2 - 0.6)).coerceIn(-5.0, 5.0)))
            is ExprNode.Var -> if (rng.nextDouble() < 0.65) {
                ExprNode.Var(RawFeature.entries[rng.nextInt(RawFeature.entries.size)])
            } else {
                ExprNode.Unary(UnaryOp.entries[rng.nextInt(UnaryOp.entries.size)], node)
            }
            is ExprNode.Unary -> if (rng.nextDouble() < 0.35) {
                ExprNode.Unary(UnaryOp.entries[rng.nextInt(UnaryOp.entries.size)], node.child)
            } else if (rng.nextDouble() < 0.15) {
                node.child // simplification structurelle (hoist)
            } else {
                ExprNode.Unary(node.op, mutateStep(node.child, rng))
            }
            is ExprNode.Binary -> if (rng.nextDouble() < 0.30) {
                ExprNode.Binary(BinaryOp.entries[rng.nextInt(BinaryOp.entries.size)], node.left, node.right)
            } else if (rng.nextBoolean()) {
                ExprNode.Binary(node.op, mutateStep(node.left, rng), node.right)
            } else {
                ExprNode.Binary(node.op, node.left, mutateStep(node.right, rng))
            }
        }
    }

    private fun trimToDepth(node: ExprNode, depthLeft: Int): ExprNode {
        if (depthLeft <= 1) {
            return when (node) {
                is ExprNode.Var, is ExprNode.Const -> node
                else -> {
                    val feats = usedFeatures(node)
                    if (feats.isNotEmpty()) ExprNode.Var(feats.first()) else ExprNode.Const(1.0)
                }
            }
        }
        return when (node) {
            is ExprNode.Var, is ExprNode.Const -> node
            is ExprNode.Unary -> ExprNode.Unary(node.op, trimToDepth(node.child, depthLeft - 1))
            is ExprNode.Binary -> ExprNode.Binary(
                node.op,
                trimToDepth(node.left, depthLeft - 1),
                trimToDepth(node.right, depthLeft - 1)
            )
        }
    }

    private fun evalConstantUnary(op: UnaryOp, x: Double): Double = when (op) {
        UnaryOp.NEG -> -x
        UnaryOp.ABS -> abs(x)
        UnaryOp.SQRT_ABS -> sqrt(abs(x))
        UnaryOp.LOG1P_ABS -> ln(1.0 + abs(x))
        UnaryOp.EXP_CLIP -> exp(x.coerceIn(-4.0, 4.0))
        UnaryOp.SIN_PI -> sin(x.coerceIn(-10.0, 10.0) * PI)
        UnaryOp.SIGMOID -> 1.0 / (1.0 + exp(-x.coerceIn(-12.0, 12.0)))
        UnaryOp.SQUARE -> x.coerceIn(-100.0, 100.0).let { it * it }
        UnaryOp.INV_SAFE -> 1.0 / (1.0 + abs(x))
    }.let { if (it.isFinite()) roundConst(it.coerceIn(-1e3, 1e3)) else 0.0 }

    private fun evalConstantBinary(op: BinaryOp, a: Double, b: Double): Double = when (op) {
        BinaryOp.ADD -> a + b
        BinaryOp.SUB -> a - b
        BinaryOp.MUL -> a * b
        BinaryOp.DIV_SAFE -> a / (abs(b) + 0.1)
        BinaryOp.MAX -> max(a, b)
        BinaryOp.MIN -> min(a, b)
        BinaryOp.POW_SAFE -> {
            val mag = abs(a).coerceIn(1e-6, 10.0).pow(abs(b).coerceIn(0.2, 2.5))
            if (a < 0.0) -mag else mag
        }
        BinaryOp.MOD_SAFE -> {
            val m = abs(b) + 0.5
            ((a % m) + m) % m
        }
    }.let { if (it.isFinite()) roundConst(it.coerceIn(-1e3, 1e3)) else 0.0 }

    private fun roundConst(v: Double): Double = kotlin.math.round(v * 1000.0) / 1000.0
}
