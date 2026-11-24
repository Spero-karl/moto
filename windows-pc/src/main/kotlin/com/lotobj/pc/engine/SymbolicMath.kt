package com.lotobj.pc.engine

import com.lotobj.pc.model.BacktestPeriod
import com.lotobj.pc.model.BinaryOp
import com.lotobj.pc.model.ExprNode
import com.lotobj.pc.model.LotoDraw
import com.lotobj.pc.model.RawFeature
import com.lotobj.pc.model.RawFeatureMatrix
import com.lotobj.pc.model.SymbolicGenome
import com.lotobj.pc.model.UnaryOp
import com.lotobj.pc.util.LotoDateUtils
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
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
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Programme RPN plat compilé depuis ExprNode pour exécution vectorisée sur GPU OpenCL C et CPU rapide.
 * Opcodes:
 *   0 = PUSH_VAR(featureIdx)
 *   1 = PUSH_CONST(constFloatIdx)
 *   2 = UNARY(unaryOrdinal)
 *   3 = BINARY(binaryOrdinal)
 */
data class CompiledRpnProgram(
    val opCodes: IntArray,
    val opArgs: IntArray,
    val constants: FloatArray,
    val usesSameDayRep: Boolean
) {
    val instructionCount: Int get() = opCodes.size

    fun evaluateCpu(
        matrix: RawFeatureMatrix,
        number: Int,
        sameDayRepNorm: Double,
        scratchStack: DoubleArray = threadLocalStack.get()
    ): Double {
        val stack = scratchStack
        var sp = 0
        for (i in opCodes.indices) {
            when (opCodes[i]) {
                OP_VAR -> {
                    val featIdx = opArgs[i]
                    val v = if (featIdx == RawFeature.SAME_DAY_REP.ordinal) {
                        sameDayRepNorm
                    } else {
                        matrix.get(featIdx, number)
                    }
                    if (sp < MAX_STACK_DEPTH) stack[sp++] = v
                }
                OP_CONST -> {
                    if (sp < MAX_STACK_DEPTH) stack[sp++] = constants[opArgs[i]].toDouble()
                }
                OP_UNARY -> {
                    if (sp < 1) continue
                    val x = stack[--sp]
                    val res = when (opArgs[i]) {
                        UnaryOp.NEG.ordinal -> -x
                        UnaryOp.ABS.ordinal -> abs(x)
                        UnaryOp.SQRT_ABS.ordinal -> sqrt(abs(x))
                        UnaryOp.LOG1P_ABS.ordinal -> ln(1.0 + abs(x))
                        UnaryOp.EXP_CLIP.ordinal -> exp(x.coerceIn(-4.0, 4.0))
                        UnaryOp.SIN_PI.ordinal -> sin(x.coerceIn(-10.0, 10.0) * PI)
                        UnaryOp.SIGMOID.ordinal -> 1.0 / (1.0 + exp(-x.coerceIn(-12.0, 12.0)))
                        UnaryOp.SQUARE.ordinal -> {
                            val c = x.coerceIn(-100.0, 100.0)
                            c * c
                        }
                        UnaryOp.INV_SAFE.ordinal -> 1.0 / (1.0 + abs(x))
                        UnaryOp.TANH.ordinal -> {
                            val e2 = exp(2.0 * x.coerceIn(-6.0, 6.0))
                            (e2 - 1.0) / (e2 + 1.0)
                        }
                        UnaryOp.COS_PI.ordinal -> cos(x.coerceIn(-10.0, 10.0) * PI)
                        UnaryOp.CBRT.ordinal -> {
                            val ax = abs(x)
                            val r = if (ax > 1e-12) ax.pow(1.0 / 3.0) else 0.0
                            if (x < 0.0) -r else r
                        }
                        UnaryOp.SOFTPLUS.ordinal -> ln(1.0 + exp(x.coerceIn(-10.0, 6.0)))
                        UnaryOp.LUCKY9_WAVE.ordinal -> sin(x.coerceIn(-4.0, 4.0) * 9.0 * PI)
                        UnaryOp.STEP_POS.ordinal -> if (x > 0.0) 1.0 else 0.0
                        else -> 0.0
                    }
                    stack[sp++] = if (res.isFinite()) res.coerceIn(-1e4, 1e4) else 0.0
                }
                OP_BINARY -> {
                    if (sp < 2) continue
                    val b = stack[--sp]
                    val a = stack[--sp]
                    val res = when (opArgs[i]) {
                        BinaryOp.ADD.ordinal -> a + b
                        BinaryOp.SUB.ordinal -> a - b
                        BinaryOp.MUL.ordinal -> a * b
                        BinaryOp.DIV_SAFE.ordinal -> a / (abs(b) + 0.1)
                        BinaryOp.MAX.ordinal -> max(a, b)
                        BinaryOp.MIN.ordinal -> min(a, b)
                        BinaryOp.POW_SAFE.ordinal -> {
                            val base = abs(a).coerceIn(1e-6, 10.0)
                            val expVal = abs(b).coerceIn(0.2, 2.5)
                            val mag = base.pow(expVal)
                            if (a < 0.0) -mag else mag
                        }
                        BinaryOp.MOD_SAFE.ordinal -> {
                            val m = abs(b) + 0.5
                            ((a % m) + m) % m
                        }
                        BinaryOp.HYPOT_NORM.ordinal -> sqrt(min(a * a + b * b, 1e8)) * 0.5
                        BinaryOp.MEAN2.ordinal -> 0.5 * (a + b)
                        BinaryOp.ATAN2_PI.ordinal -> {
                            if (abs(a) < 1e-5 && abs(b) < 1e-5) 0.0 else atan2(a, b) / PI
                        }
                        BinaryOp.TRIANG_BLEND.ordinal -> 0.5 * abs(a - b) + min(a, b)
                        BinaryOp.EXP_DIFF.ordinal -> exp(-min(abs(a - b), 8.0))
                        else -> 0.0
                    }
                    stack[sp++] = if (res.isFinite()) res.coerceIn(-1e4, 1e4) else 0.0
                }
            }
        }
        val out = if (sp > 0) stack[sp - 1] else 0.0
        return if (out.isFinite()) out.coerceIn(-1e4, 1e4) else 0.0
    }

    companion object {
        const val OP_VAR = 0
        const val OP_CONST = 1
        const val OP_UNARY = 2
        const val OP_BINARY = 3
        const val MAX_STACK_DEPTH = 64
        private val threadLocalStack = ThreadLocal.withInitial { DoubleArray(MAX_STACK_DEPTH) }
    }
}

object SymbolicFormula {

    private const val MAX_DEPTH = 5
    private const val MAX_NODES = 35

    val GPU_CAPABLE_FEATURES: List<RawFeature> = RawFeature.entries.filter { it != RawFeature.SAME_DAY_REP }

    fun pickRandomFeature(rng: Random, gpuBias: Double = 0.90): RawFeature {
        return if (rng.nextDouble() < gpuBias) {
            GPU_CAPABLE_FEATURES[rng.nextInt(GPU_CAPABLE_FEATURES.size)]
        } else {
            RawFeature.entries[rng.nextInt(RawFeature.entries.size)]
        }
    }

    val THEOREM_PARADIGMS: List<String> = listOf(
        "Congruence Base 7 & Lucky 9",
        "Cube90 & Congruence Affine",
        "Triangulation & Grille 9×10",
        "Limites Weibull & Proba Z-Score",
        "Extrapolation Exponentielle & Flux",
        "Harmoniques Fourier & Spirale",
        "Synthèse Hybride Profonde"
    )

    fun compileToRpn(node: ExprNode): CompiledRpnProgram {
        val opCodes = ArrayList<Int>(MAX_NODES)
        val opArgs = ArrayList<Int>(MAX_NODES)
        val constants = ArrayList<Float>(12)
        var usesRep = false

        fun emit(cur: ExprNode) {
            when (cur) {
                is ExprNode.Var -> {
                    if (cur.feature == RawFeature.SAME_DAY_REP) usesRep = true
                    opCodes.add(CompiledRpnProgram.OP_VAR)
                    opArgs.add(cur.feature.ordinal)
                }
                is ExprNode.Const -> {
                    val idx = constants.size
                    constants.add(cur.value.toFloat())
                    opCodes.add(CompiledRpnProgram.OP_CONST)
                    opArgs.add(idx)
                }
                is ExprNode.Unary -> {
                    emit(cur.child)
                    opCodes.add(CompiledRpnProgram.OP_UNARY)
                    opArgs.add(cur.op.ordinal)
                }
                is ExprNode.Binary -> {
                    emit(cur.left)
                    emit(cur.right)
                    opCodes.add(CompiledRpnProgram.OP_BINARY)
                    opArgs.add(cur.op.ordinal)
                }
            }
        }

        emit(node)
        return CompiledRpnProgram(
            opCodes = opCodes.toIntArray(),
            opArgs = opArgs.toIntArray(),
            constants = constants.toFloatArray(),
            usesSameDayRep = usesRep
        )
    }

    fun extractDistinctBalls(draw: LotoDraw): List<Int> {
        return listOf(draw.num1, draw.num2, draw.num3, draw.num4, draw.num5)
            .filter { it in 1..90 }
            .distinct()
    }

    fun extractRawMatrix(
        strictPast: List<LotoDraw>,
        targetHoraire: String
    ): RawFeatureMatrix {
        val featCount = RawFeature.entries.size
        val rows = Array(featCount) { DoubleArray(91) }
        val total = strictPast.size

        for (n in 1..90) {
            rows[RawFeature.X_NORM.ordinal][n] = n / 90.0
            rows[RawFeature.BASE7_NORM.ordinal][n] = (n % 7) / 6.0
            rows[RawFeature.LUCKY_ROOT_NORM.ordinal][n] = (1 + ((n - 1) % 9)) / 9.0
            rows[RawFeature.MOD10_UNIT_NORM.ordinal][n] = (n % 10) / 9.0
            rows[RawFeature.MOD11_HARMONIC.ordinal][n] = (n % 11) / 10.0
        }
        if (total == 0) return RawFeatureMatrix(rows)

        val globalCounts = IntArray(91)
        val w5Counts = IntArray(91)
        val w15Counts = IntArray(91)
        val w40Counts = IntArray(91)
        val horaireCounts = IntArray(91)
        var horaireDraws = 0

        val mod7RecentCounts = IntArray(7)
        val mod11RecentCounts = IntArray(11)
        val luckyAllCounts = IntArray(10) // 1..9
        val cube90ClassCounts = IntArray(90)

        val lastSeen = IntArray(91) { -1 }
        val prevSeen = IntArray(91) { -1 }

        val lastBalls = IntArray(5)
        var lastBallsCount = 0
        val lastDrawMask = BooleanArray(91)
        val lag2Balls = IntArray(5)
        var lag2Count = 0
        val lag2Mask = BooleanArray(91)

        val coCounts = IntArray(91)
        var lastOverlapDraws = 0

        // Pré-extraire les boules du dernier et de l'avant-dernier tirage sans allocation
        val lastDraw = strictPast[total - 1]
        fun fillBalls(d: LotoDraw, outArr: IntArray, mask: BooleanArray): Int {
            var c = 0
            val n1 = d.num1; if (n1 in 1..90 && !mask[n1]) { mask[n1] = true; outArr[c++] = n1 }
            val n2 = d.num2; if (n2 in 1..90 && !mask[n2]) { mask[n2] = true; outArr[c++] = n2 }
            val n3 = d.num3; if (n3 in 1..90 && !mask[n3]) { mask[n3] = true; outArr[c++] = n3 }
            val n4 = d.num4; if (n4 in 1..90 && !mask[n4]) { mask[n4] = true; outArr[c++] = n4 }
            val n5 = d.num5; if (n5 in 1..90 && !mask[n5]) { mask[n5] = true; outArr[c++] = n5 }
            return c
        }
        lastBallsCount = fillBalls(lastDraw, lastBalls, lastDrawMask)
        if (total >= 2) {
            lag2Count = fillBalls(strictPast[total - 2], lag2Balls, lag2Mask)
        }

        val drawMaskTmp = BooleanArray(91)
        val drawBallsTmp = IntArray(5)

        for (i in 0 until total) {
            val d = strictPast[i]
            val sameH = LotoDateUtils.normalizeHoraire(d) == targetHoraire
            if (sameH) horaireDraws++
            val distFromEnd = total - 1 - i

            var bLen = 0
            var overlapsLast = false
            fun pushBall(v: Int) {
                if (v in 1..90 && !drawMaskTmp[v]) {
                    drawMaskTmp[v] = true
                    drawBallsTmp[bLen++] = v
                    if (lastDrawMask[v]) overlapsLast = true
                }
            }
            pushBall(d.num1); pushBall(d.num2); pushBall(d.num3); pushBall(d.num4); pushBall(d.num5)

            if (overlapsLast) lastOverlapDraws++
            for (k in 0 until bLen) {
                val n = drawBallsTmp[k]
                drawMaskTmp[n] = false
                globalCounts[n]++
                if (distFromEnd < 5) {
                    w5Counts[n]++
                    mod7RecentCounts[n % 7]++
                }
                if (distFromEnd < 11) {
                    mod11RecentCounts[n % 11]++
                }
                if (distFromEnd < 15) w15Counts[n]++
                if (distFromEnd < 40) w40Counts[n]++
                if (sameH) horaireCounts[n]++
                if (overlapsLast) coCounts[n]++

                luckyAllCounts[1 + ((n - 1) % 9)]++
                val c90 = (((n.toLong() * n * n) + 9L) % 90L).toInt()
                cube90ClassCounts[c90]++

                prevSeen[n] = lastSeen[n]
                lastSeen[n] = i
            }
        }

        val lastDrawDigitalRoots = BooleanArray(10)
        for (idx in 0 until lastBallsCount) {
            lastDrawDigitalRoots[1 + ((lastBalls[idx] - 1) % 9)] = true
        }

        // Triangulation (différences |a-b|, sommes modulaires, symétries) sur le dernier tirage
        val triangCounts = DoubleArray(91)
        for (i in 0 until lastBallsCount) {
            for (j in i + 1 until lastBallsCount) {
                val a = lastBalls[i]
                val b = lastBalls[j]
                val diff = abs(b - a)
                if (diff in 1..90) triangCounts[diff] += 1.0
                val modSum = ((a + b - 1) % 90) + 1
                triangCounts[modSum] += 0.8
                val comp = 91 - ((a + b) % 90).let { if (it == 0) 90 else it }
                if (comp in 1..90) triangCounts[comp] += 0.6
            }
        }

        // Résonances affines (7k+9 mod 90, miroir 91-k, décalages ±6)
        val affineEcho = DoubleArray(91)
        for (idx in 0 until lastBallsCount) {
            val b = lastBalls[idx]
            val t7 = ((7 * b + 9) % 90) + 1
            val mir = 91 - b
            val sPlus = ((b + 6 - 1) % 90) + 1
            val sMinus = ((b - 6 + 89) % 90) + 1
            affineEcho[t7] += 0.75
            affineEcho[mir] += 0.55
            affineEcho[sPlus] += 0.45
            affineEcho[sMinus] += 0.45
        }
        for (idx in 0 until lag2Count) {
            val b = lag2Balls[idx]
            val t7 = ((7 * b + 9) % 90) + 1
            val mir = 91 - b
            affineEcho[t7] += 0.30
            affineEcho[mir] += 0.25
        }

        val invTotal = 1.0 / total.toDouble()
        val invW5 = 1.0 / min(5, total).toDouble()
        val invW11 = 1.0 / min(11, total).toDouble()
        val invW15 = 1.0 / min(15, total).toDouble()
        val invW40 = 1.0 / min(40, total).toDouble()
        val invH = if (horaireDraws > 0) 1.0 / horaireDraws.toDouble() else 0.0
        val invCo = if (lastOverlapDraws > 0) 1.0 / lastOverlapDraws.toDouble() else 0.0
        val p0 = 5.0 / 90.0
        val se = sqrt(p0 * (1.0 - p0) * invTotal).coerceAtLeast(1e-4)

        for (n in 1..90) {
            val freqGlob = globalCounts[n] * invTotal
            val freq5 = w5Counts[n] * invW5
            val freq15 = w15Counts[n] * invW15
            val freq40 = w40Counts[n] * invW40
            rows[RawFeature.FREQ_GLOBAL.ordinal][n] = freqGlob
            rows[RawFeature.FREQ_W5.ordinal][n] = freq5
            rows[RawFeature.FREQ_W15.ordinal][n] = freq15
            rows[RawFeature.FREQ_W40.ordinal][n] = freq40
            val gap = if (lastSeen[n] < 0) total else (total - 1 - lastSeen[n])
            val prevGap = when {
                lastSeen[n] < 0 -> total
                prevSeen[n] < 0 -> lastSeen[n]
                else -> lastSeen[n] - prevSeen[n]
            }
            rows[RawFeature.GAP_NORM.ordinal][n] = gap / (total + 1.0)
            rows[RawFeature.PREV_GAP_NORM.ordinal][n] = prevGap / (total + 1.0)
            rows[RawFeature.LAST_HIT.ordinal][n] = if (lastDrawMask[n]) 1.0 else 0.0
            rows[RawFeature.LAG2_HIT.ordinal][n] = if (lag2Mask[n]) 1.0 else 0.0
            rows[RawFeature.CO_LAST_RAW.ordinal][n] = coCounts[n] * invCo
            rows[RawFeature.HORAIRE_FREQ.ordinal][n] = horaireCounts[n] * invH

            // Nouvelles variables théorèmes zéro-fuite
            rows[RawFeature.MOD7_RES_DENSITY.ordinal][n] =
                (mod7RecentCounts[n % 7] * invW5 / 1.5).coerceIn(0.0, 1.0)
            val dr = 1 + ((n - 1) % 9)
            val drNorm = dr / 9.0
            val drFreqRatio = (luckyAllCounts[dr] / (total * (5.0 / 9.0)).coerceAtLeast(1.0)).coerceIn(0.0, 2.0)
            rows[RawFeature.LUCKY_ROOT_NORM.ordinal][n] =
                (0.4 * drNorm + 0.35 * (drFreqRatio * 0.5) + if (lastDrawDigitalRoots[dr]) 0.25 else 0.0).coerceIn(0.0, 1.0)
            val mod11Res = (mod11RecentCounts[n % 11] * invW11).coerceIn(0.0, 1.0)
            rows[RawFeature.MOD11_HARMONIC.ordinal][n] =
                (0.5 * ((n % 11) / 10.0) + 0.5 * mod11Res).coerceIn(0.0, 1.0)

            val c90 = (((n.toLong() * n * n) + 9L) % 90L).toInt()
            rows[RawFeature.CONGRUENCE_CUBE90.ordinal][n] =
                (cube90ClassCounts[c90] / (total * (15.0 / 90.0)).coerceAtLeast(1.0) * 0.5).coerceIn(0.0, 1.0)

            rows[RawFeature.AFFINE_MOD90_RESONANCE.ordinal][n] = affineEcho[n].coerceIn(0.0, 1.0)
            rows[RawFeature.EXTRAPOL_VELOCITY.ordinal][n] = ((freq5 - freq15) + 0.5 * (freq15 - freq40)).coerceIn(-1.0, 1.0)

            val expectedGap = if (globalCounts[n] > 0) (total.toDouble() / globalCounts[n]).coerceAtLeast(3.0) else (total.toDouble() + 18.0)
            val overdueRatio = ((gap + 0.5) / expectedGap).coerceIn(0.0, 6.0)
            rows[RawFeature.LIMIT_HAZARD.ordinal][n] = (1.0 - exp(-overdueRatio)).coerceIn(0.0, 1.0)

            val z = (freqGlob - p0) / se
            rows[RawFeature.PROB_ZSCORE.ordinal][n] = (z / 4.0).coerceIn(-1.0, 1.0)
            rows[RawFeature.TRIANG_PYRAMID_DIFF.ordinal][n] = (triangCounts[n] * 0.45).coerceIn(0.0, 1.0)

            val rowN = (n - 1) / 10
            val colN = (n - 1) % 10
            var gridScore = 0.0
            for (idx in 0 until lastBallsCount) {
                val b = lastBalls[idx]
                val rb = (b - 1) / 10
                val cb = (b - 1) % 10
                val drDist = abs(rowN - rb)
                val dcDist = abs(colN - cb)
                if ((drDist == 1 && dcDist == 2) || (drDist == 2 && dcDist == 1)) {
                    gridScore += 0.9
                } else if (drDist <= 1 && dcDist <= 1 && (drDist + dcDist > 0)) {
                    gridScore += 0.65
                }
            }
            rows[RawFeature.GRID_9X10_PROXIMITY.ordinal][n] = (gridScore * 0.4).coerceIn(0.0, 1.0)
        }
        return RawFeatureMatrix(rows)
    }

    /**
     * Extraction incrémentale O(T) exacte (parité mathématique totale avec extractRawMatrix)
     * sur une chronologie déjà triée et dédupliquée, avec gestion stricte timestamp < cible.
     */
    fun extractRawMatrixSequence(
        chronological: List<LotoDraw>,
        targetIndices: IntArray = IntArray(chronological.size) { it }
    ): Map<Int, RawFeatureMatrix> {
        val nTotal = chronological.size
        if (nTotal == 0 || targetIndices.isEmpty()) return emptyMap()

        val sortedTargets = targetIndices.distinct().filter { it in 0 until nTotal }.sorted()
        if (sortedTargets.isEmpty()) return emptyMap()
        val maxTargetIdx = sortedTargets.last()
        val featCount = RawFeature.entries.size

        // Pré-extraction sans allocation des boules distinctes et masques 2x64 bits par tirage
        val drawBallFlat = IntArray(nTotal * 5)
        val drawBallLen = IntArray(nTotal)
        val drawLoMask = LongArray(nTotal)
        val drawHiMask = LongArray(nTotal)
        val horaireNorms = Array(nTotal) { i -> LotoDateUtils.normalizeHoraire(chronological[i]) }
        val horaireIdMap = HashMap<String, Int>(12)
        val horaireIds = IntArray(nTotal) { i -> horaireIdMap.getOrPut(horaireNorms[i]) { horaireIdMap.size } }
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

        val xNorm = DoubleArray(91)
        val base7Norm = DoubleArray(91)
        val luckyIdx = IntArray(91)
        val mod10Norm = DoubleArray(91)
        val mod11StaticNorm = DoubleArray(91)
        val cubeClass = IntArray(91)
        for (n in 1..90) {
            xNorm[n] = n / 90.0
            base7Norm[n] = (n % 7) / 6.0
            luckyIdx[n] = 1 + ((n - 1) % 9)
            mod10Norm[n] = (n % 10) / 9.0
            mod11StaticNorm[n] = (n % 11) / 10.0
            cubeClass[n] = (((n.toLong() * n * n) + 9L) % 90L).toInt()
        }

        val globalCounts = IntArray(91)
        val w5Counts = IntArray(91)
        val w15Counts = IntArray(91)
        val w40Counts = IntArray(91)
        val mod7RecentCounts = IntArray(7)
        val mod11RecentCounts = IntArray(11)
        val horaireCounts = Array(numHoraires) { IntArray(91) }
        val horaireDraws = IntArray(numHoraires)
        val luckyAllCounts = IntArray(10)
        val cube90ClassCounts = IntArray(90)
        val lastSeen = IntArray(91) { -1 }
        val prevSeen = IntArray(91) { -1 }

        val coCounts = IntArray(91)
        val lastDrawMask = BooleanArray(91)
        val lag2Mask = BooleanArray(91)
        val lastDrawDigitalRoots = BooleanArray(10)
        val triangCounts = DoubleArray(91)
        val affineEcho = DoubleArray(91)

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
                    mod7RecentCounts[n % 7]++
                    mod11RecentCounts[n % 11]++
                    horaireCounts[hId][n]++
                    luckyAllCounts[luckyIdx[n]]++
                    cube90ClassCounts[cubeClass[n]]++
                    prevSeen[n] = lastSeen[n]
                    lastSeen[n] = idx
                }

                // Retrait des fenêtres glissantes W5, W11, W15, W40
                val exit5 = idx - 5
                if (exit5 >= 0) {
                    val eLen = drawBallLen[exit5]
                    val eOff = exit5 * 5
                    for (k in 0 until eLen) {
                        val n = drawBallFlat[eOff + k]
                        w5Counts[n]--
                        mod7RecentCounts[n % 7]--
                    }
                }
                val exit11 = idx - 11
                if (exit11 >= 0) {
                    val eLen = drawBallLen[exit11]
                    val eOff = exit11 * 5
                    for (k in 0 until eLen) {
                        mod11RecentCounts[drawBallFlat[eOff + k] % 11]--
                    }
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
                if (total == 0) {
                    for (n in 1..90) {
                        rows[RawFeature.X_NORM.ordinal][n] = xNorm[n]
                        rows[RawFeature.BASE7_NORM.ordinal][n] = base7Norm[n]
                        rows[RawFeature.LUCKY_ROOT_NORM.ordinal][n] = luckyIdx[n] / 9.0
                        rows[RawFeature.MOD10_UNIT_NORM.ordinal][n] = mod10Norm[n]
                        rows[RawFeature.MOD11_HARMONIC.ordinal][n] = mod11StaticNorm[n]
                    }
                    out[t] = RawFeatureMatrix(rows)
                    continue
                }

                val lastIdx = total - 1
                val lastLen = drawBallLen[lastIdx]
                val lastOff = lastIdx * 5
                val lastLo = drawLoMask[lastIdx]
                val lastHi = drawHiMask[lastIdx]

                lastDrawMask.fill(false)
                lastDrawDigitalRoots.fill(false)
                for (k in 0 until lastLen) {
                    val n = drawBallFlat[lastOff + k]
                    lastDrawMask[n] = true
                    lastDrawDigitalRoots[luckyIdx[n]] = true
                }

                lag2Mask.fill(false)
                val hasLag2 = total >= 2
                val lag2Idx = total - 2
                if (hasLag2) {
                    val l2Len = drawBallLen[lag2Idx]
                    val l2Off = lag2Idx * 5
                    for (k in 0 until l2Len) lag2Mask[drawBallFlat[l2Off + k]] = true
                }

                // Co-occurrences avec le dernier tirage strictement passé (2 instructions bitwise/tirage)
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

                triangCounts.fill(0.0)
                for (i in 0 until lastLen) {
                    val a = drawBallFlat[lastOff + i]
                    for (j in i + 1 until lastLen) {
                        val b = drawBallFlat[lastOff + j]
                        val diff = abs(b - a)
                        if (diff in 1..90) triangCounts[diff] += 1.0
                        val modSum = ((a + b - 1) % 90) + 1
                        triangCounts[modSum] += 0.8
                        val comp = 91 - ((a + b) % 90).let { if (it == 0) 90 else it }
                        if (comp in 1..90) triangCounts[comp] += 0.6
                    }
                }

                affineEcho.fill(0.0)
                for (idx in 0 until lastLen) {
                    val b = drawBallFlat[lastOff + idx]
                    val t7 = ((7 * b + 9) % 90) + 1
                    val mir = 91 - b
                    val sPlus = ((b + 6 - 1) % 90) + 1
                    val sMinus = ((b - 6 + 89) % 90) + 1
                    affineEcho[t7] += 0.75
                    affineEcho[mir] += 0.55
                    affineEcho[sPlus] += 0.45
                    affineEcho[sMinus] += 0.45
                }
                if (hasLag2) {
                    val l2Len = drawBallLen[lag2Idx]
                    val l2Off = lag2Idx * 5
                    for (idx in 0 until l2Len) {
                        val b = drawBallFlat[l2Off + idx]
                        val t7 = ((7 * b + 9) % 90) + 1
                        val mir = 91 - b
                        affineEcho[t7] += 0.30
                        affineEcho[mir] += 0.25
                    }
                }

                val tHId = horaireIds[t]
                val hDrawsCount = horaireDraws[tHId]
                val hCountsArr = horaireCounts[tHId]
                val invTotal = 1.0 / total.toDouble()
                val invW5 = 1.0 / min(5, total).toDouble()
                val invW11 = 1.0 / min(11, total).toDouble()
                val invW15 = 1.0 / min(15, total).toDouble()
                val invW40 = 1.0 / min(40, total).toDouble()
                val invH = if (hDrawsCount > 0) 1.0 / hDrawsCount.toDouble() else 0.0
                val invCo = if (lastOverlapDraws > 0) 1.0 / lastOverlapDraws.toDouble() else 0.0
                val p0 = 5.0 / 90.0
                val se = sqrt(p0 * (1.0 - p0) * invTotal).coerceAtLeast(1e-4)

                for (n in 1..90) {
                    rows[RawFeature.X_NORM.ordinal][n] = xNorm[n]
                    rows[RawFeature.BASE7_NORM.ordinal][n] = base7Norm[n]
                    rows[RawFeature.MOD10_UNIT_NORM.ordinal][n] = mod10Norm[n]

                    val freqGlob = globalCounts[n] * invTotal
                    val freq5 = w5Counts[n] * invW5
                    val freq15 = w15Counts[n] * invW15
                    val freq40 = w40Counts[n] * invW40
                    rows[RawFeature.FREQ_GLOBAL.ordinal][n] = freqGlob
                    rows[RawFeature.FREQ_W5.ordinal][n] = freq5
                    rows[RawFeature.FREQ_W15.ordinal][n] = freq15
                    rows[RawFeature.FREQ_W40.ordinal][n] = freq40

                    val ls = lastSeen[n]
                    val gap = if (ls < 0) total else (total - 1 - ls)
                    val prevGap = when {
                        ls < 0 -> total
                        prevSeen[n] < 0 -> ls
                        else -> ls - prevSeen[n]
                    }
                    rows[RawFeature.GAP_NORM.ordinal][n] = gap / (total + 1.0)
                    rows[RawFeature.PREV_GAP_NORM.ordinal][n] = prevGap / (total + 1.0)
                    rows[RawFeature.LAST_HIT.ordinal][n] = if (lastDrawMask[n]) 1.0 else 0.0
                    rows[RawFeature.LAG2_HIT.ordinal][n] = if (lag2Mask[n]) 1.0 else 0.0
                    rows[RawFeature.CO_LAST_RAW.ordinal][n] = coCounts[n] * invCo
                    rows[RawFeature.HORAIRE_FREQ.ordinal][n] = hCountsArr[n] * invH

                    rows[RawFeature.MOD7_RES_DENSITY.ordinal][n] =
                        (mod7RecentCounts[n % 7] * invW5 / 1.5).coerceIn(0.0, 1.0)
                    val dr = luckyIdx[n]
                    val drNorm = dr / 9.0
                    val drFreqRatio = (luckyAllCounts[dr] / (total * (5.0 / 9.0)).coerceAtLeast(1.0)).coerceIn(0.0, 2.0)
                    rows[RawFeature.LUCKY_ROOT_NORM.ordinal][n] =
                        (0.4 * drNorm + 0.35 * (drFreqRatio * 0.5) + if (lastDrawDigitalRoots[dr]) 0.25 else 0.0).coerceIn(0.0, 1.0)
                    val mod11Res = (mod11RecentCounts[n % 11] * invW11).coerceIn(0.0, 1.0)
                    rows[RawFeature.MOD11_HARMONIC.ordinal][n] =
                        (0.5 * mod11StaticNorm[n] + 0.5 * mod11Res).coerceIn(0.0, 1.0)

                    rows[RawFeature.CONGRUENCE_CUBE90.ordinal][n] =
                        (cube90ClassCounts[cubeClass[n]] / (total * (15.0 / 90.0)).coerceAtLeast(1.0) * 0.5).coerceIn(0.0, 1.0)

                    rows[RawFeature.AFFINE_MOD90_RESONANCE.ordinal][n] = affineEcho[n].coerceIn(0.0, 1.0)
                    rows[RawFeature.EXTRAPOL_VELOCITY.ordinal][n] = ((freq5 - freq15) + 0.5 * (freq15 - freq40)).coerceIn(-1.0, 1.0)

                    val expectedGap = if (globalCounts[n] > 0) (total.toDouble() / globalCounts[n]).coerceAtLeast(3.0) else (total.toDouble() + 18.0)
                    val overdueRatio = ((gap + 0.5) / expectedGap).coerceIn(0.0, 6.0)
                    rows[RawFeature.LIMIT_HAZARD.ordinal][n] = (1.0 - exp(-overdueRatio)).coerceIn(0.0, 1.0)

                    val z = (freqGlob - p0) / se
                    rows[RawFeature.PROB_ZSCORE.ordinal][n] = (z / 4.0).coerceIn(-1.0, 1.0)
                    rows[RawFeature.TRIANG_PYRAMID_DIFF.ordinal][n] = (triangCounts[n] * 0.45).coerceIn(0.0, 1.0)

                    val rowN = (n - 1) / 10
                    val colN = (n - 1) % 10
                    var gridScore = 0.0
                    for (idx in 0 until lastLen) {
                        val b = drawBallFlat[lastOff + idx]
                        val rb = (b - 1) / 10
                        val cb = (b - 1) % 10
                        val drDist = abs(rowN - rb)
                        val dcDist = abs(colN - cb)
                        if ((drDist == 1 && dcDist == 2) || (drDist == 2 && dcDist == 1)) {
                            gridScore += 0.9
                        } else if (drDist <= 1 && dcDist <= 1 && (drDist + dcDist > 0)) {
                            gridScore += 0.65
                        }
                    }
                    rows[RawFeature.GRID_9X10_PROXIMITY.ordinal][n] = (gridScore * 0.4).coerceIn(0.0, 1.0)
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
                if (node.feature == RawFeature.SAME_DAY_REP) sameDayRepNorm
                else matrix.get(node.feature.ordinal, number)
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
                    UnaryOp.TANH -> {
                        val e2 = exp(2.0 * x.coerceIn(-6.0, 6.0))
                        (e2 - 1.0) / (e2 + 1.0)
                    }
                    UnaryOp.COS_PI -> cos(x.coerceIn(-10.0, 10.0) * PI)
                    UnaryOp.CBRT -> {
                        val ax = abs(x)
                        val r = if (ax > 1e-12) ax.pow(1.0 / 3.0) else 0.0
                        if (x < 0.0) -r else r
                    }
                    UnaryOp.SOFTPLUS -> ln(1.0 + exp(x.coerceIn(-10.0, 6.0)))
                    UnaryOp.LUCKY9_WAVE -> sin(x.coerceIn(-4.0, 4.0) * 9.0 * PI)
                    UnaryOp.STEP_POS -> if (x > 0.0) 1.0 else 0.0
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
                    BinaryOp.HYPOT_NORM -> sqrt(min(a * a + b * b, 1e8)) * 0.5
                    BinaryOp.MEAN2 -> 0.5 * (a + b)
                    BinaryOp.ATAN2_PI -> if (abs(a) < 1e-5 && abs(b) < 1e-5) 0.0 else atan2(a, b) / PI
                    BinaryOp.TRIANG_BLEND -> 0.5 * abs(a - b) + min(a, b)
                    BinaryOp.EXP_DIFF -> exp(-min(abs(a - b), 8.0))
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
                when {
                    child is ExprNode.Const -> ExprNode.Const(evalConstantUnary(node.op, child.value))
                    node.op == UnaryOp.NEG && child is ExprNode.Unary && child.op == UnaryOp.NEG -> child.child
                    node.op == UnaryOp.ABS && child is ExprNode.Unary && child.op == UnaryOp.ABS -> child
                    else -> ExprNode.Unary(node.op, child)
                }
            }
            is ExprNode.Binary -> {
                val l = simplify(node.left)
                val r = simplify(node.right)
                when {
                    l is ExprNode.Const && r is ExprNode.Const ->
                        ExprNode.Const(evalConstantBinary(node.op, l.value, r.value))
                    node.op == BinaryOp.ADD && r is ExprNode.Const && abs(r.value) < 1e-6 -> l
                    node.op == BinaryOp.ADD && l is ExprNode.Const && abs(l.value) < 1e-6 -> r
                    node.op == BinaryOp.SUB && r is ExprNode.Const && abs(r.value) < 1e-6 -> l
                    node.op == BinaryOp.MUL && ((l is ExprNode.Const && abs(l.value) < 1e-6) || (r is ExprNode.Const && abs(r.value) < 1e-6)) ->
                        ExprNode.Const(0.0)
                    node.op == BinaryOp.MUL && r is ExprNode.Const && abs(r.value - 1.0) < 1e-6 -> l
                    node.op == BinaryOp.MUL && l is ExprNode.Const && abs(l.value - 1.0) < 1e-6 -> r
                    node.op == BinaryOp.SUB && l == r -> ExprNode.Const(0.0)
                    else -> ExprNode.Binary(node.op, l, r)
                }
            }
        }
        return if (nodeCount(s) > MAX_NODES || depth(s) > MAX_DEPTH) trimToDepth(s, MAX_DEPTH) else s
    }

    fun prettyPrint(node: ExprNode): String = when (node) {
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
                BinaryOp.MAX, BinaryOp.MIN, BinaryOp.MOD_SAFE,
                BinaryOp.HYPOT_NORM, BinaryOp.MEAN2, BinaryOp.ATAN2_PI,
                BinaryOp.TRIANG_BLEND, BinaryOp.EXP_DIFF -> "${node.op.symbol}($l, $r)"
                else -> "($l ${node.op.symbol} $r)"
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
        val simplified = simplify(expr)
        return if (usedFeatures(simplified).isEmpty()) {
            val feat = pickRandomFeature(rng)
            val c = roundConst(0.2 + rng.nextDouble() * 1.6)
            ExprNode.Binary(BinaryOp.MUL, ExprNode.Const(c), ExprNode.Var(feat))
        } else {
            simplified
        }
    }

    fun randomTheoremExpr(paradigmIdx: Int, rng: Random): ExprNode {
        val p = ((paradigmIdx % THEOREM_PARADIGMS.size) + THEOREM_PARADIGMS.size) % THEOREM_PARADIGMS.size
        val pool: List<RawFeature> = when (p) {
            0 -> listOf(
                RawFeature.BASE7_NORM, RawFeature.MOD7_RES_DENSITY, RawFeature.LUCKY_ROOT_NORM,
                RawFeature.MOD10_UNIT_NORM, RawFeature.MOD11_HARMONIC, RawFeature.GAP_NORM, RawFeature.FREQ_W15
            )
            1 -> listOf(
                RawFeature.CONGRUENCE_CUBE90, RawFeature.AFFINE_MOD90_RESONANCE, RawFeature.X_NORM,
                RawFeature.CO_LAST_RAW, RawFeature.LAST_HIT, RawFeature.LAG2_HIT, RawFeature.HORAIRE_FREQ
            )
            2 -> listOf(
                RawFeature.TRIANG_PYRAMID_DIFF, RawFeature.GRID_9X10_PROXIMITY, RawFeature.CO_LAST_RAW,
                RawFeature.GAP_NORM, RawFeature.PREV_GAP_NORM, RawFeature.LAST_HIT
            )
            3 -> listOf(
                RawFeature.LIMIT_HAZARD, RawFeature.PROB_ZSCORE, RawFeature.GAP_NORM,
                RawFeature.PREV_GAP_NORM, RawFeature.FREQ_GLOBAL, RawFeature.HORAIRE_FREQ
            )
            4 -> listOf(
                RawFeature.EXTRAPOL_VELOCITY, RawFeature.FREQ_W5, RawFeature.FREQ_W15,
                RawFeature.FREQ_W40, RawFeature.LIMIT_HAZARD, RawFeature.CO_LAST_RAW
            )
            5 -> listOf(
                RawFeature.X_NORM, RawFeature.MOD11_HARMONIC, RawFeature.BASE7_NORM,
                RawFeature.LUCKY_ROOT_NORM, RawFeature.FREQ_W15, RawFeature.GAP_NORM
            )
            else -> GPU_CAPABLE_FEATURES
        }
        fun pickFeat(): ExprNode = ExprNode.Var(pool[rng.nextInt(pool.size)])
        fun pickCoef(): ExprNode = ExprNode.Const(roundConst((rng.nextDouble() * 3.6) - 1.8))
        fun weighted(f: ExprNode): ExprNode = ExprNode.Binary(BinaryOp.MUL, pickCoef(), f)

        val raw = when (rng.nextInt(5)) {
            0 -> {
                val u = UnaryOp.entries[rng.nextInt(UnaryOp.entries.size)]
                val b1 = BinaryOp.entries[rng.nextInt(BinaryOp.entries.size)]
                val b2 = BinaryOp.entries[rng.nextInt(BinaryOp.entries.size)]
                ExprNode.Binary(
                    b1,
                    ExprNode.Unary(u, weighted(pickFeat())),
                    ExprNode.Binary(b2, pickFeat(), weighted(pickFeat()))
                )
            }
            1 -> {
                ExprNode.Binary(
                    BinaryOp.ADD,
                    ExprNode.Unary(UnaryOp.TANH, ExprNode.Binary(BinaryOp.SUB, pickFeat(), pickFeat())),
                    ExprNode.Binary(BinaryOp.HYPOT_NORM, weighted(pickFeat()), pickFeat())
                )
            }
            2 -> {
                ExprNode.Binary(
                    BinaryOp.TRIANG_BLEND,
                    ExprNode.Unary(UnaryOp.SOFTPLUS, weighted(pickFeat())),
                    ExprNode.Binary(BinaryOp.MOD_SAFE, pickFeat(), ExprNode.Const(roundConst(0.5 + rng.nextDouble() * 2.5)))
                )
            }
            3 -> {
                ExprNode.Binary(
                    BinaryOp.MUL,
                    ExprNode.Unary(UnaryOp.SIGMOID, ExprNode.Binary(BinaryOp.ADD, weighted(pickFeat()), pickCoef())),
                    ExprNode.Binary(BinaryOp.EXP_DIFF, pickFeat(), pickFeat())
                )
            }
            else -> buildRandomExpr(if (rng.nextBoolean()) 4 else 5, rng)
        }
        val simplified = simplify(raw)
        return if (usedFeatures(simplified).isEmpty()) randomExpr(4, rng) else simplified
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

    fun exprFromJson(obj: JsonObject): ExprNode = when (obj["t"]?.jsonPrimitive?.contentOrNull) {
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

    fun computePeriodRange(
        chronological: List<LotoDraw>,
        period: BacktestPeriod,
        referenceNowMs: Long = System.currentTimeMillis(),
        customDays: Int? = period.days
    ): IntRange {
        val n = chronological.size
        if (n < 4) return 1 until n
        val effectiveDays = if (period == BacktestPeriod.ALL_HISTORY && customDays == null) {
            null
        } else {
            (customDays ?: period.days)?.coerceAtLeast(1)
        }
        val shortScale = (effectiveDays != null && effectiveDays < 7) || n < 20
        val minWarmup = if (shortScale) {
            minOf(8, maxOf(1, (n * 0.20).toInt()))
        } else {
            minOf(20, maxOf(4, (n * 0.4).toInt()))
        }
        if (period == BacktestPeriod.THREE_WEEKS && effectiveDays == 21) {
            val window = LotoDateUtils.resolveThreeWeekWindow(chronological, referenceNowMs)
            val endExclusive = (chronological.indexOfLast { it.timestamp <= window.endMs } + 1)
                .takeIf { it > 1 } ?: n
            val idxByDate = chronological.subList(0, endExclusive)
                .indexOfFirst { it.timestamp >= window.startMs }
            val start = when {
                idxByDate >= 4 -> idxByDate
                idxByDate >= 0 -> minWarmup.coerceAtMost((endExclusive - 2).coerceAtLeast(1))
                else -> minWarmup.coerceAtMost((endExclusive - 2).coerceAtLeast(1))
            }
            return start until endExclusive.coerceAtLeast(start + 1).coerceAtMost(n)
        }
        val days = effectiveDays ?: return minWarmup.coerceAtMost((n - 2).coerceAtLeast(1)) until n
        val endExclusive = (chronological.indexOfLast { it.timestamp <= referenceNowMs } + 1).takeIf { it >= 2 } ?: n
        val start = if (days == 1) {
            (endExclusive - 6).coerceAtLeast(1).coerceAtMost(endExclusive - 2)
        } else {
            val anchorTs = chronological[endExclusive - 1].timestamp
            val cutoffMs = anchorTs - days.toLong() * 86_400_000L
            val timeStart = chronological.subList(0, endExclusive)
                .indexOfFirst { it.timestamp > cutoffMs }
                .takeIf { it >= 1 } ?: 1
            val countStart = endExclusive - (days * 6)
            maxOf(timeStart, countStart).coerceAtLeast(1).coerceAtMost(endExclusive - 2)
        }
        return start until endExclusive.coerceAtLeast(start + 1).coerceAtMost(n)
    }

    private fun buildRandomExpr(depthLeft: Int, rng: Random): ExprNode {
        if (depthLeft <= 1 || (depthLeft < MAX_DEPTH && rng.nextDouble() < 0.32)) {
            return if (rng.nextDouble() < 0.74) {
                ExprNode.Var(pickRandomFeature(rng))
            } else {
                ExprNode.Const(roundConst((rng.nextDouble() * 4.0) - 2.0))
            }
        }
        return if (rng.nextDouble() < 0.30) {
            val op = UnaryOp.entries[rng.nextInt(UnaryOp.entries.size)]
            ExprNode.Unary(op, buildRandomExpr(depthLeft - 1, rng))
        } else {
            val op = BinaryOp.entries[rng.nextInt(BinaryOp.entries.size)]
            ExprNode.Binary(op, buildRandomExpr(depthLeft - 1, rng), buildRandomExpr(depthLeft - 1, rng))
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
            return replaceRandomSubtree(node, buildRandomExpr(2, rng), rng)
        }
        return when (node) {
            is ExprNode.Const -> ExprNode.Const(roundConst((node.value + (rng.nextDouble() * 1.2 - 0.6)).coerceIn(-5.0, 5.0)))
            is ExprNode.Var -> if (rng.nextDouble() < 0.65) {
                ExprNode.Var(pickRandomFeature(rng))
            } else {
                ExprNode.Unary(UnaryOp.entries[rng.nextInt(UnaryOp.entries.size)], node)
            }
            is ExprNode.Unary -> if (rng.nextDouble() < 0.35) {
                ExprNode.Unary(UnaryOp.entries[rng.nextInt(UnaryOp.entries.size)], node.child)
            } else if (rng.nextDouble() < 0.15) {
                node.child
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
        UnaryOp.TANH -> {
            val e2 = exp(2.0 * x.coerceIn(-6.0, 6.0))
            (e2 - 1.0) / (e2 + 1.0)
        }
        UnaryOp.COS_PI -> cos(x.coerceIn(-10.0, 10.0) * PI)
        UnaryOp.CBRT -> {
            val ax = abs(x)
            val r = if (ax > 1e-12) ax.pow(1.0 / 3.0) else 0.0
            if (x < 0.0) -r else r
        }
        UnaryOp.SOFTPLUS -> ln(1.0 + exp(x.coerceIn(-10.0, 6.0)))
        UnaryOp.LUCKY9_WAVE -> sin(x.coerceIn(-4.0, 4.0) * 9.0 * PI)
        UnaryOp.STEP_POS -> if (x > 0.0) 1.0 else 0.0
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
        BinaryOp.HYPOT_NORM -> sqrt(min(a * a + b * b, 1e8)) * 0.5
        BinaryOp.MEAN2 -> 0.5 * (a + b)
        BinaryOp.ATAN2_PI -> if (abs(a) < 1e-5 && abs(b) < 1e-5) 0.0 else atan2(a, b) / PI
        BinaryOp.TRIANG_BLEND -> 0.5 * abs(a - b) + min(a, b)
        BinaryOp.EXP_DIFF -> exp(-min(abs(a - b), 8.0))
    }.let { if (it.isFinite()) roundConst(it.coerceIn(-1e3, 1e3)) else 0.0 }

    private fun roundConst(v: Double): Double = kotlin.math.round(v * 1000.0) / 1000.0
}
