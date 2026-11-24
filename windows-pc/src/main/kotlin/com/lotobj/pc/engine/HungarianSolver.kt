package com.lotobj.pc.engine

object HungarianSolver {

    fun cube90Transform(n: Int): Int {
        val v = (n.toLong() * n.toLong() * n.toLong()) + 9L
        return (1L + (v % 90L)).toInt()
    }

    fun bijectiveAffineTransform(n: Int): Int {
        val v = (7L * n.toLong()) + 9L
        return (1L + (((v % 90L) + 90L) % 90L)).toInt()
    }

    fun solveMaxWeightAssignment(scoreMatrix: Array<DoubleArray>): IntArray {
        val rows = 5
        val cols = 90
        require(scoreMatrix.size == rows) { "scoreMatrix doit avoir 5 lignes" }

        var maxScore = Double.NEGATIVE_INFINITY
        for (p in 0 until rows) {
            require(scoreMatrix[p].size > cols) { "Ligne $p doit couvrir 1..90" }
            for (n in 1..cols) {
                if (scoreMatrix[p][n] > maxScore) {
                    maxScore = scoreMatrix[p][n]
                }
            }
        }
        if (!maxScore.isFinite()) maxScore = 0.0

        val cost = Array(rows) { p ->
            DoubleArray(cols + 1) { n ->
                if (n == 0) 0.0 else (maxScore - scoreMatrix[p][n]) + n * 1e-9
            }
        }

        val u = DoubleArray(rows + 1)
        val v = DoubleArray(cols + 1)
        val pMatch = IntArray(cols + 1)
        val way = IntArray(cols + 1)

        for (i in 1..rows) {
            pMatch[0] = i
            var j0 = 0
            val minv = DoubleArray(cols + 1) { Double.POSITIVE_INFINITY }
            val used = BooleanArray(cols + 1) { false }

            do {
                used[j0] = true
                val i0 = pMatch[j0]
                var delta = Double.POSITIVE_INFINITY
                var j1 = 0

                for (j in 1..cols) {
                    if (!used[j]) {
                        val cur = cost[i0 - 1][j] - u[i0] - v[j]
                        if (cur < minv[j]) {
                            minv[j] = cur
                            way[j] = j0
                        }
                        if (minv[j] < delta) {
                            delta = minv[j]
                            j1 = j
                        }
                    }
                }

                for (j in 0..cols) {
                    if (used[j]) {
                        u[pMatch[j]] += delta
                        v[j] -= delta
                    } else {
                        minv[j] -= delta
                    }
                }
                j0 = j1
            } while (pMatch[j0] != 0)

            do {
                val j1 = way[j0]
                pMatch[j0] = pMatch[j1]
                j0 = j1
            } while (j0 != 0)
        }

        val result = IntArray(rows)
        for (j in 1..cols) {
            if (pMatch[j] in 1..rows) {
                result[pMatch[j] - 1] = j
            }
        }

        check(result.toSet().size == rows && result.all { it in 1..cols }) {
            "Affectation hongroise invalide: ${result.toList()}"
        }
        return result
    }
}
