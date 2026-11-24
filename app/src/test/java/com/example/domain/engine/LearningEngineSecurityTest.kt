package com.example.domain.engine

import com.example.data.local.LotoResultEntity
import com.example.domain.util.LotoDateUtils
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LearningEngineSecurityTest {

    private fun buildDraws(count: Int, seedOffset: Int = 0): List<LotoResultEntity> {
        val horaires = listOf("11H", "14H", "18H", "20H")
        val games = listOf("Fortune", "Star", "Digital")
        return (0 until count).map { i ->
            val base = ((i * 7 + seedOffset) % 80) + 1
            val horaire = horaires[i % horaires.size]
            val game = games[i % games.size]
            val ts = 1_790_000_000_000L + i * 21_600_000L
            LotoResultEntity(
                uniqueId = "draw_$i",
                date = LotoDateUtils.formatFrenchDayAndDate(ts),
                time = horaire,
                game = game,
                num1 = base,
                num2 = ((base + 5) % 90) + 1,
                num3 = ((base + 11) % 90) + 1,
                num4 = ((base + 19) % 90) + 1,
                num5 = ((base + 29) % 90) + 1,
                timestamp = ts
            )
        }
    }

    @Test
    fun cube90Formula_andBijectiveAffineTransform() {
        assertEquals(11, HungarianSolver.cube90Transform(1))
        assertEquals(10, HungarianSolver.cube90Transform(90))
        assertEquals(37, HungarianSolver.cube90Transform(3))

        val cubeOutputs = (1..90).map { HungarianSolver.cube90Transform(it) }.toSet()
        assertEquals("Cube90 doit produire exactement 30 images (collisions 3-pour-1)", 30, cubeOutputs.size)

        val affineOutputs = (1..90).map { HungarianSolver.bijectiveAffineTransform(it) }.toSet()
        assertEquals("BijectiveAffine doit couvrir exactement les 90 numeros", 90, affineOutputs.size)
    }

    @Test
    fun hungarianSolver_findsExactGlobalOptimumWithDistinctNumbers() {
        val matrix = Array(5) { DoubleArray(91) { 0.0 } }
        matrix[0][10] = 100.0
        matrix[0][20] = 90.0
        matrix[1][10] = 95.0
        matrix[1][30] = 50.0
        matrix[2][40] = 80.0
        matrix[3][50] = 70.0
        matrix[4][60] = 60.0

        val assigned = HungarianSolver.solveMaxWeightAssignment(matrix)
        assertEquals(5, assigned.toSet().size)
        assertEquals(20, assigned[0])
        assertEquals(10, assigned[1])
        assertEquals(40, assigned[2])
        assertEquals(50, assigned[3])
        assertEquals(60, assigned[4])
    }

    @Test
    fun zeroFutureLeakage_allElevenMethodsDoNotChangeWhenTargetOrFutureMutates() {
        val engine = LearningEngine()
        val datasetA = buildDraws(45, seedOffset = 0)
        val datasetB = datasetA.mapIndexed { idx, draw ->
            if (idx >= 38) {
                draw.copy(
                    num1 = ((draw.num1 + 40) % 90) + 1,
                    num2 = ((draw.num2 + 41) % 90) + 1,
                    num3 = ((draw.num3 + 42) % 90) + 1,
                    num4 = ((draw.num4 + 43) % 90) + 1,
                    num5 = ((draw.num5 + 44) % 90) + 1
                )
            } else {
                draw
            }
        }

        for (method in PredictionMethod.entries) {
            val btA = engine.backtest(datasetA, method = method)
            val btB = engine.backtest(datasetB, method = method)

            assertTrue("Causalite non verifiee pour $method", btA.allStepsCausalVerified)
            assertTrue("Causalite non verifiee pour $method", btB.allStepsCausalVerified)
            assertEquals(btA.totalTested, btB.totalTested)

            val stepsA = btA.steps.associateBy { it.drawId }
            val stepsB = btB.steps.associateBy { it.drawId }

            for (i in 25..38) {
                val id = "draw_$i"
                val sA = stepsA.getValue(id)
                val sB = stepsB.getValue(id)
                assertEquals("Fuite detectee ($method) sur $id", sA.predictedNumbers, sB.predictedNumbers)
                assertTrue(sA.maxTrainTimestamp < sA.targetTimestamp)
            }
        }
    }

    @Test
    fun frenchWeekdayName_andDailySummary_arePopulatedCorrectly() {
        val engine = LearningEngine()
        val draws = buildDraws(45)
        val bt = engine.backtest(draws, method = PredictionMethod.DIRICHLET_MULTINOMIAL)

        assertTrue(bt.totalTested > 0)
        assertTrue(bt.dailySummaries.isNotEmpty())
        val frenchWeekdays = listOf("Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche")
        for (step in bt.steps) {
            assertTrue(
                "Date sans nom de jour francais: ${step.drawDate}",
                frenchWeekdays.any { step.drawDate.startsWith(it) }
            )
            val expectedMatches = step.predictedNumbers.intersect(step.actualNumbers.toSet()).sorted()
            assertEquals(expectedMatches, step.matchedNumbers)
            assertEquals(expectedMatches.size, step.matchCount)
        }
        for (day in bt.dailySummaries) {
            assertEquals(day.drawsWithTwoPlus >= 3, day.isQualifyingDay)
        }
    }
}
