package com.example.data.local

import androidx.compose.runtime.Immutable
import androidx.room.Entity
import androidx.room.PrimaryKey

@Immutable
@Entity(tableName = "discovered_formulas")
data class DiscoveredFormulaEntity(
    @PrimaryKey
    val id: String,
    val periodName: String,
    val generation: Int,
    val parentIdsCsv: String,
    val primaryExprText: String,
    val secondaryExprText: String,
    val slotSplitLabel: String,
    val genomeJson: String,
    val usedFeaturesCsv: String,
    val status: String, // DISCOVERED, ROLLING_PASSED, REJECTED, VALIDATED
    val rejectionStage: String,
    val rejectionReason: String,
    val discoveryStreak: Int,
    val rollingStreak: Int,
    val confirmationStreak: Int,
    val sealedStreak: Int,
    val combinedStreak: Int,
    val discoveryAvgMatches: Float,
    val rollingAvgMatches: Float,
    val confirmationAvgMatches: Float,
    val sealedAvgMatches: Float,
    val pValueAdjusted: Double,
    val fitnessScore: Double,
    val updatedAt: Long
)

@Immutable
@Entity(tableName = "discovery_sessions")
data class DiscoverySessionEntity(
    @PrimaryKey
    val periodName: String,
    val phaseName: String,
    val generation: Int,
    val experimentsCount: Int,
    val confirmationTestsCount: Int,
    val sessionSalt: Long,
    val populationJson: String,
    val activeGenomeJson: String,
    val validatedGenomeJson: String?,
    val bestDiscoveryStreak: Int,
    val bestRollingStreak: Int,
    val bestConfirmationStreak: Int,
    val bestSealedStreak: Int,
    val bestCombinedStreak: Int,
    val honestStatusNote: String,
    val logsJson: String,
    val updatedAt: Long
)

@Immutable
@Entity(tableName = "approved_backtest_ledger")
data class ApprovedBacktestLedgerEntity(
    @PrimaryKey
    val ledgerKey: String, // strategySignature|periodKey
    val strategySignature: String,
    val periodKey: String,
    val frozenStepsJson: String,
    val pendingPredictionsJson: String,
    val lastApprovedTimestamp: Long,
    val updatedAt: Long
)
