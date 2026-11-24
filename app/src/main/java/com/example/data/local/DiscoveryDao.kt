package com.example.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface DiscoveryDao {
    @Query("SELECT * FROM discovered_formulas WHERE periodName = :periodName ORDER BY updatedAt DESC")
    fun observeFormulasForPeriod(periodName: String): Flow<List<DiscoveredFormulaEntity>>

    @Query("SELECT * FROM discovered_formulas WHERE periodName = :periodName AND status != 'REJECTED' AND id NOT LIKE 'checkpoint_best_%' ORDER BY combinedStreak DESC, fitnessScore DESC LIMIT :limit")
    suspend fun getTopFormulas(periodName: String, limit: Int = 12): List<DiscoveredFormulaEntity>

    @Query("SELECT * FROM discovered_formulas WHERE id = :checkpointId LIMIT 1")
    suspend fun getCheckpointFormula(checkpointId: String): DiscoveredFormulaEntity?

    @Query("SELECT * FROM discovered_formulas WHERE periodName = :periodName AND status = 'REJECTED' ORDER BY updatedAt DESC LIMIT :limit")
    suspend fun getRejectedFormulas(periodName: String, limit: Int = 12): List<DiscoveredFormulaEntity>

    @Query(
        """
        SELECT * FROM discovered_formulas
        WHERE periodName = :periodName
          AND (status = 'VALIDATED' OR status LIKE 'MILESTONE_%')
          AND id NOT LIKE 'checkpoint_best_%'
        ORDER BY combinedStreak DESC,
                 (CASE WHEN status = 'VALIDATED' THEN 1 ELSE 0 END) DESC,
                 fitnessScore DESC,
                 updatedAt DESC
        LIMIT 1
        """
    )
    suspend fun getValidatedFormula(periodName: String): DiscoveredFormulaEntity?

    @Query(
        """
        SELECT * FROM discovered_formulas
        WHERE periodName = :periodName
          AND (status = 'VALIDATED' OR status LIKE 'MILESTONE_%')
          AND id NOT LIKE 'checkpoint_best_%'
        ORDER BY combinedStreak DESC,
                 (CASE WHEN status = 'VALIDATED' THEN 1 ELSE 0 END) DESC,
                 fitnessScore DESC,
                 updatedAt DESC
        LIMIT :limit
        """
    )
    suspend fun getValidatedFormulas(periodName: String, limit: Int = 20): List<DiscoveredFormulaEntity>

    @Query("SELECT * FROM discovered_formulas WHERE id = :id LIMIT 1")
    suspend fun getFormulaById(id: String): DiscoveredFormulaEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertFormulas(items: List<DiscoveredFormulaEntity>)

    @Query(
        """
        DELETE FROM discovered_formulas
        WHERE periodName = :periodName
          AND status != 'VALIDATED'
          AND status NOT LIKE 'MILESTONE_%'
        """
    )
    suspend fun clearFormulasForPeriod(periodName: String)

    @Query("SELECT * FROM discovery_sessions WHERE periodName = :periodName LIMIT 1")
    suspend fun getSession(periodName: String): DiscoverySessionEntity?

    @Query("SELECT * FROM discovery_sessions WHERE phaseName = 'ANALYZING' ORDER BY updatedAt DESC LIMIT 1")
    suspend fun getActiveAnalyzingSession(): DiscoverySessionEntity?

    @Query(
        """
        DELETE FROM discovered_formulas
        WHERE periodName = :periodName
          AND status != 'VALIDATED'
          AND status NOT LIKE 'MILESTONE_%'
          AND id NOT LIKE 'checkpoint_best_%'
          AND id NOT IN (
              SELECT id FROM discovered_formulas
              WHERE periodName = :periodName AND status != 'REJECTED'
              ORDER BY combinedStreak DESC, fitnessScore DESC LIMIT 30
          )
          AND id NOT IN (
              SELECT id FROM discovered_formulas
              WHERE periodName = :periodName AND status = 'REJECTED'
              ORDER BY updatedAt DESC LIMIT 30
          )
        """
    )
    suspend fun pruneOldFormulas(periodName: String)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertSession(session: DiscoverySessionEntity)

    @Query("DELETE FROM discovery_sessions WHERE periodName = :periodName")
    suspend fun deleteSession(periodName: String)

    @Query("SELECT * FROM approved_backtest_ledger WHERE ledgerKey = :ledgerKey LIMIT 1")
    suspend fun getApprovedBacktestLedger(ledgerKey: String): ApprovedBacktestLedgerEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertApprovedBacktestLedger(ledger: ApprovedBacktestLedgerEntity)

    @Query("DELETE FROM approved_backtest_ledger WHERE ledgerKey = :ledgerKey")
    suspend fun deleteApprovedBacktestLedger(ledgerKey: String)
}
