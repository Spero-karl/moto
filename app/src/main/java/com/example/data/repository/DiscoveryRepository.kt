package com.example.data.repository

import com.example.data.local.ApprovedBacktestLedgerEntity
import com.example.data.local.DiscoveredFormulaEntity
import com.example.data.local.DiscoveryDao
import com.example.data.local.DiscoverySessionEntity
import com.example.domain.engine.BacktestPeriod
import kotlinx.coroutines.flow.Flow

class DiscoveryRepository(private val discoveryDao: DiscoveryDao) {

    fun observeFormulas(period: BacktestPeriod): Flow<List<DiscoveredFormulaEntity>> {
        return discoveryDao.observeFormulasForPeriod(period.name)
    }

    suspend fun loadSession(period: BacktestPeriod): DiscoverySessionEntity? {
        return discoveryDao.getSession(period.name)
    }

    suspend fun loadActiveAnalyzingSession(): DiscoverySessionEntity? {
        return discoveryDao.getActiveAnalyzingSession()
    }

    suspend fun saveSession(session: DiscoverySessionEntity) {
        discoveryDao.upsertSession(session)
    }

    suspend fun clearSessionAndFormulas(period: BacktestPeriod) {
        discoveryDao.deleteSession(period.name)
        discoveryDao.clearFormulasForPeriod(period.name)
    }

    suspend fun saveFormulas(formulas: List<DiscoveredFormulaEntity>) {
        if (formulas.isNotEmpty()) {
            discoveryDao.upsertFormulas(formulas)
        }
    }

    suspend fun pruneOldFormulas(period: BacktestPeriod) {
        discoveryDao.pruneOldFormulas(period.name)
    }

    suspend fun loadTopFormulas(period: BacktestPeriod, limit: Int = 8): List<DiscoveredFormulaEntity> {
        return discoveryDao.getTopFormulas(period.name, limit)
    }

    suspend fun loadCheckpointBestFormula(period: BacktestPeriod): DiscoveredFormulaEntity? {
        return discoveryDao.getCheckpointFormula(checkpointIdFor(period))
    }

    suspend fun saveCheckpointBestFormula(period: BacktestPeriod, entity: DiscoveredFormulaEntity) {
        discoveryDao.upsertFormulas(listOf(entity.copy(id = checkpointIdFor(period), periodName = period.name)))
    }

    suspend fun loadRejectedFormulas(period: BacktestPeriod, limit: Int = 8): List<DiscoveredFormulaEntity> {
        return discoveryDao.getRejectedFormulas(period.name, limit)
    }

    suspend fun loadValidatedFormula(period: BacktestPeriod): DiscoveredFormulaEntity? {
        return discoveryDao.getValidatedFormula(period.name)
    }

    suspend fun loadValidatedFormulas(period: BacktestPeriod, limit: Int = 20): List<DiscoveredFormulaEntity> {
        return discoveryDao.getValidatedFormulas(period.name, limit)
    }

    suspend fun loadFormulaById(id: String): DiscoveredFormulaEntity? {
        return discoveryDao.getFormulaById(id)
    }

    suspend fun loadApprovedBacktestLedger(ledgerKey: String): ApprovedBacktestLedgerEntity? {
        return discoveryDao.getApprovedBacktestLedger(ledgerKey)
    }

    suspend fun saveApprovedBacktestLedger(ledger: ApprovedBacktestLedgerEntity) {
        discoveryDao.upsertApprovedBacktestLedger(ledger)
    }

    companion object {
        fun checkpointIdFor(period: BacktestPeriod): String = "checkpoint_best_${period.name}"
    }
}
