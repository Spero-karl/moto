package com.example.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [
        LotoResultEntity::class,
        DiscoveredFormulaEntity::class,
        DiscoverySessionEntity::class,
        ApprovedBacktestLedgerEntity::class
    ],
    version = 3,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun lotoDao(): LotoDao
    abstract fun discoveryDao(): DiscoveryDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `discovered_formulas` (
                        `id` TEXT NOT NULL,
                        `periodName` TEXT NOT NULL,
                        `generation` INTEGER NOT NULL,
                        `parentIdsCsv` TEXT NOT NULL,
                        `primaryExprText` TEXT NOT NULL,
                        `secondaryExprText` TEXT NOT NULL,
                        `slotSplitLabel` TEXT NOT NULL,
                        `genomeJson` TEXT NOT NULL,
                        `usedFeaturesCsv` TEXT NOT NULL,
                        `status` TEXT NOT NULL,
                        `rejectionStage` TEXT NOT NULL,
                        `rejectionReason` TEXT NOT NULL,
                        `discoveryStreak` INTEGER NOT NULL,
                        `rollingStreak` INTEGER NOT NULL,
                        `confirmationStreak` INTEGER NOT NULL,
                        `sealedStreak` INTEGER NOT NULL,
                        `combinedStreak` INTEGER NOT NULL,
                        `discoveryAvgMatches` REAL NOT NULL,
                        `rollingAvgMatches` REAL NOT NULL,
                        `confirmationAvgMatches` REAL NOT NULL,
                        `sealedAvgMatches` REAL NOT NULL,
                        `pValueAdjusted` REAL NOT NULL,
                        `fitnessScore` REAL NOT NULL,
                        `updatedAt` INTEGER NOT NULL,
                        PRIMARY KEY(`id`)
                    )
                    """.trimIndent()
                )
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `discovery_sessions` (
                        `periodName` TEXT NOT NULL,
                        `phaseName` TEXT NOT NULL,
                        `generation` INTEGER NOT NULL,
                        `experimentsCount` INTEGER NOT NULL,
                        `confirmationTestsCount` INTEGER NOT NULL,
                        `sessionSalt` INTEGER NOT NULL,
                        `populationJson` TEXT NOT NULL,
                        `activeGenomeJson` TEXT NOT NULL,
                        `validatedGenomeJson` TEXT,
                        `bestDiscoveryStreak` INTEGER NOT NULL,
                        `bestRollingStreak` INTEGER NOT NULL,
                        `bestConfirmationStreak` INTEGER NOT NULL,
                        `bestSealedStreak` INTEGER NOT NULL,
                        `bestCombinedStreak` INTEGER NOT NULL,
                        `honestStatusNote` TEXT NOT NULL,
                        `logsJson` TEXT NOT NULL,
                        `updatedAt` INTEGER NOT NULL,
                        PRIMARY KEY(`periodName`)
                    )
                    """.trimIndent()
                )
            }
        }

        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `approved_backtest_ledger` (
                        `ledgerKey` TEXT NOT NULL,
                        `strategySignature` TEXT NOT NULL,
                        `periodKey` TEXT NOT NULL,
                        `frozenStepsJson` TEXT NOT NULL,
                        `pendingPredictionsJson` TEXT NOT NULL,
                        `lastApprovedTimestamp` INTEGER NOT NULL,
                        `updatedAt` INTEGER NOT NULL,
                        PRIMARY KEY(`ledgerKey`)
                    )
                    """.trimIndent()
                )
            }
        }

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "loto_database"
                )
                    .addMigrations(MIGRATION_1_2, MIGRATION_2_3)
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
