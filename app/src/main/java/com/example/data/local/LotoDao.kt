package com.example.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface LotoDao {
    @Query("SELECT * FROM loto_results ORDER BY timestamp DESC")
    fun getAllResults(): Flow<List<LotoResultEntity>>

    @Query("SELECT * FROM loto_results ORDER BY timestamp DESC")
    suspend fun getAllResultsOnce(): List<LotoResultEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(results: List<LotoResultEntity>)

    @Query("SELECT COUNT(*) FROM loto_results")
    suspend fun getCount(): Int
}
