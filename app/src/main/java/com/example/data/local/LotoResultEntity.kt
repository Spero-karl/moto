package com.example.data.local

import androidx.compose.runtime.Immutable
import androidx.room.Entity
import androidx.room.PrimaryKey

@Immutable
@Entity(tableName = "loto_results")
data class LotoResultEntity(
    @PrimaryKey
    val uniqueId: String,
    val date: String,
    val time: String,
    val game: String,
    val num1: Int,
    val num2: Int,
    val num3: Int,
    val num4: Int,
    val num5: Int,
    val timestamp: Long
)
