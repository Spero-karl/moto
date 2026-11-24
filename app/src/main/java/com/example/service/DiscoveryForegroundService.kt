package com.example.service

import android.Manifest
import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.example.MainActivity
import com.example.R

class DiscoveryForegroundService : Service() {

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        ensureNotificationChannels(this)
        val powerManager = getSystemService(POWER_SERVICE) as? PowerManager
        wakeLock = powerManager?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_LOCK_TAG)?.apply {
            setReferenceCounted(false)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        ensureNotificationChannels(this)

        when (intent?.action) {
            ACTION_STOP_SERVICE -> {
                releaseWakeLock()
                ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
                stopSelf()
                return START_NOT_STICKY
            }

            ACTION_SUSPEND_FROM_NOTIFICATION -> {
                AutonomousDiscoveryCoordinator.getInstance(application).suspendDiscovery()
                releaseWakeLock()
                ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
                stopSelf()
                return START_NOT_STICKY
            }

            ACTION_COMPLETED -> {
                val gen = intent.getIntExtra(EXTRA_GENERATION, 0)
                val exp = intent.getIntExtra(EXTRA_EXPERIMENTS, 0)
                postCompletionNotification(this, gen, exp)
                releaseWakeLock()
                ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
                stopSelf()
                return START_NOT_STICKY
            }

            else -> {
                val generation = intent?.getIntExtra(EXTRA_GENERATION, 0) ?: 0
                val experiments = intent?.getIntExtra(EXTRA_EXPERIMENTS, 0) ?: 0
                val bestStreak = intent?.getIntExtra(EXTRA_BEST_STREAK, 0) ?: 0
                val promoted = promoteToForeground(generation, experiments, bestStreak)
                if (!promoted) {
                    releaseWakeLock()
                    stopSelf()
                    return START_NOT_STICKY
                }
                acquireWakeLock()

                if (intent == null || intent.action == ACTION_RESUME_STICKY) {
                    AutonomousDiscoveryCoordinator.getInstance(application)
                        .resumeActiveSessionFromStorageIfNeeded()
                }
                return START_STICKY
            }
        }
    }

    override fun onTaskRemoved(rootIntent: Intent?) {
        // Ne jamais arreter la recherche quand l'utilisateur quitte l'ecran ou ferme la tache recente.
        AutonomousDiscoveryCoordinator.getInstance(application).flushCheckpointAsync()
        super.onTaskRemoved(rootIntent)
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        AutonomousDiscoveryCoordinator.getInstance(application).flushCheckpointAsync()
        releaseWakeLock()
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        releaseWakeLock()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun promoteToForeground(generation: Int, experiments: Int, bestStreak: Int): Boolean {
        val notification = buildProgressNotification(this, generation, experiments, bestStreak)
        return try {
            val fgsType = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
            } else {
                0
            }
            ServiceCompat.startForeground(this, PROGRESS_NOTIFICATION_ID, notification, fgsType)
            true
        } catch (_: Exception) {
            false
        }
    }

    @SuppressLint("WakelockTimeout")
    private fun acquireWakeLock() {
        try {
            val lock = wakeLock ?: return
            if (!lock.isHeld) {
                lock.acquire()
            }
        } catch (_: Exception) {
        }
    }

    private fun releaseWakeLock() {
        try {
            val lock = wakeLock ?: return
            if (lock.isHeld) {
                lock.release()
            }
        } catch (_: Exception) {
        }
    }

    companion object {
        const val PROGRESS_CHANNEL_ID = "loto_discovery_progress_channel"
        const val DONE_CHANNEL_ID = "loto_discovery_completed_channel"
        const val PROGRESS_NOTIFICATION_ID = 4091
        const val DONE_NOTIFICATION_ID = 4092

        const val ACTION_START_OR_UPDATE = "com.example.service.action.START_OR_UPDATE"
        const val ACTION_RESUME_STICKY = "com.example.service.action.RESUME_STICKY"
        const val ACTION_SUSPEND_FROM_NOTIFICATION = "com.example.service.action.SUSPEND_FROM_NOTIFICATION"
        const val ACTION_COMPLETED = "com.example.service.action.COMPLETED"
        const val ACTION_STOP_SERVICE = "com.example.service.action.STOP_SERVICE"

        const val EXTRA_GENERATION = "extra_generation"
        const val EXTRA_EXPERIMENTS = "extra_experiments"
        const val EXTRA_BEST_STREAK = "extra_best_streak"

        private const val WAKE_LOCK_TAG = "LotoBJ:DiscoveryEngineWakeLock"

        fun startOrUpdate(
            context: Context,
            generation: Int,
            experiments: Int,
            bestStreak: Int
        ) {
            val intent = Intent(context, DiscoveryForegroundService::class.java).apply {
                action = ACTION_START_OR_UPDATE
                putExtra(EXTRA_GENERATION, generation)
                putExtra(EXTRA_EXPERIMENTS, experiments)
                putExtra(EXTRA_BEST_STREAK, bestStreak)
            }
            try {
                ContextCompat.startForegroundService(context, intent)
            } catch (_: Exception) {
            }
        }

        fun updateProgressNotificationIfPermitted(
            context: Context,
            generation: Int,
            experiments: Int,
            bestStreak: Int
        ) {
            if (!canPostNotifications(context)) return
            try {
                ensureNotificationChannels(context)
                val manager = context.getSystemService(NOTIFICATION_SERVICE) as? NotificationManager ?: return
                manager.notify(
                    PROGRESS_NOTIFICATION_ID,
                    buildProgressNotification(context, generation, experiments, bestStreak)
                )
            } catch (_: Exception) {
            }
        }

        fun notifyCompleted(context: Context, generation: Int, experiments: Int) {
            val intent = Intent(context, DiscoveryForegroundService::class.java).apply {
                action = ACTION_COMPLETED
                putExtra(EXTRA_GENERATION, generation)
                putExtra(EXTRA_EXPERIMENTS, experiments)
            }
            try {
                context.startService(intent)
            } catch (_: Exception) {
                postCompletionNotification(context, generation, experiments)
            }
        }

        fun notifyMilestoneWhileRunning(context: Context, streak: Int, generation: Int, experiments: Int) {
            if (!canPostNotifications(context)) return
            try {
                ensureNotificationChannels(context)
                val openAppIntent = Intent(context, MainActivity::class.java).apply {
                    flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
                }
                val openPendingIntent = PendingIntent.getActivity(
                    context,
                    104,
                    openAppIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                )
                val notification = NotificationCompat.Builder(context, DONE_CHANNEL_ID)
                    .setSmallIcon(R.drawable.ic_discovery_notification)
                    .setContentTitle("Loto BJ • Stratégie ${streak}J Enregistrée !")
                    .setContentText("Gén. $generation ($experiments exp.) — Backtest débloqué, recherche continue pour les autres jours...")
                    .setPriority(NotificationCompat.PRIORITY_DEFAULT)
                    .setAutoCancel(true)
                    .setContentIntent(openPendingIntent)
                    .build()

                val manager = context.getSystemService(NOTIFICATION_SERVICE) as? NotificationManager ?: return
                manager.notify(DONE_NOTIFICATION_ID, notification)
            } catch (_: Exception) {
            }
        }

        fun stopService(context: Context) {
            val intent = Intent(context, DiscoveryForegroundService::class.java).apply {
                action = ACTION_STOP_SERVICE
            }
            try {
                context.stopService(intent)
            } catch (_: Exception) {
            }
        }

        fun ensureNotificationChannels(context: Context) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
            val manager = context.getSystemService(NOTIFICATION_SERVICE) as? NotificationManager ?: return

            if (manager.getNotificationChannel(PROGRESS_CHANNEL_ID) == null) {
                val progressChannel = NotificationChannel(
                    PROGRESS_CHANNEL_ID,
                    "Recherche Continue Loto BJ",
                    NotificationManager.IMPORTANCE_LOW
                ).apply {
                    description = "Maintient le moteur de recherche et de backtest actif même hors de l'application."
                    setShowBadge(false)
                }
                manager.createNotificationChannel(progressChannel)
            }

            if (manager.getNotificationChannel(DONE_CHANNEL_ID) == null) {
                val doneChannel = NotificationChannel(
                    DONE_CHANNEL_ID,
                    "Objectif de Découverte Atteint",
                    NotificationManager.IMPORTANCE_DEFAULT
                ).apply {
                    description = "Notification lorsqu'une formule symbolique valide les 4 niveaux sans fuite."
                }
                manager.createNotificationChannel(doneChannel)
            }
        }

        fun buildProgressNotification(
            context: Context,
            generation: Int,
            experiments: Int,
            bestStreak: Int
        ): Notification {
            val openAppIntent = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            val openPendingIntent = PendingIntent.getActivity(
                context,
                101,
                openAppIntent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            val suspendIntent = Intent(context, DiscoveryForegroundService::class.java).apply {
                action = ACTION_SUSPEND_FROM_NOTIFICATION
            }
            val suspendPendingIntent = PendingIntent.getService(
                context,
                102,
                suspendIntent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            val statusText = when {
                generation <= 0 -> "Initialisation des 4 étapes causales..."
                bestStreak > 4 -> "Gén. $generation • $experiments exp. • Gagnant ${bestStreak}j"
                else -> "Gén. $generation • $experiments exp. • Série $bestStreak/4j"
            }

            return NotificationCompat.Builder(context, PROGRESS_CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_discovery_notification)
                .setContentTitle("Loto BJ • Moteur Actif en Continu")
                .setContentText(statusText)
                .setStyle(
                    NotificationCompat.BigTextStyle().bigText(
                        "$statusText\nTourne sans interruption jusqu'à validation de l'objectif."
                    )
                )
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setContentIntent(openPendingIntent)
                .addAction(0, "Suspendre", suspendPendingIntent)
                .build()
        }

        private fun postCompletionNotification(context: Context, generation: Int, experiments: Int) {
            if (!canPostNotifications(context)) return
            try {
                ensureNotificationChannels(context)
                val openAppIntent = Intent(context, MainActivity::class.java).apply {
                    flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
                }
                val openPendingIntent = PendingIntent.getActivity(
                    context,
                    103,
                    openAppIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                )
                val notification = NotificationCompat.Builder(context, DONE_CHANNEL_ID)
                    .setSmallIcon(R.drawable.ic_discovery_notification)
                    .setContentTitle("Loto BJ • Objectif 4j Validé !")
                    .setContentText("Gén. $generation ($experiments exp.) — Touchez pour voir Backtest & Prédiction.")
                    .setPriority(NotificationCompat.PRIORITY_DEFAULT)
                    .setAutoCancel(true)
                    .setContentIntent(openPendingIntent)
                    .build()

                val manager = context.getSystemService(NOTIFICATION_SERVICE) as? NotificationManager ?: return
                manager.notify(DONE_NOTIFICATION_ID, notification)
            } catch (_: Exception) {
            }
        }

        private fun canPostNotifications(context: Context): Boolean {
            return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.POST_NOTIFICATIONS
                ) == PackageManager.PERMISSION_GRANTED
            } else {
                true
            }
        }
    }
}
