package com.example

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.ui.screens.AutonomousEngineOutputScreen
import com.example.ui.screens.ComparisonScreen
import com.example.ui.screens.PredictionsScreen
import com.example.ui.screens.ResultsScreen
import com.example.ui.theme.MyApplicationTheme
import com.example.ui.viewmodel.DiscoveryViewModel
import com.example.ui.viewmodel.LotoViewModel

private const val ROUTE_RESULTS = "results"
private const val ROUTE_PREDICTIONS = "predictions"
private const val ROUTE_ENGINE_OUTPUT = "engine_output"
private const val ROUTE_COMPARISON = "comparison"

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MyApplicationTheme {
                LotoApp()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LotoApp(
    viewModel: LotoViewModel = viewModel(),
    discoveryViewModel: DiscoveryViewModel = viewModel()
) {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route ?: ROUTE_RESULTS
    val context = LocalContext.current

    val results by viewModel.allResults.collectAsStateWithLifecycle()
    val isLoading by viewModel.isLoading.collectAsStateWithLifecycle()
    val isComparing by viewModel.isComparing.collectAsStateWithLifecycle()
    val errorMessage by viewModel.errorMessage.collectAsStateWithLifecycle()
    val statusMessage by viewModel.statusMessage.collectAsStateWithLifecycle()
    val scientificReport by viewModel.scientificReport.collectAsStateWithLifecycle()

    val selectedMethod by discoveryViewModel.selectedMethod.collectAsStateWithLifecycle()
    val selectedPeriod by discoveryViewModel.selectedPeriod.collectAsStateWithLifecycle()
    val customPeriodDays by discoveryViewModel.customPeriodDays.collectAsStateWithLifecycle()
    val selectedHoraire by discoveryViewModel.selectedHoraire.collectAsStateWithLifecycle()
    val availableHoraires by discoveryViewModel.availableHoraires.collectAsStateWithLifecycle()
    val nextDrawTarget by discoveryViewModel.nextDrawTarget.collectAsStateWithLifecycle()
    val prediction by discoveryViewModel.prediction.collectAsStateWithLifecycle()
    val backtestResult by discoveryViewModel.backtestResult.collectAsStateWithLifecycle()
    val engineState by discoveryViewModel.discoveryState.collectAsStateWithLifecycle()
    val precisionState by discoveryViewModel.precisionUiState.collectAsStateWithLifecycle()
    val isBacktestBusy by discoveryViewModel.isBacktestBusy.collectAsStateWithLifecycle()

    val snackbarHostState = remember { SnackbarHostState() }

    val archivePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            val fileName = resolveFileName(context, uri)
            val content = context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
            if (content != null) {
                viewModel.importArchive(content, fileName)
            }
        }
    }

    var pendingActionMode by remember { mutableStateOf<Int?>(null) } // 1=start(reset), 2=resume, 3=continueBeyond, 4=infiniteMode
    val notificationPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { _ ->
        val mode = pendingActionMode
        pendingActionMode = null
        when (mode) {
            1 -> discoveryViewModel.startContinuousDiscovery(reset = true)
            2 -> discoveryViewModel.resumeDiscovery()
            3 -> discoveryViewModel.continueBeyondObjective()
            4 -> discoveryViewModel.startInfiniteMode()
        }
    }

    val runDiscoveryWithNotificationPermission: (Int) -> Unit = { mode ->
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            pendingActionMode = mode
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else {
            when (mode) {
                1 -> discoveryViewModel.startContinuousDiscovery(reset = true)
                2 -> discoveryViewModel.resumeDiscovery()
                3 -> discoveryViewModel.continueBeyondObjective()
                4 -> discoveryViewModel.startInfiniteMode()
            }
        }
    }

    LaunchedEffect(errorMessage) {
        val message = errorMessage
        if (!message.isNullOrBlank()) {
            val action = snackbarHostState.showSnackbar(
                message = message,
                actionLabel = "Réessayer"
            )
            if (action == SnackbarResult.ActionPerformed) {
                viewModel.refreshData()
            } else {
                viewModel.clearError()
            }
        }
    }

    LaunchedEffect(statusMessage) {
        val message = statusMessage
        if (!message.isNullOrBlank()) {
            snackbarHostState.showSnackbar(message = message)
            viewModel.clearStatus()
        }
    }

    LaunchedEffect(currentRoute) {
        discoveryViewModel.refreshNextDrawTarget()
    }

    Scaffold(
        snackbarHost = { SnackbarHost(hostState = snackbarHostState) },
        topBar = {
            TopAppBar(
                title = {
                    val title = when (currentRoute) {
                        ROUTE_PREDICTIONS -> "Moteur & Découverte"
                        ROUTE_ENGINE_OUTPUT -> "Sortie & Confiances"
                        ROUTE_COMPARISON -> "Laboratoire Scientifique"
                        else -> if (results.isNotEmpty()) "LNB Résultats (${results.size})" else "LNB Résultats"
                    }
                    Text(title, fontWeight = FontWeight.Medium)
                },
                actions = {
                    if (currentRoute == ROUTE_RESULTS) {
                        IconButton(
                            onClick = {
                                archivePicker.launch(arrayOf("text/*", "application/json", "*/*"))
                            },
                            modifier = Modifier.testTag("import_archive_button")
                        ) {
                            Icon(Icons.Default.Add, contentDescription = "Importer archive CSV ou JSON")
                        }
                        IconButton(
                            onClick = { viewModel.refreshData() },
                            modifier = Modifier.testTag("refresh_button")
                        ) {
                            Icon(Icons.Default.Refresh, contentDescription = "Rafraîchir")
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                    titleContentColor = MaterialTheme.colorScheme.onBackground
                )
            )
        },
        bottomBar = {
            BottomNavigationBar(
                navController = navController,
                currentRoute = currentRoute
            )
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = ROUTE_RESULTS,
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            composable(ROUTE_RESULTS) {
                ResultsScreen(
                    results = results,
                    isLoading = isLoading,
                    errorMessage = errorMessage,
                    onRefresh = viewModel::refreshData
                )
            }
            composable(ROUTE_PREDICTIONS) {
                PredictionsScreen(
                    engineState = engineState,
                    prediction = prediction,
                    backtestResult = backtestResult,
                    selectedMethod = selectedMethod,
                    selectedPeriod = selectedPeriod,
                    customPeriodDays = customPeriodDays,
                    selectedHoraire = selectedHoraire,
                    availableHoraires = availableHoraires,
                    nextDrawTarget = nextDrawTarget,
                    resultsCount = results.size,
                    isLoading = isLoading || isBacktestBusy,
                    onSelectPeriod = discoveryViewModel::selectPeriod,
                    onSelectCustomPeriodDays = discoveryViewModel::selectCustomPeriodDays,
                    onSelectHoraire = discoveryViewModel::selectHoraire,
                    onStartAnalysis = { runDiscoveryWithNotificationPermission(1) },
                    onSuspendAnalysis = discoveryViewModel::suspendDiscovery,
                    onResumeAnalysis = { runDiscoveryWithNotificationPermission(2) },
                    onContinueBeyondObjective = { runDiscoveryWithNotificationPermission(3) },
                    onSelectSavedWinner = discoveryViewModel::selectSavedWinner,
                    onSelectPreferredStrategyDays = discoveryViewModel::selectPreferredStrategyDays,
                    onStopAnalysis = discoveryViewModel::stopDiscovery,
                    onLaunchBacktestAndPrediction = discoveryViewModel::launchBacktestAndPrediction,
                    onStartInfiniteMode = { runDiscoveryWithNotificationPermission(4) },
                    onRestartFromZero = { discoveryViewModel.restartDiscoveryFromZero() },
                    onSetInfiniteMode = discoveryViewModel::setInfiniteMode,
                    onRunChercheur = discoveryViewModel::runChercheurAndInject,
                    onExportStrategiesBackup = discoveryViewModel::prepareStrategiesBackupExport,
                    onImportStrategiesBackup = discoveryViewModel::importStrategiesBackup,
                    precisionState = precisionState,
                    onSetPrecisionTargetMode = discoveryViewModel::setPrecisionTargetMode,
                    onStartPrecisionDiscovery = { discoveryViewModel.startPrecisionDiscovery(false) },
                    onSuspendPrecisionDiscovery = discoveryViewModel::suspendPrecisionDiscovery,
                    onResumePrecisionDiscovery = discoveryViewModel::resumePrecisionDiscovery,
                    onStopPrecisionDiscovery = discoveryViewModel::stopPrecisionDiscovery,
                    onRestartPrecisionFromZero = discoveryViewModel::restartPrecisionFromZero,
                    onSetPrecisionInfiniteMode = discoveryViewModel::setPrecisionInfiniteMode,
                    onLaunchPrecisionBacktest = discoveryViewModel::launchPrecisionBacktestAndPrediction,
                    onOpenEngineOutput = {
                        navController.navigate(ROUTE_ENGINE_OUTPUT) {
                            popUpTo(navController.graph.startDestinationId) { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    }
                )
            }
            composable(ROUTE_ENGINE_OUTPUT) {
                AutonomousEngineOutputScreen(
                    engineState = engineState,
                    prediction = prediction,
                    selectedHoraire = selectedHoraire,
                    availableHoraires = availableHoraires,
                    nextDrawTarget = nextDrawTarget,
                    backtestResult = backtestResult,
                    isBusy = isLoading || isBacktestBusy,
                    onSelectHoraire = discoveryViewModel::selectHoraire,
                    onStartOrResumeAnalysis = {
                        runDiscoveryWithNotificationPermission(if (engineState.canResume) 2 else 1)
                    },
                    onLaunchBacktestAndPrediction = discoveryViewModel::launchBacktestAndPrediction,
                    onSelectPreferredStrategyDays = discoveryViewModel::selectPreferredStrategyDays,
                    onSelectSavedWinner = discoveryViewModel::selectSavedWinner,
                    onExportStrategiesBackup = discoveryViewModel::prepareStrategiesBackupExport,
                    onImportStrategiesBackup = discoveryViewModel::importStrategiesBackup,
                    onOpenDiscoveryConsole = {
                        navController.navigate(ROUTE_PREDICTIONS) {
                            popUpTo(navController.graph.startDestinationId) { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    }
                )
            }
            composable(ROUTE_COMPARISON) {
                ComparisonScreen(
                    report = scientificReport,
                    resultsCount = results.size,
                    isComparing = isComparing,
                    selectedPeriod = selectedPeriod,
                    customPeriodDays = customPeriodDays,
                    onRunComparison = {
                        viewModel.runScientificComparison(
                            period = selectedPeriod,
                            customDays = customPeriodDays
                        )
                    },
                    onSelectSavedModel = { modelId ->
                        discoveryViewModel.selectSavedWinner(modelId)
                        discoveryViewModel.launchBacktestAndPrediction()
                        navController.navigate(ROUTE_ENGINE_OUTPUT) {
                            popUpTo(navController.graph.startDestinationId) { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    },
                    onSelectMethod = { method ->
                        discoveryViewModel.selectFocusAxis(method)
                        navController.navigate(ROUTE_PREDICTIONS) {
                            popUpTo(navController.graph.startDestinationId) { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    }
                )
            }
        }
    }
}

private fun resolveFileName(context: android.content.Context, uri: Uri): String {
    context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
        val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
        if (nameIndex >= 0 && cursor.moveToFirst()) {
            return cursor.getString(nameIndex) ?: "archive.csv"
        }
    }
    return uri.lastPathSegment ?: "archive.csv"
}

@Composable
fun BottomNavigationBar(
    navController: NavHostController,
    currentRoute: String
) {
    NavigationBar {
        NavigationBarItem(
            icon = { Icon(Icons.AutoMirrored.Filled.List, contentDescription = "Résultats") },
            label = { Text("Résultats") },
            selected = currentRoute == ROUTE_RESULTS,
            onClick = {
                navController.navigate(ROUTE_RESULTS) {
                    popUpTo(navController.graph.startDestinationId) { saveState = true }
                    launchSingleTop = true
                    restoreState = true
                }
            },
            modifier = Modifier.testTag("nav_results")
        )
        NavigationBarItem(
            icon = { Icon(Icons.Default.Star, contentDescription = "Moteur") },
            label = { Text("Moteur") },
            selected = currentRoute == ROUTE_PREDICTIONS,
            onClick = {
                navController.navigate(ROUTE_PREDICTIONS) {
                    popUpTo(navController.graph.startDestinationId) { saveState = true }
                    launchSingleTop = true
                    restoreState = true
                }
            },
            modifier = Modifier.testTag("nav_predictions")
        )
        NavigationBarItem(
            icon = { Icon(Icons.Default.CheckCircle, contentDescription = "Sortie IA") },
            label = { Text("Sortie IA") },
            selected = currentRoute == ROUTE_ENGINE_OUTPUT,
            onClick = {
                navController.navigate(ROUTE_ENGINE_OUTPUT) {
                    popUpTo(navController.graph.startDestinationId) { saveState = true }
                    launchSingleTop = true
                    restoreState = true
                }
            },
            modifier = Modifier.testTag("nav_engine_output")
        )
        NavigationBarItem(
            icon = { Icon(Icons.Default.Info, contentDescription = "Labo") },
            label = { Text("Labo") },
            selected = currentRoute == ROUTE_COMPARISON,
            onClick = {
                navController.navigate(ROUTE_COMPARISON) {
                    popUpTo(navController.graph.startDestinationId) { saveState = true }
                    launchSingleTop = true
                    restoreState = true
                }
            },
            modifier = Modifier.testTag("nav_comparison")
        )
    }
}
