@echo off
setlocal enabledelayedexpansion
title Loto BJ - Station PC Windows 10/11 (GPU OpenCL + CPU + RAM)

cd /d "%~dp0"

set "LOG_DIR=%USERPROFILE%\.LotoBJ_PC"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "LOG_FILE=%LOG_DIR%\startup.log"

echo ============================================================== > "%LOG_FILE%"
echo Loto BJ PC - Demarrage (%DATE% %TIME%) >> "%LOG_FILE%"
echo ============================================================== >> "%LOG_FILE%"

REM 1. Recherche automatique de java.exe (PATH, JAVA_HOME, Android Studio JBR, Temurin, Oracle, Microsoft...)
set "JAVA_EXE="
where java >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%J in ('where java 2^>nul') do (
        if not defined JAVA_EXE set "JAVA_EXE=%%J"
    )
)
if not defined JAVA_EXE if defined JAVA_HOME if exist "%JAVA_HOME%\bin\java.exe" set "JAVA_EXE=%JAVA_HOME%\bin\java.exe"
if not defined JAVA_EXE if exist "%ProgramFiles%\Android\Android Studio\jbr\bin\java.exe" set "JAVA_EXE=%ProgramFiles%\Android\Android Studio\jbr\bin\java.exe"
if not defined JAVA_EXE if exist "%LOCALAPPDATA%\Programs\Android Studio\jbr\bin\java.exe" set "JAVA_EXE=%LOCALAPPDATA%\Programs\Android Studio\jbr\bin\java.exe"

if not defined JAVA_EXE (
    for %%R in ("%ProgramFiles%\Eclipse Adoptium" "%ProgramFiles%\Java" "%ProgramFiles%\Microsoft" "%ProgramFiles%\Zulu" "%ProgramFiles%\Amazon Corretto") do (
        if exist "%%~R" (
            for /d %%D in ("%%~R\*") do (
                if not defined JAVA_EXE if exist "%%~D\bin\java.exe" set "JAVA_EXE=%%~D\bin\java.exe"
            )
        )
    )
)

if not defined JAVA_EXE (
    echo [ERREUR] Aucun Java ^(JDK/JRE 17 ou 21+^) n'a ete trouve sur ce PC.
    echo [ERREUR] Java introuvable. >> "%LOG_FILE%"
    echo.
    echo Installez Eclipse Temurin JDK 21 ^(x64^) depuis :
    echo https://adoptium.net/
    echo ^(ou lancez avec Android Studio installe^).
    echo.
    pause
    exit /b 1
)

echo [INFO] Java detecte : "%JAVA_EXE%"
echo [INFO] Java : "%JAVA_EXE%" >> "%LOG_FILE%"
"%JAVA_EXE%" -version >> "%LOG_FILE%" 2>&1

REM 2. Verification version Java >= 17 (requis pour le bytecode Kotlin/Java 17+)
set "JAVA_MAJOR=0"
for /f "tokens=3" %%V in ('"%JAVA_EXE%" -version 2^>^&1 ^| findstr /i "version"') do (
    set "RAW_VER=%%~V"
    for /f "tokens=1,2 delims=._-" %%A in ("!RAW_VER!") do (
        if "%%A"=="1" (
            set "JAVA_MAJOR=%%B"
        ) else (
            set "JAVA_MAJOR=%%A"
        )
    )
)
if !JAVA_MAJOR! LSS 17 (
    echo [ERREUR] Version Java trop ancienne ^(Java !JAVA_MAJOR! detectee^). Java 17 ou 21+ est requis.
    echo Telechargez JDK 21 LTS : https://adoptium.net/
    pause
    exit /b 1
)

REM 3. Resolution du JAR tout-en-un (pre-compile dans app\ prioritaire, sinon build\libs\, sinon gradle)
set "JAR_PATH=app\LotoBJ-PC-AllInOne.jar"
if not exist "%JAR_PATH%" set "JAR_PATH=build\libs\LotoBJ-PC-AllInOne.jar"

if not exist "%JAR_PATH%" (
    echo [BUILD] Package JAR absent, compilation via Gradle...
    where gradle >nul 2>&1
    if errorlevel 1 (
        echo [ERREUR] Ni app\LotoBJ-PC-AllInOne.jar ni Gradle n'ont ete trouves.
        pause
        exit /b 1
    )
    call gradle copyBundledJar --no-daemon >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        echo [ERREUR] Echec compilation Gradle. Consultez : "%LOG_FILE%"
        type "%LOG_FILE%"
        pause
        exit /b 1
    )
    if exist "app\LotoBJ-PC-AllInOne.jar" set "JAR_PATH=app\LotoBJ-PC-AllInOne.jar"
)

if not exist "%JAR_PATH%" (
    echo [ERREUR] JAR introuvable : "%JAR_PATH%"
    pause
    exit /b 1
)

REM 4. Test rapide des options memoire securisees (sans AlwaysPreTouch qui peut crasher Windows Commit Limit)
set "HEAP_FLAGS=-Xms512m -XX:+UseG1GC -XX:MaxRAMPercentage=75.0"
"%JAVA_EXE%" %HEAP_FLAGS% -version >nul 2>&1
if errorlevel 1 (
    set "HEAP_FLAGS=-Xms256m -Xmx2048m"
)

echo [LANCEMENT] Ouverture de Loto BJ PC ^(GPU OpenCL + CPU 100%% + %HEAP_FLAGS%^)...
echo [INFO] Fermez la fenetre Loto BJ pour terminer, ou laissez ce terminal minimisé.
echo [INFO] Lancement JAR="%JAR_PATH%" FLAGS="%HEAP_FLAGS%" >> "%LOG_FILE%"

"%JAVA_EXE%" %HEAP_FLAGS% -Dfile.encoding=UTF-8 -jar "%JAR_PATH%" >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo ==============================================================
    echo [ERREUR] L'application s'est fermee avec le code %EXIT_CODE%.
    echo Journal enregistre dans : "%LOG_FILE%"
    echo --------------------------------------------------------------
    type "%LOG_FILE%"
    echo ==============================================================
    pause
    exit /b %EXIT_CODE%
)

exit /b 0
