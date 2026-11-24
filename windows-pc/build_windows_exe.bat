@echo off
setlocal enabledelayedexpansion
title Creation Executable Windows 10/11 (.exe) - Loto BJ PC

cd /d "%~dp0"

REM 1. Recherche jpackage.exe (inclus dans JDK 17/21+ complet)
set "JPACKAGE_EXE="
where jpackage >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('where jpackage 2^>nul') do (
        if not defined JPACKAGE_EXE set "JPACKAGE_EXE=%%P"
    )
)
if not defined JPACKAGE_EXE if defined JAVA_HOME if exist "%JAVA_HOME%\bin\jpackage.exe" set "JPACKAGE_EXE=%JAVA_HOME%\bin\jpackage.exe"
if not defined JPACKAGE_EXE if exist "%ProgramFiles%\Android\Android Studio\jbr\bin\jpackage.exe" set "JPACKAGE_EXE=%ProgramFiles%\Android\Android Studio\jbr\bin\jpackage.exe"
if not defined JPACKAGE_EXE if exist "%LOCALAPPDATA%\Programs\Android Studio\jbr\bin\jpackage.exe" set "JPACKAGE_EXE=%LOCALAPPDATA%\Programs\Android Studio\jbr\bin\jpackage.exe"
if not defined JPACKAGE_EXE (
    for %%R in ("%ProgramFiles%\Eclipse Adoptium" "%ProgramFiles%\Java" "%ProgramFiles%\Microsoft" "%ProgramFiles%\Zulu" "%ProgramFiles%\Amazon Corretto") do (
        if exist "%%~R" (
            for /d %%D in ("%%~R\*") do (
                if not defined JPACKAGE_EXE if exist "%%~D\bin\jpackage.exe" set "JPACKAGE_EXE=%%~D\bin\jpackage.exe"
            )
        )
    )
)

if not defined JPACKAGE_EXE (
    echo [ERREUR] jpackage.exe introuvable ^(inclus dans JDK 21+ complet^).
    echo Telechargez Eclipse Temurin JDK 21 : https://adoptium.net/
    pause
    exit /b 1
)
echo [INFO] jpackage : "%JPACKAGE_EXE%"

REM 2. Utilise le JAR pre-compile dans app\ si present, sinon compile via Gradle
set "INPUT_DIR=app"
if not exist "app\LotoBJ-PC-AllInOne.jar" (
    if exist "build\libs\LotoBJ-PC-AllInOne.jar" (
        set "INPUT_DIR=build\libs"
    ) else (
        echo [1/3] Compilation du JAR tout-en-un via Gradle...
        where gradle >nul 2>&1
        if errorlevel 1 (
            echo [ERREUR] Aucun JAR pre-compile dans app\ et Gradle est absent.
            pause
            exit /b 1
        )
        call gradle copyBundledJar --no-daemon
        if errorlevel 1 (
            echo [ERREUR] Echec gradle copyBundledJar.
            pause
            exit /b 1
        )
        set "INPUT_DIR=app"
    )
)

echo [1/3] Utilisation de "%INPUT_DIR%\LotoBJ-PC-AllInOne.jar"

if exist "dist\LotoBJ-PC" rmdir /s /q "dist\LotoBJ-PC"
if not exist "dist" mkdir "dist"

echo [2/3] Generation de l'application native Windows (.exe autonome)...
"%JPACKAGE_EXE%" ^
  --type app-image ^
  --name "LotoBJ-PC" ^
  --app-version "1.0.0" ^
  --vendor "LotoBJ Workstation" ^
  --input "%INPUT_DIR%" ^
  --main-jar "LotoBJ-PC-AllInOne.jar" ^
  --main-class "com.lotobj.pc.MainKt" ^
  --dest "dist" ^
  --add-modules "java.base,java.desktop,java.logging,java.management,jdk.management,java.naming,java.net.http,jdk.unsupported" ^
  --java-options "-Xms512m" ^
  --java-options "-XX:+UseG1GC" ^
  --java-options "-XX:MaxRAMPercentage=75.0" ^
  --java-options "-Dfile.encoding=UTF-8"

if errorlevel 1 (
    echo [ERREUR] Echec de generation jpackage.
    pause
    exit /b 1
)

echo.
echo [3/3] SUCCES ! Executable genere :
echo       dist\LotoBJ-PC\LotoBJ-PC.exe
echo Vous pouvez lancer directement dist\LotoBJ-PC\LotoBJ-PC.exe sans console.
pause
