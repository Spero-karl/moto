Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = baseDir

jarPath = baseDir & "\app\LotoBJ-PC-AllInOne.jar"
If Not fso.FileExists(jarPath) Then
    jarPath = baseDir & "\build\libs\LotoBJ-PC-AllInOne.jar"
End If

If Not fso.FileExists(jarPath) Then
    MsgBox "Package LotoBJ-PC-AllInOne.jar introuvable." & vbCrLf & "Lancez run_windows.bat une premiere fois pour diagnostiquer.", 16, "Loto BJ PC"
    WScript.Quit 1
End If

javawExe = ""
candidates = Array( _
    shell.ExpandEnvironmentStrings("%JAVA_HOME%") & "\bin\javaw.exe", _
    shell.ExpandEnvironmentStrings("%ProgramFiles%") & "\Android\Android Studio\jbr\bin\javaw.exe", _
    shell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Programs\Android Studio\jbr\bin\javaw.exe" _
)
For Each c In candidates
    If Len(c) > 15 And fso.FileExists(c) Then
        javawExe = c
        Exit For
    End If
Next

If javawExe = "" Then
    javawExe = "javaw.exe"
End If

cmd = """" & javawExe & """ -Xms256m -XX:+UseG1GC -XX:MaxRAMPercentage=65.0 -Dfile.encoding=UTF-8 -jar """ & jarPath & """"
On Error Resume Next
shell.Run cmd, 0, False
If Err.Number <> 0 Then
    MsgBox "Impossible de lancer javaw.exe (" & Err.Description & ")." & vbCrLf & "Utilisez run_windows.bat pour voir les details.", 16, "Loto BJ PC"
End If
