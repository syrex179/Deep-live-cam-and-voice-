Option Explicit

' Launch Deep-Live-Cam without displaying a PowerShell or console window.
Dim shell, fso, appFolder, command
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
appFolder = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = appFolder
command = Chr(34) & appFolder & "\venv\Scripts\pythonw.exe" & Chr(34) & _
          " " & Chr(34) & appFolder & "\run.py" & Chr(34) & _
          " --execution-provider cuda"
shell.Run command, 0, False
