Option Explicit

Dim shell, fso, appFolder, command
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
appFolder = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = appFolder
command = Chr(34) & appFolder & "\venv\Scripts\pythonw.exe" & Chr(34) & _
          " " & Chr(34) & appFolder & "\deep_live_studio.py" & Chr(34)
shell.Run command, 0, False
