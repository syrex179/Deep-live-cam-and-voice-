Option Explicit

Dim shell, fso, root, pythonw, entry
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = Chr(34) & root & "\venv\Scripts\pythonw.exe" & Chr(34)
entry = Chr(34) & root & "\deep_live_studio.py" & Chr(34)
shell.Run pythonw & " " & entry, 0, False
