' Real bug found live 2026-08-30: this file previously hardcoded the
' OLD machine's absolute path (C:\Users\mubas\PycharmProjects\Quant2\...),
' which silently broke the moment this repo ran on any other machine —
' same portability bug already found and fixed the same day in
' quant_app_tray.ps1 and the other .bat launchers. Derives its own
' directory from WScript.ScriptFullName instead, so this file works
' unmodified regardless of where the repo is checked out.
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """" & scriptDir & "\quant_clerk_execution.bat""", 0, True
Set WshShell = Nothing
