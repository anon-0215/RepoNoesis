Option Explicit
Dim fs, shell, root, command, argument
Set fs = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fs.GetParentFolderName(WScript.ScriptFullName)
command = """" & root & "\runtime\pythonw.exe"" -I -B """ & root & "\desktop\desktop_host.py"""
For Each argument In WScript.Arguments
  command = command & " """ & Replace(argument, """", """""") & """"
Next
shell.Run command, 1, False
