Option Explicit
Dim fs, shell, root, command, argument
Set fs = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fs.GetParentFolderName(WScript.ScriptFullName)
If Not fs.FileExists(root & "\app\runtime\pythonw.exe") Then
  MsgBox "请先完整解压便携 ZIP，再打开源鉴。请保留 app 文件夹与此入口在同一目录。", 16, "源鉴"
  WScript.Quit 1
End If
command = """" & root & "\app\runtime\pythonw.exe"" -I -B """ & root & "\app\desktop\desktop_host.py"""
For Each argument In WScript.Arguments
  command = command & " """ & Replace(argument, """", """""") & """"
Next
shell.CurrentDirectory = root
shell.Run command, 1, False
