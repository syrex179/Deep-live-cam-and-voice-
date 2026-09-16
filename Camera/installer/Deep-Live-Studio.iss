; Deep Live Studio — Windows installer (Inno Setup 6)
; Packages the tested local runtime, including Python and model resources.

#define AppName "Deep Live Studio"
#define AppVersion "1.0.0"
#define AppPublisher "Syrex"
#define AppLauncher "Deep Live Studio.vbs"

[Setup]
AppId={{C3AC851A-71A2-4C9C-983B-6F6B0A5E7BC5}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Deep Live Studio
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\installer-output
OutputBaseFilename=Deep-Live-Studio-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
; CUDA, TensorRT and the Python runtime exceed the 4 GB single-EXE limit.
; Inno Setup therefore emits a Setup.exe plus adjacent data volumes.
DiskSpanning=yes
DiskSliceSize=2000000000
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#AppName}

[Files]
; Runtime only: omit source repositories, diagnostics and generated caches.
Source: "..\*.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\*.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\*.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\*.vbs"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\assets\*"; DestDir: "{app}\assets"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\locales\*"; DestDir: "{app}\locales"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\modules\*"; DestDir: "{app}\modules"; Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: "__pycache__\*,*.pyc"
Source: "..\models\*"; DestDir: "{app}\models"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\venv\*"; DestDir: "{app}\venv"; Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: "__pycache__\*,*.pyc"
; The only FaceFusion files called directly by the current camera pipeline.
Source: "..\facefusion\.assets\models\u2netp.*"; DestDir: "{app}\facefusion\.assets\models"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\facefusion\.assets\models\bisenet_resnet_34.*"; DestDir: "{app}\facefusion\.assets\models"; Flags: ignoreversion skipifsourcedoesntexist
; Prebuilt full-head engines are optional; copy them when present.
Source: "..\liveportrait\pretrained_weights\liveportrait\engines\*"; DestDir: "{app}\liveportrait\pretrained_weights\liveportrait\engines"; Flags: recursesubdirs createallsubdirs ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\{#AppLauncher}"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\Deep_Live_Studio.ico"
Name: "{autodesktop}\{#AppName}"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\{#AppLauncher}"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\Deep_Live_Studio.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; GroupDescription: "Дополнительные ярлыки:"; Flags: unchecked

[Run]
Filename: "{sys}\wscript.exe"; Parameters: """{app}\{#AppLauncher}"""; Description: "Запустить {#AppName}"; Flags: nowait postinstall skipifsilent
