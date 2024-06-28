if exist C:\Users\onekey\.conda\envs\onekey (
%windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "& 'C:\ProgramData\Anaconda3\shell\condabin\conda-hook.ps1' ; conda activate onekey; jupyter-notebook"
) else (
%windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "& 'C:\ProgramData\Anaconda3\shell\condabin\conda-hook.ps1' ; conda activate %ONEKEY_HOME%onekey_envs; jupyter-notebook"
)