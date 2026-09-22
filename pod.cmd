@echo off
rem Runs scripts/pod.sh with Git Bash (NOT WSL). Usage:  pod up | wait | status | down
set "GB=%ProgramFiles%\Git\bin\bash.exe"
if not exist "%GB%" set "GB=%LocalAppData%\Programs\Git\bin\bash.exe"
if not exist "%GB%" (
  echo Git Bash not found - install Git for Windows from https://git-scm.com
  exit /b 1
)
"%GB%" "%~dp0scripts/pod.sh" %*
