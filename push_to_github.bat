@echo off
setlocal
echo ========================================================
echo   Push AI Road Damage Detection to GitHub
echo ========================================================
echo.

set /p REPO_NAME="Enter your GitHub repository name (e.g. road-pothole-pbl): "

if "%REPO_NAME%"=="" (
    echo [ERROR] Repository name cannot be empty.
    pause
    exit /b 1
)

echo.
echo Setting remote to: https://github.com/patidarprasanna1602-sys/%REPO_NAME%.git ...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/patidarprasanna1602-sys/%REPO_NAME%.git
git branch -M main

echo.
echo Pushing to GitHub (you may be prompted by GitHub for login/token)...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo   SUCCESS! Project pushed to:
    echo   https://github.com/patidarprasanna1602-sys/%REPO_NAME%
    echo ========================================================
) else (
    echo.
    echo [ERROR] Push failed. Make sure the repository exists at:
    echo https://github.com/new
)

echo.
pause
