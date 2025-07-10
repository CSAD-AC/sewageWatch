@echo off
setlocal enabledelayedexpansion

set "PORTS=1935 5173 8000 8080 8081"

for %%P in (%PORTS%) do (
    echo 检查端口 %%P 占用情况...
    netstat -ano | findstr /RC:":%%P\>" > nul
    
    if !errorlevel! neq 0 (
        echo [提示] 端口 %%P 未被占用
    ) else (
        for /f "tokens=5" %%I in ('netstat -ano ^| findstr /RC:":%%P\>"') do (
            echo [成功] 终止进程 PID: %%I (端口 %%P)
            taskkill /PID %%I /F /T > nul
        )
    )
)
echo 所有端口检查完成
pause