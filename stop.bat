@echo off
setlocal enabledelayedexpansion

set "PORTS=1935 5173 8000 8080 8081"

for %%P in (%PORTS%) do (
    echo ���˿� %%P ռ�����...
    netstat -ano | findstr /RC:":%%P\>" > nul
    
    if !errorlevel! neq 0 (
        echo [��ʾ] �˿� %%P δ��ռ��
    ) else (
        for /f "tokens=5" %%I in ('netstat -ano ^| findstr /RC:":%%P\>"') do (
            echo [�ɹ�] ��ֹ���� PID: %%I (�˿� %%P)
            taskkill /PID %%I /F /T > nul
        )
    )
)
echo ���ж˿ڼ�����
pause