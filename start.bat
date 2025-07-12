@echo off
setlocal enabledelayedexpansion

REM ����·�����ã��滻Ϊ���ʵ��·����
set "BASE=D:\��Ŀ\sewageWatch"

set "LOG_DIR=%BASE%\logs"  REM ��־ͳһĿ¼

REM ������־Ŀ¼����������ڣ�
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM ��������������
set "FFmpeg_cmd=ffmpeg -re -stream_loop -1 -i "%BASE%\Node.js\sample.mp4" -c copy -f flv rtmp://localhost:1935/live/stream1"
set "Node_cmd=node "%BASE%\Node.js\server.js""
set "Python_cmd=cd "%BASE%\sewage-watch-Python" && python main.py"
set "Vue_cmd=cd "%BASE%\Vue" && npm run dev"
set "Java_cmd=cd "%BASE%\sewage-watch-Java" && mvn spring-boot:run"

REM ��������Ŀ¼
set "Node_dir=Node.js"
set "Python_dir=sewage-watch-Python"
set "Vue_dir=Vue"
set "Java_dir=sewage-watch-Java"

REM �������з�����־���е�../logs��
for %%S in (FFmpeg, Node, Python, Vue, Java) do (
    echo [%%S] ��������...
    start /B /MIN cmd /c "cd /d "%BASE%\!%%S_dir!" && !%%S_cmd! > "%LOG_DIR%\%%S.log" 2>&1"
)

echo ���з�������������־�ļ�λ��: %LOG_DIR%\*.log
echo �رձ����ڼ���ֹͣ���з���
pause