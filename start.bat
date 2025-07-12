@echo off
setlocal enabledelayedexpansion

REM 基础路径设置（替换为你的实际路径）
set "BASE=D:\项目\sewageWatch"

set "LOG_DIR=%BASE%\logs"  REM 日志统一目录

REM 创建日志目录（如果不存在）
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM 各服务启动命令
set "FFmpeg_cmd=ffmpeg -re -stream_loop -1 -i "%BASE%\Node.js\sample.mp4" -c copy -f flv rtmp://localhost:1935/live/stream1"
set "Node_cmd=node "%BASE%\Node.js\server.js""
set "Python_cmd=cd "%BASE%\sewage-watch-Python" && python main.py"
set "Vue_cmd=cd "%BASE%\Vue" && npm run dev"
set "Java_cmd=cd "%BASE%\sewage-watch-Java" && mvn spring-boot:run"

REM 各服务工作目录
set "Node_dir=Node.js"
set "Python_dir=sewage-watch-Python"
set "Vue_dir=Vue"
set "Java_dir=sewage-watch-Java"

REM 启动所有服务（日志集中到../logs）
for %%S in (FFmpeg, Node, Python, Vue, Java) do (
    echo [%%S] 正在启动...
    start /B /MIN cmd /c "cd /d "%BASE%\!%%S_dir!" && !%%S_cmd! > "%LOG_DIR%\%%S.log" 2>&1"
)

echo 所有服务已启动！日志文件位于: %LOG_DIR%\*.log
echo 关闭本窗口即可停止所有服务！
pause