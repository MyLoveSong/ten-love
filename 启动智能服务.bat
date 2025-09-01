@echo off
echo 🚀 启动血糖预测智能服务...
echo.
echo 📊 系统将自动检查已训练模型
echo 🌐 API服务将在 http://localhost:8000 启动
echo 📖 API文档将在 http://localhost:8000/docs 可用
echo.
echo 按任意键开始...
pause > nul

python smart_integrated_service.py

echo.
echo 服务已停止，按任意键退出...
pause > nul 