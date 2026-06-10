

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接启动学术级集成系统
跳过所有检查，直接启动Web服务
"""

import sys
import uvicorn
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    print("=" * 60)
    print("学术级智能健康监测集成系统 - 直接启动")
    print("=" * 60)

    try:
        # 直接导入并启动系统
        from academic_integrated_system import AcademicIntegratedSystem

        print("正在初始化系统...")
        system = AcademicIntegratedSystem()

        print("系统初始化成功！")
        print("\n" + "=" * 60)
        print("系统正在启动，请稍候...")
        print("=" * 60)
        print("启动完成后，可以通过以下方式访问：")
        print("- Web界面: http://localhost:8000")
        print("- API文档: http://localhost:8000/docs")
        print("- 健康检查: http://localhost:8000/health")
        print("\n按 Ctrl+C 停止系统")
        print("=" * 60)

        # 启动Web服务
        uvicorn.run(
            system.app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有依赖包已正确安装")
        return False
    except Exception as e:
        print(f"❌ 启动错误: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        input("\n按回车键退出...")
        sys.exit(1)

__all__ = ["'current_dir'", "'main'"]
