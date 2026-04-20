#!/usr/bin/env python3
"""
车道检测系统启动脚本（项目根目录运行）
Lane Detection System Launcher (run from project root)
"""

import sys
import os

# 添加integrated_system目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'integrated_system'))

# 导入并运行
import integrated_system.main_pipeline as main_pipeline

if __name__ == "__main__":
    try:
        main_pipeline.main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
