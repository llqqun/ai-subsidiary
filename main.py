import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QTextCursor
from ui.main_window import AISubsidiaryApp

def main():
    # 确保必要的目录存在
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 启动应用
    app = QApplication(sys.argv)
    
    # 注册QTextCursor类型
    from PyQt5.QtCore import QMetaType
    QMetaType.type('QTextCursor')
    
    window = AISubsidiaryApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
