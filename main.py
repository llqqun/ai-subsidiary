import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QTextEdit, QFileDialog, QTabWidget,
                             QLineEdit, QGroupBox, QStackedWidget)
from PyQt5.QtCore import Qt

# 导入自定义模块
from modules.training import TrainingManager
from modules.computer_control import ComputerController
from modules.models import ModelManager
from modules.game_config import GameConfig

class AISubsidiaryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI辅助训练系统")
        self.setGeometry(100, 100, 1000, 700)
        
        # 初始化组件
        self.training_manager = TrainingManager()
        self.computer_controller = ComputerController()
        self.model_manager = ModelManager()
        self.game_config = GameConfig()
        
        # 导入强化学习模块
        from modules.reinforcement_learning import RLTrainingManager
        self.rl_training_manager = RLTrainingManager()
        self.rl_training_manager.set_log_callback(self.add_rl_log)
        
        # 设置UI
        self.init_ui()
        
    def init_ui(self):
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建选项卡
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # 训练选项卡
        training_tab = QWidget()
        tabs.addTab(training_tab, "训练")
        
        # 操作选项卡
        operation_tab = QWidget()
        tabs.addTab(operation_tab, "操作")
        
        # 设置选项卡
        settings_tab = QWidget()
        tabs.addTab(settings_tab, "设置")
        
        # 设置训练选项卡内容
        self.setup_training_tab(training_tab)
        
        # 设置操作选项卡内容
        self.setup_operation_tab(operation_tab)
        
        # 设置设置选项卡内容
        self.setup_settings_tab(settings_tab)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
    
    def setup_training_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # 创建训练方式选择
        training_mode_layout = QHBoxLayout()
        training_mode_label = QLabel("训练方式:")
        self.training_mode_combo = QComboBox()
        self.training_mode_combo.addItems(["传统训练", "游戏自主训练"])
        self.training_mode_combo.currentIndexChanged.connect(self.switch_training_mode)
        training_mode_layout.addWidget(training_mode_label)
        training_mode_layout.addWidget(self.training_mode_combo)
        training_mode_layout.addStretch()
        layout.addLayout(training_mode_layout)
        
        # 创建堆叠部件用于切换不同训练方式的UI
        self.training_stack = QStackedWidget()
        
        # 传统训练UI
        traditional_training_widget = QWidget()
        traditional_layout = QVBoxLayout(traditional_training_widget)
        
        # 领域选择区域
        domain_layout = QHBoxLayout()
        domain_label = QLabel("训练领域:")
        self.domain_combo = QComboBox()
        self.domain_combo.addItems(["游戏", "法律", "医疗", "教育", "自定义"])
        domain_layout.addWidget(domain_label)
        domain_layout.addWidget(self.domain_combo)
        domain_layout.addStretch()
        traditional_layout.addLayout(domain_layout)
        
        # 训练数据区域
        data_layout = QHBoxLayout()
        data_label = QLabel("训练数据:")
        self.data_path_label = QLabel("未选择")
        data_button = QPushButton("选择文件/文件夹")
        data_button.clicked.connect(self.select_training_data)
        data_layout.addWidget(data_label)
        data_layout.addWidget(self.data_path_label)
        data_layout.addWidget(data_button)
        traditional_layout.addLayout(data_layout)
        
        # 训练参数区域
        params_label = QLabel("训练参数:")
        traditional_layout.addWidget(params_label)
        self.params_edit = QTextEdit()
        self.params_edit.setPlaceholderText("在此输入训练参数 (JSON格式)")
        self.params_edit.setText('{"learning_rate": 0.001, "epochs": 100, "batch_size": 32}')
        traditional_layout.addWidget(self.params_edit)
        
        # 游戏自主训练UI
        rl_training_widget = QWidget()
        rl_layout = QVBoxLayout(rl_training_widget)
        
        # 游戏选择区域
        game_layout = QHBoxLayout()
        game_label = QLabel("选择游戏:")
        self.game_combo = QComboBox()
        self.refresh_games_button = QPushButton("刷新")
        self.refresh_games_button.clicked.connect(self.refresh_games)
        game_layout.addWidget(game_label)
        game_layout.addWidget(self.game_combo)
        game_layout.addWidget(self.refresh_games_button)
        rl_layout.addLayout(game_layout)
        
        # RL训练参数区域
        rl_params_label = QLabel("强化学习参数:")
        rl_layout.addWidget(rl_params_label)
        self.rl_params_edit = QTextEdit()
        self.rl_params_edit.setPlaceholderText("在此输入强化学习参数 (JSON格式)")
        self.rl_params_edit.setText('{"learning_rate": 0.001, "episodes": 1000, "batch_size": 32, "epsilon_start": 1.0, "epsilon_min": 0.1, "epsilon_decay": 0.995, "gamma": 0.99, "buffer_capacity": 10000}')
        rl_layout.addWidget(self.rl_params_edit)
        
        # 添加两种训练方式的UI到堆叠部件
        self.training_stack.addWidget(traditional_training_widget)
        self.training_stack.addWidget(rl_training_widget)
        layout.addWidget(self.training_stack)
        
        # 训练控制区域
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("开始训练")
        self.start_button.clicked.connect(self.start_training)
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self.pause_training)
        self.pause_button.setEnabled(False)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        layout.addLayout(control_layout)
        
        # 训练日志区域
        log_label = QLabel("训练日志:")
        layout.addWidget(log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        # 初始刷新游戏列表
        self.refresh_games()
    
    def setup_operation_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # 模型选择区域
        model_layout = QHBoxLayout()
        model_label = QLabel("选择模型:")
        self.model_combo = QComboBox()
        self.refresh_models_button = QPushButton("刷新")
        self.refresh_models_button.clicked.connect(self.refresh_models)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.refresh_models_button)
        layout.addLayout(model_layout)
        
        # 操作控制区域
        control_layout = QHBoxLayout()
        self.start_operation_button = QPushButton("开始操作")
        self.start_operation_button.clicked.connect(self.start_operation)
        self.stop_operation_button = QPushButton("停止操作")
        self.stop_operation_button.clicked.connect(self.stop_operation)
        self.stop_operation_button.setEnabled(False)
        control_layout.addWidget(self.start_operation_button)
        control_layout.addWidget(self.stop_operation_button)
        layout.addLayout(control_layout)
        
        # 操作日志区域
        log_label = QLabel("操作日志:")
        layout.addWidget(log_label)
        self.operation_log_text = QTextEdit()
        self.operation_log_text.setReadOnly(True)
        layout.addWidget(self.operation_log_text)
        
        # 初始加载模型列表
        self.refresh_models()
    
    def setup_settings_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # 模型存储路径设置
        model_path_layout = QHBoxLayout()
        model_path_label = QLabel("模型存储路径:")
        self.model_path_edit = QTextEdit()
        self.model_path_edit.setMaximumHeight(30)
        self.model_path_edit.setText(os.path.join(os.getcwd(), "models"))
        model_path_button = QPushButton("浏览")
        model_path_button.clicked.connect(self.select_model_path)
        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(model_path_button)
        layout.addLayout(model_path_layout)
        
        # 游戏配置区域
        game_config_group = QGroupBox("游戏配置")
        game_config_layout = QVBoxLayout()
        
        # 游戏名称
        game_name_layout = QHBoxLayout()
        game_name_label = QLabel("游戏名称:")
        self.game_name_edit = QLineEdit()
        game_name_layout.addWidget(game_name_label)
        game_name_layout.addWidget(self.game_name_edit)
        game_config_layout.addLayout(game_name_layout)
        
        # 游戏类型
        game_type_layout = QHBoxLayout()
        game_type_label = QLabel("游戏类型:")
        self.game_type_combo = QComboBox()
        self.game_type_combo.addItems(["动作", "冒险", "策略", "角色扮演", "其他"])
        game_type_layout.addWidget(game_type_label)
        game_type_layout.addWidget(self.game_type_combo)
        game_config_layout.addLayout(game_type_layout)
        
        # 操作按键
        keys_label = QLabel("操作按键:")
        game_config_layout.addWidget(keys_label)
        self.keys_edit = QTextEdit()
        self.keys_edit.setPlaceholderText("请输入游戏使用的按键，每行一个，例如:\nup\ndown\nleft\nright\nspace")
        self.keys_edit.setMaximumHeight(100)
        game_config_layout.addWidget(self.keys_edit)
        
        # 保存游戏配置按钮
        save_game_config_button = QPushButton("保存游戏配置")
        save_game_config_button.clicked.connect(self.save_game_config)
        game_config_layout.addWidget(save_game_config_button)
        
        game_config_group.setLayout(game_config_layout)
        layout.addWidget(game_config_group)
        
        # 保存设置按钮
        save_button = QPushButton("保存设置")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)
        layout.addStretch()
    
    def select_training_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择训练数据文件", "", "所有文件 (*);;文本文件 (*.txt);;CSV文件 (*.csv)", options=options)
        if file_path:
            self.data_path_label.setText(file_path)
            self.log_text.append(f"已选择训练数据: {file_path}")
    
    def select_model_path(self):
        options = QFileDialog.Options()
        dir_path = QFileDialog.getExistingDirectory(self, "选择模型存储路径", options=options)
        if dir_path:
            self.model_path_edit.setText(dir_path)
    
    def switch_training_mode(self, index):
        """切换训练模式"""
        self.training_stack.setCurrentIndex(index)
    
    def refresh_games(self):
        """刷新可用游戏列表"""
        games = self.game_config.get_available_games()
        self.game_combo.clear()
        if games:
            self.game_combo.addItems(games)
            self.start_button.setEnabled(True)
        else:
            self.game_combo.addItem("无可用游戏配置")
            self.start_button.setEnabled(False)
    
    def add_rl_log(self, message):
        """添加强化学习日志"""
        self.log_text.append(message)
    
    def start_training(self):
        """开始训练"""
        # 根据当前选择的训练模式执行不同的训练方法
        if self.training_mode_combo.currentIndex() == 0:
            # 传统训练
            domain = self.domain_combo.currentText()
            data_path = self.data_path_label.text()
            
            if data_path == "未选择":
                self.log_text.append("错误: 请先选择训练数据")
                return
            
            try:
                params = json.loads(self.params_edit.toPlainText())
            except json.JSONDecodeError:
                self.log_text.append("错误: 训练参数格式不正确，请使用有效的JSON格式")
                return
            
            self.log_text.append(f"开始训练 - 领域: {domain}")
            self.log_text.append(f"训练参数: {params}")
            
            # 启动训练过程
            self.training_manager.set_log_callback(lambda msg: self.log_text.append(msg))
            self.training_manager.start_training(domain, data_path, params)
        else:
            # 游戏自主训练
            game_name = self.game_combo.currentText()
            if game_name == "无可用游戏配置":
                self.log_text.append("错误: 请先创建游戏配置")
                return
            
            try:
                params = json.loads(self.rl_params_edit.toPlainText())
            except json.JSONDecodeError:
                self.log_text.append("错误: 强化学习参数格式不正确，请使用有效的JSON格式")
                return
            
            self.log_text.append(f"开始游戏自主训练 - 游戏: {game_name}")
            self.log_text.append(f"训练参数: {params}")
            
            # 加载游戏配置
            try:
                self.game_config.load_game_config(game_name)
                # 设置强化学习环境
                self.rl_training_manager.setup(self.computer_controller, self.game_config, params)
                # 启动强化学习训练
                self.rl_training_manager.start_training(params)
            except Exception as e:
                self.log_text.append(f"错误: 启动训练失败 - {str(e)}")
                return
        
        # 更新UI状态
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage("训练中...")
    
    def pause_training(self):
        """暂停训练"""
        if self.training_mode_combo.currentIndex() == 0:
            self.training_manager.pause_training()
        else:
            self.rl_training_manager.pause_training()
            
        self.log_text.append("训练已暂停")
        self.pause_button.setText("继续")
        self.pause_button.clicked.disconnect()
        self.pause_button.clicked.connect(self.resume_training)
        self.statusBar().showMessage("训练已暂停")
    
    def resume_training(self):
        """继续训练"""
        if self.training_mode_combo.currentIndex() == 0:
            self.training_manager.resume_training()
        else:
            self.rl_training_manager.resume_training()
            
        self.log_text.append("训练已继续")
        self.pause_button.setText("暂停")
        self.pause_button.clicked.disconnect()
        self.pause_button.clicked.connect(self.pause_training)
        self.statusBar().showMessage("训练中...")
    
    def stop_training(self):
        """停止训练"""
        if self.training_mode_combo.currentIndex() == 0:
            self.training_manager.stop_training()
        else:
            self.rl_training_manager.stop_training()
            
        self.log_text.append("训练已停止")
        
        # 更新UI状态
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.pause_button.setText("暂停")
        self.pause_button.clicked.disconnect()
        self.pause_button.clicked.connect(self.pause_training)
        self.statusBar().showMessage("就绪")
    
    def refresh_models(self):
        # 获取可用模型列表
        models = self.model_manager.get_available_models()
        
        # 更新下拉列表
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
            self.start_operation_button.setEnabled(True)
        else:
            self.model_combo.addItem("无可用模型")
            self.start_operation_button.setEnabled(False)
    
    def start_operation(self):
        model_name = self.model_combo.currentText()
        if model_name == "无可用模型":
            return
        
        # 加载选定的模型
        model = self.model_manager.load_model(model_name)
        if not model:
            self.operation_log_text.append(f"错误: 无法加载模型 {model_name}")
            return
        
        # 启动操作
        self.computer_controller.set_model(model)
        self.computer_controller.start()
        
        # 更新UI
        self.operation_log_text.append(f"开始使用模型 {model_name} 进行操作")
        self.start_operation_button.setEnabled(False)
        self.stop_operation_button.setEnabled(True)
        self.statusBar().showMessage("AI正在操作中...")
    
    def stop_operation(self):
        # 停止操作
        self.computer_controller.stop()
        
        # 更新UI
        self.operation_log_text.append("操作已停止")
        self.start_operation_button.setEnabled(True)
        self.stop_operation_button.setEnabled(False)
        self.statusBar().showMessage("就绪")
    
    def save_settings(self):
        model_path = self.model_path_edit.toPlainText()
        
        # 确保目录存在
        if not os.path.exists(model_path):
            try:
                os.makedirs(model_path)
            except Exception as e:
                self.statusBar().showMessage(f"错误: 无法创建目录 - {str(e)}")
                return
        
        # 保存设置
        settings = {
            "model_path": model_path
        }
        
        try:
            with open("settings.json", "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
            self.statusBar().showMessage("设置已保存")
        except Exception as e:
            self.statusBar().showMessage(f"错误: 无法保存设置 - {str(e)}")
    
    def save_game_config(self):
        game_name = self.game_name_edit.text().strip()
        game_type = self.game_type_combo.currentText()
        control_keys = [key.strip() for key in self.keys_edit.toPlainText().split('\n') if key.strip()]
        
        if not game_name:
            self.statusBar().showMessage("错误: 游戏名称不能为空")
            return
        
        if not control_keys:
            self.statusBar().showMessage("错误: 请至少设置一个操作按键")
            return
        
        try:
            game_config = GameConfig()
            game_config.create_game_config(game_name, game_type, control_keys)
            self.statusBar().showMessage("游戏配置已保存")
        except Exception as e:
            self.statusBar().showMessage(f"错误: 保存游戏配置失败 - {str(e)}")


def main():
    # 确保必要的目录存在
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 启动应用
    app = QApplication(sys.argv)
    window = AISubsidiaryApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
