import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QTextEdit, QFileDialog, QTabWidget,
                             QLineEdit, QGroupBox, QStackedWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

# 导入自定义模块
from modules.training import TrainingManager
from modules.computer_control import ComputerController
from modules.models import ModelManager
from modules.game_config import GameConfig
from modules.reinforcement_learning import RLTrainingManager
from modules.hot_reload import HotReloader

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
        
        # 初始化强化学习管理器
        self.rl_training_manager = RLTrainingManager()
        self.rl_training_manager.set_log_callback(self.add_rl_log)
        
        # 初始化热更新管理器
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.hot_reloader = HotReloader(base_path, self.on_module_changed)
        self.hot_reloader.start()
        
        # 设置UI
        self.init_ui()
        
        # 加载已有游戏配置
        self.refresh_game_configs()
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        self.hot_reloader.stop()
        event.accept()
        self.setWindowTitle("AI辅助训练系统")
        self.setGeometry(100, 100, 1000, 700)
        
        # 初始化组件
        self.training_manager = TrainingManager()
        self.computer_controller = ComputerController()
        self.model_manager = ModelManager()
        self.game_config = GameConfig()
        
        # 初始化强化学习管理器
        self.rl_training_manager = RLTrainingManager()
        self.rl_training_manager.set_log_callback(self.add_rl_log)
        
        # 设置UI
        self.init_ui()
        
        # 加载已有游戏配置
        self.refresh_game_configs()
    
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
        
        # 游戏配置选择
        config_select_layout = QHBoxLayout()
        config_select_label = QLabel("已有配置:")
        self.config_select_combo = QComboBox()
        self.config_select_combo.currentIndexChanged.connect(self.load_selected_config)
        config_select_layout.addWidget(config_select_label)
        config_select_layout.addWidget(self.config_select_combo)
        game_config_layout.addLayout(config_select_layout)
        
        # 导入配置按钮
        import_config_button = QPushButton("导入游戏配置")
        import_config_button.clicked.connect(self.import_game_config)
        game_config_layout.addWidget(import_config_button)
        
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
        self.game_type_combo.addItems(["冒险", "动作", "策略", "角色扮演", "其他"])
        game_type_layout.addWidget(game_type_label)
        game_type_layout.addWidget(self.game_type_combo)
        game_config_layout.addLayout(game_type_layout)
        
        # 操作按键
        keys_label = QLabel("操作按键:")
        game_config_layout.addWidget(keys_label)
        
        # 创建按键设置容器
        self.keys_container = QWidget()
        keys_container_layout = QVBoxLayout(self.keys_container)
        keys_container_layout.setSpacing(5)
        game_config_layout.addWidget(self.keys_container)
        
        # 添加按键行的按钮
        add_key_button = QPushButton("添加按键")
        add_key_button.clicked.connect(self.add_key_row)
        game_config_layout.addWidget(add_key_button)
        
        # 初始添加一行按键设置
        self.add_key_row()
        
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
    
    def refresh_game_configs(self):
        """刷新游戏配置列表"""
        games = self.game_config.get_available_games()
        self.config_select_combo.clear()
        if games:
            self.config_select_combo.addItems(games)
    
    def load_selected_config(self, index):
        """加载选中的游戏配置"""
        if index < 0:
            return
        
        game_name = self.config_select_combo.currentText()
        try:
            config = self.game_config.load_game_config(game_name)
            # 更新界面显示
            self.game_name_edit.setText(config['game_name'])
            self.game_type_combo.setCurrentText(config['game_type'])
            
            # 清除现有的按键行
            while self.keys_container.layout().count():
                item = self.keys_container.layout().takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # 为每个控制键添加新的按键行
            for key_config in config['control_keys']:
                action, key = key_config.split(': ', 1)
                row_widget = self.add_key_row()
                action_input = row_widget.layout().itemAt(0).widget()
                key_display = row_widget.layout().itemAt(1).widget()
                action_input.setText(action)
                key_display.setText(key)
                
        except Exception as e:
            self.statusBar().showMessage(f"错误: 加载配置失败 - {str(e)}")
    
    def import_game_config(self):
        """导入游戏配置"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "JSON文件 (*.json)", options=options)
        if file_path:
            try:
                self.game_config.import_game_config(file_path)
                self.refresh_game_configs()
                # 选择新导入的配置
                index = self.config_select_combo.findText(self.game_config.current_game)
                if index >= 0:
                    self.config_select_combo.setCurrentIndex(index)
                self.statusBar().showMessage("配置导入成功")
            except Exception as e:
                self.statusBar().showMessage(f"错误: 导入配置失败 - {str(e)}")
    
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
    
    def start_key_recording(self, key_display):
        """开始录制按键"""
        key_display.setText("按下任意键...")
        key_display.setFocus()
        key_display.installEventFilter(self)
        self.recording_key_display = key_display
    
    def eventFilter(self, obj, event):
        """事件过滤器，用于捕获按键事件"""
        if hasattr(self, 'recording_key_display') and obj == self.recording_key_display:
            if event.type() == event.KeyPress:
                key_text = event.text().upper() if event.text() else QKeySequence(event.key()).toString()
                self.recording_key_display.setText(key_text)
                self.recording_key_display.removeEventFilter(self)
                self.recording_key_display = None
                return True
        return super().eventFilter(obj, event)
    
    def add_key_row(self):
        """添加一行按键设置"""
        # 创建一行的容器
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setSpacing(5)
        
        # 操作效果输入框
        action_input = QLineEdit()
        action_input.setPlaceholderText("操作效果（如：向上、跳跃等）")
        row_layout.addWidget(action_input)
        
        # 按键显示框
        key_display = QLineEdit()
        key_display.setPlaceholderText("按下按键")
        key_display.setReadOnly(True)
        row_layout.addWidget(key_display)
        
        # 录制按钮
        record_button = QPushButton("录制")
        record_button.clicked.connect(lambda: self.start_key_recording(key_display))
        row_layout.addWidget(record_button)
        
        # 删除按钮
        delete_button = QPushButton("删除")
        delete_button.clicked.connect(lambda: row_widget.deleteLater())
        row_layout.addWidget(delete_button)
        
        # 将行添加到容器中
        self.keys_container.layout().addWidget(row_widget)
    
    def import_game_config(self):
        """导入游戏配置"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择游戏配置文件", "", "JSON文件 (*.json)", options=options)
        if file_path:
            try:
                print('导入配置')
                self.game_config.import_game_config(file_path)
                self.statusBar().showMessage("游戏配置导入成功")
                self.refresh_games()
                self.refresh_game_configs()  # 刷新配置列表
            except Exception as e:
                self.statusBar().showMessage(f"错误: {str(e)}")
    
    def save_game_config(self):
        """保存游戏配置"""
        game_name = self.game_name_edit.text()
        game_type = self.game_type_combo.currentText()
        
        # 从按键行容器中获取所有按键设置
        control_keys = []
        for i in range(self.keys_container.layout().count()):
            row_widget = self.keys_container.layout().itemAt(i).widget()
            if row_widget:
                action_input = row_widget.layout().itemAt(0).widget()
                key_display = row_widget.layout().itemAt(1).widget()
                if action_input.text() and key_display.text():
                    control_keys.append(f"{action_input.text()}: {key_display.text()}")
        
        if not game_name:
            self.statusBar().showMessage("错误: 请输入游戏名称")
            return
        
        if not control_keys:
            self.statusBar().showMessage("错误: 请至少添加一个操作按键")
            return
        
        try:
            self.game_config.create_game_config(game_name, game_type, control_keys)
            self.statusBar().showMessage("游戏配置已保存")
            self.refresh_games()
            self.refresh_game_configs()  # 刷新配置列表
        except Exception as e:
            self.statusBar().showMessage(f"错误: 无法保存游戏配置 - {str(e)}")
    
    def on_module_changed(self, module_path):
        """热更新回调函数，在模块发生变化时调用"""
        try:
            HotReloader.reload_module(module_path)
            self.log_text.append(f"模块已更新: {os.path.basename(module_path)}")
        except Exception as e:
            self.log_text.append(f"模块更新失败: {str(e)}")