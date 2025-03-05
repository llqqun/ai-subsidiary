import json
import os

class GameConfig:
    """游戏配置管理器，用于存储和管理游戏的基本设置"""
    
    def __init__(self):
        # 使用项目根目录作为基准路径
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.base_path, 'configs')
        os.makedirs(self.config_path, exist_ok=True)
        self.current_config = None
        self.current_game = None
        
        # 确保默认配置文件存在
        default_config_path = os.path.join(self.config_path, 'default.json')
        if not os.path.exists(default_config_path):
            self.create_game_config('默认配置', '通用', [
                '上移: W',
                '下移: S',
                '左移: A',
                '右移: D',
                '确认: Space',
                '取消: Esc',
                '互动: E',
                '跳跃: Space',
                '攻击: Mouse1',
                '防御: Mouse2'
            ])
    
    def import_game_config(self, file_path):
        """从外部文件导入游戏配置
        
        Args:
            file_path (str): 配置文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 验证配置文件格式
            required_fields = ['game_name', 'game_type', 'control_keys']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f'配置文件缺少必要字段: {field}')
            
            # 保存到配置目录
            game_name = config['game_name']
            config_file = os.path.join(self.config_path, f'{game_name}.json')
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            
            self.current_config = config
            self.current_game = game_name
            return config
        except Exception as e:
            raise ValueError(f'导入配置文件失败: {str(e)}')
    
    def create_game_config(self, game_name, game_type, control_keys):
        """创建新的游戏配置
        
        Args:
            game_name (str): 游戏名称
            game_type (str): 游戏类型
            control_keys (list): 可用的操作按键列表
        """
        config = {
            'game_name': game_name,
            'game_type': game_type,
            'control_keys': control_keys,
            'reward_rules': [],
            'state_indicators': []
        }
        
        # 保存配置文件
        config_file = os.path.join(self.config_path, f'{game_name}.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        
        self.current_config = config
        self.current_game = game_name
        return config
    
    def load_game_config(self, game_name):
        """加载已有的游戏配置
        
        Args:
            game_name (str): 游戏名称
        """
        config_file = os.path.join(self.config_path, f'{game_name}.json')
        if not os.path.exists(config_file):
            raise FileNotFoundError(f'游戏配置文件不存在: {game_name}')
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.current_config = config
        self.current_game = game_name
        return config
    
    def on_reload(self):
        """热更新回调函数，在模块重新加载时调用"""
        if self.current_game:
            self.load_game_config(self.current_game)
    
    def add_reward_rule(self, condition, reward):
        """添加奖励规则
        
        Args:
            condition (dict): 触发奖励的条件
            reward (float): 奖励值
        """
        if not self.current_config:
            raise ValueError('未加载游戏配置')
        
        rule = {
            'condition': condition,
            'reward': reward
        }
        self.current_config['reward_rules'].append(rule)
        self._save_current_config()
    
    def add_state_indicator(self, name, region):
        """添加状态指示器
        
        Args:
            name (str): 指示器名称
            region (tuple): 屏幕区域 (x, y, width, height)
        """
        if not self.current_config:
            raise ValueError('未加载游戏配置')
        
        indicator = {
            'name': name,
            'region': region
        }
        self.current_config['state_indicators'].append(indicator)
        self._save_current_config()
    
    def get_available_games(self):
        """获取所有已配置的游戏列表"""
        games = []
        for file in os.listdir(self.config_path):
            if file.endswith('.json'):
                games.append(file[:-5])  # 移除.json后缀
        return games
    
    def _save_current_config(self):
        """保存当前配置到文件"""
        if not self.current_config or not self.current_game:
            return
        
        config_file = os.path.join(self.config_path, f'{self.current_game}.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_config, f, ensure_ascii=False, indent=4)