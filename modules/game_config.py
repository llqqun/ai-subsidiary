import json
import os

class GameConfig:
    """游戏配置管理器，用于存储和管理游戏的基本设置"""
    
    def __init__(self):
        self.config_path = os.path.join(os.getcwd(), 'configs')
        os.makedirs(self.config_path, exist_ok=True)
        self.current_config = None
        self.current_game = None
    
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