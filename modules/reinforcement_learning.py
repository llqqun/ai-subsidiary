import os
import time
import threading
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
import random
from PIL import ImageGrab
import cv2

class GameEnvironment:
    """游戏环境类，负责与游戏交互并提供强化学习所需的接口"""
    
    def __init__(self, computer_controller, game_config):
        self.controller = computer_controller
        self.game_config = game_config
        self.state = None
        self.reward_history = []
        self.episode_rewards = 0
        self.episode_steps = 0
        
    def reset(self):
        """重置游戏环境"""
        # 这里可以实现游戏重置逻辑，例如按ESC键返回主菜单，然后重新开始游戏
        # 或者直接重启游戏进程等
        
        # 简单示例：按ESC键
        self.controller._send_key('esc')
        time.sleep(1)  # 等待游戏响应
        
        # 获取初始状态
        self.state = self._get_state()
        self.episode_rewards = 0
        self.episode_steps = 0
        return self.state
    
    def step(self, action):
        """执行动作并获取结果
        
        Args:
            action: 要执行的动作索引
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # 将动作索引转换为实际操作
        action_key = self._index_to_action(action)
        
        # 执行动作
        self.controller._send_key(action_key)
        time.sleep(0.1)  # 等待游戏响应
        
        # 获取新状态
        next_state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward(next_state)
        self.episode_rewards += reward
        self.episode_steps += 1
        
        # 判断游戏是否结束
        done = self._is_done(next_state)
        
        # 更新当前状态
        self.state = next_state
        
        # 返回结果
        info = {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps
        }
        return next_state, reward, done, info
    
    def _get_state(self):
        """获取当前游戏状态"""
        # 捕获屏幕画面
        screen = ImageGrab.grab()
        # 转换为OpenCV格式
        screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        # 调整大小以提高处理速度
        screen = cv2.resize(screen, (84, 84))
        # 转换为灰度图
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        # 归一化
        normalized = gray / 255.0
        return normalized
    
    def _index_to_action(self, action_index):
        """将动作索引转换为实际操作按键"""
        control_keys = self.game_config.current_config['control_keys']
        if 0 <= action_index < len(control_keys):
            return control_keys[action_index]
        return None
    
    def _calculate_reward(self, state):
        """根据游戏状态计算奖励"""
        reward = 0
        
        # 如果有配置奖励规则，则根据规则计算奖励
        if 'reward_rules' in self.game_config.current_config:
            for rule in self.game_config.current_config['reward_rules']:
                # 这里需要根据具体规则实现奖励计算逻辑
                # 例如：检测特定区域的像素变化、游戏分数变化等
                pass
        
        # 默认给予小的正向奖励，鼓励探索
        reward += 0.1
        
        return reward
    
    def _is_done(self, state):
        """判断游戏是否结束"""
        # 这里需要根据游戏特性实现结束条件判断
        # 例如：检测游戏结束画面、生命值为0等
        
        # 简单示例：如果步数超过1000，则认为一个回合结束
        if self.episode_steps >= 1000:
            return True
        
        return False


class DQNModel(nn.Module):
    """深度Q网络模型"""
    
    def __init__(self, input_channels, action_size):
        super(DQNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_size)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样经验"""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class RLTrainingManager:
    """强化学习训练管理器"""
    
    def __init__(self):
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.env = None
        self.replay_buffer = None
        self.train_thread = None
        self.is_training = False
        self.is_paused = False
        self.log_callback = None
        self.episode_count = 0
        self.total_steps = 0
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # 折扣因子
        self.update_target_every = 5  # 每5个回合更新目标网络
    
    def set_log_callback(self, callback):
        """设置日志回调函数"""
        self.log_callback = callback
    
    def log(self, message):
        """记录日志"""
        print(message)
        if self.log_callback:
            self.log_callback(message)
    
    def setup(self, computer_controller, game_config, params):
        """设置训练环境和参数"""
        # 创建游戏环境
        self.env = GameEnvironment(computer_controller, game_config)
        
        # 获取动作空间大小
        action_size = len(game_config.current_config['control_keys'])
        
        # 创建DQN模型
        self.model = DQNModel(1, action_size)  # 1个输入通道（灰度图）
        self.target_model = DQNModel(1, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 设置优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=params.get('learning_rate', 0.001))
        
        # 创建经验回FFER区
        self.replay_buffer = ReplayBuffer(params.get('buffer_capacity', 10000))
        
        # 设置训练参数
        self.epsilon = params.get('epsilon_start', 1.0)
        self.epsilon_min = params.get('epsilon_min', 0.1)
        self.epsilon_decay = params.get('epsilon_decay', 0.995)
        self.gamma = params.get('gamma', 0.99)
        self.update_target_every = params.get('update_target_every', 5)
    
    def start_training(self, params):
        """开始训练过程"""
        if not self.env:
            self.log("错误: 环境未设置，请先调用setup方法")
            return
        
        # 创建新线程进行训练
        self.train_thread = threading.Thread(target=self._train_process, args=(params,))
        self.is_training = True
        self.is_paused = False
        self.train_thread.start()
    
    def pause_training(self):
        """暂停训练"""
        self.is_paused = True
    
    def resume_training(self):
        """继续训练"""
        self.is_paused = False
    
    def stop_training(self):
        """停止训练"""
        self.is_training = False
        if self.train_thread and self.train_thread.is_alive():
            self.train_thread.join(timeout=1.0)
    
    def _train_process(self, params):
        """训练过程的实现"""
        try:
            self.log("开始强化学习训练")
            
            # 训练参数
            batch_size = params.get('batch_size', 32)
            episodes = params.get('episodes', 1000)
            
            for episode in range(episodes):
                if not self.is_training:
                    break
                
                # 暂停逻辑
                while self.is_paused and self.is_training:
                    time.sleep(0.1)
                
                if not self.is_training:
                    break
                
                # 重置环境
                state = self.env.reset()
                state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
                done = False
                episode_reward = 0
                
                while not done and self.is_training:
                    # 选择动作（epsilon-贪婪策略）
                    if random.random() <= self.epsilon:
                        action = random.randrange(len(self.env.game_config.current_config['control_keys']))
                    else:
                        with torch.no_grad():
                            q_values = self.model(state)
                            action = torch.argmax(q_values).item()
                    
                    # 执行动作
                    next_state, reward, done, info = self.env.step(action)
                    next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)
                    
                    # 存储经验
                    self.replay_buffer.add(state, action, reward, next_state, done)
                    
                    # 更新状态和累积奖励
                    state = next_state
                    episode_reward += reward
                    self.total_steps += 1
                    
                    # 从经验回放中学习
                    if len(self.replay_buffer) > batch_size:
                        self._learn(batch_size)
                    
                    # 暂停逻辑
                    while self.is_paused and self.is_training:
                        time.sleep(0.1)
                    
                    if not self.is_training:
                        break
                
                # 更新探索率
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                # 更新目标网络
                if episode % self.update_target_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                
                # 记录训练进度
                self.episode_count += 1
                self.log(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.4f}")
                
                # 每100个回合保存一次模型
                if (episode + 1) % 100 == 0:
                    self._save_model()
            
            # 训练结束，保存最终模型
            self._save_model()
            self.log("训练完成")
            
        except Exception as e:
            self.log(f"训练过程中出错: {str(e)}")
        finally:
            self.is_training = False
    
    def _learn(self, batch_size):
        """从经验回放中学习"""
        # 采样经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 计算当前Q值
        current_q = self.model(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _save_model(self):
        """保存模型"""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        model_name = f"RL_{self.env.game_config.current_game}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
        model_path = os.path.join("models", model_name)
        
        # 保存模型状态
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps
        }, model_path)
        
        self.log(f"模型已保存: {model_path}")
    
    def load_model(self, model_path):
        """加载已保存的模型"""
        if not os.path.exists(model_path):
            self.log(f"错误: 模型文件不存在 {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.episode_count = checkpoint.get('episode_count', 0)
            self.total_steps = checkpoint.get('total_steps', 0)
            
            self.log(f"成功加载模型: {model_path}")
            return True
        except Exception as e:
            self.log(f"加载模型失败: {str(e)}")
            return False