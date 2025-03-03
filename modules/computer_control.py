import time
import threading
import cv2
import numpy as np
import torch
from PIL import ImageGrab
import win32api
import win32con
import win32gui

class ComputerController:
    """计算机控制器，负责执行AI的操作指令"""
    
    def __init__(self):
        self.model = None
        self.is_running = False
        self.control_thread = None
        # 使用win32api获取屏幕尺寸
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        
        # 安全设置
        self.failsafe = True
        self.pause = 0.1
    
    def set_model(self, model):
        """设置要使用的AI模型"""
        self.model = model
    
    def start(self):
        """开始执行操作"""
        if not self.model:
            raise ValueError("未设置AI模型")
        
        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.start()
    
    def stop(self):
        """停止执行操作"""
        self.is_running = False
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)
    
    def _control_loop(self):
        """控制循环，持续执行AI的决策"""
        try:
            while self.is_running:
                # 获取屏幕截图
                screen = self._capture_screen()
                
                # 使用模型进行决策
                action = self._get_action(screen)
                
                # 执行操作
                self._execute_action(action)
                
                # 短暂休息，避免过于频繁的操作
                time.sleep(0.05)
        except Exception as e:
            print(f"操作执行出错: {str(e)}")
            self.is_running = False
    
    def _capture_screen(self):
        """捕获屏幕画面"""
        # 使用PIL的ImageGrab获取屏幕截图
        screen = ImageGrab.grab()
        # 转换为OpenCV格式
        screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        # 调整大小以提高处理速度
        screen = cv2.resize(screen, (800, 600))
        return screen
    
    def _get_action(self, screen):
        """使用AI模型获取下一步操作"""
        # 这里需要根据具体的模型实现来处理图像数据
        # 并返回具体的操作指令
        try:
            # 预处理图像
            processed_screen = self._preprocess_screen(screen)
            
            # 使用模型预测动作
            with torch.no_grad():
                action = self.model(processed_screen)
            
            return action
        except Exception as e:
            print(f"获取动作失败: {str(e)}")
            return None
    
    def _preprocess_screen(self, screen):
        """预处理屏幕图像"""
        # 转换为灰度图
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        # 调整大小
        resized = cv2.resize(gray, (84, 84))
        # 归一化
        normalized = resized / 255.0
        # 转换为张量
        tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
        return tensor
    
    def _execute_action(self, action):
        """执行具体的操作"""
        if action is None:
            return
        
        try:
            # 这里需要根据模型输出的动作类型来执行具体操作
            # 例如：移动鼠标、点击、按键等
            action_type = self._interpret_action(action)
            
            if action_type['type'] == 'mouse_move':
                # 使用win32api移动鼠标
                win32api.SetCursorPos((action_type['x'], action_type['y']))
                time.sleep(self.pause)  # 模拟pyautogui的暂停
            elif action_type['type'] == 'mouse_click':
                # 获取当前鼠标位置
                x, y = win32api.GetCursorPos()
                # 鼠标左键点击
                clicks = action_type.get('clicks', 1)
                interval = action_type.get('interval', 0.1)
                
                for _ in range(clicks):
                    # 鼠标按下
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
                    # 鼠标释放
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
                    if clicks > 1 and _ < clicks - 1:
                        time.sleep(interval)
                time.sleep(self.pause)
            elif action_type['type'] == 'key_press':
                self._send_key(action_type['key'])
                time.sleep(self.pause)
            elif action_type['type'] == 'key_hold':
                self._key_down(action_type['key'])
                time.sleep(self.pause)
            elif action_type['type'] == 'key_release':
                self._key_up(action_type['key'])
                time.sleep(self.pause)
        except Exception as e:
            print(f"执行动作失败: {str(e)}")
    
    def _send_key(self, key):
        """发送按键"""
        # 键盘按键映射到win32con虚拟键码
        key_to_vk = {
            'up': win32con.VK_UP,
            'down': win32con.VK_DOWN,
            'left': win32con.VK_LEFT,
            'right': win32con.VK_RIGHT,
            'space': win32con.VK_SPACE,
            'enter': win32con.VK_RETURN,
            'esc': win32con.VK_ESCAPE
        }
        
        if key in key_to_vk:
            vk_code = key_to_vk[key]
            # 按下并释放按键
            win32api.keybd_event(vk_code, 0, 0, 0)  # 按下
            time.sleep(0.05)
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)  # 释放
    
    def _key_down(self, key):
        """按下按键"""
        key_to_vk = {
            'up': win32con.VK_UP,
            'down': win32con.VK_DOWN,
            'left': win32con.VK_LEFT,
            'right': win32con.VK_RIGHT,
            'space': win32con.VK_SPACE,
            'enter': win32con.VK_RETURN,
            'esc': win32con.VK_ESCAPE
        }
        
        if key in key_to_vk:
            vk_code = key_to_vk[key]
            win32api.keybd_event(vk_code, 0, 0, 0)  # 按下
    
    def _key_up(self, key):
        """释放按键"""
        key_to_vk = {
            'up': win32con.VK_UP,
            'down': win32con.VK_DOWN,
            'left': win32con.VK_LEFT,
            'right': win32con.VK_RIGHT,
            'space': win32con.VK_SPACE,
            'enter': win32con.VK_RETURN,
            'esc': win32con.VK_ESCAPE
        }
        
        if key in key_to_vk:
            vk_code = key_to_vk[key]
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)  # 释放
    
    def _interpret_action(self, action):
        """解释模型输出的动作"""
        # 将模型输出的张量转换为numpy数组
        action_np = action.cpu().numpy()
        
        # 获取最大值的索引作为动作类型
        action_type_idx = np.argmax(action_np)
        
        # 根据索引确定动作类型
        # 假设前3个索引对应鼠标操作，后面的对应键盘操作
        if action_type_idx == 0:  # 鼠标移动
            # 从动作张量中提取坐标信息
            # 假设动作张量的第1和第2个元素分别表示x和y的相对位置（范围在-1到1之间）
            x_rel = float(action_np[0][1])
            y_rel = float(action_np[0][2])
            
            # 将相对位置转换为屏幕坐标
            x = int((x_rel + 1) / 2 * self.screen_width)
            y = int((y_rel + 1) / 2 * self.screen_height)
            
            return {
                'type': 'mouse_move',
                'x': x,
                'y': y
            }
        elif action_type_idx == 1:  # 鼠标点击
            return {
                'type': 'mouse_click'
            }
        elif action_type_idx == 2:  # 鼠标双击
            return {
                'type': 'mouse_click',
                'clicks': 2,
                'interval': 0.1
            }
        else:  # 键盘操作
            # 键盘按键映射
            key_mapping = {
                3: 'up',
                4: 'down',
                5: 'left',
                6: 'right',
                7: 'space',
                8: 'enter',
                9: 'esc'
            }
            
            if action_type_idx in key_mapping:
                return {
                    'type': 'key_press',
                    'key': key_mapping[action_type_idx]
                }
            else:
                # 默认返回空操作
                return None