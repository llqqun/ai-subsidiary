import os
import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class HotReloadHandler(FileSystemEventHandler):
    """文件变更处理器，用于检测并响应文件变化"""
    
    def __init__(self, callback=None):
        self.last_modified = 0
        self.callback = callback
    
    def on_modified(self, event):
        if event.is_directory:
            return
            
        # 检查文件扩展名
        if not event.src_path.endswith('.py'):
            return
            
        # 防止重复触发
        current_time = time.time()
        if current_time - self.last_modified < 1:
            return
        self.last_modified = current_time
        
        # 执行回调
        if self.callback:
            self.callback(event.src_path)

class HotReloader:
    """热更新管理器，负责监控文件变化并重新加载模块"""
    
    def __init__(self, watch_path, callback=None):
        self.watch_path = watch_path
        self.observer = None
        self.handler = HotReloadHandler(callback)
    
    def start(self):
        """启动文件监控"""
        if self.observer is None:
            self.observer = Observer()
            self.observer.schedule(self.handler, self.watch_path, recursive=True)
            self.observer.start()
            logging.info(f'开始监控目录: {self.watch_path}')
    
    def stop(self):
        """停止文件监控"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logging.info('停止文件监控')
    
    @staticmethod
    def reload_module(module_path):
        """重新加载指定模块
        
        Args:
            module_path (str): 模块文件路径
        """
        try:
            # 获取模块名称
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            
            # 查找已加载的模块
            for name, module in list(sys.modules.items()):
                if name.endswith(module_name):
                    # 重新加载模块
                    if hasattr(module, 'on_reload'):
                        module.on_reload()
                    reload_result = __import__(name, fromlist=['*'])
                    sys.modules[name] = reload_result
                    logging.info(f'成功重新加载模块: {name}')
        except Exception as e:
            logging.error(f'重新加载模块失败: {str(e)}')