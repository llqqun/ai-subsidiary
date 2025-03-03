import os
import torch
import glob
import json

class ModelManager:
    """模型管理器，负责模型的加载和管理"""
    
    def __init__(self):
        self.models_dir = "models"
        self.settings_file = "settings.json"
        self.load_settings()
    
    def load_settings(self):
        """加载设置"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                    self.models_dir = settings.get("model_path", "models")
        except Exception as e:
            print(f"加载设置失败: {str(e)}")
    
    def get_available_models(self):
        """获取可用的模型列表"""
        try:
            # 确保目录存在
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir)
            
            # 查找所有PyTorch模型文件
            model_files = glob.glob(os.path.join(self.models_dir, "*.pth"))
            
            # 提取模型名称
            model_names = [os.path.basename(file) for file in model_files]
            
            return model_names
        except Exception as e:
            print(f"获取模型列表失败: {str(e)}")
            return []
    
    def load_model(self, model_name):
        """加载指定的模型"""
        try:
            model_path = os.path.join(self.models_dir, model_name)
            
            if not os.path.exists(model_path):
                print(f"模型文件不存在: {model_path}")
                return None
            
            # 根据模型名称确定模型类型和参数
            model_type, params = self._determine_model_type(model_name)
            
            # 创建模型实例
            model = self._create_model_instance(model_type, params)
            
            # 加载模型权重
            model.load_state_dict(torch.load(model_path))
            model.eval()  # 设置为评估模式
            
            print(f"成功加载模型: {model_name}")
            return model
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            return None
    
    def _determine_model_type(self, model_name):
        """根据模型名称确定模型类型和参数"""
        # 这里可以根据命名规则或元数据文件来确定模型类型和参数
        # 简单示例：假设模型名称格式为 "领域_时间戳.pth"
        domain = model_name.split('_')[0]
        
        # 默认参数
        params = {
            "input_size": 100,
            "hidden_size": 128,
            "output_size": 10
        }
        
        # 根据领域调整参数
        if domain == "游戏":
            params["input_size"] = 84 * 84  # 假设输入是84x84的图像
            params["output_size"] = 10  # 假设有10种可能的操作
        elif domain == "法律":
            params["input_size"] = 1000  # 假设输入是1000维的文本特征
            params["output_size"] = 5  # 假设有5种可能的决策
        
        return domain, params
    
    def _create_model_instance(self, model_type, params):
        """创建模型实例"""
        # 导入模型类
        from modules.training import SimpleModel
        
        # 创建模型实例
        model = SimpleModel(
            input_size=params["input_size"],
            hidden_size=params["hidden_size"],
            output_size=params["output_size"]
        )
        
        return model
    
    def delete_model(self, model_name):
        """删除指定的模型"""
        try:
            model_path = os.path.join(self.models_dir, model_name)
            
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"已删除模型: {model_name}")
                return True
            else:
                print(f"模型不存在: {model_name}")
                return False
        except Exception as e:
            print(f"删除模型失败: {str(e)}")
            return False
    
    def export_model(self, model_name, export_path):
        """导出模型到指定路径"""
        try:
            model_path = os.path.join(self.models_dir, model_name)
            
            if not os.path.exists(model_path):
                print(f"模型文件不存在: {model_path}")
                return False
            
            # 确保导出目录存在
            export_dir = os.path.dirname(export_path)
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            # 复制模型文件
            import shutil
            shutil.copy2(model_path, export_path)
            
            print(f"模型已导出到: {export_path}")
            return True
        except Exception as e:
            print(f"导出模型失败: {str(e)}")
            return False
    
    def import_model(self, import_path):
        """从指定路径导入模型"""
        try:
            if not os.path.exists(import_path):
                print(f"导入文件不存在: {import_path}")
                return False
            
            # 确保模型目录存在
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir)
            
            # 生成目标路径
            model_name = os.path.basename(import_path)
            target_path = os.path.join(self.models_dir, model_name)
            
            # 复制模型文件
            import shutil
            shutil.copy2(import_path, target_path)
            
            print(f"模型已导入: {model_name}")
            return True
        except Exception as e:
            print(f"导入模型失败: {str(e)}")
            return False