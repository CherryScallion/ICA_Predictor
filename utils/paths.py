# utils/paths.py
"""
统一路径管理模块 - 确保所有路径都基于项目根目录 (HachimiNetV1/)
"""
import os
from pathlib import Path

def get_project_root():
    """
    获取项目根目录 (HachimiNetV1/)。
    
    策略：
    1. 检查环境变量 HACHIMINET_ROOT（如果设置）
    2. 从当前文件向上查找包含 'HachimiNetV1' 的目录
    3. 如果都没找到，返回当前工作目录（假设在项目根目录运行）
    
    Returns:
        Path: 项目根目录的 Path 对象
    """
    # 1. 环境变量优先
    env_root = os.environ.get('HACHIMINET_ROOT')
    if env_root:
        root = Path(env_root).resolve()
        if root.exists():
            return root
    
    # 2. 从当前文件位置向上查找
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if parent.name == 'HachimiNetV1':
            return parent
    
    # 3. 尝试查找包含 main.py 和 configs/ 的目录
    for parent in current_file.parents:
        if (parent / 'main.py').exists() and (parent / 'configs').exists():
            return parent
    
    # 4. 回退：假设当前工作目录就是项目根目录
    cwd = Path.cwd()
    if (cwd / 'main.py').exists() and (cwd / 'configs').exists():
        return cwd
    
    # 5. 最后回退：返回包含 utils/ 的目录的父目录（如果从子目录调用）
    if current_file.parent.name == 'utils':
        return current_file.parent.parent
    
    # 如果都没找到，返回当前文件所在目录的父目录
    # （假设在 utils/ 下调用，那么上级就是项目根目录）
    return current_file.parent.parent

# 全局项目根目录
PROJECT_ROOT = get_project_root()

def get_config_path():
    """获取配置文件路径"""
    return PROJECT_ROOT / 'configs' / 'train_config.yaml'

def get_data_dir(subdir=''):
    """获取数据目录路径"""
    if subdir:
        return PROJECT_ROOT / 'data' / subdir
    return PROJECT_ROOT / 'data'

def get_checkpoint_dir():
    """获取 checkpoints 目录路径"""
    return PROJECT_ROOT / 'checkpoints'

def get_template_dir():
    """获取模板目录路径"""
    return PROJECT_ROOT / 'data' / 'templates'

def resolve_path(path_str):
    """
    解析路径字符串，如果是相对路径则基于项目根目录解析。
    
    Args:
        path_str: 路径字符串，可以是绝对路径或相对路径
        
    Returns:
        Path: 解析后的绝对路径
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    # 如果是相对路径，基于项目根目录解析
    return PROJECT_ROOT / path

