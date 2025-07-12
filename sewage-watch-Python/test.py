import logging
import xml.etree.ElementTree as ET
from pathlib import Path
import mysql.connector
from mysql.connector import Error

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """从XML配置文件中加载配置"""
    try:
        config_path = Path(__file__).parent / "../config.xml"
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return None
        
        tree = ET.parse(config_path)
        root = tree.getroot()
        
        # 构建配置字典
        config = {}
        
        # 尝试加载数据库配置
        try:
            config['database'] = {
                'host': root.find('.//database/host').text,
                'port': int(root.find('.//database/port').text),
                'user': root.find('.//database/user').text,
                'password': root.find('.//database/password').text,
                'name': root.find('.//database/name').text
            }
            logger.info("成功加载数据库配置")
        except AttributeError as e:
            logger.error("数据库配置不完整: %s", e)
            return None
        
        return config
        
    except Exception as e:
        logger.error(f"读取配置文件时出错: {e}")
        return None

def test_db_connection(config):
    """测试数据库连接"""
    if not config or 'database' not in config:
        logger.error("无效的数据库配置")
        return False
    
    try:
        logger.info("尝试连接数据库...")
        connection = mysql.connector.connect(
            host=config['database']['host'],
            port=config['database']['port'],
            user=config['database']['user'],
            password=config['database']['password'],
            database=config['database']['name']
        )
        
        if connection.is_connected():
            db_info = connection.get_server_info()
            logger.info(f"成功连接到MySQL服务器，版本: {db_info}")
            
            # 执行简单查询测试
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            record = cursor.fetchone()
            logger.info(f"测试查询结果: {record}")
            
            cursor.close()
            connection.close()
            logger.info("数据库连接已关闭")
            return True
            
    except Error as e:
        logger.error(f"数据库连接错误: {e}")
        return False

if __name__ == "__main__":
    logger.info("开始数据库连接测试")
    
    # 加载配置
    config = load_config()
    if not config:
        logger.error("无法加载配置，测试终止")
        exit(1)
    
    # 测试连接
    if test_db_connection(config):
        logger.info("数据库连接测试成功")
    else:
        logger.error("数据库连接测试失败")