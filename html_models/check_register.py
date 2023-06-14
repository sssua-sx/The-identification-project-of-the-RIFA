# encoding=gbk
from .config import creat_db
import yaml
import threading

conn,cur = creat_db()
lock = threading.Lock()

def add_user(username, password):
    # sql commands 将新用户存入数据库
    lock.acquire()
    sql = "INSERT INTO user(username, password) VALUES ('%s','%s')" %(username, password)
    # execute(sql)
    conn.ping(reconnect=True)
    cur.execute("use ExistedUsers")
    cur.execute(sql)
    # commit
    conn.commit()  # 对数据库内容有改变，需要commit()
    conn.close()
    lock.release()
    # 将新用户备份到yaml文件
    new_user = {username: password}
    with open('E:/图像检测平台/webyolox-main/run/user.yaml', 'a', encoding='utf-8', ) as f:
        yaml.dump(new_user, f, encoding='utf-8', allow_unicode=True)
