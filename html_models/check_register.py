# encoding=gbk
from .config import creat_db
import yaml
import threading

conn,cur = creat_db()
lock = threading.Lock()

def add_user(username, password):
    # sql commands �����û��������ݿ�
    lock.acquire()
    sql = "INSERT INTO user(username, password) VALUES ('%s','%s')" %(username, password)
    # execute(sql)
    conn.ping(reconnect=True)
    cur.execute("use ExistedUsers")
    cur.execute(sql)
    # commit
    conn.commit()  # �����ݿ������иı䣬��Ҫcommit()
    conn.close()
    lock.release()
    # �����û����ݵ�yaml�ļ�
    new_user = {username: password}
    with open('E:/ͼ����ƽ̨/webyolox-main/run/user.yaml', 'a', encoding='utf-8', ) as f:
        yaml.dump(new_user, f, encoding='utf-8', allow_unicode=True)
