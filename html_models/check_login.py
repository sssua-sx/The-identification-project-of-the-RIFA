# encoding=gbk
from .config import creat_db
import threading

lock = threading.Lock()

conn,cur = creat_db()
def is_null(username,password):
    if username == '' or password == '':
        return True
    else:
        return False

def is_existed(username,password):
    sql="SELECT * FROM user WHERE username ='%s' and password ='%s'" %(username,password)
    lock.acquire()
    conn.ping(reconnect=True)
    cur.execute("use ExistedUsers")
    cur.execute(sql)
    conn.commit()
    lock.release()
    result = cur.fetchall()
    if len(result) == 0:
        return False
    else:
        return True

def exist_user(username):
    sql = "SELECT * FROM user WHERE username ='%s'" % (username)
    lock.acquire()
    conn.ping(reconnect=True)
    cur.execute("use ExistedUsers")
    cur.execute(sql)
    conn.commit()
    lock.release()
    result = cur.fetchall()
    if len(result) == 0:
        return False
    else:
        return True