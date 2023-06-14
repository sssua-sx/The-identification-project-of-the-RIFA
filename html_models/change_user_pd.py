# encoding=gbk
from .config import creat_db
import threading

lock = threading.Lock()

conn,cur = creat_db()
def change_user(old_username,new_username,new_password):
    lock.acquire()
    conn.ping(reconnect=True)
    cur.execute("use ExistedUsers")
    cur.execute("delete from user where username='%s'"%old_username)
    conn.commit()
    cur.execute("insert into user (username,password) value ('%s','%s')"%(new_username,new_password))
    conn.commit()
    conn.close()
    lock.release()
