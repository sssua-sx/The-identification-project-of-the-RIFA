# encoding=gbk
#���ݿ���������
import pymysql

def creat_db():
        conn = pymysql.connect(
                host='127.0.0.1',
                port=3306,
                user='root',
                password='690823sgj'
                )
        cur = conn.cursor()
        try:
                cur.execute("use ExistedUsers")
        except:
                cur.execute("create database ExistedUsers")
                cur.execute("use ExistedUsers")
        try:
                cur.execute("create table user(username varchar(20),password varchar(20))")
                print('�����ɹ�')
        except:
                pass
        return conn,cur