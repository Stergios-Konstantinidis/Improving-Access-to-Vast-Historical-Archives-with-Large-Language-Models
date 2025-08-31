import mysql.connector as sql


def sqlConnection():
    connection = sql.connect(
        host="YOUR_HOST",
        database="YOUR_DATABASE",
        user="YOUR_USER",
        password="YOUR_PASSWORD",
        port=3306,
        autocommit=True
    )
    return connection