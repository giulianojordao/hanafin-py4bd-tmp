import sqlite3           # SQLite3 탑재

db_name = 'my_books.db'

# 테이블 생성
def create_table(db_name, db_sql):
    """
    데이터베이스 테이블을 생성하는 함수
    Args:
        db_name : Database Name
        db_sql  : Query for creating Table
    Returns : None
    """
    # 데이터베이스 커넥션 생성
    conn = sqlite3.connect(db_name)

    # 커서 확보
    cur = conn.cursor()

    # 테이블 생성
    cur.execute(db_sql)

    # 데이터베이스 반영
    conn.commit()

    # 데이터베이스 커넥션 닫기
    conn.close()


if __name__ == "__main__":  # 외부에서 호출 시

    db_sql = '''
    CREATE TABLE my_books (
        title text,
        published_date text,
        publisher text,
        pages integer,
        recommendation integer
    )
    '''

    create_table(db_name, db_sql)

