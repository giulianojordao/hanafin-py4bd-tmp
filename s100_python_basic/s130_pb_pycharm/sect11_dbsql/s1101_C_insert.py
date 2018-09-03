import sqlite3           # SQLite3 탑재

db_name = 'my_books.db'

# 데이터 등록
def insert_books(db_name):
    """
    데이터베이스 테이블에 데이터를 등록하는 함수
    Args:
        db_name : Database Name
    Returns : None
    """

    # 데이터베이스 커넥션 생성
    conn = sqlite3.connect(db_name)

    # 커서 확보
    cur = conn.cursor()

    # 데이터 입력 SQL1
    db_sql = "INSERT INTO my_books VALUES ('메가트랜드', '2002.03.02','A', 200, 0)"
    cur.execute(db_sql)

    # 데이터 입력 SQL2
    db_sql = 'INSERT INTO my_books VALUES (?, ?, ?, ?, ?)'
    cur.execute(db_sql, ('인더스트리 4.0', '2016.07.09', 'B', 584, 1))

    # # 데이터 입력 SQL3
    books = [
        ('유니콘 스타트업', '2011.07.15', 'A', 248, 1),
        ('빅데이터 마케팅', '2012.08.25', 'A', 296, 1),
        ('사물인터넷 전망', '2013.08.22', 'B', 526, 0)
    ]
    cur.executemany(db_sql, books)

    # 데이터베이스 반영
    conn.commit()

    # 커넥션 닫기
    conn.close()


if __name__ == "__main__":          # 외부에서 호출 시
    insert_books(db_name)           # 데이터 입력 함수 호출


