import sqlite3           # SQLite3 탑재

db_name = 'my_books.db'


def select_all_books(db_name):
    """
    전체 데이터를 조회하는 함수
    Args:
        db_name : Database Name
    Returns : None
    """

    # 데이터베이스 커넥션 생성
    conn = sqlite3.connect(db_name)

    # 커서 확보
    cur = conn.cursor()

    # 조회용 SQL 실행
    db_sql = "SELECT * FROM my_books"
    cur.execute(db_sql)

    # 조회한 데이터 불러오기
    print('[1] 전체 데이터 출력하기')
    books = cur.fetchall()

    # 데이터 출력하기
    for book in books:
        print(book)

    # 커넥션 닫기
    conn.close()


if __name__ == "__main__":
    select_all_books(db_name)
    print('=============================================')


# 일부 조회용 함수
def select_some_books(db_name, number):
    """
    일부 데이터를 조회하는 함수
    Args:
        db_name : Database Name
        number  : Count of data to query
    Returns : None
    """

    # 데이터베이스 커넥션 생성
    conn = sqlite3.connect(db_name)

    # 커서 확보
    cur = conn.cursor()

    # 조회용 SQL 실행
    db_sql = "SELECT * FROM my_books"
    cur.execute(db_sql)

    # 조회한 데이터 일부 불러오기
    print('[2] 데이터 일부 출력하기')
    books = cur.fetchmany(number)

    # 데이터 출력하기
    for book in books:
        print(book)

    # 커넥션 닫기
    conn.close()


if __name__ == "__main__":         # 외부에서 호출 시
    select_some_books(db_name, number=3)
    print('=============================================')


# 1개 조회용 함수
def select_one_book(db_name):
    """
    최상단 하나의 데이터를 조회하는 함수
    Args:
        db_name : Database Name
    Returns : None
    """

    # 데이터베이스 커넥션 생성
    conn = sqlite3.connect(db_name)

    # 커서 확보
    cur = conn.cursor()

    # 조회용 SQL 실행
    db_sql = "SELECT * FROM my_books"
    cur.execute(db_sql)

    # 데이터 한개 출력하기
    print('[3] 1개 데이터 출력하기')
    print(cur.fetchone())

    # 커넥션 닫기
    conn.close()


if __name__ == "__main__":        # 외부에서 호출 시
    select_one_book(db_name)
print('=============================================')


# 쪽수 많은 책 조회용 함수
def find_big_books(db_name):
    """
    조건에 맞는 데이터를 조회하는 함수
    조건 : 페이지수가 300쪽보다 큰 데이터
    Args:
        db_name : Database Name
    Returns : None
    """

    # 데이터베이스 커넥션 생성
    conn = sqlite3.connect(db_name)

    # 커서 확보
    cur = conn.cursor()

    # 조회용 SQL 실행
    db_sql = "SELECT title, pages FROM my_books "
    db_sql += "WHERE pages > 300"
    cur.execute(db_sql)

    # 조회한 데이터 불러오기
    print('[4] 페이지 많은 책 출력하기')
    books = cur.fetchall()

    # 데이터 출력하기
    for book in books:
        print(book)

    # 커넥션 닫기
    conn.close()

if __name__ == "__main__":          # 외부에서 호출 시
    find_big_books(db_name)
    print('=============================================')