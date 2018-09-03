import sqlite3           # SQLite3 탑재
from sect11_dbsql.s1102_R_select import select_one_book

db_name = 'my_books.db'

# 다른 디렉토리에 있는 파일을 import하려면 path를 설정
# :: sys.path에 path를 추가
# import sys
# sys.path.insert(0, '/Python/workspace/s1_prog_basics_v36/sect11_dbsql/s1102_R_select')

# http://brownbears.tistory.com/296
# from . import s1102_R-select.py.select_one_book

# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


def update_books(db_name):
    """
    데이터를 수정하는 함수
    Args:
        db_name : Database Name
    Returns : None
    """

    # 데이터베이스 커넥션 생성
    conn = sqlite3.connect(db_name)

    # 커서 확보
    cur = conn.cursor()

    # 데이터 수정 SQL ( 제목이 ? 인 책의 추천 유무를 ? 로 변경하라 )
    db_sql = "UPDATE my_books SET recommendation=? WHERE title=? "

    # 수정 SQL 실행
    cur.execute(db_sql, (1, '메가트랜드'))

    # 데이터베이스 반영
    conn.commit()

    # 커넥션 닫기
    conn.close()

if __name__ == "__main__":        # 외부에서 호출 시
    select_one_book(db_name)
    update_books(db_name)
    print('[데이터 수정 완료] ')
    print('=============================================')
    select_one_book(db_name)

