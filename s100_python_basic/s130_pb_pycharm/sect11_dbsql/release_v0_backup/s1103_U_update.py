import sqlite3                                   # sqlite3 모듈 탑재
# from s1102_R_select import select_one_book     # 데이터 조회용 함수 탑재
from s1_prog_basics_v36.sect11_dbsql.s1102_R_select import select_one_book

# 다른 디렉토리에 있는 파일을 import하려면 path를 설정
# :: sys.path에 path를 추가
# import sys
# sys.path.insert(0, '/Python/workspace/s1_prog_basics_v36/sect11_dbsql/s1102_R_select')

# http://brownbears.tistory.com/296
# from . import s1102_R-select.py.select_one_book

# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))



# 데이터 수정용 함수
def update_books():
    conn = sqlite3.connect('my_books.db')       # 데이터베이스 커넥션 생성
    cur = conn.cursor()  # 커서 확보

    # 데이터 수정 SQL ( 제목이 ? 인 책의 추천 유무를 ? 로 변경하라 )
    update_sql = 'UPDATE my_books SET recommendation=? WHERE title=?'

    # 수정 SQL 실행
    cur.execute(update_sql, (1, '메가트랜드'))

    conn.commit()                                   # 데이터베이스 반영
    conn.close()                                    # 커넥션 닫기


if __name__ == "__main__":		                # 외부에서 호출 시
    select_one_book()
    update_books()                                  # 데이터 수정 함수 호출
    print('[데이터 수정 완료] ================== ')
    select_one_book()
