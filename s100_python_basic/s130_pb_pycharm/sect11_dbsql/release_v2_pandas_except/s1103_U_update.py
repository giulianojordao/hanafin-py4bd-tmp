import sqlite3           # SQLite3 탑재
from sect11_dbsql.s1102_R_select import select_one_book

db_name = './database/my_books.db'

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
    Returns :
        is_success : Boolean
    """
    is_success = True

    try:
        # 데이터베이스 커넥션 생성
        conn = sqlite3.connect(db_name)

        # 커서 확보
        cur = conn.cursor()

        # 데이터 수정 SQL ( 제목이 ? 인 책의 추천 유무를 ? 로 변경하라 )
        db_sql = "UPDATE my_books SET recommendation=? WHERE title=? "

        # 수정 SQL 실행
        cur.execute(db_sql, (1, '메가트랜드'))

    except:
        is_success = False
        print("Database Error!")

    finally:
        if is_success:
            # 데이터베이스 반영
            conn.commit()
        else:
            # 데이터베이스 철회
            conn.rollback()

        # 데이터베이스 커넥션 닫기
        conn.close()

    return is_success


if __name__ == "__main__":        # 외부에서 호출 시
    is_success, books_df1 = select_one_book(db_name)

    if update_books(db_name):
        print('데이터가 성공적으로 수정되었습니다.')
    else:
        print('데이터가 수정되지 않았습니다')

    is_success, books_df2 = select_one_book(db_name)

    books_df = pd.concat([books_df1, books_df2], axis=0)
    books_df['update'] = ['수정전', '수정후']
    books_df.set_index('update', inplace=True)
    print(books_df)
