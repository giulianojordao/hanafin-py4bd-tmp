import sqlite3           # SQLite3 탑재
from sect11_dbsql.s1102_R_select import select_all_books

db_name = './database/my_books.db'


# 데이터 삭제용 함수
def delete_books_by_title(db_name, title):
    """
    책제목에 해당하는 데이터를 삭제하는 함수
    Args:
        db_name : Database Name
        title   : Title of the book to be removed
    Returns :
        is_success : Boolean
    """
    is_success = True

    try:
        # 데이터베이스 커넥션 생성
        conn = sqlite3.connect(db_name)

        # 커서 확보
        cur = conn.cursor()

        # 데이터 삭제 SQL
        db_sql = "DELETE FROM my_books "
        db_sql += "WHERE title = ?      "

        # 수정 SQL 실행
        # print('db_sql:', db_sql)
        # print('title:', title)
        cur.execute(db_sql, (title,))
        # count = cur.execute(db_sql, (title,))
        # print('count:', type(count), count)

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
    title = '메가트랜드'
    if delete_books_by_title(db_name, title):
        print('데이터가 성공적으로 삭제되었습니다.')
    else:
        print('데이터가 삭제되지 않았습니다')

    is_success, books_df = select_all_books(db_name)
    print(books_df)


def delete_books(db_name, col_name, col_val):
    """
    조건에 맞는 데이터를 삭제하는 함수
    Args:
        db_name  : Database Name
        col_name : Column Name
        col_val  : Column Value
    Returns :
        is_success : Boolean
    """
    is_success = True

    try:
        # 데이터베이스 커넥션 생성
        conn = sqlite3.connect(db_name)

        # 커서 확보
        cur = conn.cursor()

        # 데이터 삭제 SQL
        # db_sql = "DELETE FROM my_books "
        # db_sql+= "WHERE {} = '{}' "
        # db_sql = db_sql.format(col_name, col_val)
        # cur.execute(db_sql)

        # # 데이터 삭제 SQL
        db_sql = 'DELETE FROM my_books '
        db_sql += 'WHERE {} = ? '
        db_sql = db_sql.format(col_name)

        # 수정 SQL 실행
        cur.execute(db_sql, (col_val,))

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
    is_success, books_df = select_all_books(db_name)
    print(books_df)

    col_name = 'publisher'
    col_val  = 'A'
    if delete_books(db_name, col_name, col_val):
        print('데이터가 성공적으로 삭제되었습니다.')
    else :
        print('데이터가 삭제되지 않았습니다')

    is_success, books_df = select_all_books(db_name)
    print(books_df)

    col_name = 'title'
    col_val = '사물인터넷 전망'
    if delete_books(db_name, col_name, col_val):
        print('데이터가 성공적으로 삭제되었습니다.')
    else:
        print('데이터가 삭제되지 않았습니다')

    is_success, books_df = select_all_books(db_name)
    print(books_df)
