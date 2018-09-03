import sqlite3

db_name = './database/my_books.db'


def create_table(db_name, db_sql):
    """
    데이터베이스 테이블을 생성하는 함수
    Args:
        db_name : Database Name
        db_sql  : Query for creating Table
    Returns :
        is_success : Boolean
    """
    is_success = True

    try:
        # 데이터베이스 커넥션 생성
        conn = sqlite3.connect(db_name)

        # 커서 확보
        cur = conn.cursor()

        # 테이블 생성
        cur.execute(db_sql)

    # except OperationalError as e:
    #     is_success = False
    #     print('Error:', e)

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
        # print('Finish process of function.')
        conn.close()

    return is_success

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
    if create_table(db_name, db_sql):
        print('테이블이 성공적으로 생성되었습니다.')
    else:
        print('테이블이 생성되지 않았습니다')

