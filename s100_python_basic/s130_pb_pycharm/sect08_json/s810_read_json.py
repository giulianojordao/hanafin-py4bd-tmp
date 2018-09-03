import json
from pprint import pprint

json_data = {
    'firstname' : '길동',
    'lastname'  : '홍',
    'age'       : 20,
    'country'   : '율도국'
}

json_code = json.JSONEncoder().encode(json_data)
print(json_code)

check = json.dumps('한글')
print(check)

check = json.dumps('한글', ensure_ascii=False)
print(check)

check = json.dumps(json_data, ensure_ascii=False)
print(check)

json_code = json.JSONDecoder().decode(check)
print(json_code)


result = json_code['country']
print(result)

result = "{}{}은 {}살 이고, {}에 살고 있습니다.".format(
    json_code['lastname'],
    json_code['firstname'],
    json_code['age'],
    json_code['country'],
)
print(result)
