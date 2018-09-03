import re

# 주민등록번호 : 숫자 6자리 - 숫자 7자리
# 데이터에서 주민등록번호만 찾아서, 뒷자리를 암호화(*)

text = """
    김찬영 020822-3069110
    김준영 040825-3069115
    김채영 110715-4063111
"""

print('-'*70, '\n# 주민등록번호 찾기')
pattern = re.compile("\d{6}-?\d{7}")
jumin_no = pattern.findall(text)
print(jumin_no)

print('-'*70, '\n# 주민등록번호 찾기2')
# 정규표현식 GROUP
# 1. 생년월일 그룹 <birth>
# 2. 주민등록번호 뒷자리 그룹 <secret>
# pattern = re.compile("\d{6}-?\d{7}")
pattern = re.compile("\d{6}-\d{7}")
pattern = re.compile("(?P<birth>\d{6})-(?P<secret>\d{7})")

# 김채영 110715-******* => 김채영(110715-*******)
jumin_no = pattern.findall(text)
print(jumin_no)


print('-'*70, '\n# 주민등록번호 찾기3', end=' ')
result = pattern.sub("\g<birth>-*******", text)
print(result)


print('-'*70, '\n# 주민등록번호 찾기4')
pattern = re.compile("(?P<name>\w{3}) (?P<birth>\d{6})-(?P<secret>\d{7})")
result = pattern.findall(text)
print(result)
# print(text)

print('-'*70, '\n# 주민등록번호 찾기5', end='')
pattern = re.compile("(?P<name>\w{3}) (?P<birth>\d{6})-(?P<secret>\d{7})")
result = pattern.sub("\g<name>(\g<birth>-*******)", text)
print(result)

print('-'*70, '\n# 주민등록번호 찾기6')
result = result.split('\n')
print(result)

result.pop(0)
print(result)

result.pop(-1)
print(result)


print('-'*70, '\n# 주민등록번호 찾기7 - 최종출력 !!')
for idx, val in enumerate(result):
    val = val.replace(" ", "")
    # val[2] = "O"
    # val.pop(2)
    # val.insert(2, "O")
    print(idx, val)

