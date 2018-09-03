import re

# 010-5670-3847
# 010-오륙칠공-3847
# 공일빵 5670 3팔4칠
# cf. 0.5% 정도는 저런사람 있쥐요~

text = """
    010-5670-3847   # space, -, . => []
    옛날에는 011-1052-3847 이랬는데..
    010 5670 3847
    010.-5670 3847
    사는동네가 자이아파트 516동512호
    그리구, 사무실번호는 02-360-4047이고
    우편번호는 100-791, 청파로 463번지
"""

print('-'*70, '\n# 핸드폰 번호 찾기')
pattern = re.compile("\d{3}[ -.]?\d{4}[ -.]?\d{4}")
phone_no = pattern.findall(text)
print(phone_no)

print('-'*70, '\n# 핸드폰 번호 찾기2')
# pattern = re.compile("\d{3}[ -.]?\d{4}[ -.]?\d{4}")
pattern = re.compile("\d{3}[ -\.]{1,2}\d{3,4}[ -\.]?\d{4}")
phone_no = pattern.findall(text)
print(phone_no)

print('-'*70, '\n# 핸드폰 번호 찾기3')
# pattern = re.compile("\d{3}[ -.]?\d{4}[ -.]?\d{4}")
# pattern = re.compile("\d{3}[ -\.]{1,2}\d{3,4}[ -\.]?\d{4}")
pattern = re.compile("\d{2,3}[ -\.]?\d{3,4}[ -\.]?\d{4}")
phone_no = pattern.findall(text)
print(phone_no)
