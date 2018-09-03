import re

# 010-5670-3847
# 010-오륙칠공-3847
# 공일빵 5670 3팔4칠
# cf. 0.5% 정도는 저런사람 있쥐요~
text = """
    010-5670-3847   # space, -, . => []
    010 5670 3847
    010.5670 3847
"""
print('-'*70, '\n# 핸드폰 번호 찾기')
pattern = re.compile("\d{3}[ -.]?\d{4}[ -.]?\d{4}")
phone_no = pattern.findall(text)
print(phone_no)
