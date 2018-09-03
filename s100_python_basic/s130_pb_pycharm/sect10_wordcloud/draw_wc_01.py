from collections import Counter
from konlpy.tag import Twitter
import pytagcloud

fp = open('data/president_speech.txt')
speech = fp.read()
fp.close()

# print(speech)

nlp = Twitter()
nouns = nlp.nouns(speech)

count = Counter(nouns)
print(count)

# 위 데이터를 common_tag 변수에 담아 pytagcloud.make_tags() 를 이용하여 태그를 생성
common_tag = count.most_common(30)
tag_list = pytagcloud.make_tags(common_tag, maxsize=80)
print(tag_list)

# save_img = 'wordcloud/president_speech.jpg'
# pytagcloud.create_tag_image(tag_list, save_img, size=(900, 600), fontname='Nobile', rectangular=False)

# save_img = 'wordcloud/president_speech2.jpg'
# pytagcloud.create_tag_image(tag_list, save_img, size=(900, 600), fontname='Korean', rectangular=False)

save_img = 'wordcloud/president_speech3.jpg'
pytagcloud.create_tag_image(tag_list, save_img, size=(500, 500), fontname='Korean', rectangular=True)

