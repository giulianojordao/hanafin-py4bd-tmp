from collections import Counter
from konlpy.tag import Twitter
import pytagcloud

fp = open('data/president_speech.txt')
speech = fp.read()
fp.close()

nlp = Twitter()
nouns = nlp.nouns(speech)

count = Counter(nouns)
common_tag = count.most_common(30)
tag_list = pytagcloud.make_tags(common_tag, maxsize=80)

save_img = 'wordcloud/president_moon.jpg'
pytagcloud.create_tag_image(tag_list, save_img, size=(900, 600), fontname='Korean', rectangular=False)

