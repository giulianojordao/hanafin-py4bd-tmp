
# Industry 4.0 의 중심, BigData

<div align='right'><font size=2 color='gray'>Data Processing Based Python @ <font color='blue'><a href='https://www.facebook.com/jskim.kr'>FB / jskim.kr</a></font>, [김진수](bigpycraft@gmail.com)</font></div>
<hr>

## Python Review : Pythonic Coding Craft

> 
<font color='#CC0000'> 
믿기 어렵겠지만 25년이나 지났는데도 사람들은 아직도 파이썬에 열광한다.
<br> － 마이클 페일린(Michael Palin)
</font>

## The Zen of Python
> 파이썬에서 설계 원칙에 대한 일종의 교리
- 무엇을 하든 그것을 할 수 있는 하나의, 가급적이면 단 하나의 당연한 방법이 존재해야 한다.


```python
import this
```

    The Zen of Python, by Tim Peters
    
    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!
    

### 들여쓰기


```python
for num in [1, 2, 3, 4, 5]:
    print(num)
```

    1
    2
    3
    4
    5
    


```python
# 한줄에 결과값 출력하기
for num in range(10):
    print(num, end=' ')
```

    0 1 2 3 4 5 6 7 8 9 

### 모듈

#### 1. 만약 코드에서 re를 사용하고 있다면 별칭(alias)을 사용할 수 있다.


```python
import re as regex
my_regex = regex.compile('[0-9]+', regex.I)
my_regex
```




    re.compile(r'[0-9]+', re.IGNORECASE|re.UNICODE)



#### 2. 모듈 하나에서 특정 기능만 필요하다면, 전체 모듈을 불러오지 않고 해당 기능만 명시해서 불러온다.


```python
from collections import defaultdict, Counter
lookup = defaultdict(int)
my_counter = Counter()
```

#### 3. 가장 안좋은 습관중 하나는 모듈의 기능을 통째로 불러와서 기존의 변수들을 덮어쓰는것이다.


```python
match = 10
from re import *
print(match)
```

    <function match at 0x00000299AF84E378>
    


```python
# 연산
from __future__ import division
num = 5/2
num
```




    2.5



### 함수


```python
def sum(x, y):
    return x+y

sum(2, 3)
```




    5




```python
plus = lambda x, y : x+y

plus(2, 3)
```




    5




```python
minus = lambda x, y : x-y
minus(2, 3)
```




    -1




```python
def minus(x, y): return x-y
minus(2, 3)
```




    -1


cf. 똑같은 한줄인데, lambda 보다는 def 가 더 직관적이네~^^
### 예외 처리


```python
try:
    num = 5/0
except ZeroDivisionError:
    print("Can't divide zero !")
```

    Can't divide zero !
    


```python
try:
    num = 5/0
except:
    print("Can't divide zero !")
```

    Can't divide zero !
    

## 데이터구조문 : list, tuple, set, dict

###  list


```python
numbers = list(range(10))
numbers
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
numbers[:5]
```




    [0, 1, 2, 3, 4]




```python
numbers[5:]
```




    [5, 6, 7, 8, 9]




```python
numbers[-3:]
```




    [7, 8, 9]




```python
numbers[1:-1]     # 처음과 끝만 제외
```




    [1, 2, 3, 4, 5, 6, 7, 8]




```python
integers1 = numbers
integers1, id(integers1)==id(numbers)
```




    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], True)




```python
integers2 = numbers[:]
integers2, id(integers2)==id(numbers)
```




    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], False)




```python
fruits = ['apple', 'grape', 'melon']
```


```python
'apple' in fruits
```




    True




```python
'orange' not in fruits
```




    True




```python
fruits.append('orange')
fruits
```




    ['apple', 'grape', 'melon', 'orange']




```python
fruits.extend(['strawberry', 'pear'])
fruits
```




    ['apple', 'grape', 'melon', 'orange', 'strawberry', 'pear']




```python
fruits.append(['strawberry', 'pear'])
fruits
```




    ['apple',
     'grape',
     'melon',
     'orange',
     'strawberry',
     'pear',
     ['strawberry', 'pear']]




```python
# unpack 
x, y = [1, 2]
x, y
```




    (1, 2)




```python
# 첫번째는 신경쓰지 않을때
_, y = [1, 2]
y
```




    2




```python
_
```




    2



### tuple


```python
tp_num1 = (1, 2, 3)
tp_num1
```




    (1, 2, 3)




```python
tp_num2 =  4, 5, 6
tp_num2
```




    (4, 5, 6)




```python
ls_num1 = list(tp_num1)
ls_num1
```




    [1, 2, 3]




```python
ls_num1[1] = 7
ls_num1
```




    [1, 7, 3]




```python
# tp_num2[1] = 7
# tp_nump2

try:
    tp_num2[1] = 7
except TypeError:
    print("Can't modify a tuple!!")
```

    Can't modify a tuple!!
    


```python
# tuple 강점1. 함수에서 여러값을 반환할때 
def getSumMultiply(x, y):
    return (x+y), (x*y)
```


```python
hap_gop = getSumMultiply(2, 3)
hap_gop
```




    (5, 6)




```python
hap, gop = getSumMultiply(2, 3)
hap, gop
```




    (5, 6)




```python
# tuple 강점2. 변수를 교환할 때
x, y = 1, 2
x, y
```




    (1, 2)




```python
x, y = y, x     # It's really pythonic.
x, y
```




    (2, 1)



### dict


```python
document = 'forgiveness is better than permission'
```


```python
# 방법1. 단어를 key로, 빈도수를 value로 지정하는 dict 생성
word_counts = {}
for word in document:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1
        
# word_counts
```


```python
# 방법2. 예외처리를 하면서 dict를 생성하는 방법
word_counts = {}
for word in document:
    try:
        word_counts[word] += 1
    except KeyError:
        word_counts[word] = 1

# word_counts
```


```python
# 방법3. 존재하지 않는 key를 적절하게 처리해 주는 get을 사용해서 dict를 생성
word_counts = {}
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1

# word_counts
```

### defaultdict
> 
- 만약 존재하지 않는 key가 주어진다면 defaultdict는 이 key와 인자에서 주어진 값으로 dict에 새로운 항목을 추가해준다.
- defaultdict를 사용하기 위해서는 먼저 collections 모듈에서 defaultdict를 불러와야 한다.


```python
# 방법4. defaultdict 를 사용하는 방법
from collections import defaultdict

word_counts = defaultdict(int)    # int()는 0을 생성

for word in document:
    word_counts[word] += 1

word_counts
```




    defaultdict(int,
                {' ': 4,
                 'a': 1,
                 'b': 1,
                 'e': 5,
                 'f': 1,
                 'g': 1,
                 'h': 1,
                 'i': 4,
                 'm': 1,
                 'n': 3,
                 'o': 2,
                 'p': 1,
                 'r': 3,
                 's': 5,
                 't': 3,
                 'v': 1})




```python
type(word_counts)
```




    collections.defaultdict




```python
dd_list = defaultdict(list)    # list()는 빈 list 를 생성
dd_list
```




    defaultdict(list, {})




```python
dd_list[2].append(1)
dd_list
```




    defaultdict(list, {2: [1]})




```python
dd_dict = defaultdict(dict)    # dict()는 빈 dict 를 생성 
dd_dict
```




    defaultdict(dict, {})




```python
dd_dict['대한민국']['수도'] = '서울'
dd_dict
```




    defaultdict(dict, {'대한민국': {'수도': '서울'}})




```python
dd_dict['대한민국']['인구'] = 50000000
dd_dict
```




    defaultdict(dict, {'대한민국': {'수도': '서울', '인구': 50000000}})




```python
dd_pair = defaultdict(lambda: [0, 0])
dd_pair
```




    defaultdict(<function __main__.<lambda>>, {})




```python
dd_pair[0][0] = 1
dd_pair
```




    defaultdict(<function __main__.<lambda>>, {0: [1, 0]})




```python
dd_pair[1][0] = 2
dd_pair
```




    defaultdict(<function __main__.<lambda>>, {0: [1, 0], 1: [2, 0]})




```python
dd_pair[2][1] = 3
dd_pair
```




    defaultdict(<function __main__.<lambda>>, {0: [1, 0], 1: [2, 0], 2: [0, 3]})




```python
dd_pair[1][1] = 4
dd_pair
```




    defaultdict(<function __main__.<lambda>>, {0: [1, 0], 1: [2, 4], 2: [0, 3]})



#### Counter
> 
- 연속된 값을 defaultdict(int)와 유사한 객체로 변환
- key 와 value 의 빈도를 연결시켜 준다
- Histogram 을 그릴 때 사용
- PyFormat  using Using % and .format() : https://pyformat.info/


```python
from collections import Counter
cnt = Counter([1, 2, 3, 2, 2, 1])
cnt
```




    Counter({1: 2, 2: 3, 3: 1})




```python
word_counts = Counter(document)
word_counts
```




    Counter({' ': 4,
             'a': 1,
             'b': 1,
             'e': 5,
             'f': 1,
             'g': 1,
             'h': 1,
             'i': 4,
             'm': 1,
             'n': 3,
             'o': 2,
             'p': 1,
             'r': 3,
             's': 5,
             't': 3,
             'v': 1})




```python
# 가장 자주 나오는 단어 10개와 이 단어들의 빈도수를 출력
for word, cnt in word_counts.most_common(10):
    print('%s : %d' %(word, cnt))
```

    e : 5
    s : 5
    i : 4
      : 4
    r : 3
    n : 3
    t : 3
    o : 2
    f : 1
    g : 1
    


```python
for word, cnt in word_counts.most_common(10):
    print('{} = {}'.format(word, cnt))
```

    e = 5
    s = 5
    i = 4
      = 4
    r = 3
    n = 3
    t = 3
    o = 2
    f = 1
    g = 1
    

#### PyFormat  using Using % and .format() 
> ref. https://pyformat.info/

### set 의 사용이유
> 
- 이유1. 특정항목의 존재여부 확인 : in은 set에서 굉장히 빠르게 작동 
- 이유2. 중복된 원소를 제거


```python
pocket = set()
pocket.add(1)
pocket.add(2)
pocket.add(1)
pocket
```




    {1, 2}




```python
cnt = len(pocket)
cnt
```




    2




```python
exist_0 = 0 in pocket
exist_0
```




    False




```python
exist_1 = 1 in pocket
exist_1
```




    True




```python
import time
import os

def chk_processting_time(start_time, end_time):
    process_time = end_time - start_time
    p_time = int(process_time)
    p_min = p_time // 60
    p_sec = p_time %  60
    print('경과시간 : {p_min}분 {p_sec}초 경과되었습니다.'.format(
            p_min = p_min, 
            p_sec = p_sec
        ))
    return process_time

def chk_processting_micro_sec(start_time, end_time):
    process_time = end_time - start_time
    print('경과시간 : {}초 경과되었습니다.'.format(process_time))
    return process_time
```


```python
scope = pow(10, 7)
scope
```




    10000000




```python
otherwords_list = list()
for num in range(scope):
    temp = '단어' + str(num)
    otherwords_list.append(temp)

print(otherwords_list[:5], '• • •', otherwords_list[-5:], end=' ')
```

    ['단어0', '단어1', '단어2', '단어3', '단어4'] • • • ['단어9999995', '단어9999996', '단어9999997', '단어9999998', '단어9999999'] 


```python
stopwords_list = ['가', '는', '에게'] + otherwords_list + ['나', '너', '그']
print(stopwords_list[:5], '• • •', stopwords_list[-5:], end=' ')
```

    ['가', '는', '에게', '단어0', '단어1'] • • • ['단어9999998', '단어9999999', '나', '너', '그'] 


```python
time1 = time.time()
check = '당신' in stopwords_list
time2 = time.time()

chk_processting_micro_sec(time1, time2)
check
```

    경과시간 : 0.3495817184448242초 경과되었습니다.
    




    False




```python
stopwords_set = set(stopwords_list)
```


```python
time1 = time.time()
check = '당신' in stopwords_set
time2 = time.time()

chk_processting_micro_sec(time1, time2)
check
```

    경과시간 : 0.0초 경과되었습니다.
    




    False




```python
# 중복원소 제거하기
item_list = [1, 2, 3, 1, 2, 3]
item_list
```




    [1, 2, 3, 1, 2, 3]




```python
num_items = len(item_list)
num_items
```




    6




```python
item_set = set(item_list)
item_set
```




    {1, 2, 3}




```python
num_distinct_items = len(item_set)
num_distinct_items
```




    3




```python
distinct_item_list = list(item_set)
distinct_item_list
```




    [1, 2, 3]




```python
def getDistinctList(item_list):
    item_set = set(item_list)
    return list(item_set)
```


```python
item_list = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
getDistinctList(item_list)
```




    [0, 1, 2, 3, 4, 5]



### 흐름 제어


```python
score = 90
if score >= 90:
    grade = 'A'
elif score >= 80:
    grade = 'B'
else:
    grade = 'F'

print('점수는 {}, 평점은 {}'.format(score, grade))
```

    점수는 90, 평점은 A
    


```python
score = 85
grade = 'A' if score >= 90 else 'B' if score >= 80 else 'C'

print('점수는 {}, 평점은 {}'.format(score, grade))
```

    점수는 85, 평점은 B
    

### 불리언(boolean) : True 와 False


```python
True == False
```




    False




```python
var = None
print(var == None)    # Not Pythonic
print(var is None)    # Now Pythonic
```

    True
    True
    


```python
all_false = [ False, None, [], {}, set(), 0, 0.0, '', ' ']
all_false
```




    [False, None, [], {}, set(), 0, 0.0, '', ' ']




```python
for is_chk in all_false:
    print(bool(is_chk), end=' | ')
```

    False | False | False | False | False | False | False | False | True | 


```python
def get_string():
    result = 'test'
    return result

def get_number():
    result = 12345
    return result
```


```python
ret_str = get_string()
ret_str
```




    'test'




```python
if ret_str:
    first_char = ret_str[0]
else:
    first_char = ''
```


```python
first_char = ret_str and ret_str[0]
first_char
```




    't'




```python
ret_num = get_number()
ret_num
```




    12345




```python
x = get_number()
safe_x = x or 0
safe_x
```




    12345




```python
x = get_string()
safe_x = x or 0
safe_x
```




    'test'



### 정렬


```python
x = [1, 3, 5, 2, 4]
x
```




    [1, 3, 5, 2, 4]




```python
y = sorted(x)
y
```




    [1, 2, 3, 4, 5]




```python
x.sort()
x
```




    [1, 2, 3, 4, 5]




```python
# 절대값의 내림차순으로 list를 정렬
num = [-5, 2, -3, 4, -7]
x = sorted(num, key=abs, reverse=True)
x
```




    [-7, -5, 4, -3, 2]




```python
document = '대한사람 대한으로 길이 보존하세'
document = '생각이란 생각하면 생각할수록 생각나는것이 생각인다'
```


```python
# 방법3. 존재하지 않는 key를 적절하게 처리해 주는 get을 사용해서 dict를 생성
word_counts = {}
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1
```


```python
word_counts
```




    {' ': 4,
     '각': 5,
     '것': 1,
     '나': 1,
     '는': 1,
     '다': 1,
     '란': 1,
     '록': 1,
     '면': 1,
     '생': 5,
     '수': 1,
     '이': 2,
     '인': 1,
     '하': 1,
     '할': 1}




```python
word_counts.items()
```




    dict_items([('생', 5), ('각', 5), ('이', 2), ('란', 1), (' ', 4), ('하', 1), ('면', 1), ('할', 1), ('수', 1), ('록', 1), ('나', 1), ('는', 1), ('것', 1), ('인', 1), ('다', 1)])




```python
word_counts.keys()
```




    dict_keys(['생', '각', '이', '란', ' ', '하', '면', '할', '수', '록', '나', '는', '것', '인', '다'])




```python
word_counts.values()
```




    dict_values([5, 5, 2, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])




```python
word, count = word_counts.keys(), word_counts.values()
```


```python
cnt = lambda word, count : count
cnt
```




    <function __main__.<lambda>>




```python
import operator
```


```python
word = word_counts.items()
word
```




    dict_items([('생', 5), ('각', 5), ('이', 2), ('란', 1), (' ', 4), ('하', 1), ('면', 1), ('할', 1), ('수', 1), ('록', 1), ('나', 1), ('는', 1), ('것', 1), ('인', 1), ('다', 1)])




```python
# value를 기준으로 정렬
v_sorted_word = sorted(word, key=operator.itemgetter(1), reverse=True)
v_sorted_word
```




    [('생', 5),
     ('각', 5),
     (' ', 4),
     ('이', 2),
     ('란', 1),
     ('하', 1),
     ('면', 1),
     ('할', 1),
     ('수', 1),
     ('록', 1),
     ('나', 1),
     ('는', 1),
     ('것', 1),
     ('인', 1),
     ('다', 1)]




```python
# key를 기준으로 정렬
k_sorted_word = sorted(word, key=operator.itemgetter(0))
k_sorted_word
```




    [(' ', 4),
     ('각', 5),
     ('것', 1),
     ('나', 1),
     ('는', 1),
     ('다', 1),
     ('란', 1),
     ('록', 1),
     ('면', 1),
     ('생', 5),
     ('수', 1),
     ('이', 2),
     ('인', 1),
     ('하', 1),
     ('할', 1)]



### List Comprehension


```python
even_numbers = [num for num in range(5) if num%2 == 0]
even_numbers
```




    [0, 2, 4]




```python
zeroes = [0 for _ in even_numbers]
zeroes
```




    [0, 0, 0]




```python
pairs = [(x, y)
         for x in range(1, 3)
         for y in range(1, 10)
        ]
pairs
```




    [(1, 1),
     (1, 2),
     (1, 3),
     (1, 4),
     (1, 5),
     (1, 6),
     (1, 7),
     (1, 8),
     (1, 9),
     (2, 1),
     (2, 2),
     (2, 3),
     (2, 4),
     (2, 5),
     (2, 6),
     (2, 7),
     (2, 8),
     (2, 9)]




```python
pairs = [(x, y)
         for x in range(1, 3)
         for y in range(x+1, 10)
        ]
pairs
```




    [(1, 2),
     (1, 3),
     (1, 4),
     (1, 5),
     (1, 6),
     (1, 7),
     (1, 8),
     (1, 9),
     (2, 3),
     (2, 4),
     (2, 5),
     (2, 6),
     (2, 7),
     (2, 8),
     (2, 9)]



### Generator 와 iterator
> 
- Generator(생성자)는 반복할 수 있다. (주로 for문을 통해서) 
- generator의 각 항목은 필요한 순간에 그때그때 생성된다.
- Generator의 단점은 generator를 단 한번만 반복할 수 있다는 점이다.
- 만약 데이터를 여러번 반복해야 한다면 매번 generator를 다시 만들거나 list를 사용해야 한다.


```python
def lazy_range(n):
    '''range와 똑같은 기능을 하는 generator'''
    i = 0
    while i < n:
        yield i
        i += 1
    
```


```python
for i in lazy_range(5):
    print(i)
```

    0
    1
    2
    3
    4
    


```python
def natural_numbers():
    '''1, 2, 3, ...을 반환'''
    n = 1
    while True:
        yield n
        n += 1
```


```python
natural_numbers()
```




    <generator object natural_numbers at 0x00000299F23D4E08>




```python
lazy_evens_below_20 = (i for i in lazy_range(20) if i%2==0)
lazy_evens_below_20
```




    <generator object <genexpr> at 0x00000299F23D4570>




```python
natural_numbers
```




    <function __main__.natural_numbers>



### 난수 생성


```python
import random

five_uniform_randoms = [random.random() for _ in range(5)]
five_uniform_randoms
```




    [0.6478277233948211,
     0.046238345934104474,
     0.5539884783451832,
     0.9533612437313497,
     0.7689547736885814]




```python
random.seed(10)
random.random()
```




    0.5714025946899135




```python
random.random()
```




    0.4288890546751146




```python
random.seed(10)
random.random()
```




    0.5714025946899135




```python
random.random()
```




    0.4288890546751146




```python
random.random()
```




    0.5780913011344704




```python
random.randrange(10)
```




    3




```python
random.randrange(3, 6)
```




    4




```python
random.randint(1, 3)
```




    2




```python
# 실행할때마다 다르다.
up_to_ten = list(range(10))
random.shuffle(up_to_ten)
up_to_ten
```




    [6, 1, 5, 7, 8, 3, 9, 0, 2, 4]




```python
star_list = ['이영애', '신민아', '전지현', '김태희', '강소라']
my_best_star = random.choice(star_list)
my_best_star
```




    '이영애'




```python
lottery_numbers = list(range(46))
winning_numbers = random.sample(lottery_numbers, 6)
winning_numbers
```




    [26, 8, 38, 22, 24, 45]



### 정규표현식(Regular Expression)


```python
import re
```


```python
print([                                             # 전부 True
    not not re.match('a', 'apple'),                 # a로 시작느냐
    bool(re.match('a', 'apple')),
    not re.match('b', 'apple'),
    bool(re.search('p', 'apple')),                  # p가 존재하느냐
    not re.search('b', 'apple'),
    7 ==  len(re.split('[ab]', 'Abracadabra')),     # a 혹은 b 기준으로 나눈다
    'A-B-' == re.sub('[0-9]', '-', 'A1B4')          # 숫자를 '-' 로 대체
])
```

    [True, True, True, True, True, True, True]
    


```python
print(re.match('a', 'apple'))      # a로 시작하느냐
```

    <_sre.SRE_Match object; span=(0, 1), match='a'>
    


```python
print(re.match('p', 'apple'))
```

    None
    


```python
print(re.search('p', 'apple'))     # p가 존재하느냐
```

    <_sre.SRE_Match object; span=(1, 2), match='p'>
    


```python
print(re.search('b', 'apple'))
```

    None
    


```python
print(re.split('[ab]', 'Abracadabra'))    # a 혹은 b 기준으로 나눈다
```

    ['A', 'r', 'c', 'd', '', 'r', '']
    


```python
print(re.sub('[0-9]', '-', 'A1B4'))       # 숫자를 '-' 로 대체
```

    A-B-
    

### OOP(Object-Oriented Programming)


```python
# 클래스 이름은 카멜표기법으로 표기
class Set:
    
    # 멤버 함수들을 정의
    # 모든 멤버 함수의 첫번째 인자는 'self' (관습중 하나)
    # 'self'는 현재 사용되는 클래스, Set 객체를 의미
    
    def __init__(self, values=None):
        ''' 클래스의 생성자(constructor)이다.
            새로운 Set을 만들면 가장 먼저 호출된다.
        '''
        self.dict = {}     # 모든 Set의 인스턴스는 자체적으로 dict를 유지
        
        if values is not None:
            for value in values:
                self.add(value)
                
                
    def __repr(self):
        ''' 이 함수를 입력하거나 str()으로 보내주면 
            Set 객체를 문자열로 표현
        '''
        return 'Set: ' + str(self.dict.keys())
    
    # self.dict에서 항목과 True를 각각 key와 value로 사용해서
    # Set 안에 존재하는 항목을 표현
    def add(self, value):
        self.dict[value] = True
        
    # 만약 항목이 dict의 key라면 항목은 Set 안에 존재함
    def contains(self, value):
        return value in self.dict
    
    def remove(self, value):
        del self.dict[value]
        
```


```python
s1 = Set()
s1
```




    <__main__.Set at 0x299f23dc630>




```python
s2 = Set([1, 2, 2, 3])
s2
```




    <__main__.Set at 0x299f23dcd30>




```python
s2.add(4)
```


```python
s2.contains(4)
```




    True




```python
s2.remove(2)
```


```python
s2.contains(2)
```




    False



### enumerate
> 
- list를 반복하면서 list의 항목과 인덱스가 모두 필요한 경우가 가끔 있다.
- 가장 파이썬스러운 방법은 (인덱스, 항목) 형태의 tuple을 생성해 주는 enumerate를 활용하는 것이다.


```python
documents = "동해물과 백두산이"
```


```python
# not pythonic
for i in range(len(documents)):
    document = documents[i]
    print(i, document)
```

    0 동
    1 해
    2 물
    3 과
    4  
    5 백
    6 두
    7 산
    8 이
    


```python
# not pythonic too
i = 0
for document in documents:
    print(i, document)
    i += 1
```

    0 동
    1 해
    2 물
    3 과
    4  
    5 백
    6 두
    7 산
    8 이
    


```python
# really pythonic 
for i, document in enumerate(documents):
    print(i, document)
```

    0 동
    1 해
    2 물
    3 과
    4  
    5 백
    6 두
    7 산
    8 이
    

### zip 과 argument unpacking
> 
- 2개 이상의 list를 서로 묶어 주고 싶을때가 있다.
- zip은 여러개의 list를 서로 상응하는 항목의 tuple로 구성된 list를 변환


```python
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]

zipped = zip(list1, list2)
zipped
```




    <zip at 0x299f23ee7c8>




```python
list(zipped)
```




    [('a', 1), ('b', 2), ('c', 3)]




```python
pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)
```


```python
letters
```




    ('a', 'b', 'c')




```python
numbers
```




    (1, 2, 3)




```python
list3 = list('가나다')
list3
```




    ['가', '나', '다']




```python
zipped = zip(list2, list3)
list(zipped)
```




    [(1, '가'), (2, '나'), (3, '다')]




```python
z_key, z_val = zip(*zip(list2, list3))
z_key==list2 and z_val==list3
```




    False



### args 와 kwargs


```python
def doubled(param):
    def resfunc(num):
        return 2*param(num)
    return resfunc

def plusone(num):
    return 1+num

my_func = doubled(plusone)
```


```python
my_func(4)
```




    10




```python
my_func(-1)
```




    0




```python
def sum(x, y):
    return x+y

my_func2 = doubled(sum)
```


```python
try:
    my_func2(3, 4)
except TypeError:
    print('TypeError: resfunc() takes 1 positional argument but 2 were given')
```

    TypeError: resfunc() takes 1 positional argument but 2 were given
    


```python
def magic(*args, **kwargs):
    print('unnamed args:', args)
    print('keyword args:', kwargs)
```


```python
magic(1, 2)
```

    unnamed args: (1, 2)
    keyword args: {}
    


```python
magic(key1=1, key2=2)
```

    unnamed args: ()
    keyword args: {'key1': 1, 'key2': 2}
    


```python
magic(1, 2, key1=1, key2=2)
```

    unnamed args: (1, 2)
    keyword args: {'key1': 1, 'key2': 2}
    


```python
def doubled2(param):
    def resfunc(*args, **kwargs):
        return 2*param(*args, **kwargs)
    return resfunc

def sum(x, y):
    return x+y

my_func2 = doubled2(sum)
```


```python
my_func2(3, 4)
```




    14



<hr>
<marquee><font size=3 color='brown'>The BigpyCraft find the information to design valuable society with Technology & Craft.</font></marquee>
<div align='right'><font size=2 color='gray'> &lt; The End &gt; </font></div>
