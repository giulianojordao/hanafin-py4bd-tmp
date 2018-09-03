
# Industry 4.0 의 중심, BigData

<div align='right'><font size=2 color='gray'>Data Processing Based Python @ <font color='blue'><a href='https://www.facebook.com/jskim.kr'>FB / jskim.kr</a></font>, [김진수](bigpycraft@gmail.com)</font></div>
<hr>

### <font color='blue'>실습. Histogram 함수 작성해보기</font>
> 리스트를 받아서, 히스토그램을 그리는 함수
- 입력값 : ["cat", "cat", "cat", "sheep", "sheep", "duck", "duck", "duck", "duck" ]
# 1. 히스토그램을 그리는 함수 
  input: {'cat': 3, 'duck': 4, 'sheep': 2} => output: 그림 

# 2. 리스트를 받아서, 숫자를 세는 함수
  input: list => output: histogram dict

```python
from functools import *

data = ["cat", "cat", "cat", "sheep", "sheep", "duck", "duck", "duck", "duck" ]
```


```python
result_dict = {'cat': 3, 'duck': 4, 'sheep': 2}
result_hist = """
  dog   ====
  cat   ==
  bird  ===
"""

print(result_dict)
print('-'*40, result_hist)
```

    {'cat': 3, 'duck': 4, 'sheep': 2}
    ---------------------------------------- 
      dog   ====
      cat   ==
      bird  ===
    
    

### # 각자 reduce를 활용하여  아래와 같은 Histogram이 출력되도록 함수를 구현해보세요 !!!


```python
def get_count(result, element):
    
    if result.get(element):
        result[element] += 1
    else:
        result[element] = 1
    
    # check process
    print(element, '\t:', result)
    
    return result

```


```python
init_dict = dict()
print('result \t:', init_dict, '\n--------------------------------------------')

reduce(
    get_count,
    data,
    init_dict,
)
```

    result 	: {} 
    --------------------------------------------
    cat 	: {'cat': 1}
    cat 	: {'cat': 2}
    cat 	: {'cat': 3}
    sheep 	: {'cat': 3, 'sheep': 1}
    sheep 	: {'cat': 3, 'sheep': 2}
    duck 	: {'cat': 3, 'sheep': 2, 'duck': 1}
    duck 	: {'cat': 3, 'sheep': 2, 'duck': 2}
    duck 	: {'cat': 3, 'sheep': 2, 'duck': 3}
    duck 	: {'cat': 3, 'sheep': 2, 'duck': 4}
    




    {'cat': 3, 'duck': 4, 'sheep': 2}




```python
def get_histogram_dict(data):
    def get_count(result, element):
        if result.get(element):
            result[element] += 1
        else:
            result[element] = 1
        return result

    return reduce(
        get_count,
        data,
        {},
    )

def print_histogram(histogram_dict):    
    rows = []
    for key, value in histogram_dict.items():
        row = "{key}{spaces}{value_count}".format(
            key         =  key,
            spaces      =  " " * (7 - len(key)),
            value_count =  "=" * value,
        )
        rows.append(row)
        
    print("\n".join(rows))
    

def draw_histogram(data):
    histogram_dict = get_histogram_dict(data)
    print_histogram(histogram_dict)

```


```python
data = ["cat", "cat", "cat", "sheep", "sheep", "duck", "duck", "duck", "duck" ]
histogram_dict = get_histogram_dict(data)
histogram_dict
```




    {'cat': 3, 'duck': 4, 'sheep': 2}




```python
print_histogram(histogram_dict)
```

    cat    ===
    sheep  ==
    duck   ====
    

### # 투표 결과 Random Histogram !!!


```python
from random import *

candidate = ['빨강생 후보', '노랑색 후보', '파랑색 후보', '초록색 후보', '하얀색 후보']
num_of_voters = 100
```


```python
voting_result = list()
for _ in range(num_of_voters):
    c_num = randint(0, 4)
    voting_result.append(candidate[c_num])

draw_histogram(voting_result)
```

    노랑색 후보 ======================
    파랑색 후보 ==========================
    빨강생 후보 =================
    초록색 후보 ====================
    하얀색 후보 ===============
    

<hr>
<marquee><font size=3 color='brown'>The BigpyCraft find the information to design valuable society with Technology & Craft.</font></marquee>
<div align='right'><font size=2 color='gray'> &lt; The End &gt; </font></div>
