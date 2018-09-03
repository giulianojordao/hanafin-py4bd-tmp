
# Industry 4.0 의 중심, BigData

<div align='right'><font size=2 color='gray'>Data Processing Based Python @ <font color='blue'><a href='https://www.facebook.com/jskim.kr'>FB / jskim.kr</a></font>, [김진수](bigpycraft@gmail.com)</font></div>
<hr>

## Python Review : 제어문, 함수


```python
a = 30
if a > 0:
    print("양수다.")
else:
    # 0 또는 음수
    if a == 0:
        print("0이다.")
    else:
        print("음수다.")
            
```

    양수다.
    


```python
a = 30
if a > 0:
    print("양수다.")
elif a == 0:
    print("0이다.")
else:
    print("음수다.")

```

    양수다.
    


```python
# x 라는 argument 를 받아서,
# x * 2 라는 결과를 주는 함수
```


```python
def double(x):    # 함수를 정의할 때 "인자" => Parameter
    return x * 2
```


```python
double(100)      # 함수를 호출할 때, "인자" => Arguments
```




    200




```python
def add(a, b):
    return a + b
```


```python
add(10, 20)
```




    30




```python
# add ( 더한다 )

# add2(10, 20) => 30
# add2(10, 20, 30) => 60

# add3([10, 20, 30]) => 60
```


```python
def add(a, b, c):
    return a + b + c
```


```python
# 일반적으로 두개를 다 되게 하려면 둘다 정의
def add(a, b):
    return a + b

def add(a, b, c):
    return a + b + c
```


```python
# 기본 자료형
def add3(numbers):   # 숫자 리스트를 받아서, 합을 리턴하는 함수
    pass
```


```python
# 기본 자료형
def add3(numbers):   # 숫자 리스트를 받아서, 합을 리턴하는 함수
    sum = 0
    for num in numbers:
        sum += num
        
    return sum
```


```python
add3([10, 20, 30, 40])
```




    100




```python
def add(a, b):
    return a + b

add(10, 20)
```




    30




```python
data = (10, 15)  #Tuple
# data 를 add 에 넣어서, 25라는 결과를 가지고 싶다.
```


```python
add(data)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-16-111f2035281f> in <module>()
    ----> 1 add(data)
    

    TypeError: add() missing 1 required positional argument: 'b'



```python
add(*data)     # unpacking
```




    25




```python
def add2(*numbers):                  # 1, 2, 3, 4, => (1, 2, 3, 4) (packing)
                                     #                serialie, deserialzie
    result =  0                    
    for number in numbers:
        result += number        
    return result
```


```python
add2(1, 2, 3, 4)
```




    10




```python
# *args
# **kvargs

def introduce(name, greeting):
    return "{name}님, {greeting}".format(
        name=name,
        greeting=greeting,
    )
```


```python
introduce_dict = {
    "name" : "김진수",
    "greeting" : "안녕하세요",
}
```


```python
introduce(**introduce_dict)
```




    '김진수님, 안녕하세요'



<hr>
<marquee><font size=3 color='brown'>The BigpyCraft find the information to design valuable society with Technology & Craft.</font></marquee>
<div align='right'><font size=2 color='gray'> &lt; The End &gt; </font></div>
