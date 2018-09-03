
# Industry 4.0 의 중심, AI - ML&DL

<div align='right'><font size=2 color='gray'>Machine Learning & Deep Learning with TensorFlow @ <font color='blue'><a href='https://www.facebook.com/jskim.kr'>FB / jskim.kr</a></font>, 김진수</font></div>
<hr>

## Tensorflow Programming Model

### <font color='brown'>TF01_Hello</font>


```python
import tensorflow as tf

hello = tf.constant(" Hello TensorFlow!")
sess = tf.Session()

sess.run(hello)
```

    /Users/bigpycraft/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    




    b' Hello TensorFlow!'



### <font color='brown'>TF02_Variable</font>


```python
#first_session_only_tensorflow.py
import tensorflow as tf

x = tf.constant(100, name='x')
y = tf.Variable(x*2, name='y')

model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    print("constant x : {x} \nVariable y : {y}".format(
        x = sess.run(x), 
        y = sess.run(y)
    ))
```

    constant x : 100 
    Variable y : 200
    

### <font color='brown'>TF03_placeholder</font>


```python
import tensorflow as tf

# 변수 a, b는 동적으로 지정
a = tf.placeholder("int32")
b = tf.placeholder("int32")

# multiply 함수는 입력된 정수 a와 b의 곱셈을 반환한다.
y = tf.multiply(a,b)

sess = tf.Session()

print("tf.multiply({a},{b}) : {y}".format(
    a = 20, b = 30,
    y = sess.run(y , feed_dict={a: 20, b: 30})
))
```

    tf.multiply(20,30) : 600
    

### <font color='brown'>TF04_Tensorboard</font>


```python
import tensorflow as tf

a = tf.constant(10,name="a")
b = tf.constant(20,name="b")
y = tf.Variable(a**2+b**2, name="y")

model = tf.global_variables_initializer()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./graph/board", sess.graph)
    sess.run(model)
    print(" constant a = {a} \n constant b = {b} \n Variable y = {y}".format(
        a = sess.run(a), 
        b = sess.run(b),
        y = sess.run(y)
    )) 
```

     constant a = 10 
     constant b = 20 
     Variable y = 500
    
<!--
(Anaconda3) > tensorboard --logdir=graph/board --port=9999

http://localhost:9999/#graphs
//-->
### <font color='brown'>TF05_Tensor_Operation</font>


```python
import numpy as np

tensor_1d = np.array([1.2, 3.4, 5.6, 7.8])
tensor_2d = np.arange(16).reshape((4,4))
```


```python
import tensorflow as tf

tf_tensor=tf.convert_to_tensor(tensor_1d,dtype=tf.float64)
with tf.Session() as sess:
    print(sess.run(tf_tensor))
    print(sess.run(tf_tensor[0]))
    print(sess.run(tf_tensor[-1]))
```

    [ 1.2  3.4  5.6  7.8]
    1.2
    7.8
    


```python
import tensorflow as tf 

tf_tensor_1d = tf.convert_to_tensor(tensor_1d,dtype=tf.float64)
tf_tensor_2d = tf.convert_to_tensor(tensor_2d,dtype=tf.float64)
sess = tf.Session()
```


```python
sess.run(tf_tensor_1d)
```




    array([ 1.2,  3.4,  5.6,  7.8])




```python
sess.run(tf_tensor_1d[0])
```




    1.2




```python
sess.run(tf_tensor_1d[2:])
```




    array([ 5.6,  7.8])




```python
sess.run(tf_tensor_2d)
```




    array([[  0.,   1.,   2.,   3.],
           [  4.,   5.,   6.,   7.],
           [  8.,   9.,  10.,  11.],
           [ 12.,  13.,  14.,  15.]])




```python
sess.run(tf_tensor_2d[3][3])
```




    15.0




```python
sess.run(tf_tensor_2d[1:3,1:3])
```




    array([[  5.,   6.],
           [  9.,  10.]])



<hr>
<marquee><font size=3 color='brown'>The BigpyCraft find the information to design valuable society with Technology & Craft.</font></marquee>
<div align='right'><font size=2 color='gray'> &lt; The End &gt; </font></div>
