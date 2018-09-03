class Person:
    name = str()
    age = int()
    hometown = str()

    def __init__(self, name, age, hometown):
        self.name = name
        self.age = age
        self.hometown = hometown

    def to_string(self):
        # str = '나의 살던 고향은 ' + self.hometown + '입니다.'
        str = '%s의 나이는 %d살이고, 고향은 %s입니다.' % (self.name, self.age, self.hometown)
        return str


theif1 = Person("홍길동", 20, "율도국");
theif2 = Person("임꺽정", 35, "구월산");

str1 = theif1.to_string()
str2 = theif2.to_string()

print(str1)
print(str2)

