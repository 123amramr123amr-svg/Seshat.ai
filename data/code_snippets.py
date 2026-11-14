"""
أكواد التدريب — كل عنصر: (اللغة، المهمة بالعربي، المهمة بالإنجليزي، الكود)
المهمة بالعربي بتبدأ بفعل مضارع (يجمع، يطبع...) وبالإنجليزي بفعل أمر (add, print...)
زوّد براحتك بنفس الشكل، وdataset.py هيولّد صيغ الأسئلة لوحده.
"""

SNIPPETS = []


def add(lang, ar, en, code):
    SNIPPETS.append((lang, ar, en, code.strip("\n")))


# ============================================================
# Python
# ============================================================
add("python", "يطبع hello world", "print hello world", '''
print("Hello, World!")
''')

add("python", "يجمع رقمين", "add two numbers", '''
def add(a, b):
    return a + b

print(add(3, 5))
''')

add("python", "يحسب المضروب", "calculate the factorial of a number", '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
''')

add("python", "يطبع متتالية فيبوناتشي", "print the fibonacci sequence", '''
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a)
        a, b = b, a + b

fibonacci(10)
''')

add("python", "يعكس نص", "reverse a string", '''
def reverse(text):
    return text[::-1]

print(reverse("hello"))
''')

add("python", "يتأكد إن الكلمة باليندروم", "check if a string is a palindrome", '''
def is_palindrome(text):
    return text == text[::-1]

print(is_palindrome("level"))
''')

add("python", "يتأكد إن الرقم أولي", "check if a number is prime", '''
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

print(is_prime(17))
''')

add("python", "يتأكد إن الرقم زوجي ولا فردي", "check if a number is even or odd", '''
n = int(input("Enter a number: "))
if n % 2 == 0:
    print("Even")
else:
    print("Odd")
''')

add("python", "يجيب أكبر رقم في ليست", "find the largest number in a list", '''
numbers = [4, 9, 2, 7]
largest = numbers[0]
for n in numbers:
    if n > largest:
        largest = n
print(largest)
''')

add("python", "يجيب أصغر رقم في ليست", "find the smallest number in a list", '''
numbers = [4, 9, 2, 7]
smallest = numbers[0]
for n in numbers:
    if n < smallest:
        smallest = n
print(smallest)
''')

add("python", "يجمع أرقام ليست", "sum the numbers in a list", '''
numbers = [1, 2, 3, 4, 5]
total = 0
for n in numbers:
    total += n
print(total)
''')

add("python", "يحسب المتوسط", "calculate the average of a list", '''
numbers = [10, 20, 30]
average = sum(numbers) / len(numbers)
print(average)
''')

add("python", "يرتب ليست", "sort a list", '''
numbers = [5, 2, 9, 1]
numbers.sort()
print(numbers)
''')

add("python", "يطبع الأرقام من 1 لـ 10", "print the numbers from 1 to 10", '''
for i in range(1, 11):
    print(i)
''')

add("python", "يطبع جدول الضرب", "print a multiplication table", '''
n = 5
for i in range(1, 11):
    print(n, "x", i, "=", n * i)
''')

add("python", "يعد الحروف المتحركة في نص", "count the vowels in a string", '''
text = "hello world"
count = 0
for ch in text:
    if ch in "aeiou":
        count += 1
print(count)
''')

add("python", "يعد الكلمات في جملة", "count the words in a sentence", '''
sentence = "I love Egypt very much"
words = sentence.split()
print(len(words))
''')

add("python", "يقرأ ملف", "read a file", '''
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()
print(content)
''')

add("python", "يكتب في ملف", "write to a file", '''
with open("data.txt", "w", encoding="utf-8") as f:
    f.write("Hello from Seshat")
''')

add("python", "ياخد اسم المستخدم ويرحب بيه", "ask for the user's name and greet them", '''
name = input("What is your name? ")
print("Hello, " + name + "!")
''')

add("python", "يحول من سيلزيوس لفهرنهايت", "convert celsius to fahrenheit", '''
def to_fahrenheit(c):
    return c * 9 / 5 + 32

print(to_fahrenheit(25))
''')

add("python", "يبدل قيمة متغيرين", "swap two variables", '''
a = 1
b = 2
a, b = b, a
print(a, b)
''')

add("python", "يشيل المتكرر من ليست", "remove duplicates from a list", '''
items = [1, 2, 2, 3, 3, 3]
unique = list(set(items))
print(unique)
''')

add("python", "يعمل كلاس", "define a class", '''
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print("Hi, I am " + self.name)

p = Person("Ali", 20)
p.greet()
''')

add("python", "يولد رقم عشوائي", "generate a random number", '''
import random

number = random.randint(1, 100)
print(number)
''')

add("python", "يحسب مساحة الدايرة", "calculate the area of a circle", '''
import math

radius = 5
area = math.pi * radius ** 2
print(area)
''')

add("python", "يعمل بحث ثنائي", "do a binary search", '''
def binary_search(items, target):
    low, high = 0, len(items) - 1
    while low <= high:
        mid = (low + high) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

print(binary_search([1, 3, 5, 7, 9], 7))
''')

add("python", "يعمل ترتيب الفقاعات", "implement bubble sort", '''
def bubble_sort(items):
    n = len(items)
    for i in range(n):
        for j in range(n - i - 1):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
    return items

print(bubble_sort([5, 2, 9, 1]))
''')

add("python", "يحسب القاسم المشترك الأكبر", "find the greatest common divisor", '''
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

print(gcd(48, 18))
''')

add("python", "يتأكد إن السنة كبيسة", "check if a year is a leap year", '''
def is_leap(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

print(is_leap(2024))
''')

add("python", "يعمل قاموس ويطبعه", "create a dictionary and loop over it", '''
ages = {"Ali": 20, "Mona": 25}
for name, age in ages.items():
    print(name, age)
''')

add("python", "يعمل آلة حاسبة", "make a simple calculator", '''
a = float(input("First number: "))
op = input("Operation (+ - * /): ")
b = float(input("Second number: "))

if op == "+":
    print(a + b)
elif op == "-":
    print(a - b)
elif op == "*":
    print(a * b)
elif op == "/":
    print(a / b)
''')

add("python", "يعمل لعبة تخمين الرقم", "make a number guessing game", '''
import random

secret = random.randint(1, 10)
guess = 0
while guess != secret:
    guess = int(input("Guess: "))
    if guess < secret:
        print("Too low")
    elif guess > secret:
        print("Too high")
print("Correct!")
''')

add("python", "يحسب مجموع الأرقام من 1 لـ n", "sum the numbers from 1 to n", '''
n = 100
total = 0
for i in range(1, n + 1):
    total += i
print(total)
''')

add("python", "يحول النص لحروف كبيرة", "convert a string to uppercase", '''
text = "hello"
print(text.upper())
''')

add("python", "يعد تكرار كل حرف في نص", "count how often each character appears", '''
text = "banana"
counts = {}
for ch in text:
    counts[ch] = counts.get(ch, 0) + 1
print(counts)
''')

add("python", "يتعامل مع الأخطاء", "handle errors with try and except", '''
try:
    x = int(input("Enter a number: "))
    print(10 / x)
except ValueError:
    print("That is not a number")
except ZeroDivisionError:
    print("Cannot divide by zero")
''')

add("python", "يجيب الأرقام الزوجية من ليست", "get the even numbers from a list", '''
numbers = [1, 2, 3, 4, 5, 6]
evens = [n for n in numbers if n % 2 == 0]
print(evens)
''')

add("python", "يحسب الأس", "calculate the power of a number", '''
def power(base, exp):
    result = 1
    for _ in range(exp):
        result *= base
    return result

print(power(2, 10))
''')

add("python", "يدمج ليستتين", "merge two lists", '''
a = [1, 2, 3]
b = [4, 5, 6]
merged = a + b
print(merged)
''')

add("python", "يعرض التاريخ والوقت", "show the current date and time", '''
from datetime import datetime

now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
''')

add("python", "يعمل عد تنازلي", "make a countdown", '''
import time

for i in range(10, 0, -1):
    print(i)
    time.sleep(1)
print("Go!")
''')

add("python", "يحل فيز باز", "solve fizzbuzz", '''
for i in range(1, 101):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
''')

add("python", "يعمل ستاك", "implement a stack", '''
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

s = Stack()
s.push(1)
s.push(2)
print(s.pop())
''')

add("python", "يحسب طول نص", "find the length of a string", '''
text = "Seshat"
print(len(text))
''')

add("python", "يعمل حلقة while", "write a while loop", '''
i = 1
while i <= 5:
    print(i)
    i += 1
''')

add("python", "يعمل سيرفر فلاسك", "create a flask server", '''
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from Flask!"

app.run(port=5000)
''')

# ============================================================
# JavaScript
# ============================================================
add("javascript", "يطبع hello world", "print hello world", '''
console.log("Hello, World!");
''')

add("javascript", "يجمع رقمين", "add two numbers", '''
function add(a, b) {
  return a + b;
}

console.log(add(3, 5));
''')

add("javascript", "يحسب المضروب", "calculate the factorial of a number", '''
function factorial(n) {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}

console.log(factorial(5));
''')

add("javascript", "يعكس نص", "reverse a string", '''
function reverse(text) {
  return text.split("").reverse().join("");
}

console.log(reverse("hello"));
''')

add("javascript", "يتأكد إن الكلمة باليندروم", "check if a string is a palindrome", '''
function isPalindrome(text) {
  return text === text.split("").reverse().join("");
}

console.log(isPalindrome("level"));
''')

add("javascript", "يتأكد إن الرقم أولي", "check if a number is prime", '''
function isPrime(n) {
  if (n < 2) return false;
  for (let i = 2; i * i <= n; i++) {
    if (n % i === 0) return false;
  }
  return true;
}

console.log(isPrime(17));
''')

add("javascript", "يتأكد إن الرقم زوجي ولا فردي", "check if a number is even or odd", '''
function evenOrOdd(n) {
  return n % 2 === 0 ? "Even" : "Odd";
}

console.log(evenOrOdd(7));
''')

add("javascript", "يجيب أكبر رقم في ليست", "find the largest number in a list", '''
const numbers = [4, 9, 2, 7];
const largest = Math.max(...numbers);
console.log(largest);
''')

add("javascript", "يجمع أرقام ليست", "sum the numbers in a list", '''
const numbers = [1, 2, 3, 4, 5];
const total = numbers.reduce((a, b) => a + b, 0);
console.log(total);
''')

add("javascript", "يرتب ليست", "sort a list", '''
const numbers = [5, 2, 9, 1];
numbers.sort((a, b) => a - b);
console.log(numbers);
''')

add("javascript", "يطبع الأرقام من 1 لـ 10", "print the numbers from 1 to 10", '''
for (let i = 1; i <= 10; i++) {
  console.log(i);
}
''')

add("javascript", "يحل فيز باز", "solve fizzbuzz", '''
for (let i = 1; i <= 100; i++) {
  if (i % 15 === 0) console.log("FizzBuzz");
  else if (i % 3 === 0) console.log("Fizz");
  else if (i % 5 === 0) console.log("Buzz");
  else console.log(i);
}
''')

add("javascript", "يغير نص عنصر في الصفحة", "change the text of an element", '''
const title = document.getElementById("title");
title.textContent = "Hello from Seshat";
''')

add("javascript", "يظهر رسالة لما ندوس على زرار", "show an alert when a button is clicked", '''
const button = document.querySelector("button");
button.addEventListener("click", () => {
  alert("Button clicked!");
});
''')

add("javascript", "يجيب بيانات من API", "fetch data from an api", '''
async function getData() {
  const response = await fetch("https://example.com/api");
  const data = await response.json();
  console.log(data);
}

getData();
''')

add("javascript", "يولد رقم عشوائي", "generate a random number", '''
const number = Math.floor(Math.random() * 100) + 1;
console.log(number);
''')

add("javascript", "يشيل المتكرر من ليست", "remove duplicates from a list", '''
const items = [1, 2, 2, 3, 3, 3];
const unique = [...new Set(items)];
console.log(unique);
''')

add("javascript", "يعمل عد تنازلي", "make a countdown", '''
let count = 10;
const timer = setInterval(() => {
  console.log(count);
  count--;
  if (count < 0) clearInterval(timer);
}, 1000);
''')

add("javascript", "يعمل كلاس", "define a class", '''
class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }

  greet() {
    console.log("Hi, I am " + this.name);
  }
}

const p = new Person("Ali", 20);
p.greet();
''')

add("javascript", "يعرض التاريخ والوقت", "show the current date and time", '''
const now = new Date();
console.log(now.toLocaleString());
''')

# ============================================================
# C
# ============================================================
add("c", "يطبع hello world", "print hello world", '''
#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}
''')

add("c", "يجمع رقمين", "add two numbers", '''
#include <stdio.h>

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    printf("%d\\n", a + b);
    return 0;
}
''')

add("c", "يحسب المضروب", "calculate the factorial of a number", '''
#include <stdio.h>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    printf("%d\\n", factorial(5));
    return 0;
}
''')

add("c", "يتأكد إن الرقم زوجي ولا فردي", "check if a number is even or odd", '''
#include <stdio.h>

int main() {
    int n;
    scanf("%d", &n);
    if (n % 2 == 0)
        printf("Even\\n");
    else
        printf("Odd\\n");
    return 0;
}
''')

add("c", "يطبع الأرقام من 1 لـ 10", "print the numbers from 1 to 10", '''
#include <stdio.h>

int main() {
    for (int i = 1; i <= 10; i++) {
        printf("%d\\n", i);
    }
    return 0;
}
''')

add("c", "يجيب أكبر رقم في ليست", "find the largest number in a list", '''
#include <stdio.h>

int main() {
    int numbers[] = {4, 9, 2, 7};
    int largest = numbers[0];
    for (int i = 1; i < 4; i++) {
        if (numbers[i] > largest) largest = numbers[i];
    }
    printf("%d\\n", largest);
    return 0;
}
''')

add("c", "يبدل قيمة متغيرين", "swap two variables", '''
#include <stdio.h>

int main() {
    int a = 1, b = 2, temp;
    temp = a;
    a = b;
    b = temp;
    printf("%d %d\\n", a, b);
    return 0;
}
''')

# ============================================================
# C++
# ============================================================
add("c++", "يطبع hello world", "print hello world", '''
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
''')

add("c++", "يجمع رقمين", "add two numbers", '''
#include <iostream>
using namespace std;

int main() {
    int a, b;
    cin >> a >> b;
    cout << a + b << endl;
    return 0;
}
''')

add("c++", "يحسب المضروب", "calculate the factorial of a number", '''
#include <iostream>
using namespace std;

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    cout << factorial(5) << endl;
    return 0;
}
''')

add("c++", "يتأكد إن الرقم زوجي ولا فردي", "check if a number is even or odd", '''
#include <iostream>
using namespace std;

int main() {
    int n;
    cin >> n;
    if (n % 2 == 0)
        cout << "Even" << endl;
    else
        cout << "Odd" << endl;
    return 0;
}
''')

add("c++", "يطبع الأرقام من 1 لـ 10", "print the numbers from 1 to 10", '''
#include <iostream>
using namespace std;

int main() {
    for (int i = 1; i <= 10; i++) {
        cout << i << endl;
    }
    return 0;
}
''')

add("c++", "يعمل كلاس", "define a class", '''
#include <iostream>
#include <string>
using namespace std;

class Person {
public:
    string name;
    int age;

    void greet() {
        cout << "Hi, I am " << name << endl;
    }
};

int main() {
    Person p;
    p.name = "Ali";
    p.age = 20;
    p.greet();
    return 0;
}
''')

# ============================================================
# Java
# ============================================================
add("java", "يطبع hello world", "print hello world", '''
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
''')

add("java", "يجمع رقمين", "add two numbers", '''
public class Main {
    static int add(int a, int b) {
        return a + b;
    }

    public static void main(String[] args) {
        System.out.println(add(3, 5));
    }
}
''')

add("java", "يحسب المضروب", "calculate the factorial of a number", '''
public class Main {
    static int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }

    public static void main(String[] args) {
        System.out.println(factorial(5));
    }
}
''')

add("java", "يتأكد إن الرقم زوجي ولا فردي", "check if a number is even or odd", '''
public class Main {
    public static void main(String[] args) {
        int n = 7;
        if (n % 2 == 0) {
            System.out.println("Even");
        } else {
            System.out.println("Odd");
        }
    }
}
''')

add("java", "يطبع الأرقام من 1 لـ 10", "print the numbers from 1 to 10", '''
public class Main {
    public static void main(String[] args) {
        for (int i = 1; i <= 10; i++) {
            System.out.println(i);
        }
    }
}
''')

add("java", "يعكس نص", "reverse a string", '''
public class Main {
    public static void main(String[] args) {
        String text = "hello";
        String reversed = new StringBuilder(text).reverse().toString();
        System.out.println(reversed);
    }
}
''')

# ============================================================
# SQL
# ============================================================
add("sql", "ينشئ جدول", "create a table", '''
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER
);
''')

add("sql", "يضيف صف في جدول", "insert a row into a table", '''
INSERT INTO users (name, age)
VALUES ('Ali', 20);
''')

add("sql", "يجيب كل البيانات من جدول", "select all rows from a table", '''
SELECT * FROM users;
''')

add("sql", "يجيب البيانات بشرط", "select rows with a condition", '''
SELECT name, age
FROM users
WHERE age > 18;
''')

add("sql", "يعدل بيانات في جدول", "update a row in a table", '''
UPDATE users
SET age = 21
WHERE name = 'Ali';
''')

add("sql", "يمسح صف من جدول", "delete a row from a table", '''
DELETE FROM users
WHERE id = 1;
''')

add("sql", "يرتب النتايج", "sort the results", '''
SELECT * FROM users
ORDER BY age DESC;
''')

add("sql", "يعد الصفوف في جدول", "count the rows in a table", '''
SELECT COUNT(*) FROM users;
''')

# ============================================================
# HTML / CSS
# ============================================================
add("html", "يعمل صفحة بسيطة", "make a simple page", '''
<!DOCTYPE html>
<html>
<head>
  <title>My Page</title>
</head>
<body>
  <h1>Hello, World!</h1>
  <p>This is my first page.</p>
</body>
</html>
''')

add("html", "يعمل زرار", "make a button", '''
<button onclick="alert('Hello!')">Click me</button>
''')

add("html", "يعمل لينك", "make a link", '''
<a href="https://example.com">Visit the site</a>
''')

add("html", "يعرض صورة", "show an image", '''
<img src="photo.jpg" alt="My photo" width="300">
''')

add("html", "يعمل جدول", "make a table", '''
<table border="1">
  <tr>
    <th>Name</th>
    <th>Age</th>
  </tr>
  <tr>
    <td>Ali</td>
    <td>20</td>
  </tr>
</table>
''')

add("html", "يعمل فورم تسجيل دخول", "make a login form", '''
<form action="/login" method="post">
  <input type="text" name="username" placeholder="Username">
  <input type="password" name="password" placeholder="Password">
  <button type="submit">Login</button>
</form>
''')

add("html", "يعمل قايمة", "make a list", '''
<ul>
  <li>First item</li>
  <li>Second item</li>
  <li>Third item</li>
</ul>
''')

add("css", "يوسّط عنصر في الصفحة", "center a div", '''
.container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}
''')

add("css", "يغير لون الخلفية", "change the background color", '''
body {
  background-color: #1a1a2e;
  color: white;
}
''')

add("css", "ينسق زرار", "style a button", '''
button {
  background-color: gold;
  border: none;
  padding: 10px 20px;
  border-radius: 8px;
  cursor: pointer;
}
''')

# ============================================================
# Bash
# ============================================================
add("bash", "يطبع hello world", "print hello world", '''
#!/bin/bash
echo "Hello, World!"
''')

add("bash", "يطبع الأرقام من 1 لـ 10", "print the numbers from 1 to 10", '''
#!/bin/bash
for i in {1..10}; do
  echo $i
done
''')

add("bash", "يعرض الملفات اللي في الفولدر", "list the files in a folder", '''
#!/bin/bash
ls -la
''')

add("bash", "يتأكد إن الملف موجود", "check if a file exists", '''
#!/bin/bash
if [ -f "data.txt" ]; then
  echo "File exists"
else
  echo "File not found"
fi
''')
