i=1
j=1
while i<=9:
    while j<=i:
        f=i*j
        print(f"{j}*{i}={f}\t",end='')
        j+=1
    j=1
    i+=1
    print()


for x in range(1,10):
    for y in range (1,10):
        if y<=x:
            print(f"{y}*{x}={y*x} ",end='')
    print()


money=10000
j=int(money/1000)
i=0
for x in range(1,21):
    import random
    num = random.randint(1,10)
    if num>=5:
        if money>0:
            money=money-1000
            print(f"员工{x},", end='')
            print(f"绩效为:{num}",end='')
            print("获取工资1000元")
        i+=1
if i>j:
    print("钱不够了",end='')
    print(f"还有{i - j}人没有工资")
    print(i)
else:
    print("公司余额为:",end='')
    print(money)


money=10000
j=int(money/1000)
i=0
for x in range(1,21):
    import random
    num = random.randint(1,10)
    if num>=5:
        i += 1
        if money>0:
            money=money-1000
            print(f"员工{x},", end='')
            print(f"绩效为:{num}",end='')
            print("获取工资1000元")
        else:
            if i > j:
                print(f"{x}号 ",end='')
if money<=0:
    print("没有工资")


my_list=[21,25,21,23,22,20]
my_list.append(31)
my_list.extend([29,33,30])
del my_list[0]
del my_list[8]
i=0
for x in my_list:
    if x!=31:
        i+=1
    else:
        print(i+1)
print(my_list)




t1=("zhou",11,["ahh","all"])
index=t1.index(11)
t1[2][0]="wll"
del t1[2][0]
t1[2].append("wsfw")
i=0
while i<len(t1):
    print(t1[i])
    i+=1





my_list=[1,2,3,4,5,6,7,8,9,10]
my_list1=[]
def while_func(x,y):
    index=0
    while index<len(x):
        element=x[index]
        if element%2==0:
            print(element)
            y.append(element)
        index+=1
    print(y)
def for_func(x):
        index=0
        for i in x:
            if index%2!=0:
                print(x[index])
            index+=1
while_func(my_list,my_list1)
for_func(my_list)




my_list=[1,2,3,4,5,6,7,8,9,10]
def while_func(x):
    index=0
    while index<len(x):
        element=x[index]
        if element%2==0:
            print(element)
        index+=1
def for_func(x):
        index=0
        for i in x:
            if index%2!=0:
                print(x[index])
            index+=1
while_func(my_list)
for_func(my_list)




my_list = [1, 2, 3, 4, "whhh", 6]
def while_print_func(x):
    index=0
    while index<len(x):
        z=x[index]
        index+=1
        print(z)
def for_print_func(x):
    index=0
    for i in x:
        print(x[index])
        index+=1
for_print_func(my_list)
while_print_func(my_list)





def main():
    print("----------welcome----------")
    print("选择你的业务:")
    print("查询余额\t[方式1]")
    print("存款\t\t[方式2]")
    print("取款\t\t[方式3]")
    print("退出\t\t[方式4]")
def back():
    print("请继续选择业务或者按4离开")
money=5000
def select(x):
    global money
    if x==1:
        print(f"您的余额为：{money}")
    elif x==2:
        print(f"您银行账户现有{money}元，请输入你要存入的钱")
        y=float(input())
        money+=y
        print(f"您银行账户现有{money}元")
    elif x==3:
        print(f"您银行账户现有{money}元，请输入你要取出的钱")
        y=float(input())
        if y<=money:
            print(f"余额还有{money-y}元")
            money-=y
        else:
            print("余额不足！")
    elif x==4:
        print("本账户已经退出")
    else:
        print("请选择正确的服务")
main()
while 1:
    z=int(input())
    select(z)
    if z==4:
        break
    back()

