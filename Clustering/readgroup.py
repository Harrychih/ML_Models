import csv

f = open("datasets/HAC/0.11_average.y.csv")
lines = f.readlines()
dic = dict()

for i in range(11):
    dic[i] = []

for i in range(len(lines)):

    dic[int(lines[i])].append(i)

print(dic)