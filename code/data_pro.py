from collections import defaultdict
import re

dd = defaultdict()
# with open("test.txt",'r') as fin:
#     lines = fin.readlines()
#     for l in lines:
#         if l[0] == "1":
#             print(l)
pos = "/home/wang/PycharmProjects/tianchi/pos_tags.txt"
neg = "/home/wang/PycharmProjects/tianchi/neg_tags.txt"


pos_list = []
neg_list = []
pos_tags = []
neg_tags = []

with open(pos,'r') as fin:
    lines = fin.readlines()
    for line in lines:
        tag = line.split(',')[0]
        pos_list.append(tag)
        tt = re.split('_| ', tag)
        # print(pos_tags)
        # print(tt)
        pos_tags.extend(tt)
pos_tags = [t.upper() for t in pos_tags]
pos_tags = list(set(pos_tags))
print(pos_tags)

with open(neg,'r') as fin:
    lines = fin.readlines()
    for line in lines:
        tag = line.split(',')[0]
        neg_list.append(tag)

        tt = re.split('_| ',tag)
        neg_tags.extend(tt)
neg_tags = [t.upper() for t in neg_tags]
neg_tags = list(set(neg_tags))
print(neg_tags)
print("####################pos tag")
for p in pos_tags:
    if p not in neg_tags:
        print(p)
print("##########################")
for n in neg_tags:
    if n not in pos_tags:
        print(n)


# for p in pos_list:
#     if p not in neg_list:
#
#         print(p)
#
#
# print("\n")
# print("############neg tags ")
# for n in neg_list:
#     if n not in pos_list and "T2" in n.upper():
#         print(n)
