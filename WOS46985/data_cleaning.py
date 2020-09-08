#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

X_file = open("X.txt")
Xfile_contents = X_file.read()
Xcontents_split = Xfile_contents.splitlines()

# print(len(Xcontents_split))
# 5736

Y_file = open("Y.txt")
Yfile_contents = Y_file.read()
Ycontents_split = Yfile_contents.splitlines()

print(len(Ycontents_split))
# 5736

d = {'X': Xcontents_split, 'Y': Ycontents_split}
df = pd.DataFrame(data=d)
df.to_csv('xydata.csv')