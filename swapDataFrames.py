import pandas as pd
import csv

csv_file1 = 'sheet2.1.csv'
csv_file2 = 'sheet2.2.csv'
csv_file3 = 'sheet2.3.csv'

df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)
myFile = open(csv_file3, 'w')

with myFile:
    myFields = ['Policy', 'Year']
    writer = csv.DictWriter(myFile, fieldnames = myFields)
    writer.writeheader()

    for index2, row2 in df2.iterrows():
        policy2 = row2['Policy']
        for index1, row1 in df1.iterrows():
            policy1 = row1['Policy']
            if(policy1 == policy2):
                writer.writerow({'Policy' : policy2, 'Year': row1['Year']})
                break
