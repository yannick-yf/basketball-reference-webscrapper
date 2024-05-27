# # WebScrapNbaSalary.py

# from urllib.request import urlopen
# from bs4 import BeautifulSoup
# import pandas as pd
# import requests
# import sys
# import numpy as np
# from urllib.error import HTTPError
# import re
# import lxml.html as lh
# import time

# def ScrapePage(url):
#     r = requests.get(url)
#     # Get number of pages
#     soup = BeautifulSoup(r.content, features='html.parser')
#     page_total = soup.find(class_="page-numbers").get_text()
#     page_total = re.sub('.*of', '', page_total).strip()
#     page_total = int(page_total)
#     r = requests.get(url)
#     doc = lh.fromstring(r.content)
#     tr_elements = doc.xpath('//tr')
#     # Create empty list
#     col = []
#     i = 0
#     # For each row, store each first element (header) and an empty list
#     for t in tr_elements[0]:
#         i += 1
#         name = t.text_content()
#         col.append((name, []))
#     for j in range(len(tr_elements)):
#         # T is our j'th row
#         T = tr_elements[j]
#         # i is the index of our column
#         i = 0
#         # Iterate through each element of the row
#         for t in T.iterchildren():
#             data = t.text_content()
#             # Check if row is empty
#             if i > 0:
#                 # Convert any numerical value to integers
#                 try:
#                     data = int(data)
#                 except:
#                     pass
#             # Append the data to the empty list of the i'th column
#             col[i][1].append(data)
#             # Increment i for the next column
#             i += 1
#     dict = ({title: column for (title, column) in col})
#     return dict, page_total

# def GetSalaries(year):
#     df = []
#     dict = {}
#     url = 'http://www.espn.com/nba/salaries/_/year/' + year + '/'
#     data, page_total = ScrapePage(url)
#     df.append(pd.DataFrame(data))
    
#     #Get Salary Data
#     for k in range(2, page_total + 1):
#         url = 'http://www.espn.com/nba/salaries/_/year/' + year + '/page/' + str(k)
#         Dict = ScrapePage(url)[0]
#         df.append(pd.DataFrame(Dict))

#         time.sleep(5)

#     df = pd.concat(df)
#     df['year'] = int(year)
#     df = df[df['RK'] != "RK"]
#     df.reset_index(inplace=True, drop=True)
#     #Convert salary to numeric
#     for i in range(len(df)):
#         df.loc[i, 'name'] = re.sub(',.*', '', df['NAME'][i])
#         df.loc[i, 'salary'] = re.sub('\$', '', df['SALARY'][i])
#         df.loc[i, 'salary'] = re.sub(',', '', df['salary'][i])
#     #Get Rid of text rows
#         df['salary'] = pd.to_numeric(df['salary'])
#     return df