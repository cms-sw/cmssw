#!/usr/bin/env python
import lxml
from bs4 import BeautifulSoup
import sys
url=sys.argv[1]
page=open(url)
soup=BeautifulSoup(page.read(),features="lxml")
seen=set()
tables=soup.find_all('table',recursive=True)
rows = tables[2].findChildren('tr')
for row in rows:
    cells=row.findChildren('td')
    key=str(cells[2])+str(cells[4])+str(cells[5])
    if key in seen:
        row.decompose()
    else:
        seen.add(key)
print soup.prettify("latin1")
