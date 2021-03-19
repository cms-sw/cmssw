#!/usr/bin/env python
import lxml
from bs4 import BeautifulSoup
import sys, os
url=os.path.abspath(sys.argv[1])
report_dir = os.path.dirname(url)
page=open(url)
soup=BeautifulSoup(page.read(),features="lxml")
seen=dict()
tables=soup.find_all('table',recursive=True)

header=tables[2].findAll('thead')
rowheader=header[0].findAll('tr')
rowheaders=rowheader[0].findAll('td')
rowheaders[2].append(' (# reports)')

rows = tables[2].findChildren('tr')
for row in rows:
    cells=row.findChildren('td')
    key=str(cells[2])+str(cells[4])+str(cells[5])
    if key in seen.keys():
        seen[key]=seen[key]+1
        href = cells[6].find('a',href=True)
        if href:
          report = href['href'].split("#")[0]
          report_file = os.path.join(report_dir, report)
          if report.startswith("report-") and os.path.exists(report_file):
            os.remove(report_file)
        row.decompose()
    else:
        seen[key]=1


rows = tables[2].find('tbody').findChildren('tr')
for row in rows:
    cells=row.findChildren('td')
    key=str(cells[2])+str(cells[4])+str(cells[5])
    if not key==rowheaders[2]:
        cells[2].append(' ({})'.format(seen[key]))
print(soup.prettify("latin1"))
