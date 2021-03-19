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

rowheader=tables[2].find('thead')
rowheaders=rowheader.find_all('tr')
htag = soup.new_tag('td')
htag.string='Num reports'
htag['class']='Q'
rowheaders[-1].insert(7,htag)

rowsbody = tables[2].find('tbody')
rows=rowsbody.find_all('tr')
for row in rows:
    cells=row.find_all('td')
    key=str(cells[2])+str(cells[3])+str(cells[4])
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


rowsbody = tables[2].find('tbody')
rows=rowsbody.find_all('tr')
for row in rows:
    cells=row.find_all('td')
    key=str(cells[2])+str(cells[3])+str(cells[4])
    tag = soup.new_tag('td')
    tag.string='{}'.format(seen[key])
    tag['class']='Q'
    row.insert(3,tag)
print(soup.prettify("latin1"))
