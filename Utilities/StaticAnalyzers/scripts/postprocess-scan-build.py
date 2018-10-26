#!/usr/bin/env python
import lxml
from bs4 import BeautifulSoup
import sys, os
url=os.path.abspath(sys.argv[1])
report_dir = os.path.dirname(url)
page=open(url)
soup=BeautifulSoup(page.read(),features="lxml")
seen=set()
tables=soup.find_all('table',recursive=True)
rows = tables[2].findChildren('tr')
for row in rows:
    cells=row.findChildren('td')
    key=str(cells[2])+str(cells[4])+str(cells[5])
    if key in seen:
        href = cells[6].find('a',href=True)
        if href:
          report = href['href'].split("#")[0]
          report_file = os.path.join(report_dir, report)
          if report.startswith("report-") and os.path.exists(report_file):
            os.remove(report_file)
        row.decompose()
    else:
        seen.add(key)
print soup.prettify("latin1")
