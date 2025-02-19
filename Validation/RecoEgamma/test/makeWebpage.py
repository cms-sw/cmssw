#!/usr/bin/python

## This script writes an index.html page
## displaying all plots contained in the 
## directory.
##
## The file: makeWebpage.html is needed
## to be kept in the same place as the
## script and is used as a template.
## -------------------------------------
## Usage: ./makeWebpage.py DIR
## -------------------------------------
## with DIR = the path to the directoy
## containig the plots.
## DIR will also be the title of the
## webpage.
##
## For the moment it works for .gif's.
## For other formats, change line 42
## accordingly.
##
## After running the script, copy the
## folder conatining the plots and the
## index.html to an apropriate place.
## (This can be autmated too, if needed.)

import sys
import os
import re

directory=sys.argv[1]
oldV=sys.argv[2]
newV=sys.argv[3]
sample=sys.argv[4]

# get the plots in the directory
list = os.listdir(directory)

# open a tmeplate file and the new .html file
template=open("makeWebpage.html","r")
page=open(directory+"/index.html","w")

# write the common part of the index.html
for line in template:
    if(re.search(r'blablabla',line)!=None):
        page.write(sys.argv[1])
    else:
        page.write(line)

page.write('<h1> Track based conversions: '+sys.argv[3]+' vs '+sys.argv[2]+' Validation<br> Sample used: '+sys.argv[4]+ '<br><h3>In all plots below, '+sys.argv[2]+' is in purple, '+sys.argv[3]+' in black<br> Responsible: N. Marinelli</h3>')

# add all the plots in the directory
for i in range(len(list)):
    if(re.search(r'gif',list[i])!=None):
        print list[i]
        page.write('<a href=" '+list[i])
        page.write('" onMouseOver="ThumbnailPlot(')
        page.write("'"+list[i]+"'), ClearNotes(),  ThumbnailNote() ")
        page.write('"> <img src="'+list[i]+' " style="height:22px; width:25px;"><small>'+list[i]+' </small></a> <br> \n')

# write the last lines
page.write('<br>\n')
page.write('<div id="thumb_gen"><a href="#" id="thumblink_l"><img src="" id="thumb_l" width=0 height=0 border=0></a></div> \n')
page.write('</body> \n')
page.write(' </html>\n')


# now copy everything on the wwweth server...
#os.system("scp -r "+directory+" wwweth.cern.ch:JetMETPromptAnalysis/")













