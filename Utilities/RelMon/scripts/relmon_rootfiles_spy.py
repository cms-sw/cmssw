#! /usr/bin/env python
################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
# $Author: dpiparo $
# $Date: 2012/07/03 05:20:05 $
# $Revision: 1.2 $
#
#                                                                              
# Danilo Piparo CERN - danilo.piparo@cern.ch                                   
#                                                                              
################################################################################

"""
Just a draft of the real program...It is very ugly still.
"""


from os.path import basename
from optparse import OptionParser
from re import search
from sys import exit
from urllib2  import Request,build_opener,urlopen

import os
if os.environ.has_key("RELMON_SA"):
  from authentication import X509CertOpen
  from definitions import server
  from utils import wget  
else:  
  from Utilities.RelMon.authentication import X509CertOpen
  from Utilities.RelMon.definitions import server
  from Utilities.RelMon.utils import wget

def extract_list(page_html,the_server,display_url):
  contents=[]  
  for line in page_html.split("<tr><td>")[1:]:
    name=""
    #link
    link_start=line.find("href='")+6
    link_end=line.find("'>")    
    #name
    name_start=link_end+2
    name_end=line.find("</a>")
    if display_url:
      contents.append(the_server+line[link_start:link_end])
    else:
      contents.append(line[name_start:name_end])
  return contents
    
def get_page(url):
  """ Get the web page listing the rootfiles. Use the X509 auth.
  """  
  opener=build_opener(X509CertOpen())  
  datareq = Request(url)
  datareq.add_header('authenticated_wget', "The ultimate wgetter")    
  filename=basename(url)  
  return opener.open(datareq).read()

if __name__=="__main__":

  parser = OptionParser(usage="usage: %prog [options] dirtolist")

  parser.add_option("-d","--dev",
                  action="store_true",
                  dest="development",
                  default=False,
                  help="Select the development GUI instance.")

  parser.add_option("--offline",
                  action="store_true",
                  dest="offline",
                  default=False,
                  help="Select the Offline GUI instance.")
                  
  parser.add_option("-o","--online",
                  action="store_true",
                  dest="online",
                  default=False,
                  help="Select the Online GUI instance.")

  parser.add_option("-r","--relval",
                  action="store_true",
                  dest="relval",
                  default=True,
                  help="Select the RelVal GUI instance.")

  parser.add_option("-u","--show_url",
                  action="store_true",
                  dest="show_url",
                  default=False,
                  help="Show the full URL of the file.")

  parser.add_option("-g","--get",
                  action="store_true",
                  dest="get",
                  default=False,
                  help="Get the files.")

  parser.add_option("-p","--path",
                  action="store",
                  dest="path",
                  default="",
                  help="The path to be matched before getting.")

  (options, args) = parser.parse_args()

  if not(options.development or options.offline or options.online or options.relval):
    print "Select development or online instance!"
    exit(-1)

  lenargs=len(args)
  if lenargs>1:
    print "Please specify only one directory to list!"
    exit(-1)

  dirtolist=""
  if lenargs==1:
    dirtolist=args[0]
  
  mode="relval"
  if options.online:
    mode="online"
  if options.development:
    mode="dev"
  
    
  directory="%s/dqm/%s/data/browse/%s" %(server,mode,dirtolist)
  print "peeping ",directory  
  contents=extract_list(get_page(directory),server,options.show_url)
  
  if len(contents)==0:
    print "No contents found!"
  
  for content in contents:
    if not options.get and search(options.path,content):
      print content
    if options.get and options.show_url and len(options.path)>0 and search(options.path,content):
      if not search('pre',options.path) and search('pre',content):
        continue
      bcontent=basename(content)
      print "Getting %s" %bcontent
      wget(content)
      print "Got %s!!" %bcontent
  
  
