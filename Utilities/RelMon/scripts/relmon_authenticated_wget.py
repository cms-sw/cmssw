#! /usr/bin/env python
################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
# $Author: dpiparo $
# $Date: 2011/06/08 15:47:04 $
# $Revision: 1.3 $
#
#                                                                              
# Danilo Piparo CERN - danilo.piparo@cern.ch                                   
#                                                                              
################################################################################

from sys import argv,exit
from utils import wget

if __name__=="__main__":
  argc=len(argv)
  if argc!=2:
    print "Usage %prog url-to-fetch"
    exit(1)
  
  url=argv[1]
  wget(url)