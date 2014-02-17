#! /usr/bin/env python
################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
# $Author: dpiparo $
# $Date: 2012/06/12 12:25:28 $
# $Revision: 1.1 $
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