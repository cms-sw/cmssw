#! /usr/bin/env python3
################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
#
#                                                                              
# Danilo Piparo CERN - danilo.piparo@cern.ch                                   
#                                                                              
################################################################################

from __future__ import print_function
from sys import argv,exit
from utils import wget

if __name__=="__main__":
  argc=len(argv)
  if argc!=2:
    print("Usage %prog url-to-fetch")
    exit(1)
  
  url=argv[1]
  wget(url)