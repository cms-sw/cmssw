#!/usr/bin/env python


import shutil, sys, valtools

from optparse import OptionParser 


parser = OptionParser()
parser.usage = "usage: %prog pattern"
parser.add_option("-a", "--afs", dest="afs",
                  action="store_true",
                  help="print afs folder",
                  default=False)
parser.add_option("-u", "--url", dest="url",
                  action="store_true",
                  help="print url",
                  default=False)

(options,args) = parser.parse_args()
 
if len(args)!=1:
    parser.print_help()
    sys.exit(1)

pattern = args[0]
website = valtools.website()

website.listBenchmarks( pattern, options.afs, options.url)

