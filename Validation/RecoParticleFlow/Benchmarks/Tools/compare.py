#!/usr/bin/env python
# to submit a benchmark webpage to the validation website
# author: Colin

import shutil, sys, os, valtools

from optparse import OptionParser


parser = OptionParser()
parser.usage = "usage: %prog other_release"

parser.add_option("-e", "--extension", dest="extension",
                  help="adds an extension to the name of this benchmark",
                  default=None)


(options,args) = parser.parse_args()
 
if len(args)!=1:
    parser.print_help()
    sys.exit(1)

otherRelease = args[0]

website = valtools.website()

bench0 = valtools.benchmark( options.extension )
bench1 = valtools.benchmark( options.extension, otherRelease) 

print 'comparing', bench0.release_, 'and', bench1.release_

# link the 2 root files locally
print bench0.rootFileOnWebSite( website )
print bench1.rootFileOnWebSite( website )

link0 = 'benchmark_0.root'
link1 = 'benchmark_1.root'

if os.path.isfile(link0):
    os.unlink( link0 )
if os.path.isfile(link1):
    os.unlink( link1 )

os.symlink( bench0.rootFileOnWebSite( website ), 'benchmark_0.root')
os.symlink( bench1.rootFileOnWebSite( website ), 'benchmark_1.root')

os.system( 'root -l compare.C' )

# check that the user can write in the website
website.writeAccess()


