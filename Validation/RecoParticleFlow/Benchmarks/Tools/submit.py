#!/usr/bin/env python
# to submit a benchmark webpage to the validation website
# author: Colin

import shutil, sys, os, valtools

from optparse import OptionParser


parser = OptionParser()
parser.usage = "usage: %prog"

parser.add_option("-e", "--extension", dest="extension",
                  help="adds an extension to the name of this benchmark",
                  default=None)


(options,args) = parser.parse_args()
 
if len(args)!=0:
    parser.print_help()
    sys.exit(1)


website = valtools.website()
bench = valtools.benchmark( options.extension ) 

print 'submitting benchmark:', bench

# check that the user can write in the website
website.writeAccess()
bench.makeRelease( website )

if( bench.exists( website ) == True ):
    print 'please use the -e option to add an extention, or choose another extension'
    print '  e.g: submit.py -e Feb10'
else:
    print bench, bench.benchmarkOnWebSite(website)
    shutil.copytree(bench.__str__(), bench.benchmarkOnWebSite(website) ) 
    print 'done. Access your benchmark here:'
    print bench.benchmarkUrl( website )
    
