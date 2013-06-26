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

parser.add_option("-f", "--force", dest="force",action="store_true",
                  help="force the submission. Be careful!",
                  default=False)


(options,args) = parser.parse_args()


 
if len(args)!=0:
    parser.print_help()
    sys.exit(1)


website = valtools.website()
bench = valtools.benchmark( options.extension ) 
localBench = valtools.benchmark()
print 'submitting  from local: ', localBench
print '                    to: ', bench

comparisons = website.listComparisons( bench )
if len(comparisons)>0:
    print 'You are about to make the following list of comparison pages obsolete. These pages will thus be removed:' 
    print comparisons

    answer = None
    while answer != 'y' and answer != 'n':
        answer = raw_input('do you agree? [y/n]')

        if answer == 'n':
            sys.exit(0)

# check that the user can write in the website
website.writeAccess()
bench.makeRelease( website )


if bench.exists( website ) == True:
    if options.force == False:
        print 'please use the -e option to choose another extension'
        print '  e.g: submit.py -e Feb10'
        print 'or force it.'
        sys.exit(1)
    else:
        print 'overwriting...'
        shutil.rmtree(bench.benchmarkOnWebSite(website))


# local benchmark. this one does not have an extension!

shutil.copytree(localBench.fullName(), bench.benchmarkOnWebSite(website) )
print 'done. Access your benchmark here:'
print bench.benchmarkUrl( website )
    
# removing comparisons
# COMPARISONS COULD ALSO BE REDONE. 
for comparison in comparisons:
    rm = 'rm -rf '+comparison
    os.system(rm)
    
