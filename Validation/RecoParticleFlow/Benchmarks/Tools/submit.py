#!/usr/bin/env python
# to submit a benchmark webpage to the validation website
# author: Colin

import shutil, sys, os

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


website = '/afs/cern.ch/cms/Physics/particleflow/Validation/cms-project-pfvalidation/Releases'
url = 'http://cern.ch/pfvalidation/Releases'
release = os.environ['CMSSW_VERSION']
benchmark = os.path.basename( os.getcwd() )
benchmarkWithExt = benchmark
if( options.extension != None ):
     benchmarkWithExt = '%s_%s' % (benchmark, options.extension)

releaseOnWebSite = '%s/%s' % (website, release)
benchmarkOnWebSite = '%s/%s'  % (releaseOnWebSite, benchmarkWithExt)
releaseUrl = '%s/%s' % (url, release)
benchmarkUrl = '%s/%s'  % (releaseUrl, benchmarkWithExt)

print 'submitting benchmark:'
print benchmarkOnWebSite


# check that the user can write in the website

if( os.access(website, os.W_OK)==False ):
     print 'cannot write to the website. Please ask Colin to give you access.'
     sys.exit(1)

if( os.path.isdir( releaseOnWebSite )==False):
     print 'creating release %s' % release
     print releaseOnWebSite
     os.mkdir( releaseOnWebSite )

if( os.path.isdir( benchmarkOnWebSite )):
     print 'benchmark %s already exists for release %s' % (benchmark, release)
     print 'please use the -e option to add an extention, or choose another extension'
     print '  e.g: submit.py -e Feb10'
else:
     shutil.copytree(benchmark, benchmarkOnWebSite) 
     print 'done. Access your benchmark here:'
     print benchmarkUrl
     sys.exit(0)



