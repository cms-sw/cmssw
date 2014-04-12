#!/usr/bin/env python


import shutil, sys, os, re, valtools, string

from string import Template

from optparse import OptionParser 
from subprocess import Popen,PIPE


def processBenchmark( path, outputRootFile ):
    (release, bname, extension ) = valtools.decodePath( path )
    if bname != webpage.benchmarkName_:
        print "sorry, you have to go to the",bname,"directory to produce this comparison. Note that you cannot compare different benchmarks."
        sys.exit(4)
    benchmark = valtools.benchmark( extension )
    benchmark.release_ = release
    print benchmark.benchmarkUrl( website ) 
    root = benchmark.rootFileOnWebSite( website )
    shutil.copy(root, outputRootFile)
    print 'retrieved ', root
    return benchmark
    

webpage = valtools.webpage()
webpage.parser_.usage = "example: %prog CMSSW_3_1_0_pre7/TauBenchmarkGeneric_ZTT_FastSim_IDEAL CMSSW_3_1_0_pre7/TauBenchmarkGeneric_TEST\nThe list of benchmarks can be obtained using the listBenchmarks.py command."
webpage.parser_.add_option("-m", "--macro", dest="macro",
                           help="specify the ROOT macro to be used for comparison. If empty, skip the plotting stage", default="compare.C")
webpage.parser_.add_option("-s", "--submit", dest="submit",
                           action="store_true",
                           help="submit the comparison to the web site",
                           default=False)
webpage.parser_.add_option("-S", "--submit-force", dest="submitForce",
                           action="store_true",
                           help="force the submission of the comparison to the web site",
                           default=False)
webpage.parseArgs()

if len(webpage.args_)!=2:
    webpage.parser_.print_help()
    sys.exit(1)


website = valtools.website()

macro = webpage.options_.macro
templateFile = 'indexCompare.html'
indexhtml = "%s/%s" % (webpage.templates_,templateFile)

# setting up benchmarks
print 
benchmark1 = processBenchmark( webpage.args_[0],
                               'benchmark_0.root' )
print
benchmark2 = processBenchmark( webpage.args_[1],
                               'benchmark_1.root' )


webpage.setOutputDir(benchmark2.fullName())

# do the plots
if webpage.options_.macro != "":
    os.system('root -b ' + macro)

valtools.testFileType(indexhtml, ".html")
infonotfoundhtml = "%s/%s" % (webpage.templates_,"infoNotFound.html")
valtools.testFileType(infonotfoundhtml, ".html")

images = webpage.readCaptions('c_captions.txt')

title = webpage.benchmarkName_

benchmark1Link = benchmark1.benchmarkUrl( website )
benchmark1Name = benchmark1.fullName()

benchmark2Link = benchmark2.benchmarkUrl( website )
benchmark2Name = benchmark2.fullName()

macroLink = valtools.processFile( macro, webpage.outputDir_  )
macroName = os.path.basename(macro)

comments = webpage.options_.comments
username = os.environ['USER']

ifile = open( indexhtml )
indexTemplate = ifile.read()
s = Template(indexTemplate)
subst = s.substitute(title = title,
                     benchmark1Link = benchmark1Link,
                     benchmark1Name = benchmark1Name,
                     benchmark2Link = benchmark2Link,
                     benchmark2Name = benchmark2Name,
                     macroLink = macroLink,
                     macroName = macroName, 
                     comments = comments,
                     images = images, 
                     username = username,
                     date = webpage.date_
                     )
ofile = open( '%s/index.html' % webpage.outputDir_, 'w' )
ofile.write( subst )
ofile.close()

ifile = open( infonotfoundhtml )
infoNotFoundTemplate = ifile.read()
s2 = Template(infoNotFoundTemplate)
subst2 = s2.substitute( username = os.environ['USER'] )
ofile2 = open( '%s/infoNotFound.html' % webpage.outputDir_, 'w' )
ofile2.write( subst2 )
ofile2.close()

# if submission is forced, it means that the user does want
# to submit. 
if  webpage.options_.submitForce:
    webpage.options_.submit = True

if (webpage.options_.submit == True):
    remoteName = benchmark1.benchmarkOnWebSite(website) + '/' + webpage.outputDir_
    comparison = valtools.comparison( benchmark1, webpage.outputDir_)
    comparison.submit( website,
                       webpage.options_.submitForce)
    benchmark1.addLinkToComparison( website, comparison )
    benchmark2.addLinkToComparison( website, comparison )

    
