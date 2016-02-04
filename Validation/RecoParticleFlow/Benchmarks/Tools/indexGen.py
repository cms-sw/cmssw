#!/usr/bin/env python


import shutil, sys, os, re, valtools

from string import Template

from optparse import OptionParser



webpage = valtools.webpage()

webpage.parser_.usage = "usage: %prog. Run from your Benchmark directory"
webpage.parser_.add_option("-r", "--recipe", dest="recipe",
                           help="url pointing to a recipe",
                           default="None")
webpage.parser_.add_option("-t", "--title", dest="title",
                           help="Benchmark title",
                           default="")
webpage.parser_.add_option("-g", "--gensource", dest="pyGenSource",
                           help="python file for the CMSSW source of the generated events, which is used in input to your simulation and reconstruction process",
                           default="None")
webpage.parser_.add_option("-s", "--simulation", dest="pySim",
                           help="python file for your CMSSW simulation and/or reconstruction process.",
                           default="None")
webpage.parseArgs()


if len(webpage.args_)!=0:
    webpage.parser_.print_help()
    sys.exit(1)

benchmark = valtools.benchmark()

webpage.setOutputDir( benchmark.fullName() )

macro = 'plot.C'
recipe = webpage.options_.recipe
genConfig = webpage.options_.pyGenSource
simConfig = webpage.options_.pySim
benchmarkConfig = 'benchmark_cfg.py'

# information about CMSSW
cmssw = os.environ['CMSSW_VERSION']
showTags = os.popen( 'showtags -t -r -u').read()

title = webpage.benchmarkName_

templateFile = 'index.html'

outputDir = webpage.outputDir_
indexhtml = "%s/%s" % (webpage.templates_,templateFile)


valtools.testFileType(genConfig, ".py")
valtools.testFileType(simConfig, ".py")
valtools.testFileType(benchmarkConfig, ".py")
valtools.testFileType(macro, ".C")

valtools.testFileType(indexhtml, ".html")
infonotfoundhtml = "%s/%s" % (webpage.templates_,"infoNotFound.html")
valtools.testFileType(infonotfoundhtml, ".html")


recipeLink = valtools.processFile( recipe, webpage.outputDir_ )
genConfigLink = valtools.processFile(genConfig, webpage.outputDir_  )
simConfigLink = valtools.processFile( simConfig, webpage.outputDir_ )
benchmarkConfigLink = valtools.processFile( benchmarkConfig,
                                            webpage.outputDir_ )
macroLink = valtools.processFile(macro, webpage.outputDir_  )
rootFileLink = valtools.processFile(webpage.rootFile_, webpage.outputDir_  )

comments = webpage.options_.comments
images = webpage.readCaptions('captions.txt')

ifile = open( indexhtml )
indexTemplate = ifile.read()
s = Template(indexTemplate)
subst = s.substitute(title = title,
                     recipe = recipe,
                     recipeLink = recipeLink,
                     genConfig = os.path.basename(genConfig),
                     genConfigLink = genConfigLink,
                     simConfig = os.path.basename(simConfig),
                     simConfigLink = simConfigLink,
                     benchmarkConfig = os.path.basename(benchmarkConfig),
                     benchmarkConfigLink = benchmarkConfigLink,
                     macro =  os.path.basename(macro), 
                     macroLink = macroLink,
                     rootFile =  os.path.basename(webpage.rootFile_), 
                     rootFileLink =  rootFileLink, 
                     comments = comments,
                     cmssw = cmssw,
                     showTags = showTags,
                     images = images, 
                     username = os.environ['USER'],
                     date = webpage.date_
                     )

ofile = open( '%s/index.html' % outputDir, 'w' )
ofile.write( subst )

ifile = open( infonotfoundhtml )
infoNotFoundTemplate = ifile.read()
s2 = Template(infoNotFoundTemplate)
subst2 = s2.substitute( username = os.environ['USER'] )
ofile2 = open( '%s/infoNotFound.html' % outputDir, 'w' )
ofile2.write( subst2 )

print 'webpage directory successfully created in', outputDir

