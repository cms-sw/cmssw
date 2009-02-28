#!/usr/bin/env python


import shutil, sys, os, re, valtools

from string import Template

from optparse import OptionParser


def testFileType( file, ext ):

     if file == "None":
          return
     
     if os.path.isfile( file ) == False:
          print '%s is not a file' % file
          sys.exit(2)
     
     (fileroot, fileext) = os.path.splitext( file )
     if fileext != ext:
          print '%s does not end with %s' % (file, ext) 
          sys.exit(3)

def processFile( file, outputDir ):
 
     if file == "None":
          return 'infoNotFound.html'
     else:
          if os.path.isfile(file):
               shutil.copy(file, outputDir)
               return os.path.basename(file)
          else:
               return file
     

def readCaption( line ):

     if( re.compile('^\s*$').match(line) ):
          raise Exception
          
     p = re.compile('^\s*(\S+)\s*\"(.*)\"');
     m = p.match(line)
     if m:
          pic = m.group(1)
          caption = m.group(2)
          return (pic, caption)
     else:
          print 'bad caption format: "%s"' % line
          raise Exception

parser = OptionParser()
parser.usage = "usage: %prog. Run from your Benchmark directory"


parser.add_option("-c", "--compare", dest="compare",
                  help="name of another release",
                  default=None)
parser.add_option("-r", "--recipe", dest="recipe",
                  help="url pointing to a recipe",
                  default="None")
parser.add_option("-t", "--title", dest="title",
                  help="Benchmark title",
                  default="")
parser.add_option("-g", "--gensource", dest="pyGenSource",
                  help="python file for the CMSSW source of the generated events, which is used in input to your simulation and reconstruction process",
                  default="None")

parser.add_option("-s", "--simulation", dest="pySim",
                  help="python file for your CMSSW simulation and/or reconstruction process.",
                  default="None")

#parser.add_option("-b", "--benchmark", dest="pyBenchmark",
#                  help="python file for the production of the benchmark root files",
#                  default="None")

#parser.add_option("-m", "--macro", dest="macro",
#                  help="root macro used for the benchmark plots",
#                  default="None")

# the benchmark root file is NOT an option!
rootFile = 'benchmark.root'

(options,args) = parser.parse_args()
 
if len(args)!=0:
    parser.print_help()
    sys.exit(1)

benchCompare = None
dirPlots = './'
templates = '../Tools/templates'
recipe = options.recipe
genConfig = options.pyGenSource
simConfig = options.pySim
benchmarkConfig = 'benchmark_cfg.py'
date =  os.popen( 'date' ).read()

# information about CMSSW
cmssw = os.environ['CMSSW_VERSION']
#print cmssw

showTags = os.popen( 'showtags -t -r -u').read()
#print showTags

macro = 'plot.C'
captions = 'captions.txt'
benchmarkName = os.path.basename( os.getcwd() ) 
title = benchmarkName
templateFile = 'index.html'
compareLine = None
if options.compare != None:
     title = '%s_%s_VS_%s' %  ( benchmarkName, cmssw, options.compare)
     print title
     macro = 'compare.C'
     captions = 'c_captions.txt'
     website = valtools.website()
     benchCompare = valtools.benchmark( None, options.compare) 
     compareLine = '<b>Compared with</b> <a href="%s">%s</a>'%(benchCompare.benchmarkUrl( website ), options.compare) 

outputDir = title
indexhtml = "%s/%s" % (templates,templateFile)


# get the pictures
pictures = []
if os.path.isdir(dirPlots): 
     #print 'getting pictures from %s' % dirPlots
     tmppictures = os.listdir( dirPlots )
     #print tmppictures
     for pic in tmppictures:
          (root, ext) = os.path.splitext(pic)
          if ext == '.jpg' or ext == '.png' or ext == '.gif':
               pictures.append( '%s/%s' % (dirPlots,pic) )
else:
     sys.exit(2)

pictures.sort()
#print 'pictures: ', pictures

testFileType(genConfig, ".py")
testFileType(simConfig, ".py")
testFileType(benchmarkConfig, ".py")
testFileType(macro, ".C")


testFileType(indexhtml, ".html")
infonotfoundhtml = "%s/%s" % (templates,"infoNotFound.html")
testFileType(infonotfoundhtml, ".html")


if os.path.isdir( outputDir ):
     print outputDir, "already exists"
     sys.exit(3)
else:
     os.mkdir( outputDir )


recipeLink = processFile( recipe, outputDir )
genConfigLink = processFile(genConfig, outputDir  )
simConfigLink = processFile( simConfig, outputDir )
benchmarkConfigLink = processFile( benchmarkConfig, outputDir )
macroLink = processFile(macro, outputDir  )
rootFileLink = processFile(rootFile, outputDir  )

comments = 'no comment'


imgTemplate = '<IMG src="%s" width="500" align="left" border="0"><br clear="ALL">'
images = ''

# open legend file

captionsContents = open( captions )
for line in captionsContents:
     try:
          (picfile, caption) = readCaption( line )
          img = imgTemplate % os.path.basename(picfile)
          images = "%s<h3>%s:</h3>\n%s\n" % (images, caption, img)
          # what to do if the file's not there? 
          # : print a warning
          shutil.copy(picfile, outputDir) 
     except Exception:
          raise

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
                     rootFile =  os.path.basename(rootFile), 
                     rootFileLink =  rootFileLink, 
                     comments = comments,
                     cmssw = cmssw,
                     showTags = showTags,
                     compareLine = compareLine,
                     images = images, 
                     username = os.environ['USER'],
                     date = date
                     )
ofile = open( '%s/index.html' % outputDir, 'w' )
ofile.write( subst )


ifile = open( infonotfoundhtml )
infoNotFoundTemplate = ifile.read()
s2 = Template(infoNotFoundTemplate)
subst2 = s2.substitute( username = os.environ['USER'] )
ofile2 = open( '%s/infoNotFound.html' % outputDir, 'w' )
ofile2.write( subst2 )



