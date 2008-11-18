#!/bin/env python


import shutil, sys, os

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
     
     


parser = OptionParser()
parser.usage = "usage: %prog <dir with plots> <template>"

parser.add_option("-r", "--recipe", dest="recipe",
                  help="url pointing to a recipe",
                  default="None")
parser.add_option("-t", "--title", dest="title",
                  help="Benchmark title",
                  default="")
parser.add_option("-g", "--gensource", dest="pyGenSource",
                  help="python file for the source of the generated events",
                  default="None")

parser.add_option("-s", "--simulation", dest="pySim",
                  help="python file for the simulation",
                  default="None")

parser.add_option("-b", "--benchmark", dest="pyBenchmark",
                  help="python file for the production of the benchmark root files",
                  default="None")

parser.add_option("-m", "--macro", dest="macro",
                  help="root macro used for the benchmark plots",
                  default="None")




(options,args) = parser.parse_args()
 
if len(args)!=2:
    parser.print_help()
    sys.exit(1)


dirPlots = args[0]
templates = args[1]

recipe = options.recipe
genConfig = options.pyGenSource
simConfig = options.pySim
benchmarkConfig = options.pyBenchmark
macro = options.macro

# information about CMSSW
cmssw = os.environ['CMSSW_VERSION']
#print cmssw

showTags = os.popen( 'showtags -t -r').read()
#print showTags

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

indexhtml = "%s/%s" % (templates,"index.html")
testFileType(indexhtml, ".html")
infonotfoundhtml = "%s/%s" % (templates,"infoNotFound.html")
testFileType(infonotfoundhtml, ".html")


title = os.path.basename( os.getcwd() ) 
outputDir = title

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

comments = 'no comment'

imgTemplate = '<IMG src="%s" width="500" align="left" border="0">'
images = ''
for pic in pictures:
    img = imgTemplate % os.path.basename(pic)
    #print img
    images = "%s\t%s\n" % (images, img)
    shutil.copy(pic, outputDir) 




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
                     comments = comments,
                     cmssw = cmssw,
                     showTags = showTags,
                     images = images, 
                     )
ofile = open( '%s/index.html' % outputDir, 'w' )
ofile.write( subst )


ifile = open( infonotfoundhtml )
infoNotFoundTemplate = ifile.read()
s2 = Template(infoNotFoundTemplate)
subst2 = s2.substitute( username = os.environ['USER'] )
ofile2 = open( '%s/infoNotFound.html' % outputDir, 'w' )
ofile2.write( subst2 )



