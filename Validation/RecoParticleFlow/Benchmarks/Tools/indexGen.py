#!/bin/env python


import shutil, sys, os

from string import Template

from optparse import OptionParser


def testFileType( file, ext ):
     if os.path.isfile( file ) == False:
          print '%s is not a file' % file
          sys.exit(2)
     
     (fileroot, fileext) = os.path.splitext( file )
     if fileext != ext:
          print '%s does not end with %s' % (file, ext) 
          sys.exit(3)


parser = OptionParser()
parser.usage = "usage: %prog <dir with plots> <py generator config> <py sim config> <by benchmark config> <.C plot macro> <template>"


parser.add_option("-t", "--title", dest="title",
                  help="Benchmark title",
                  default="")

(options,args) = parser.parse_args()
 
if len(args)!=6:
    parser.print_help()
    sys.exit(1)


dirPlots = args[0]
genConfig = args[1]
simConfig = args[2]
benchmarkConfig = args[3]
macro = args[4]
template = args[5]

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
testFileType(template, ".html")


title = os.path.basename( os.getcwd() ) 
outputDir = title

if os.path.isdir( outputDir ):
     print outputDir, "already exists"
     sys.exit(3)
else:
     os.mkdir( outputDir )


ifile = open( template )
indexTemplate = ifile.read()

for file in ( genConfig, simConfig, benchmarkConfig, macro):
     shutil.copy(file, outputDir) 
   

comments = 'no comment'

imgTemplate = '<IMG src="%s" width="500" align="left" border="0">'
images = ''
for pic in pictures:
    img = imgTemplate % os.path.basename(pic)
    #print img
    images = "%s\t%s\n" % (images, img)
    shutil.copy(pic, outputDir) 

#print images 

s = Template(indexTemplate)
subst = s.substitute(title = title, 
                     genConfig = os.path.basename(genConfig),
                     images = images, 
                     simConfig = os.path.basename(simConfig),
                     benchmarkConfig = os.path.basename(benchmarkConfig),
                     macro =  os.path.basename(macro), 
                     comments = comments,
                     cmssw = cmssw,
                     showTags = showTags
                     )



ofile = open( '%s/index.html' % outputDir, 'w' )
ofile.write( subst )

