#!/usr/bin/env python


import shutil, sys, os, re

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
parser.usage = "usage: %prog"

parser.add_option("-e", "--extension", dest="extension",
                  help="adds an extension to the name of this benchmark",
                  default=None)


(options,args) = parser.parse_args()
 
if len(args)!=0:
    parser.print_help()
    sys.exit(1)


website = '/afs/cern.ch/cms/Physics/particleflow/Validation/cms-project-pflow-validation/Releases'
release = os.environ['CMSSW_VERSION']
benchmark = os.path.basename( os.getcwd() ) 
releaseOnWebSite = '%s/%s' % (website, release)
benchmarkOnWebSite = '%s/%s'  % (releaseOnWebSite, benchmark)

if( options.extension != None ):
     benchmarkOnWebSite = '%s_%s' % (benchmarkOnWebSite, options.extension)

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
     print 'done'
     sys.exit(0)



