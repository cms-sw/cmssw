import os
import sys
import glob
from string import Template


validationToolsDir = os.environ['VALTOOLS']
validationTestDir = os.environ['VALTEST']

indexhtml = os.path.join(validationToolsDir, "templates", "index.html")

def CheckFile(myFile):
   if not os.path.isfile(myFile):
      raise IOError, "Error! Can't stat %s!" % myFile
   else:
      return myFile

# Get our path relative to the Validation/RecoTau/test directory
#title = os.path.relpath(os.cwd(), validationTestDir) python > 2.6
currentpath = os.path.abspath(os.getcwd())
title = currentpath.replace(os.path.abspath(validationTestDir), '')

#Get our current release
cmssw = os.environ['CMSSW_VERSION']

#Get current date
date =  os.popen( 'date' ).read()

#Our showtags
showTags = os.popen("cat %s" % CheckFile('Config/showtags.txt')).read()

#Get the diffs to the release
difftoreleaselink = CheckFile('Config/diffToVanillaRelease.patch')
difftorelease = "cvs diff -r %s" % cmssw

difftotagslink = CheckFile('Config/diffToTags.patch')
difftotags     = 'cvs diff'

cfgdumplink    = 'Config/cfgDump.py'
cfgdump        = 'cfgDump.py'
if not os.path.isfile(cfgdumplink):
   cfgdumplink = 'Config/cfgDump_0.py'
   cfgdump        = 'cfgDump_0.py (batch job)'
if os.path.isfile('Config/crab.cfg') and not os.path.isfile(cfgdumplink):
   cfgdumplink = 'Config/crab.cfg'
   cfgdump     = 'crab.cfg (grid job)'
else:
   print 'Did you forget to copy the crab.cfg in Config/ ?'

print cfgdumplink
CheckFile(cfgdumplink)

if title.find('fastsim') != -1:
   genConfigLink = cfgdumplink
   genConfig     = "fastsim (see cfgDump)"

elif title.find('fullsim') != -1:
   genConfigLink = cfgdumplink
   genConfig     = "fullsim (see cfgDump)"
else:
   genConfigLink = "Config/DataSource_cff.py"
   genConfig     = "DataSource_cff.py"

rootFiles = glob.glob("*root")
if len(rootFiles) != 1:
   print "There must be one, and only one root file in the directory to correctly build the webpage!"
   sys.exit()

rootFileLink = os.path.basename(rootFiles[0])
rootFile=rootFileLink

#imgTemplate = '<IMG src="%s" width="500" align="left" border="0"><br clear="ALL">'
imgTemplate = '<IMG src="%s" width="500" align="left" border="0">'
fourImages       = '<table style="text-align: left; " border="1" cellpadding="2" cellspacing="0">\n\
                     <tbody>\n\
                     <tr>\n\
                     <td style="width: 350px;"><IMG src="%s" width="350" align="left" border="0"></td>\
                     <td style="width: 350px;"><IMG src="%s" width="350" align="left" border="0"></td>\
                     <td style="width: 350px;"><IMG src="%s" width="350" align="left" border="0"></td>\
                     <td style="width: 350px;"><IMG src="%s" width="350" align="left" border="0"></td>\
                     </tr></tbody></table>\n\
                     '

images = ''

# Get summary plots
StepByStepPlotDirectories = filter(os.path.isdir, glob.glob("SummaryPlots/*"))
StepByStepPlots = []
for aDir in StepByStepPlotDirectories:
   producerName = os.path.basename(aDir)
   images += "<hr><h3>" + producerName + "</h3>\n"
   # get the plots
   getByVar = lambda x: glob.glob(os.path.join(aDir, '*%s.png' % x))[0]
   images += fourImages % (getByVar("pt"), getByVar("eta"), getByVar("energy"), getByVar("phi"))

# open legend file
captionsContents = ""
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

comments = ""

ComparisionDirectories = filter(os.path.isdir, glob.glob("ComparedTo*"))
comparisonlinks = ""
for aComp in ComparisionDirectories:
   comparisonlinks += '<h3>%s <a href="%s">(config)</a></h3>\n\n' % (aComp.replace('ComparedTo', 'Compared to '), os.path.join(aComp, "ReferenceData"))
   comparisonlinks += "<ul>\n"
   for anHtmlFile in filter(os.path.isfile, glob.glob(os.path.join(aComp, "Plots/*.html"))):
      comparisonlinks += '<li> <a href="%s">%s</a> </li>' % (anHtmlFile, os.path.basename(anHtmlFile).replace(".html",""))
   comparisonlinks += "</ul>\n"

ifile = open( indexhtml )
indexTemplate = ifile.read()

s = Template(indexTemplate)
subst = s.substitute(title = title,
                     difftotagslink=difftotagslink,
                     difftotags=difftotags,
                     difftoreleaselink=difftoreleaselink,
                     difftorelease=difftorelease,
                     cfgdumplink=cfgdumplink,
                     cfgdump=cfgdump,
                     comparisonlinks=comparisonlinks,
                     genConfig = os.path.basename(genConfig),
                     genConfigLink = genConfigLink,
                     rootFile =  os.path.basename(rootFile), 
                     rootFileLink =  rootFileLink, 
                     comments = comments,
                     cmssw = cmssw,
                     showTags = showTags,
                     images = images, 
                     username = os.environ['USER'],
                     date = date
                     )
ofile = open( 'index.html', 'w' )
ofile.write( subst )
