#! /usr/bin/env python
################################################################################
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
# $Author: anorkus $
# $Date: 2012/10/23 15:10:14 $
# $Revision: 1.5 $
#
#
# Danilo Piparo CERN - danilo.piparo@cern.ch
#
################################################################################


def getInfoFromFilename(filename):
  prefix,sample,cmssw_release,tier = filename[:-5].split("__")[:5]
  run=int(prefix.split("_")[-1][1:])
  return run,sample,cmssw_release,tier

from sys import argv,exit
import os

# Default Configuration Parameters ---------------------------------------------

stat_test="Chi2"
test_threshold=1e-5


#run="1"

dir_name=""
outdir_name=""

compare=False
report=False

do_pngs=False

black_list_str=""

#-------------------------------------------------------------------------------

from optparse import OptionParser

parser = OptionParser(usage="usage: %prog file1 file2 [options]")


#parser.add_option("-r","--run ",
                  #action="store",
                  #dest="run",
                  #default=run,
                  #help="The run to be checked \n(default is %s)" %run)

parser.add_option("-d","--dir_name",
                  action="store",
                  dest="dir_name",
                  default=dir_name,
                  help="The 'directory' to be checked in the DQM \n(default is %s)" %dir_name)

parser.add_option("-o","--outdir_name",
                  action="store",
                  dest="outdir_name",
                  default=outdir_name,
                  help="The directory where the output will be stored \n(default is %s)" %outdir_name)

parser.add_option("-p","--do_pngs",
                  action="store_true",
                  dest="do_pngs",
                  default=False,
                  help="Do the pngs of the comparison (takes 50%% of the total running time) \n(default is %s)" %False)

parser.add_option("--no_successes",
                  action="store_true",
                  dest="no_successes",
                  default=False,
                  help="Do not draw successes. Default is False.")

parser.add_option("-P","--pickle",
                  action="store",
                  dest="pklfile",
                  default="",
                  help="Pkl file of the dir structure ")

parser.add_option("--sample",
                  action="store",
                  dest="sample",
                  default="Sample",
                  help="The name of the sample to be displayed")

parser.add_option("--metas",
                  action="store",
                  dest="metas",
                  default="",
                  help="The Metas describing the two files (separated by @@@)")

parser.add_option("-t","--test_threshold",
                  action="store",
                  dest="test_threshold",
                  default=test_threshold,
                  help="Threshold for the statistical test \n(default is %s)" %test_threshold)

parser.add_option("-s","--stat_test",
                  action="store",
                  dest="stat_test",
                  default=stat_test,
                  help="Statistical test (KS or Chi2) \n(default is %s)" %stat_test)  

parser.add_option("-C","--compare",
                  action="store_true",
                  dest="compare",
                  default=compare,
                  help="Make the comparison \n(default is %s)" %compare)

parser.add_option("-R","--Report",
                  action="store_true",
                  dest="report",
                  default=report,
                  help="Make the html report \n(default is %s)" %report)

parser.add_option("--specify_run",
                  action="store_true",
                  dest="specify_run",
                  default=False,
                  help="Append the run number to the output dir for data")


parser.add_option("-B","--black_list",
                  action="store",
                  dest="black_list",
                  default=black_list_str,
                  help="Blacklist elements. form is name@hierarchy_level (i.e. HLT@1) \n(default is %s)" %black_list_str)
                  
##---HASHING---##
parser.add_option("--hash_name",
                  action="store_true",
                  dest="hash_name",
                  default=False,
                  help="Set if you want to minimize & hash the output HTML files.")
##--Blacklist File --##                  
parser.add_option("--use_black_file",
                  action="store_true",
                  dest="blacklist_file",
                  default=False,
                  help="Use a black list file of histograms located @ /RelMon/data")

def blackListedHistos():
        ##GET a black-list file of histograms##
    if os.environ.has_key("RELMON_SA"):
        black_list_file="../data/blacklist.txt"
    else:
        black_list_file="%s/src/Utilities/RelMon/data/blacklist.txt"%(os.environ["CMSSW_BASE"])
    bListFile = open(black_list_file,'r')
    black_listed_histograms = bListFile.read()
    bListFile.close()

    histogramArray = black_listed_histograms.split("\n")
    histogramArray.remove("")  #remove the last element which is empty line
    newarray = []
    for elem in histogramArray:
        tmp = elem.split("/")  #screw windows as it is being run on lxbuild machines with Linux
        tmp.insert(1,"Run summary")  #insert "Run summary" dir in path as in ROOT files they exists but user haven't defined them
        newarray.append(("/").join(tmp))
    return newarray
    ##------##
    
(options, args) = parser.parse_args()

if len(args)!=2 and options.compare:
  print "Wrong number of RootFiles specified (%s)" %len(args)
  print args
  
#-------------------------------------------------------------------------------
original_pickle_name=""
if options.compare:
  
  if os.environ.has_key("RELMON_SA"):
    import definitions  
    from dqm_interfaces import DirID,DirWalkerFile,string2blacklist
    from dirstructure import Directory
  else:
    import Utilities.RelMon.definitions as definitions  
    from Utilities.RelMon.dqm_interfaces import DirID,DirWalkerFile,string2blacklist
    from Utilities.RelMon.dirstructure import Directory

  import cPickle
  from os import mkdir,chdir,getcwd
  from os.path import exists

  #-------------------------------------------------------------------------------
  # Guess Releases and sample from filename
  rootfilename1,rootfilename2 = args

  run1=-1
  sample1=''
  cmssw_release1=''
  tier1=''
  run2=-1
  sample2=''
  cmssw_release2=''
  tier2=''

  if options.metas=='':
    run1,sample1,cmssw_release1,tier1= getInfoFromFilename(rootfilename1)
    run2,sample2,cmssw_release2,tier2= getInfoFromFilename(rootfilename2)
  else:
    print "Reading meta from commandline"
    sample1=sample2=options.sample
    cmssw_release1,cmssw_release2=options.metas.split('@@@')
    
  # check if the sample is the same
  if sample1!=sample2:
    print "I am puzzled. Did you choose two different samples?"
    exit(1)
  sample = sample1

  # check if the run is the same
  if run1!=run2:
    print "I am puzzled. Did you choose two different runs?"
#    exit(1)  
  run=run1

  fulldirname=options.outdir_name
  if len(fulldirname)==0:
    fulldirname=options.dir_name
  if len(fulldirname)==0:
    fulldirname="%s_%s_%s" %(sample1,cmssw_release1,cmssw_release2)


  black_list=string2blacklist(options.black_list)
  
  if options.blacklist_file:
    black_listed = blackListedHistos()
  else:
    black_listed = []
      
#-------------------------------------------------------------------------------

  print "Analysing Histograms located in directory %s at: " %options.dir_name
  for filename in rootfilename1,rootfilename2:
    print " o %s" %filename


  if len(black_list)>0:
    print "We have a Blacklist:"
    for dirid in black_list:
      print " o %s" %dirid

  # Set up the fake directory structure
  directory=Directory(options.dir_name)
  dirwalker=DirWalkerFile(fulldirname,
                          options.dir_name,
                          rootfilename1,rootfilename2,
                          run,
                          black_list,
                          options.stat_test,
                          options.test_threshold,
                          not options.no_successes,
                          options.do_pngs,
                          set(black_listed)
                          )
                          
  # Start the walker
  outdir_name=options.outdir_name
  if run>1 and options.specify_run:
    outdir_name+="_%s" %run
    fulldirname+="_%s" %run
  print "+"*30
  print "Output Directory will be ", outdir_name
  options.outdir_name=outdir_name
  if not exists(outdir_name) and len(outdir_name )>0:
    mkdir(outdir_name)
  if len(outdir_name)>0:
    chdir(outdir_name)
  dirwalker.walk()

  run =  dirwalker.run


  # Fetch the directory from the walker
  directory=dirwalker.directory

  # Set some meta for the page generation
  directory.meta.sample=sample
  directory.meta.run1=run1
  directory.meta.run2=run2
  directory.meta.release1=cmssw_release1
  directory.meta.release2=cmssw_release2
  directory.meta.tier1=tier1
  directory.meta.tier2=tier2

  # Print a summary Report on screen
  directory.print_report(verbose=True)

  # Remove this DQM FW reminescence.
  directory.prune("Run summary")

  # Dump the directory structure on disk in a pickle
  original_pickle_name="%s.pkl" %fulldirname
  print "Pickleing the directory as %s in dir %s" %(original_pickle_name,getcwd())
  output = open(original_pickle_name,"w")
  cPickle.dump(directory, output, -1)# use highest protocol available for the pickle
  output.close()

#-------------------------------------------------------------------------------
if options.report:
  
  if os.environ.has_key("RELMON_SA"):
    from directories2html import directory2html
    from dirstructure import Directory
  else:
    from Utilities.RelMon.directories2html import directory2html
    from Utilities.RelMon.dirstructure import Directory

  from os.path import exists
  from os import chdir,mkdir
  import os
  import cPickle    
  
  pickle_name=options.pklfile
  if len(options.pklfile)==0:
    pickle_name=original_pickle_name  

  print "Reading directory from %s" %(pickle_name)
  ifile=open(pickle_name,"rb")
  directory=cPickle.load(ifile)
  ifile.close()

  if not options.compare:
    if not os.path.exists(options.outdir_name):
      mkdir(options.outdir_name)

  if exists(options.outdir_name) and len(directory.name)==0:
    chdir(options.outdir_name)
  
  # Calculate the results of the tests for each directory
  print "Calculating stats for the directory..."
  directory.calcStats()
  
  print "Producing html..."
  directory2html(directory, options.hash_name)

if not (options.report or options.compare):
  print "Neither comparison nor report to be executed. A typo?"


