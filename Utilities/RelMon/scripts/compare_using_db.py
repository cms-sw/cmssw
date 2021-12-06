#! /usr/bin/env python3
################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
#
#                                                                              
# Danilo Piparo CERN - danilo.piparo@cern.ch                                   
#                                                                              
################################################################################

from __future__ import print_function
from sys import argv,exit
from optparse import OptionParser
import cPickle
import os

# Default Configuration Parameters ---------------------------------------------
dqm_server='https://cmsweb.cern.ch/dqm/relval'

cmssw_release1="CMSSW_5_3_0-START53_V4-v1" 
cmssw_release2="CMSSW_5_3_1-START53_V5-v1"

stat_test="Chi2"
test_threshold=0.00001

sample = "RelValZMM"

run1="1"
run2="1"

dir_name="00 Shift"
outdir_name=""

do_pngs=False

compare=False
report=False

black_list_str=""

tiers="DQM,DQM"

#-------------------------------------------------------------------------------

parser = OptionParser(usage="usage: %prog [options]")

parser.add_option("-1","--release1",
                  action="store",
                  dest="cmssw_release1",
                  default=cmssw_release1,
                  help="The main CMSSW release \n(default is %s)" %cmssw_release1)

parser.add_option("-2","--release2",
                  action="store",
                  dest="cmssw_release2",
                  default=cmssw_release2,
                  help="The CMSSW release for the regression \n(default is %s)" %cmssw_release2)

parser.add_option("-S","--sample",
                  action="store",
                  dest="sample",
                  default=sample,
                  help="The Sample upon which you want to run \n(default is %s)" %sample)

parser.add_option("-o","--outdir_name",
                  action="store",
                  dest="outdir_name",
                  default=outdir_name,
                  help="The directory where the output will be stored \n(default is %s)" %outdir_name)

parser.add_option("-D","--dqm_server",
                  action="store",
                  dest="dqm_server",
                  default=dqm_server,
                  help="The DQM server \n(default is %s)" %dqm_server)

parser.add_option("-a","--run1 ",
                  action="store",
                  dest="run1",
                  default=run1,
                  help="The run of the first sample to be checked \n(default is %s)" %run1)

parser.add_option("-b","--run2",
                  action="store",
                  dest="run2",
                  default=run2,
                  help="The run of the second sample to be checked \n(default is %s)" %run2)

parser.add_option("-d","--dir_name",
                  action="store",
                  dest="dir_name",
                  default=dir_name,
                  help="The 'directory' to be checked in the DQM \n(default is %s)" %dir_name)

parser.add_option("-p","--do_pngs",
                  action="store_true",
                  dest="do_pngs",
                  default=False,
                  help="EXPERIMENTAL!!! Do the pngs of the comparison (takes 50%% of the total running time) \n(default is %s)" %False)

parser.add_option("-P","--pickle",
                  action="store",
                  dest="pklfile",
                  default="",
                  help="Pkl file of the dir structure ")
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

parser.add_option("-T","--Tiers",
                  action="store",
                  dest="tiers",
                  default=tiers,
                  help="Data tiers (comma separated list) \n(default is %s)" %tiers)         

parser.add_option("-B","--black_list",
                  action="store",
                  dest="black_list",
                  default=black_list_str,
                  help="Blacklist elements. form is name@hierarchy_level (i.e. HLT@1) \n(default is %s)" %black_list_str)                     

(options, args) = parser.parse_args()

#-------------------------------------------------------------------------------
original_pickle_name=""
if options.compare:

  if "RELMON_SA" in os.environ:
    from dqm_interfaces import DirID,DQMcommunicator,DirWalkerDB
    from dirstructure import Directory
  else:  
    from Utilities.RelMon.dqm_interfaces import DirID,DQMcommunicator,DirWalkerDB
    from Utilities.RelMon.dirstructure import Directory


  # Pre-process the inputs
  fulldirname=options.outdir_name
  if len(fulldirname)==0:
    fulldirname=options.dir_name
  if len(fulldirname)==0:
    fulldirname="%s_%s_%s" %(sample1,cmssw_release1,cmssw_release2)
  

  black_list=[]
  black_list_str=options.black_list
  if len(black_list_str)>0:
    for ele in black_list_str.split(","):
      dirname,level=ele.split("@")
      level=int(level)
      black_list.append(DirID(dirname,level))

  db_base_url="/data/json/archive/"
  base1="%s/%s/%s/%s/DQM/" %(db_base_url,options.run1,options.sample,options.cmssw_release1)
  base2="%s/%s/%s/%s/DQM/" %(db_base_url,options.run2,options.sample,options.cmssw_release2)


  print("Analysing Histograms located in directory %s at: " %options.dir_name)
  for base in base1,base2:
    print(" o %s (server= %s)" %(base,options.dqm_server))

  # Set up the communicators
  comm1 = DQMcommunicator(server=options.dqm_server)
  comm2 = DQMcommunicator(server=options.dqm_server)

  # Set up the fake directory structure
  directory=Directory(options.dir_name)
  dirwalker=DirWalkerDB(comm1,comm2,base1,base2,directory)
  
  # Set the production of pngs on and off
  dirwalker.do_pngs=options.do_pngs
  
  # set the stat test
  dirwalker.stat_test=options.stat_test
  dirwalker.test_threshold=options.test_threshold

  # Set the blacklist, if needed
  if len(black_list)>0:
    print("We have a Blacklist:")
    for dirid in black_list:
      print(" o %s" %dirid)
    dirwalker.black_list=black_list

  # Start the walker
  if not os.path.exists(options.outdir_name) and len(options.outdir_name )>0:
    os.mkdir(options.outdir_name)
  if len(options.outdir_name)>0:
    os.chdir(options.outdir_name)

  # Since the walker is a thread, run it!
  dirwalker.start()
  # And wait until it is finished :)
  dirwalker.join()

  # Fetch the directory from the walker
  directory=dirwalker.directory

  # Set some meta for the page generation
  directory.meta.sample=options.sample
  directory.meta.run1=options.run1
  directory.meta.run2=options.run2
  directory.meta.release1=options.cmssw_release1
  directory.meta.release2=options.cmssw_release2
    
  directory.meta.tier1,directory.meta.tier2 = options.tiers.split(",")
  
  # Print a summary Report on screen
  directory.print_report()

  # Dump the directory structure on disk in a pickle
  original_pickle_name="%s.pkl" %fulldirname
  print("Pickleing the directory as %s in dir %s" %(original_pickle_name,os.getcwd()))
  output = open(original_pickle_name,"w")
  cPickle.dump(directory, output, -1)# use highest protocol available for the pickle
  output.close()

#-------------------------------------------------------------------------------
if options.report:
  if "RELMON_SA" in os.environ:  
    from directories2html import directory2html
    from dirstructure import Directory
  else:
    from Utilities.RelMon.directories2html import directory2html
    from Utilities.RelMon.dirstructure import Directory    
  
  pickle_name=options.pklfile
  if len(options.pklfile)==0:
    pickle_name=original_pickle_name  

  print("Reading directory from %s" %(pickle_name))
  ifile=open(pickle_name,"rb")
  directory=cPickle.load(ifile)
  ifile.close()

  if os.path.exists(options.outdir_name) and len(directory.name)==0:
    os.chdir(options.outdir_name)
  
  # Calculate the results of the tests for each directory
  print("Calculating stats for the directory...")
  directory.calcStats()
  
  print("Producing html...")
  directory2html(directory)
  
if not (options.report or options.compare):
  print("Neither comparison nor report to be executed. A typo?")

