#! /usr/bin/env python
################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
# $Author: dpiparo $
# $Date: 2012/07/24 12:31:47 $
# $Revision: 1.6 $
#
#                                                                              
# Danilo Piparo CERN - danilo.piparo@cern.ch                                   
#                                                                              
################################################################################

from optparse import OptionParser

import os
import cPickle
import glob
from re import search
from subprocess import call,PIPE
from multiprocessing import Pool
from sys import exit

import sys
argv=sys.argv
sys.argv=[]
if os.environ.has_key("RELMON_SA"):
  import definitions as definitions
  from dqm_interfaces import DirWalkerFile,string2blacklist,DirWalkerFile_thread_wrapper
  from dirstructure import Directory
  from directories2html import directory2html,make_summary_table
  from utils import ask_ok, unpickler, make_files_pairs
else:
  import Utilities.RelMon.definitions as definitions
  from Utilities.RelMon.dqm_interfaces import DirWalkerFile,string2blacklist,DirWalkerFile_thread_wrapper
  from Utilities.RelMon.dirstructure import Directory
  from Utilities.RelMon.directories2html import directory2html,make_summary_table
  from Utilities.RelMon.utils import ask_ok, unpickler, make_files_pairs
sys.argv=argv

#-------------------------------------------------------------------------------

def name2sample(filename):
  namebase=os.path.basename(filename)
  return namebase.split("__")[1]

def name2version(filename):
  namebase=os.path.basename(filename)
  return namebase.split("__")[2]
  
def name2run(filename):
  namebase=os.path.basename(filename)
  return namebase.split("__")[0].split("_")[2]  

def name2runskim(filename):
  run=name2run(filename)
  skim=name2version(filename).split("_")[-1]
  # remove skim version
  if "-v" in skim:
    skim = skim[:skim.rfind('-v')]
  return "%s_%s"%(run,skim)

#-------------------------------------------------------------------------------  

def guess_params(ref_filenames,test_filenames):
  
  if len(ref_filenames)*len(test_filenames)==0:
    print "Empty reference and test filenames lists!"
    return [],"",""
  
  samples=[]
  ref_versions=[]
  test_versions=[]
    
  for ref, test in zip(map(os.path.basename,ref_filenames),map(os.path.basename,test_filenames)):
    
    ref_sample=name2sample(ref)
    ref_version=name2version(ref)
    test_sample=name2sample(test)
    test_version=name2version(test)
          
    if ref_sample!=test_sample:
      print "Files %s and %s do not seem to be relative to the same sample." %(ref, test)
      exit(2)

    # Slightly modify for data
    if search("20[01]",ref_version)!=None:
      ref_sample+=ref_version.split("_")[-1]
    samples.append(ref_sample)
 
    # append the versions
    ref_versions.append(ref_version)
    test_versions.append(test_version)

  # Check if ref and test versions are always the same.
  ref_versions=list(set(ref_versions))
  test_versions=list(set(test_versions))
  
  #for versions in ref_versions,test_versions:
    #if len(versions)!=1:
      #print "More than one kind of CMSSW version selected (%s)" %versions
      #exit(2)  
  
  cmssw_version1=ref_versions[0]
  cmssw_version2=test_versions[0]
  
  return samples,cmssw_version1,cmssw_version2
  

#-------------------------------------------------------------------------------

def check_root_files(names_list):
  for name in names_list:
    if not name.endswith(".root"):
      print "File %s does not seem to be a rootfile. Please check."
      return False
  return True

#-------------------------------------------------------------------------------

def add_to_blacklist(blacklist, pattern, target, blist_piece):
  int_pattern=pattern
  int_pattern=pattern.strip()  
  flip_condition=False
  if int_pattern[0]=='!':
    int_pattern=int_pattern[1:]
    flip_condition=True

  condition = search(int_pattern,target)!=None
  if flip_condition:
    condition = not condition

  if condition:
    #print "Found %s in %s" %(pattern,target)
    if blacklist!="": # if not the first, add a comma
      blacklist+=","
    blacklist+=blist_piece
  #else:
    #print "  NOT Found %s in %s" %(pattern,target)
  return blacklist

#-------------------------------------------------------------------------------

def guess_blacklists(samples,ver1,ver2,hlt):
  """Build a blacklist for each sample accordind to a set of rules
  """
  blacklists={}
  for sample in samples:
    blacklists[sample]="FED@1,AlcaBeamMonitor@1,Physics@1,Info@-1,HLT@1,AlCaReco@1"
    
    # HLT
    if hlt: #HLT
      blacklists[sample]+=",AlCaEcalPi0@2"
      if not search("2010+|2011+",ver1):
        print "We are treating MC files for the HLT"
        for pattern,blist in definitions.hlt_mc_pattern_blist_pairs:
          blacklists[sample]=add_to_blacklist(blacklists[sample],pattern,sample,blist)
#          print 'HLT '+pattern
#          print 'HLT '+sample
#          print 'HLT '+blacklists[sample]   
      else:
        print "We are treating Data files for the HLT"    
        # at the moment it does not make sense since hlt is ran already
    
    else: #RECO
      #Monte Carlo
      if not search("2010+|2011+",ver1):
        print "We are treating MC files"        
        
        for pattern,blist in definitions.mc_pattern_blist_pairs:
          blacklists[sample]=add_to_blacklist(blacklists[sample],pattern,sample,blist)
#          print "MC RECO"
          #print blacklists[sample]
          
      # Data
      else:
        print "We are treating Data files:"      
        blacklists[sample]+=",By__Lumi__Section@-1,AlCaReco@1"                                         
        for pattern,blist in definitions.data_pattern_blist_pairs:
          blacklists[sample]=add_to_blacklist(blacklists[sample],pattern,ver1,blist)
#         print "DATA RECO: %s %s %s -->%s" %( ver1, pattern, blist, blacklists[sample])


  return blacklists

#-------------------------------------------------------------------------------  

def get_roofiles_in_dir(directory):  
  print directory
  files_list = filter(lambda s: s.endswith(".root"), os.listdir(directory))
  files_list_path=map(lambda s: os.path.join(directory,s), files_list)
  
  return files_list_path
  
#-------------------------------------------------------------------------------  

def get_filenames_from_pool(all_samples):
  
  # get a list of the files
  files_list=get_roofiles_in_dir(all_samples)
  
  if len(files_list)==0:
    print "Zero files found in directory %s!" %all_samples
    return [],[]
  
  # Are they an even number?
  for name in files_list:
    print "* ",name  
  if len(files_list)%2!=0:
    print "The numbuer of file is not even... Trying to recover a catastrophe."
    
  files_list=make_files_pairs(files_list)
  
  # Try to couple them according to their sample
  ref_filenames=[]
  test_filenames=[]
  #files_list.sort(key=name2version)
  #files_list.sort(key=name2sample) 
  #files_list.sort(key=name2run)
  for iname in xrange(len(files_list)):
    filename=files_list[iname]
    if iname%2==0:
      ref_filenames.append(filename)
    else:
      test_filenames.append(filename)
      
  print "The guess would be the following:"
  for ref,test in zip(ref_filenames,test_filenames):
    refbasedir=os.path.dirname(ref)
    testbasedir=os.path.dirname(test)
    dir_to_print=refbasedir
    if refbasedir!=testbasedir:
      dir_to_print="%s and %s" %(refbasedir,testbasedir)
    print "* Directory: %s " %dir_to_print
    refname=os.path.basename(ref)
    testname=os.path.basename(test)
    print "  o %s" %refname
    print "  o %s" %testname
  
  #is_ok=ask_ok("Is that ok?")
  #if not is_ok:
    #print "Manual input needed then!"
    #exit(2)
      
  
  return ref_filenames,test_filenames
  

#-------------------------------------------------------------------------------

def get_clean_fileanames(ref_samples,test_samples):
  # Process the samples starting from the names
  ref_filenames=map(lambda s:s.strip(),ref_samples.split(","))
  test_filenames=map(lambda s:s.strip(),test_samples.split(","))

  if len(ref_filenames)!=len(test_filenames):
    print "The numebr of reference and test files does not seem to be the same. Please check."
    exit(2)

  if not (check_root_files(ref_filenames) and check_root_files(test_filenames)):
    exit(2)
  return ref_filenames,test_filenames

#-------------------------------------------------------------------------------

def count_alive_processes(p_list):
  return len(filter(lambda p: p.returncode==None,p_list))

#-------------------------------------------------------------------------------

def call_compare_using_files(args):
  """Creates shell command to compare two files using compare_using_files.py
  script and calls it."""
  sample, ref_filename, test_filename, options = args
  blacklists=guess_blacklists([sample],name2version(ref_filename),name2version(test_filename),options.hlt)
  command = " compare_using_files.py "
  command+= "%s %s " %(ref_filename,test_filename)
  command+= " -C -R "
  if options.do_pngs:
    command+= " -p "
  command+= " -o %s " %sample
  # Change threshold to an experimental and empirical value of 10^-5
  command+= " --specify_run "
  command+= " -t %s " %options.test_threshold
  command+= " -s %s " %options.stat_test

  # Inspect the HLT directories
  if options.hlt:
    command+=" -d HLT "
  
  if options.hash_name:
    command += " --hash_name "  

  if len(blacklists[sample]) >0:
    command+= '-B %s ' %blacklists[sample]
  print "\nExecuting --  %s" %command

  process=call(filter(lambda x: len(x)>0,command.split(" ")))
  return process
  

#--------------------------------------------------------------------------------

def do_comparisons_threaded(options):

  n_processes= int(options.n_processes)

  ref_filenames=[]
  test_filenames=[]
  
  if len(options.all_samples)>0:
    ref_filenames,test_filenames=get_filenames_from_pool(options.all_samples)  
  else:
    ref_filenames,test_filenames=get_clean_fileanames(options.ref_samples,options.test_samples)
 
  # make the paths absolute
  ref_filenames=map(os.path.abspath,ref_filenames)
  test_filenames=map(os.path.abspath,test_filenames)
  
  samples,cmssw_version1,cmssw_version2=guess_params(ref_filenames,test_filenames)
  
  if len(samples)==0:
    print "No Samples found... Quitting"
    return 0
  
#  blacklists=guess_blacklists(samples,cmssw_version1,cmssw_version2,options.hlt)

  # Launch the single comparisons
  original_dir=os.getcwd()

  outdir=options.out_dir
  if len(outdir)==0:
    print "Creating automatic outdir:",
    outdir="%sVS%s" %(cmssw_version1,cmssw_version2)
    print outdir
  if len(options.input_dir)==0:
    print "Creating automatic indir:",
    options.input_dir=outdir
    print options.input_dir
  
  if not os.path.exists(outdir):
    os.mkdir(outdir)
  os.chdir(outdir)  
  
  # adjust the number of threads
  n_comparisons=len(ref_filenames)
  if n_comparisons < n_processes:
    print "Less comparisons than possible processes: reducing n processes to",
    n_processes=n_comparisons
  #elif n_processes/n_comparisons == 0:
    #print "More comparisons than possible processes, can be done in N rounds: reducing n processes to",    
    #original_nprocesses=n_processes
    #first=True
    #n_bunches=0
    #while first or n_processes > original_nprocesses:
      #n_processes=n_comparisons/2
      #if n_comparisons%2 !=0:
        #n_processes+=1
      #first=False
      
    #print n_processes
  #print n_processes
  
  # Test if we treat data
  skim_name=""
  if search("20[01]",cmssw_version1)!=None:
    skim_name=cmssw_version1.split("_")[-1]
    
  running_subprocesses=[]
  process_counter=0
  #print ref_filenames

  ## Compare all pairs of root files
  pool = Pool(n_processes)
  args_iterable = [list(args) + [options] for args in zip(samples, ref_filenames, test_filenames)]
  pool.map(call_compare_using_files, args_iterable) 
  # move the pickles on the top, hack
  os.system("mv */*pkl .")
  
  os.chdir("..")
#-------------------------------------------------------------------------------
def do_reports(indir):
  #print indir
  os.chdir(indir)
  pkl_list=filter(lambda x:".pkl" in x, os.listdir("./"))
  running_subprocesses=[]
  n_processes=int(options.n_processes)
  process_counter=0
  for pklfilename in pkl_list:
    command = "compare_using_files.py " 
    command+= "-R "
    if options.do_pngs:
      command+= " -p "
    command+= "-P %s " %pklfilename
    command+= "-o %s " %pklfilename[:-4]
    print "Executing %s" %command
    process=call(filter(lambda x: len(x)>0,command.split(" ")))
    process_counter+=1
    # add it to the list
    running_subprocesses.append(process)   
    if process_counter>=n_processes:
      process_counter=0
      for p in running_subprocesses:
        #print "Waiting for %s" %p.name
        p.wait()
        
  os.chdir("..")
  
#-------------------------------------------------------------------------------
def do_html(options, hashing_flag):

  if options.reports:
    print "Preparing reports for the single files..."
    do_reports(options.input_dir)
  # Do the summary page
  aggregation_rules={}
  aggregation_rules_twiki={}
  # check which aggregation rules are to be used
  if options.hlt:
    print "Aggregating directories according to HLT rules"
    aggregation_rules=definitions.aggr_pairs_dict['HLT']
    aggregation_rules_twiki=definitions.aggr_pairs_twiki_dict['HLT']
  else:
    aggregation_rules=definitions.aggr_pairs_dict['reco']
    aggregation_rules_twiki=definitions.aggr_pairs_twiki_dict['reco']
  table_html = make_summary_table(options.input_dir,aggregation_rules,aggregation_rules_twiki, hashing_flag)

  # create summary html file
  ofile = open("RelMonSummary.html","w")
  ofile.write(table_html)
  ofile.close()

#-------------------------------------------------------------------------------

if __name__ == "__main__":

  #-----------------------------------------------------------------------------
  ref_samples=""
  test_samples=""
  all_samples=""
  n_processes=1
  out_dir=""
  in_dir=""
  n_threads=1 # do not change this
  run=-1
  stat_test="Chi2"
  test_threshold=0.00001
  hlt=False
  #-----------------------------------------------------------------------------


  parser = OptionParser(usage="usage: %prog [options]")

  parser.add_option("-R","--ref_samples ",
                    action="store",
                    dest="ref_samples",
                    default=ref_samples,
                    help="The samples that act as reference (comma separated list)")

  parser.add_option("-T","--test_samples",
                    action="store",
                    dest="test_samples",
                    default=test_samples,
                    help="The samples to be tested (comma separated list)")

  parser.add_option("-a","--all_samples",
                    action="store",
                    dest="all_samples",
                    default=all_samples,
                    help="EXPERIMENTAL: Try to sort all samples selected (wildacrds) and organise a comparison")

  parser.add_option("-o","--out_dir",
                    action="store",
                    dest="out_dir",
                    default=out_dir,
                    help="The outdir other than <Version1>VS<Version2>")

  parser.add_option("-p","--do_pngs",
                    action="store_true",
                    dest="do_pngs",
                    default=False,
                    help="EXPERIMENTAL!!! Do the pngs of the comparison (takes 50%% of the total running time) \n(default is %s)" %False)

  parser.add_option("-r","--run ",
                    action="store",
                    dest="run",
                    default=run,
                    help="The run to be checked \n(default is %s)" %run)

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
  
  parser.add_option("-N","--numberOfProcesses",
                    action="store",
                    dest="n_processes",
                    default=n_processes,
                    help="Number of parallel processes to be run. Be Polite! \n(default is %s)" %n_processes)  
                    
  parser.add_option("--HLT",
                    action="store_true",
                    dest="hlt",
                    default=False,
                    help="Analyse HLT histograms\n(default is %s)" %hlt)
                    
  parser.add_option("-i","--input_dir",
                    action="store",
                    dest="input_dir",
                    default=in_dir,
                    help="Input directory for html creation \n(default is %s)" %in_dir)
  
  parser.add_option("--reports",
                    action="store_true",
                    dest="reports",
                    default=False,
                    help="Do the reports for the pickles \n(default is %s)" %in_dir)
##---HASHING---##
  parser.add_option("--hash_name",
                    action="store_true",
                    dest="hash_name",
                    default=False,
                    help="Set if you want to minimize & hash the output HTML files.")                    

  (options, args) = parser.parse_args()

  if len(options.test_samples)*len(options.ref_samples)+len(options.all_samples)==0 and len(options.input_dir)==0:
    print "No samples given as input."
    parser.print_help()
    exit(2)

  if len(options.all_samples)>0 or (len(options.ref_samples)*len(options.test_samples)>0):
    do_comparisons_threaded(options)
  if len(options.input_dir)>0:
    do_html(options, options.hash_name)












