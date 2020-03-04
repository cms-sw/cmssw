from __future__ import print_function
from __future__ import absolute_import
################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
#
#                                                                              
# Danilo Piparo CERN - danilo.piparo@cern.ch                                   
#                                                                              
################################################################################


from builtins import range
import array
import os
import re
import sys
from pickle import load
from os.path import dirname,basename,join,isfile
from threading import Thread
from time import asctime

theargv=sys.argv
sys.argv=[]
import ROOT
ROOT.gErrorIgnoreLevel=1001
ROOT.gROOT.SetBatch(True)
sys.argv=theargv

if sys.version_info[0]==2:
  from urllib2  import Request,build_opener,urlopen
else:
  from urllib.request  import Request,build_opener,urlopen

if "RELMON_SA" in os.environ:
  from .definitions import *
  from .authentication import X509CertOpen
  from .utils import __file__ as this_module_name  
else:
  from Utilities.RelMon.definitions import *
  from Utilities.RelMon.authentication import X509CertOpen
  from Utilities.RelMon.utils import __file__ as this_module_name

#ROOT.gErrorIgnoreLevel=1001


_log_level=10
def logger(msg_level,message):
  if msg_level>=_log_level:
    print("[%s] %s" %(asctime(),message))

#-------------------------------------------------------------------------------
def setTDRStyle():  
  this_dir=dirname(this_module_name)
  this_dir_one_up=this_dir[:this_dir.rfind("/")+1]
  #this_dir_two_up=this_dir_one_up[:this_dir_one_up.rfind("/")+1]
  style_file=''
  if "RELMON_SA" in os.environ:
    style_file=this_dir_one_up+"data/tdrstyle_mod.C"
  else:
    style_file="%s/src/Utilities/RelMon/data/tdrstyle_mod.C"%(os.environ["CMSSW_BASE"])
  try:
    gROOT.ProcessLine(".L %s" %style_file)
    gROOT.ProcessLine("setTDRStyle()")
  except:
    "Print could not set the TDR style. File %s not found?" %style_file
    

#-------------------------------------------------------------------------------
def literal2root (literal,rootType):
  bitsarray = array.array('B')
  bitsarray.fromstring(literal.decode('hex'))

  tbuffer=0
  try:  
      tbuffer = TBufferFile(TBufferFile.kRead, len(bitsarray), bitsarray, False,0)
  except:
      print("could not transform to object array:")
      print([ i for i in  bitsarray ])
  
  # replace a couple of shortcuts with the real root class name
  if rootType == 'TPROF':
      rootType = 'TProfile'
  if rootType == 'TPROF2D':
      rootType = 'TProfile2D'
  
  root_class=eval(rootType+'.Class()')
  
  return tbuffer.ReadObject(root_class)
  
#-------------------------------------------------------------------------------

def getNbins(h):
  """
  To be used in loops on bin number with range()
  For each dimension there are GetNbinsX()+2 bins including underflow 
  and overflow, and range() loops starts from 0. So the total number
  of bins as upper limit of a range() loop already includes the next 
  to last value needed.
  """
  biny=h.GetNbinsY()
  if biny>1: biny+=2
  binz=h.GetNbinsZ()
  if binz>1:binz+=2
  return (h.GetNbinsX()+2)*(biny)*(binz)

#-------------------------------------------------------------------------------


class StatisticalTest(object):
  def __init__(self,threshold):
    self.name=""
    self.h1=None
    self.h2=None
    self.threshold=float(threshold)
    self.rank=-1
    self.is_init=False

  def set_operands(self,h1,h2):
    self.h1=h1
    self.h2=h2

  def get_rank(self):
    if not self.is_init:
      if self.rank < 0:
        type1=type(self.h1)
        type2=type(self.h2)
        if (type1 != type2):
          logger(1,"*** ERROR: object types in comparison don't match: %s!=%s" %(type1,type2))
          self.rank=test_codes["DIFF_TYPES"]
        elif not self.h2.InheritsFrom("TH1"):
          logger(1,"*** ERROR: object type is not histogram but a %s" %(type1))
          self.rank=test_codes["NO_HIST"]    
        # if histos are empty
        #elif self.h1.InheritsFrom("TH2") and not "BinToBin" in self.name:
          ## 2D!
          #return test_codes["2D"]
        else:
          is_empty1=is_empty(self.h1)
          is_empty2=is_empty(self.h2)
          are_empty=is_empty1 and is_empty2
          one_empty=is_empty1 or is_empty2

          Nbins1= getNbins(self.h1)
          Nbins2= getNbins(self.h2)

          if are_empty:
            #return -103
            # Conversation with JeanRoch and David 5 April
            return 1
          elif one_empty:
            # Due conversation with Giovanni on 2015-09-10
            return 0

          # if histos have different number of bins
          if Nbins1!=Nbins2:
            return test_codes["DIFF_BIN"]

      self.rank=self.do_test()
    self.is_init=True
    return self.rank

  def get_status(self):
    status = SUCCESS
    if self.get_rank()<0:
      status=NULL
      logger(0,"+++ Test %s FAILED: rank is %s and threshold is %s ==> %s" %(self.name, self.rank, self.threshold, status))  
    elif self.get_rank() < self.threshold:
      status=FAIL
    logger(0,"+++ Test %s: rank is %s and threshold is %s ==> %s" %(self.name, self.rank, self.threshold, status))
    return status

  def do_test(self):
    pass

#-------------------------------------------------------------------------------

def is_empty(h):
  for i in range(0,getNbins(h)):
    if h.GetBinContent(i)!=0: return False
  return True
  #return h.GetSumOfWeights()==0

#-------------------------------------------------------------------------------

def is_sparse(h):
  filled_bins=0.
  nbins=h.GetNbinsX()
  for ibin in range(0,nbins+2):
    if h.GetBinContent(ibin)>0:
      filled_bins+=1
  #print "%s %s --> %s" %(filled_bins,nbins,filled_bins/nbins)
  if filled_bins/nbins < .5:
    return True
  else:
    return False

#-------------------------------------------------------------------------------

class KS(StatisticalTest):
  def __init__(self, threshold):
   StatisticalTest.__init__(self,threshold)
   self.name="KS"

  def do_test(self):

    # Calculate errors if not there...
    for h in self.h1,self.h2:
      w2s=h.GetSumw2()
      if w2s.GetSize()==0:
        h.Sumw2()

    ## If errors are 0:
    #zero_errors=True
    #for h in self.h1,self.h2:
      #for ibin in xrange(Nbins1):
        #if h.GetBinError(ibin+1) >0:
          #zero_errors=False
          #break
    #if zero_errors:
      #return test_codes["ZERO_ERR"]

    return self.h1.KolmogorovTest(self.h2)

#-------------------------------------------------------------------------------
import array
def profile2histo(profile):
  if not profile.InheritsFrom("TH1"):
    return profile
    
  bin_low_edges=[]
  n_bins=profile.GetNbinsX()
  
  for ibin in range(1,n_bins+2):
    bin_low_edges.append(profile.GetBinLowEdge(ibin))
  bin_low_edges=array.array('f',bin_low_edges)
  histo=TH1F(profile.GetName(),profile.GetTitle(),n_bins,bin_low_edges)
  for ibin in range(0,n_bins+2):
    histo.SetBinContent(ibin,profile.GetBinContent(ibin))
    histo.SetBinError(ibin,profile.GetBinError(ibin))    
  
  return histo
#-------------------------------------------------------------------------------

class Chi2(StatisticalTest):
  def __init__(self, threshold):
    StatisticalTest.__init__(self,threshold)
    self.name="Chi2"     

  def check_filled_bins(self,min_filled):
    nbins=self.h1.GetNbinsX()
    n_filled_l=[]
    for h in self.h1,self.h2:
      nfilled=0.
      for ibin in range(0,nbins+2):
        if h.GetBinContent(ibin)>0:
          nfilled+=1
      n_filled_l.append(nfilled)
    return len([x for x in n_filled_l if x>=min_filled] )>0

  def absval(self):
    nbins=getNbins(self.h1)
    binc=0
    for i in range(0,nbins):
      for h in self.h1,self.h2:
        binc=h.GetBinContent(i)
        if binc<0:
          h.SetBinContent(i,-1*binc)
        if h.GetBinError(i)==0 and binc!=0:
          #print "Histo ",h.GetName()," Bin:",i,"-Content:",h.GetBinContent(i)," had zero error"
          h.SetBinContent(i,0)

  def check_histograms(self, histogram):
      if histogram.InheritsFrom("TProfile") or  (histogram.GetEntries()!=histogram.GetSumOfWeights()):
          return 'W'
      else:
          return 'U'

  def do_test(self):
    self.absval()
    if self.check_filled_bins(3):
      #if self.h1.InheritsFrom("TProfile") or  (self.h1.GetEntries()!=self.h1.GetSumOfWeights()):
      #  chi2=self.h1.Chi2Test(self.h2,'WW')
      #  #if chi2==0: print "DEBUG",self.h1.GetName(),"Chi2 is:", chi2
      #  return chi2
      #else:
      #  return self.h1.Chi2Test(self.h2,'UU')
      hist1 = self.check_histograms(self.h1)
      hist2 = self.check_histograms(self.h2)
      if hist1 =='W' and hist2 =='W': ##in case 
          chi2 = self.h1.Chi2Test(self.h2,'WW')     ## the both histograms are weighted
          return chi2
      elif hist1 == 'U' and hist2 == 'U':
          chi2 = self.h1.Chi2Test(self.h2,'UU')    ##the both histograms are unweighted
          return chi2
      elif hist1 == 'U' and hist2 == 'W':
          chi2 = self.h1.Chi2Test(self.h2,'UW')   ## 1st histogram is unweighted, 2nd weighted
          return chi2
      elif hist1 == 'W' and hist2 == 'U':
          chi2 = self.h2.Chi2Test(self.h1,'UW')   ## 1 is wieghted, 2nd unweigthed. so flip order to make a UW comparison
          return chi2
    else:
      return 1
      #return test_codes["FEW_BINS"]

#-------------------------------------------------------------------------------

class BinToBin(StatisticalTest):
  """The bin to bin comparison builds a fake pvalue. It is 0 if the number of 
  bins is different. It is % of corresponding bins otherwhise.
  A threshold of 1 is needed to require a 1 to 1 correspondance between 
  hisograms.
  """
  def __init__(self, threshold=1):
    StatisticalTest.__init__(self, threshold)
    self.name='BinToBin'
    self.epsilon= 0.000001

  def checkBinningMatches(self):
    if self.h1.GetNbinsX() != self.h2.GetNbinsX() \
           or self.h1.GetNbinsY() != self.h2.GetNbinsY() \
           or self.h1.GetNbinsZ() != self.h2.GetNbinsZ() \
           or abs(self.h1.GetXaxis().GetXmin() - self.h2.GetXaxis().GetXmin()) >self.epsilon \
           or abs(self.h1.GetYaxis().GetXmin() - self.h2.GetYaxis().GetXmin()) >self.epsilon \
           or abs(self.h1.GetZaxis().GetXmin() - self.h2.GetZaxis().GetXmin()) >self.epsilon \
           or abs(self.h1.GetXaxis().GetXmax() - self.h2.GetXaxis().GetXmax()) >self.epsilon \
           or abs(self.h1.GetYaxis().GetXmax() - self.h2.GetYaxis().GetXmax()) >self.epsilon \
           or abs(self.h1.GetZaxis().GetXmax() - self.h2.GetZaxis().GetXmax()) >self.epsilon:
      return False
    return True

  def do_test(self):
    # fist check that binning matches
    if not self.checkBinningMatches():
      return test_codes["DIFF_BIN"]
    # then do the real check
    equal = 1
    nbins = getNbins(self.h1)
    n_ok_bins=0.0
    for ibin in range(0, nbins):
      h1bin=self.h1.GetBinContent(ibin)
      h2bin=self.h2.GetBinContent(ibin)
      bindiff=h1bin-h2bin

      binavg=.5*(h1bin+h2bin)

      if binavg==0 or abs(bindiff) < self.epsilon:
        n_ok_bins+=1
        #print("Bin %ibin: bindiff %s" %(ibin,bindiff))
      else:
        print("Bin %ibin: bindiff %s" %(ibin,bindiff))

      #if abs(bindiff)!=0 :
        #print "Bin %ibin: bindiff %s" %(ibin,bindiff)
    
    rank=n_ok_bins/nbins
    
    if rank!=1:
      print("Histogram %s differs: nok: %s ntot: %s" %(self.h1.GetName(),n_ok_bins,nbins))
    
    return rank

#-------------------------------------------------------------------------------

class BinToBin1percent(StatisticalTest):
  """The bin to bin comparison builds a fake pvalue. It is 0 if the number of 
  bins is different. It is % of corresponding bins otherwhise.
  A threshold of 1 is needed to require a 1 to 1 correspondance between 
  hisograms.
  """
  def __init__(self, threshold=1):
    StatisticalTest.__init__(self, threshold)
    self.name='BinToBin1percent'
    self.epsilon= 0.000001
    self.tolerance= 0.01

  def checkBinningMatches(self):
    if self.h1.GetNbinsX() != self.h2.GetNbinsX() \
           or self.h1.GetNbinsY() != self.h2.GetNbinsY() \
           or self.h1.GetNbinsZ() != self.h2.GetNbinsZ() \
           or abs(self.h1.GetXaxis().GetXmin() - self.h2.GetXaxis().GetXmin()) >self.epsilon \
           or abs(self.h1.GetYaxis().GetXmin() - self.h2.GetYaxis().GetXmin()) >self.epsilon \
           or abs(self.h1.GetZaxis().GetXmin() - self.h2.GetZaxis().GetXmin()) >self.epsilon \
           or abs(self.h1.GetXaxis().GetXmax() - self.h2.GetXaxis().GetXmax()) >self.epsilon \
           or abs(self.h1.GetYaxis().GetXmax() - self.h2.GetYaxis().GetXmax()) >self.epsilon \
           or abs(self.h1.GetZaxis().GetXmax() - self.h2.GetZaxis().GetXmax()) >self.epsilon:
      return False
    return True

  def do_test(self):
    # fist check that binning matches
    if not self.checkBinningMatches():
      return test_codes["DIFF_BIN"]
    # then do the real check
    equal = 1
    nbins = getNbins(self.h1)
    n_ok_bins=0.0
    for ibin in range(0,nbins):
      ibin+=1
      h1bin=self.h1.GetBinContent(ibin)
      h2bin=self.h2.GetBinContent(ibin)
      bindiff=h1bin-h2bin

      binavg=.5*(h1bin+h2bin)

      if binavg==0 or 100*abs(bindiff)/binavg < self.tolerance:
        n_ok_bins+=1
        #print "Bin %i bin: bindiff %s" %(ibin,bindiff)
      else:
        print("-->Bin %i bin: bindiff %s (%s - %s )" %(ibin,bindiff,h1bin,h2bin))

      #if abs(bindiff)!=0 :
        #print "Bin %ibin: bindiff %s" %(ibin,bindiff)
    
    rank=n_ok_bins/nbins
    
    if rank!=1:
      print("%s nok: %s ntot: %s" %(self.h1.GetName(),n_ok_bins,nbins))
    
    return rank
#-------------------------------------------------------------------------------
Statistical_Tests={"KS":KS,
                   "Chi2":Chi2,
                   "BinToBin":BinToBin,
                   "BinToBin1percent":BinToBin1percent,
                   "Bin2Bin":BinToBin,
                   "b2b":BinToBin,}
#-------------------------------------------------------------------------------  

def ask_ok(prompt, retries=4, complaint='yes or no'):
    while True:
        ok = raw_input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no'):
            return False
        retries = retries - 1
        if retries < 0:
            raise IOError('refusenik user')
        print(complaint)

#-------------------------------------------------------------------------------

class unpickler(Thread):
  def __init__(self,filename):
    Thread.__init__(self)
    self.filename=filename
    self.directory=""

  def run(self):
    print("Reading directory from %s" %(self.filename))
    ifile=open(self.filename,"rb")
    self.directory=load(ifile) 
    ifile.close()

#-------------------------------------------------------------------------------    

def wget(url):
  """ Fetch the WHOLE file, not in bunches... To be optimised.
  """
  opener=build_opener(X509CertOpen())  
  datareq = Request(url)
  datareq.add_header('authenticated_wget', "The ultimate wgetter")    
  bin_content=None
  try:
    filename=basename(url)  
    print("Checking existence of file %s on disk..."%filename)
    if not isfile("./%s"%filename):      
      bin_content=opener.open(datareq).read()
    else:
      print("File %s exists, skipping.." %filename)
  except ValueError:
    print("Error: Unknown url %s" %url)
  
  if bin_content!=None:  
    ofile = open(filename, 'wb')
    ofile.write(bin_content)
    ofile.close()

#-------------------------------------------------------------------------------
##-----------------   Make files pairs:  RelValData utils   --------------------

def get_relvaldata_id(file):
    """Returns unique relvaldata ID for a given file."""
    run_id = re.search('R\d{9}', file)
    run = re.search('_RelVal_([\w\d]*)-v\d__', file)
    if not run:
        run = re.search('GR_R_\d*_V\d*C?_([\w\d]*)-v\d__', file)
    if run_id and run:
        return (run_id.group(), run.group(1))
    return None

def get_relvaldata_cmssw_version(file):
    """Returns tuple (CMSSW release, GR_R version) for specified RelValData file."""
    cmssw_release = re.findall('(CMSSW_\d*_\d*_\d*(?:_[\w\d]*)?)-', file)
    gr_r_version = re.findall('-(GR_R_\d*_V\d*\w?)(?:_RelVal)?_', file)
    if not gr_r_version:
        gr_r_version = re.findall('CMSSW_\d*_\d*_\d*(?:_[\w\d]*)?-(\w*)_RelVal_', file)
    if cmssw_release and gr_r_version:
        return (cmssw_release[0], gr_r_version[0])

def get_relvaldata_version(file):
    """Returns tuple (CMSSW version, run version) for specified file."""
    cmssw_version = re.findall('DQM_V(\d*)_', file)
    run_version = re.findall('_RelVal_[\w\d]*-v(\d)__', file)
    if not run_version:
        run_version = re.findall('GR_R_\d*_V\d*C?_[\w\d]*-v(\d)__', file)
    if cmssw_version and run_version:
        return (int(cmssw_version[0]), int(run_version[0]))

def get_relvaldata_max_version(files):
    """Returns file with maximum version at a) beggining of the file,
    e.g. DQM_V000M b) at the end of run, e.g. _run2012-vM. M has to be max."""
    max_file = files[0]
    max_v = get_relvaldata_version(files[0])
    for file in files:
        file_v = get_relvaldata_version(file)
        if file_v[1] > max_v[1] or ((file_v[1] == max_v[1]) and (file_v[0] > max_v[0])):
            max_file = file
            max_v = file_v
    return max_file

##-------------------   Make files pairs:  RelVal utils   ---------------------
def get_relval_version(file):
    """Returns tuple (CMSSW version, run version) for specified file."""
    cmssw_version = re.findall('DQM_V(\d*)_', file)
    run_version = re.findall('CMSSW_\d*_\d*_\d*(?:_[\w\d]*)?-[\w\d]*_V\d*\w?(?:_[\w\d]*)?-v(\d*)__', file)
    if cmssw_version and run_version:
        return (int(cmssw_version[0]), int(run_version[0]))

def get_relval_max_version(files):
    """Returns file with maximum version at a) beggining of the file,
    e.g. DQM_V000M b) at the end of run, e.g. _run2012-vM. M has to be max."""
    max_file = files[0]
    max_v = get_relval_version(files[0])
    for file in files:
        file_v = get_relval_version(file)
        if file_v[1] > max_v[1] or ((file_v[1] == max_v[1]) and (file_v[0] > max_v[0])):
            max_file = file
            max_v = file_v
    return max_file

def get_relval_cmssw_version(file):
    cmssw_release = re.findall('(CMSSW_\d*_\d*_\d*(?:_[\w\d]*)?)-', file)
    gr_r_version = re.findall('CMSSW_\d*_\d*_\d*(?:_[\w\d]*)?-([\w\d]*)_V\d*\w?(_[\w\d]*)?-v', file)
    if cmssw_release and gr_r_version:
        if "PU" in gr_r_version[0][0] and not "FastSim" in file:
            __gt = re.sub('^[^_]*_', "", gr_r_version[0][0])
            __process_string = gr_r_version[0][1]
            return (__gt, __process_string)
        elif "PU" in gr_r_version[0][0] and "FastSim" in file:   #a check for FastSimPU samples
            return (cmssw_release[0], "PU_")                     #with possibly different GT's
        return (cmssw_release[0], gr_r_version[0])

def get_relval_id(file):
    """Returns unique relval ID (dataset name) for a given file."""
    dataset_name = re.findall('R\d{9}__([\w\D]*)__CMSSW_', file)
    __process_string = re.search('CMSSW_\d*_\d*_\d*(?:_[\w\d]*)?-([\w\d]*)_V\d*\w?(_[\w\d]*)?-v', file)
    _ps = ""
    if __process_string:
        if "PU" in __process_string.group(1) and not "FastSim" in file:
            _ps = re.search('^[^_]*_', __process_string.group(1)).group()
        elif "PU" in __process_string.group(1) and "FastSim" in file:
            return dataset_name[0]+"_", _ps ##some testing is needed
    return dataset_name[0], _ps

##-------------------------  Make files pairs --------------------------
def is_relvaldata(files):
    is_relvaldata_re = re.compile('_RelVal_')
    return any([is_relvaldata_re.search(filename) for filename in files])

def make_files_pairs(files, verbose=True):
    ## Select functions to use
    if is_relvaldata(files):
        is_relval_data = True
        get_cmssw_version = get_relvaldata_cmssw_version
        get_id = get_relvaldata_id
        get_max_version = get_relvaldata_max_version
        # print 'Pairing Data RelVal files.'
    else:
        is_relval_data = False
        get_cmssw_version = get_relval_cmssw_version
        get_id = get_relval_id
        get_max_version = get_relval_max_version
        # print 'Pairing Monte Carlo RelVal files.'

    ## Divide files into groups
    versions_files = dict()
    for file in files:
        version = get_cmssw_version(file)
        if version in versions_files:
            versions_files[version].append(file)
        else:
            versions_files[version] = [file]

    ## Print the division into groups
    if verbose:
        print('\nFound versions:')
        for version in versions_files:
            print('%s: %d files' % (str(version),  len(versions_files[version])))

    if len(versions_files) <= 1:
        print('\nFound too little versions, there is nothing to pair. Exiting...\n')
        exit()

    ## Select two biggest groups.
    versions = versions_files.keys()
    sizes = [len(value) for value in versions_files.values()]
    v1 = versions[sizes.index(max(sizes))]
    versions.remove(v1)
    sizes.remove(max(sizes))
    v2 = versions[sizes.index(max(sizes))]

    ## Print two biggest groups.
    if verbose:
        print('\nPairing %s (%d files) and %s (%d files)' % (str(v1),
                len(versions_files[v1]), str(v2), len(versions_files[v2])))

    ## Pairing two versions
    print('\nGot pairs:')
    pairs = []
    for unique_id in set([get_id(file) for file in versions_files[v1]]):
        if is_relval_data:
            dataset_re = re.compile(unique_id[0]+'_')
            run_re = re.compile(unique_id[1])
            c1_files = [file for file in versions_files[v1] if dataset_re.search(file) and run_re.search(file)]
            c2_files = [file for file in versions_files[v2] if dataset_re.search(file) and run_re.search(file)]
        else:
            dataset_re = re.compile(unique_id[0]+'_')
            ps_re = re.compile(unique_id[1])
            ##compile a PU re and search also for same PU
            c1_files = [file for file in versions_files[v1] if dataset_re.search(file) and ps_re.search(file)]
            c2_files = [file for file in versions_files[v2] if dataset_re.search(file) and ps_re.search(file)]

        if len(c1_files) > 0 and len(c2_files) > 0:
            first_file = get_max_version(c1_files)
            second_file = get_max_version(c2_files)
            print('%s\n%s\n' % (first_file, second_file))
            pairs.extend((first_file, second_file))
    if verbose:
        print("Paired and got %d files.\n" % len(pairs))
    return pairs
