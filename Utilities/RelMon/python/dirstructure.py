################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
# $Author: anorkus $
# $Date: 2013/07/05 09:45:01 $
# $Revision: 1.5 $
#
#                                                                              
# Danilo Piparo CERN - danilo.piparo@cern.ch                                   
#                                                                              
################################################################################

from array import array
from copy import deepcopy
from os import chdir,getcwd,listdir,makedirs,rmdir
from os.path import exists,join

import sys
argv=sys.argv
from ROOT import *
sys.argv=argv

from definitions import *
from utils import setTDRStyle


# Something nice and familiar
setTDRStyle()

# Do not display the canvases
gROOT.SetBatch(kTRUE)


#-------------------------------------------------------------------------------
_log_level=5
def logger(msg_level,message):
  if msg_level>=_log_level:
    print "[%s] %s" %(asctime(),message)

#-------------------------------------------------------------------------------

class Weighted(object):
  def __init__(self,name,weight=1):
    self.name=name
    self.weight=weight


#-------------------------------------------------------------------------------
class CompInfo(object):
  def __init__(self,sample1="",sample2="",release1="",release2="",run1="",run2="",tier1=0,tier2=0):
    self.sample1=sample1
    self.sample2=sample2
    self.release1=release1
    self.release2=release2
    self.run1=run1
    self.run2=run2
    self.tier1=tier1
    self.tier2=tier2
    
#-------------------------------------------------------------------------------
class Directory(Weighted):
  def __init__(self,name,mother_dir="",meta=CompInfo(),draw_success=False,do_pngs=False):
    self.mother_dir=mother_dir
    self.meta=meta
    self.subdirs=[]
    self.comparisons=[]   
    self.n_fails=0
    self.n_successes=0
    self.n_nulls=0
    self.n_skiped = 0
    self.n_comp_skiped = 0
    self.n_comp_fails=0
    self.n_comp_successes=0
    self.n_comp_nulls=0 
    self.weight=0
    self.stats_calculated=False
    Weighted.__init__(self,name)
    self.draw_success=draw_success
    self.do_pngs=do_pngs
    self.rank_histo=TH1I("rh%s"%name,"",50,-0.01,1.001)
    self.rank_histo.SetDirectory(0)
    self.different_histograms = {}
    self.different_histograms['file1']= {}
    self.different_histograms['file2']= {}
    self.filename1 = ""
    self.filename2 = ""
    self.n_missing_objs = 0
    self.full_path = ""
    
  def is_empty(self):
    if len(self.subdirs)==0 and len(self.comparisons)==0:
      return True
    return False
  
  def calcStats(self,make_pie=True):
    '''Walk all subdirs and calculate weight,fails and successes.
    Moreove propagate the sample and releases names.
    '''
    if self.stats_calculated:
      return 0
    
    self.n_fails=0
    self.n_successes=0
    self.n_nulls=0
    self.n_comp_fails=0
    self.n_comp_successes=0
    self.n_comp_nulls=0  
    self.weight=0
    
    self.n_skiped = 0
    self.n_comp_skiped = 0
    self.n_missing_objs = len(self.different_histograms['file1'].keys())+len(self.different_histograms['file2'].keys())
    if self.n_missing_objs != 0:
      print "    [*] Missing in %s: %s" %(self.filename1, self.different_histograms['file1'])
      print "    [*] Missing in %s: %s" %(self.filename2, self.different_histograms['file2'])
    # clean from empty dirs    
    self.subdirs = filter(lambda subdir: not subdir.is_empty(),self.subdirs)    
    
    for comp in self.comparisons:
      if comp.status == SKIPED: #in case its in black list & skiped 
          self.n_skiped += 1
          self.n_comp_skiped += 1
          self.weight+=1
      else: #else original code -> to check for Fails and Successes
          self.rank_histo.Fill(comp.rank)
          self.weight+=1
          if comp.status == FAIL:
              self.n_fails+=1
              self.n_comp_fails+=1
          elif comp.status == SUCCESS:
              self.n_successes+=1
              self.n_comp_successes+=1
          else:
              self.n_nulls+=1
              self.n_comp_nulls+=1

    for subdir in self.subdirs:
      subdir.mother_dir=join(self.mother_dir,self.name)
      subdir.full_path = join(self.mother_dir,self.name).replace("/Run summary","")
      subdir.calcStats(make_pie)
      subdir.meta=self.meta 
      self.weight+=subdir.weight
      self.n_fails+=subdir.n_fails
      self.n_successes+=subdir.n_successes
      self.n_nulls+=subdir.n_nulls
      
      self.n_skiped+=subdir.n_skiped
      self.n_missing_objs += subdir.n_missing_objs
      
      self.rank_histo.Add(subdir.rank_histo)

    self.stats_calculated=True
    self.full_path = join(self.mother_dir,self.name).replace("/Run summary","")
    #if make_pie:
      #self.__create_pie_image()

  def get_subdirs_dict(self):
    subdirdict={}
    for subdir in self.subdirs:
      subdirdict[subdir.name]=subdir
    return subdirdict

  def get_subdirs_names(self):
    subdirnames=[]
    for subdir in self.subdirs:
      subdirnames.append(subdir.name)
    return subdirnames

  def get_summary_chart_ajax(self,w=400,h=300):
    """Emit the ajax to build a pie chart using google apis...
    """
    url = "https://chart.googleapis.com/chart?"
    url+= "cht=p3" # Select the 3d chart
    #url+= "&chl=Success|Null|Fail" # give labels
    url+= "&chco=00FF00|FFFF00|FF0000|7A7A7A" # give colours to labels
    url+= "&chs=%sx%s" %(w,h)
    #url+= "&chtt=%s" %self.name
    url+= "&chd=t:%.2f,%.2f,%.2f,%.2f"%(self.get_success_rate(),self.get_null_rate(),self.get_fail_rate(),self.get_skiped_rate())
    
    return url

  def print_report(self,indent="",verbose=False):
    if len(indent)==0:
      self.calcStats(make_pie=False)
    # print small failure report
    if verbose:
      fail_comps=filter(lambda comp:comp.status==FAIL,self.comparisons)
      fail_comps=sorted(fail_comps,key=lambda comp:comp.name )    
      if len(fail_comps)>0:
        print indent+"* %s/%s:" %(self.mother_dir,self.name)
        for comp in fail_comps:
          print indent+" - %s: %s Test Failed (pval = %s) " %(comp.name,comp.test_name,comp.rank)
      for subdir in self.subdirs:
        subdir.print_report(indent+"  ",verbose)
    
    if len(indent)==0:
      print "\n%s - summary of %s tests:" %(self.name,self.weight)
      print " o Failiures: %.2f%% (%s/%s)" %(self.get_fail_rate(),self.n_fails,self.weight)
      print " o Nulls: %.2f%% (%s/%s) " %(self.get_null_rate(),self.n_nulls,self.weight)
      print " o Successes: %.2f%% (%s/%s) " %(self.get_success_rate(),self.n_successes,self.weight)
      print " o Skipped: %.2f%% (%s/%s) " %(self.get_skiped_rate(),self.n_skiped,self.weight)
      print " o Missing objects: %s" %(self.n_missing_objs)

  def get_skiped_rate(self):
    if self.weight == 0: return 0
    return 100.*self.n_skiped/self.weight
  def get_fail_rate(self):
    if self.weight == 0:return 0
    return 100.*self.n_fails/self.weight
    
  def get_success_rate(self):
    if self.weight == 0:return 1    
    return 100.*self.n_successes/self.weight
    
  def get_null_rate(self):
    if self.weight == 0:return 0    
    return 100.*self.n_nulls/self.weight

  def __get_full_path(self):
    #print "Mother is %s" %self.mother_dir
    if len(self.mother_dir)==0:
      return self.name
    return join(self.mother_dir,self.name)
    
  def __create_on_disk(self):
    if not exists(self.mother_dir) and len(self.mother_dir)!=0:
      makedirs(self.mother_dir)
    full_path=self.__get_full_path()    
    if not exists(full_path) and len(full_path)>0:
      makedirs(full_path)

  def get_summary_chart_name(self):
    return join(self.__get_full_path(),"summary_chart.png") 

  def __create_pie_image(self):
    self.__create_on_disk()
    vals=[]
    colors=[]
    for n,col in zip((self.n_fails,self.n_nulls,self.n_successes,self.n_skiped),(kRed,kYellow,kGreen,kBlue)):
      if n!=0:
        vals.append(n)
        colors.append(col)
    valsa=array('f',vals)
    colorsa=array('i',colors)
    can = TCanvas("cpie","TPie test",100,100);
    try:
      pie = TPie("ThePie",self.name,len(vals),valsa,colorsa);
      label_n=0
      if self.n_fails!=0:
        pie.SetEntryLabel(label_n, "Fail: %.1f(%i)" %(self.get_fail_rate(),self.n_fails) );
        label_n+=1
      if self.n_nulls!=0:
        pie.SetEntryLabel(label_n, "Null: %.1f(%i)" %(self.get_null_rate(),self.n_nulls) );      
        label_n+=1
      if self.n_successes!=0:
        pie.SetEntryLabel(label_n, "Success: %.1f(%i)" %(self.get_success_rate(),self.n_successes) );
      if self.n_skiped!=0:
        pie.SetEntryLabel(label_n, "Skipped: %.1f(%i)" %(self.get_skiped_rate(),self.n_skiped));
      pie.SetY(.52);
      pie.SetAngularOffset(0.);    
      pie.SetLabelsOffset(-.3);
      #pie.SetLabelFormat("#splitline{%val (%perc)}{%txt}");
      pie.Draw("3d  nol");
      can.Print(self.get_summary_chart_name());    
    except:
      print "self.name = %s" %self.name
      print "len(vals) = %s (vals=%s)" %(len(vals),vals)
      print "valsa = %s" %valsa
      print "colorsa = %s" %colorsa

  def prune(self,expandable_dir):
    """Eliminate from the tree the directory the expandable ones.
    """
    #print "pruning %s" %self.name
    exp_index=-1
    counter=0
    for subdir in self.subdirs:      
      # Eliminate any trace of the expandable path in the mother directories
      # for depths higher than 1
      subdir.mother_dir=subdir.mother_dir.replace("/"+expandable_dir,"")
      if subdir.name==expandable_dir:        
        exp_index=counter
      counter+=1
    
    # Did we find an expandable?
    if exp_index>=0:
      exp_dir=self.subdirs[exp_index]
      for subsubdir in exp_dir.subdirs:
        #print "*******",subsubdir.mother_dir,
        subsubdir.mother_dir=subsubdir.mother_dir.replace("/"+expandable_dir,"")
        while "//" in subsubdir.mother_dir:
          print subsubdir.mother_dir
          subsubdir.mother_dir=subsubdir.mother_dir.replace("//","/") 
        #print "*******",subsubdir.mother_dir
        self.subdirs.append(subsubdir)
          
        for comp in exp_dir.comparisons:
          comp.mother_dir=comp.mother_dir.replace("/"+expandable_dir,"")        
          while "//" in comp.mother_dir:
              comp.mother_dir
              comp.mother_dir=comp.mother_dir.replace("/")
          if not comp in self.comparisons:  #in case not to  append same comparisons few times
              self.comparisons.append(comp)  # add a comparison
              self.n_comp_fails = exp_dir.n_comp_fails  #copy to-be removed directory
              self.n_comp_nulls = exp_dir.n_comp_nulls  # numbers to parent directory
              self.n_comp_successes = exp_dir.n_comp_successes
              self.n_comp_skiped = exp_dir.n_comp_skiped
        
      del self.subdirs[exp_index]
      self.prune(expandable_dir)
    
    for subdir in self.subdirs:
      subdir.prune(expandable_dir)

  def __repr__(self):
    if self.is_empty():
      return "%s seems to be empty. Please check!" %self.name
    content="%s , Rates: Success %.2f%%(%s) - Fail %.2f%%(%s) - Null %.2f%%(%s)\n" %(self.name,self.get_success_rate(),self.n_successes,self.get_fail_rate(),self.n_fails,self.get_null_rate(),self.n_nulls)   
    for subdir in self.subdirs:
      content+=" %s\n" % subdir
    for comp in self.comparisons:
      content+=" %s\n" % comp
    return content
    
#-------------------------------------------------------------------------------
from multiprocessing import Process
def print_multi_threaded(canvas,img_name):
    canvas.Print(img_name)

tcanvas_print_processes=[]
#-------------------------------------------------------------------------------

class Comparison(Weighted):
  canvas_xsize=500
  canvas_ysize=400
  def __init__(self,name,mother_dir,h1,h2,stat_test,draw_success=False,do_pngs=False, skip=False):
    self.name=name
    self.png_name="placeholder.png"
    self.mother_dir=mother_dir
    self.img_name=""
    #self.draw_success=draw_success
    Weighted.__init__(self,name)

    stat_test.set_operands(h1,h2)
    if skip:
        self.status = SKIPED
        self.test_name=stat_test.name
        self.test_name=stat_test.name
        self.test_thr=stat_test.threshold
        self.rank = 0
    else:
        self.status=stat_test.get_status()
        self.rank=stat_test.get_rank()
        self.test_name=stat_test.name
        self.test_thr=stat_test.threshold
        self.do_pngs=do_pngs
        self.draw_success=draw_success or not do_pngs
        if ((self.status==FAIL or self.status==NULL or self.status == SKIPED or self.draw_success) and self.do_pngs):
            self.__make_image(h1,h2)      
        #self.__make_image(h1,h2)

  def __make_img_dir(self):    
    if not exists(self.mother_dir):
      makedirs(self.mother_dir)
    
  def __get_img_name(self):
    #self.__make_img_dir()    
    #print "MOTHER: ",self.mother_dir
    self.img_name="%s/%s.png"%(self.mother_dir,self.name)
    self.img_name=self.img_name.replace("Run summary","")
    self.img_name=self.img_name.replace("/","_")
    self.img_name=self.img_name.strip("_")
    #print "IMAGE NAME: %s " %self.img_name
    return self.img_name

  def tcanvas_slow(self,canvas):
    #print "About to print %s" %self.img_name
    #print_multi_threaded(canvas,self.img_name)
    #print "-->Printed"

    p = Process(target=print_multi_threaded, args=(canvas,self.img_name))
    p.start()
    tcanvas_print_processes.append(p)
    n_proc=len(tcanvas_print_processes)
    if n_proc>3:
      p_to_remove=[]
      for iprocess in xrange(0,n_proc):
        p=tcanvas_print_processes[iprocess]
        p.join()
        p_to_remove.append(iprocess)

      adjustment=0
      for iprocess in p_to_remove:
        tcanvas_print_processes.pop(iprocess-adjustment)
        adjustment+=1

  def __make_image(self,obj1,obj2):
    self.img_name=self.__get_img_name()
    if self.rank==-1:
      return 0
   
    canvas=TCanvas(self.name,self.name,Comparison.canvas_xsize,Comparison.canvas_ysize)
    objs=(obj1,obj2)

    # Add some specifics for the graphs
    obj1.SetTitle(self.name)
    
    if obj1.GetNbinsY()!=0 and not "2" in obj1.ClassName() :
      obj1 .SetLineWidth(2)
      obj2 .SetLineWidth(2)

      obj1.SetMarkerStyle(8)
      obj1.SetMarkerSize(.8)

      obj2.SetMarkerStyle(8)
      obj2.SetMarkerSize(.8)

      obj1.SetMarkerColor(kBlue)
      obj1.SetLineColor(kBlue)

      obj2.SetMarkerColor(kRed)
      obj2.SetLineColor(kRed)

      obj1.Draw("EP")
      #Statsbox      
      obj2.Draw("HistSames")
      #gPad.Update()
      #if 'stats' in map(lambda o: o.GetName(),list(gPad.GetListOfPrimitives())):
        #st = gPad.GetPrimitive("stats")      
        #st.SetY1NDC(0.575)
        #st.SetY2NDC(0.735)
        #st.SetLineColor(kRed)
        #st.SetTextColor(kRed)
        #print st      
    else:
      obj1.Draw("Colz")
      gPad.Update()
      #if 'stats' in map(lambda o: o.GetName(),list(gPad.GetListOfPrimitives())):
        #st = gPad.GetPrimitive("stats")      
        #st.SetY1NDC(0.575)
        #st.SetY2NDC(0.735)
        #st.SetLineColor(kRed)
        #st.SetTextColor(kRed)
        #print st
      obj2.Draw("ColSame")

    # Put together the TLatex for the stat test if possible    
    color=kGreen+2 # which is green, as everybody knows
    if self.status==FAIL:
      print "This comparison failed %f" %self.rank
      color=kRed
    elif self.status==NULL:
      color=kYellow
    elif self.status==SKIPED:
      color=kBlue #check if kBlue exists ;)
    
    lat_text="#scale[.7]{#color[%s]{%s: %2.2f}}" %(color,self.test_name,self.rank)
    lat=TLatex(.1,.91,lat_text)
    lat.SetNDC()
    lat.Draw()
  
    # Put also the stats together!
    n1=obj1.GetEntries()
    if n1> 100000:
      n1="%e"%n1
    else:
      n1="%s"%n1
    n2=obj2.GetEntries()
    if n2> 100000:
      n2="%e"%n2
    else:
      n2="%s"%n2

    lat_text1="#scale[.7]{#color[%s]{Entries: %s}}" %(obj1.GetLineColor(),n1)
    lat1=TLatex(.3,.91,lat_text1)
    lat1.SetNDC()
    lat1.Draw()
        
    
    lat_text2="#scale[.7]{#color[%s]{Entries: %s}}" %(obj2.GetLineColor(),n2)
    lat2=TLatex(.6,.91,lat_text2)
    lat2.SetNDC()
    lat2.Draw()
    

    self.tcanvas_slow(canvas)

  def __repr__(self):
    return "%s , (%s=%s). IMG=%s. status=%s" %(self.name,self.test_name,self.rank,self.img_name,self.status)

#-------------------------------------------------------------------------------  
