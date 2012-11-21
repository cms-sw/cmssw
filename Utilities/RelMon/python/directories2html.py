################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
# $Author: anorkus $
# $Date: 2012/10/23 15:10:13 $
# $Revision: 1.4 $

#
#                                                                              
# Danilo Piparo CERN - danilo.piparo@cern.ch                                   
#                                                                              
################################################################################

from os import chdir,getcwd,listdir,makedirs
from os.path import basename,join,exists
import cgi

import sys
theargv=sys.argv
sys.argv=[]
from ROOT import TCanvas,gStyle,TH1F,TGaxis,gPad,kRed
sys.argv=theargv

import os
if os.environ.has_key("RELMON_SA"):
  from dirstructure import Comparison,Directory
  from definitions import *
  from utils import unpickler
else:
  from Utilities.RelMon.dirstructure import Comparison,Directory
  from Utilities.RelMon.definitions import *
  from Utilities.RelMon.utils import unpickler
  
  import hashlib
#-------------------------------------------------------------------------------

def encode_obj_url(url):
  for old,new in url_encode_dict.items():
    url=url.replace(old,new)
  return url

def plot_size(h=250,w=200):
  return "w=%s;h=%s" %(h,w)

def build_obj_addr(run,sample,version,plot_path,tier):
  slash="/"
  obj_url="archive/%s/%s/%s/%s/" %(run,sample,version,tier)
  obj_url+=plot_path
  while obj_url.endswith(slash):
    obj_url=obj_url[:-1]  
  while slash*2 in obj_url:
    obj_url=obj_url.replace(slash*2,slash)
  return obj_url
  
def build_obj(run,sample,version,plot_path,tier):
  obj_url="obj=%s;" %build_obj_addr(run,sample,version,plot_path,tier)
  return encode_obj_url(obj_url)

def fairy_url(run,sample,version1,version2,plot_path,tier1,tier2,draw_opts="",h=250,w=200):
  fairy_url = "%s/%s/plotfairy/overlay?" %(server,base_url)
  fairy_url+= build_obj(run,sample,version1,plot_path,tier1)
  fairy_url+= build_obj(run,sample,version2,plot_path,tier2)
  if len(draw_opts)>0:
    fairy_url+="drawopts=%s;" %draw_opts
  fairy_url+= plot_size(h,w)
  return fairy_url

def fairy_url_single(run,sample,version,plot_path,tier,draw_opts="",h=250,w=200):
  fairy_url = "%s/%s/plotfairy/" %(server,base_url)
  fairy_url+=build_obj_addr(run,sample,version,plot_path,tier)
  fairy_url+= "?%s"%plot_size(h,w)  
  if len(draw_opts)>0:
    fairy_url+="drawopts=%s;" %draw_opts  
  return fairy_url
 
#-------------------------------------------------------------------------------
style_location="http://cms-service-reldqm.web.cern.ch/cms-service-reldqm/"
def get_page_header(directory=None,additional_header=""):
  javascripts=''
  style=''
  if directory!=None and len(directory.comparisons)>0:
    meta=directory.meta
    style='img.fail {border:1px solid #ff0000;}\n'+\
          'img.succes {border:1px solid #00ff00;}\n'+\
          'img.null {border:1px solid #ffff00;}\n'+\
          'img.skiped {border:1px solid #7a7a7a;}\n'+\
          'a.black_link:link {color: #333333}\n'+\
          'a.black_link:hover {color: #737373}\n'+\
          'a.black_link:visited {color: #333333}\n'+\
          'a.black_link:active {color: #333333}\n'
  javascripts=""
  
  
  html='<html>'+\
       '<head>'+\
       '<title>RelMon Summary</title>'+\
       '<link rel="stylesheet" href="%s/style/blueprint/screen.css" type="text/css" media="screen, projection">'%style_location+\
       '<link rel="stylesheet" href="%s/style/blueprint/print.css" type="text/css" media="print">'%style_location+\
       '<link rel="stylesheet" href="%s/style/blueprint/plugins/fancy-type/screen.css" type="text/css" media="screen, projection">'%style_location+\
       '<style type="text/css">'+\
       '.rotation {display: block;-webkit-transform: rotate(-90deg);-moz-transform: rotate(-90deg); }'+\
       '%s'%style+\
       '</style>'+\
       '%s'%javascripts+\
       '%s'%additional_header+\
       '</head>'+\
       '<body>'+\
       '<div class="container">'
  
  return html
  
#-------------------------------------------------------------------------------

def get_page_footer():
    return '</div></body></html>'

#-------------------------------------------------------------------------------

def get_title_section(directory, hashing_flag, depth=2):
  mother_name=basename(directory.mother_dir)
  mother_file_name=""
  if depth==1:
    mother_file_name="../RelMonSummary.html"
    if mother_name!="":
      mother_file_name="%s.html" %(hash_name(mother_name, hashing_flag))
  elif depth==2:
    #mother_file_name="RelMonSummary.html"
    mother_file_name="%s.html" %(hash_name("RelMonSummary", hashing_flag))
    if mother_name!="":
      mother_file_name="%s.html" %(hash_name(mother_name, hashing_flag))
  else:
      if hashing_flag:
          files = directory.mother_dir.split("/")
          if len(files) != 1:
              dir_name = files[-2] ##return the mother directory name only as the html file name by it
          else:
              dir_name = files[-1]
          mother_file_name="%s.html" %(hash_name(dir_name, hashing_flag))
      else:
          mother_file_name="%s.html" %directory.mother_dir.replace("/","_")
          mother_file_name=mother_file_name.strip("_")
      
  link_to_mother='<a href="%s">..</a>' %mother_file_name
  html= '<div class="span-20">'+\
        '<h1>%s</h1>'%directory.name+\
        '</div>'+\
        '<div class="span-1">'+\
        '<h1>%s</h1>'%link_to_mother+\
        '</div>'+\
        '<div class="span-3 last">'+\
        '<img src="http://cms-service-reldqm.web.cern.ch/cms-service-reldqm/style/CMS.gif" class="top right" width="54" hight="54">'+\
        '</div>'+\
        '<hr>' 
  if len(mother_name)>0:
    html+='<h2 class="alt">%s</h2>'% directory.mother_dir+\
          '<hr>' 
  return html
 
#-------------------------------------------------------------------------------
def get_dir_stats(directory):
  html='<p><span class="caps alt">%s comparisons:</span></p>'%directory.weight
  html+='<ul>'
  if directory.n_successes>0:
    html+='<li><span class="caps">Success: %.1f%% (%s)</span></li>'%(directory.get_success_rate(),directory.n_successes)
  if directory.n_nulls>0:
    html+='<li><span class="caps">Null: %.1f%% (%s)</span></li>'%(directory.get_null_rate(),directory.n_nulls)
  if directory.n_fails>0:
    html+='<li><span class="caps">Fail: %.1f%% (%s)</span></li>'%(directory.get_fail_rate(),directory.n_fails)
  if directory.n_skiped>0:
    html+='<li><span class="caps">Skipped: %.1f%% (%s)</span></li>'%(directory.get_skiped_rate(),directory.n_skiped)
  html+='</ul>'
  return html

#-------------------------------------------------------------------------------

def get_subdirs_section(directory, hashing_flag): 
  if len(directory.subdirs)==0:
    return ""
  html= '<div class="span-20 colborder">'
  html+='<h2 class="alt">Sub-Directories</h2>'
  # sort subdirs according to the number of fails and number of nulls and then alphabveticaly
  # so reverse :)
  sorted_subdirs= sorted(directory.subdirs, key= lambda subdir: subdir.name)
  sorted_subdirs= sorted(sorted_subdirs, key= lambda subdir: subdir.n_nulls, reverse=True)
  sorted_subdirs= sorted(sorted_subdirs, key= lambda subdir: subdir.n_fails, reverse=True)
  for subdir in sorted_subdirs:
    name=subdir.name
    if hashing_flag:
        link = "%s.html" %(hash_name(name, hashing_flag))
    else:
        link="%s_%s_%s.html" %(directory.mother_dir.replace("/","_"),directory.name.replace("/","_"),name)
        link=link.strip("_")
    html+='<div class="span-4 prepend-2 colborder">'
    html+='<h3>%s</h3>'%name
    html+='</div>'
    
    html+='<div class="span-7">'
    html+=get_dir_stats(subdir)
    html+='</div>'
    
    html+='<div class="span-6 last">'
    html+='<a href="%s"><img src="%s" class="top right"></a>'%(link,subdir.get_summary_chart_ajax(150,100))
    html+='</div>'
    
    html+='<hr>'
  return html+'</div>'

 
#-------------------------------------------------------------------------------

def get_summary_section(directory,matrix_page=True):
  
  # Hack find the first comparison and fill test and threshold
  # shall this be put in meta?
  test_name=""
  test_threshold=0
  for comparison in directory.comparisons:
      test_name=comparison.test_name
      test_threshold=comparison.test_thr
      break
  if len(test_name)==0:
    for subdir in directory.subdirs:  
      for comparison in subdir.comparisons:
        test_name=comparison.test_name
        test_threshold=comparison.test_thr
        break
      if len(test_name)!=0:break      
      if len(test_name)==0:  
        for subsubdir in subdir.subdirs:          
          for comparison in subsubdir.comparisons:
            test_name=comparison.test_name
            test_threshold=comparison.test_thr
            break
          if len(test_name)!=0:break
        if len(test_name)==0:      
          for subsubsubdir in subsubdir.subdirs:
            for comparison in subsubsubdir.comparisons:
              test_name=comparison.test_name
              test_threshold=comparison.test_thr
              break
            if len(test_name)!=0:break


  meta=directory.meta
  
  html= '<div class="span-6">'+\
        '<h3>Summary</h3>'
  html+=get_dir_stats(directory)
  html+='<a href="%s/%s">To the DQM GUI...</a>' %(server,base_url)
  html+='</div>'
        
  html+='<div class="span-7 colborder">'+\
        '<img src="%s" class="top right">'%directory.get_summary_chart_ajax(200,200)+\
        '</div>'+\
        '<div class="span-9 last">'
  if matrix_page:
    html+='<h3>Sample:</h3>'+\
          '<p class="caps">%s</p>'%meta.sample+\
          '<h3>Run1 and Run2:</h3>'+\
          '<p class="caps">%s - %s</p>'%(meta.run1,meta.run2)
  html+='<h3>Releases:</h3>'+\
        '<ul><li><p>%s</p></li><li><p>%s</p></li></ul>'%(meta.release1,meta.release2)+\
        '<h3>Statistical Test (Pvalue threshold):</h3>'+\
        '<ul><li><p class="caps">%s (%s)</p></li></ul>'%(test_name,test_threshold)+\
        '</div>'+\
        '<hr>'
  return html

#-------------------------------------------------------------------------------

def get_comparisons(category,directory):
  """Prepare the comparisons between histograms and organise them in the page.
  Moreover create separate pages with the overlay and the single plots.
  """
  counter=1
  tot_counter=1
  
  # get the right ones
  comparisons= filter (lambda comp: comp.status == cat_states[category] , directory.comparisons) 
  n_comparisons=len(comparisons)    

  is_reverse=True
  if category == FAIL:
    is_reverse=False
  comparisons=sorted(comparisons, key=lambda comp:comp.rank, reverse=is_reverse)

  
  dir_abs_path="%s/%s/" %(directory.mother_dir,directory.name)
  html_comparisons=""
  for comparison in comparisons:
    plot_name=comparison.img_name
    if "http://" not in plot_name:
      plot_name= basename(comparison.img_name)
    class_type="colborder"    
    if counter==3 or tot_counter==n_comparisons:
      class_type=" colborder last"
    comp_abs_path="%s/%s" %(dir_abs_path,comparison.name)


    if directory.do_pngs:
      png_link=comparison.img_name
      html_comparisons+='<div class="span-6 %s"><p>%s</p>' %(class_type,comparison.name)+\
                      '<p class="alt">%s: %.2E</p>' %(comparison.test_name,comparison.rank)+\
                      '<a href="%s"><img src="%s" width="250" height="200" class="top center %s"></a></div>'%(png_link,png_link,cat_classes[category])
    else:
      big_fairy=fairy_url(directory.meta.run1,
                        directory.meta.sample,
                        directory.meta.release1,
                        directory.meta.release2,
                        comp_abs_path,
                        directory.meta.tier1,
                        directory.meta.tier2,
                        "",600,600)
      small_fairy=fairy_url(directory.meta.run1,
                        directory.meta.sample,
                        directory.meta.release1,
                        directory.meta.release2,
                        comp_abs_path,
                        directory.meta.tier1,
                        directory.meta.tier2)    

      single_fairy1=fairy_url_single(directory.meta.run1,
                                   directory.meta.sample,
                                   directory.meta.release1,
                                   comp_abs_path,
                                   directory.meta.tier1,
                                   "",500,500)
      single_fairy2=fairy_url_single(directory.meta.run2,
                                   directory.meta.sample,
                                   directory.meta.release2,
                                   comp_abs_path,
                                   directory.meta.tier2,
                                   "",500,500)

      html_comparisons+='<div class="span-6 %s"><p>%s</p>' %(class_type,comparison.name)+\
                      '<p class="alt">%s: %.2E</p>' %(comparison.test_name,comparison.rank)+\
                      '<div><a class="black_link" href="%s">%s</a></div>'%(single_fairy1,directory.meta.release1)+\
                      '<div><a href="%s">%s</a></div>'%(single_fairy2,directory.meta.release2)+\
                      '<a href="%s"><img src="%s" class="top center %s"></a></div>'%(big_fairy,small_fairy,cat_classes[category])

    if counter==3:                    
      html_comparisons+="<hr>"
      counter=0
    counter+=1
    tot_counter+=1

  if len(html_comparisons)!=0:
    html='<div class="span-20"><h2 class="alt">%s Comparisons</h2></div>' %cat_names[category]
    html+=html_comparisons
    html+='<hr>'
    return html
  return ""

#-------------------------------------------------------------------------------

def get_rank_section(directory):
# do Rank histo png
    imgname="RankSummary.png"
    gStyle.SetPadTickY(0)
    c=TCanvas("ranks","ranks",500,400)
    #gStyle.SetOptStat(0)
    c.cd()

    h=directory.rank_histo
    rank_histof=TH1F(h.GetName(),"",h.GetNbinsX(),h.GetXaxis().GetXmin(),h.GetXaxis().GetXmax())
    rank_histof.SetLineWidth(2)
    for i in xrange(0,h.GetNbinsX()+1):
      rank_histof.SetBinContent(i,h.GetBinContent(i))
    h.SetTitle("Ranks Summary;Rank;Frequency")
    h.Draw("Hist")
    c.Update()
    rank_histof.ComputeIntegral()
    integral = rank_histof.GetIntegral()
    rank_histof.SetContent(integral)

    rightmax = 1.1*rank_histof.GetMaximum()
    scale = gPad.GetUymax()/rightmax
    rank_histof.SetLineColor(kRed)
    rank_histof.Scale(scale)
    rank_histof.Draw("same")

    #draw an axis on the right side
    axis = TGaxis(gPad.GetUxmax(),gPad.GetUymin(),gPad.GetUxmax(), gPad.GetUymax(),0,rightmax,510,"+L")
    axis.SetTitle("Cumulative")
    axis.SetTitleColor(kRed)
    axis.SetLineColor(kRed)
    axis.SetLabelColor(kRed)
    axis.Draw()

    rank_histof.Draw("Same");
    

    c.Print(imgname)

    page_html='<div class="span-20"><h2 class="alt"><a name="rank_summary">Ranks Summary</a></h2>'
    page_html+='<div class="span-19"><img src="%s"></div>' %imgname
    page_html+='</div> <a href="#top">Top...</a><hr>'

    return page_html
    
#-------------------------------------------------------------------------------

def directory2html(directory, hashing, depth=0):
  """Converts a directory tree into html pages, very nice ones.
  """
  #print "d2html: depth", str(depth)," dir ",directory.name
  depth+=1
  #old_cwd=getcwd()
  #if not exists(directory.name) and len(directory.name)>0:
    #makedirs(directory.name)
  
  #if len(directory.name)>0:
    #chdir(directory.name)
  
  for subdir in directory.subdirs:
    directory2html(subdir,hashing, depth)
  
  page_html=get_page_header(directory)+\
            get_title_section(directory,hashing, depth)+\
            get_summary_section(directory)+\
            get_subdirs_section(directory, hashing)

  for do_cat,cat in ((directory.n_comp_fails >0,FAIL ),
                     (directory.n_comp_nulls >0,NULL ),
                     
                     (directory.n_comp_successes >0 and directory.draw_success,SUCCESS ),
                     (directory.n_comp_skiped >0,SKIPED)):
    if do_cat:
      page_html+=get_comparisons(cat,directory)

  # Distribution of ranks

  if depth==1:
    page_html+=get_rank_section(directory)


  page_html+=get_page_footer()

  page_name=directory.name

  if len(page_name)==0:
    page_name="RelMonSummary"
  if hashing:
      ofilename = "%s.html" %(hash_name(page_name, hashing))
  else:
      ofilename="%s_%s.html" %(directory.mother_dir.replace("/","_"),page_name)
      ofilename=ofilename.strip("_")
  #print "Writing on %s" %ofilename
  ofile=open(ofilename,"w")
  ofile.write(page_html)
  ofile.close()

  #chdir(old_cwd)

#-------------------------------------------------------------------------------

def build_gauge(total_success_rate,minrate=.80,small=False,escaped=False):
  total_success_rate_scaled=(total_success_rate-minrate)
  total_success_rate_scaled_repr=total_success_rate_scaled/(1-minrate)
  if total_success_rate_scaled_repr<0:
    total_success_rate_scaled_repr=0
  size_s="200x100"
  if small:
    size_s="40x30"
  #print "Total success rate %2.2f and scaled %2.2f "%(total_success_rate,total_success_rate_scaled)
  gauge_link ="https://chart.googleapis.com/chart?chs=%s&cht=gom"%size_s
  gauge_link+="&chd=t:%2.1f"%(total_success_rate_scaled_repr*100.)
  if not small:
    gauge_link+="&chxt=x,y&chxl=0:|%2.1f%%|1:|%i%%|%i%%|100%%"%(total_success_rate*100,minrate*100.,(1+minrate)*50)
    gauge_link+="&chma=10,10,10,0"
  img_tag= '<img src="%s">'%gauge_link
  if escaped:
    img_tag=cgi.escape(img_tag)    
  return img_tag

#-------------------------------------------------------------------------------

def get_aggr_pairs_info(dir_dict,the_aggr_pairs=[]):
  # Let's make a summary for an overview in categories act on the global dir
  aggr_pairs_info=[]#(name,{directories names:{nsucc: nsucc,weight:weight}},weight,success_rate)

  list_of_names=[]
  if the_aggr_pairs==[]:
    for samplename,sampledir in dir_dict.items():
      for subsysdirname in sorted(sampledir.get_subdirs_dict().keys()):
        if not subsysdirname in list_of_names:
          list_of_names.append(subsysdirname)
          the_aggr_pairs.append((subsysdirname,[subsysdirname]))  
          
  #print the_aggr_pairs
  for cat_name, subdir_list in the_aggr_pairs:
    total_successes=0.
    total_directory_successes=0
    total_weight=0.    
    present_subdirs={}
    total_ndirs=0
    # Loop on samples
    for dirname, sampledir in dir_dict.items():
      # Loop on directories
      for subdirname,subdir in sampledir.get_subdirs_dict().items():        
        if subdirname in subdir_list:          
          nsucc=subdir.n_successes
          total_successes+=nsucc
          weight=subdir.weight
          total_weight+=weight
          total_ndirs+=1
          
          total_directory_successes+= float(nsucc)/weight
          if present_subdirs.has_key(subdirname):
            this_dir_dict=present_subdirs[subdirname]
            this_dir_dict["nsucc"]+=nsucc
            this_dir_dict["weight"]+=weight
          else:
            present_subdirs[subdirname]={"nsucc":nsucc,"weight":weight}
        # Make it usable also for subdirectories
        for subsubdirname,subsubdir in subdir.get_subdirs_dict().items():          
          for pathname in filter(lambda name:"/" in name,subdir_list):           
            selected_subdirname,selected_subsubdirname = pathname.split("/")
            if selected_subdirname == subdirname and selected_subsubdirname==subsubdirname:
              #print "Studying directory ",subsubdirname," in directory ",subdirname
              nsucc=subsubdir.n_successes
              total_successes+=nsucc
              weight=subsubdir.weight
              total_weight+=weight
              total_ndirs+=1              
              total_directory_successes+= float(nsucc)/weight
              
              if present_subdirs.has_key(subsubdirname):
                this_dir_dict=present_subdirs[subsubdirname]
                this_dir_dict["nsucc"]+=nsucc
                this_dir_dict["weight"]+=weight
              else:
                present_subdirs[subsubdirname]={"nsucc":nsucc,"weight":weight}      

    if total_ndirs == 0:
      print "No directory of the category %s is present in the samples: skipping." %cat_name
      continue
    
    average_success_rate=total_directory_successes/(total_ndirs)
    aggr_pairs_info.append((cat_name,present_subdirs,total_weight,average_success_rate))
    
  return aggr_pairs_info

#-------------------------------------------------------------------------------

def make_categories_summary(dir_dict,aggregation_rules):
    
  aggr_pairs_info= get_aggr_pairs_info(dir_dict,aggregation_rules)
  
  #print aggr_pairs_info
  
  # Now Let's build the HTML
  
  html= '<div class="span-20 colborder">'
  html+='<h2 class="alt"><a name="categories">Categories:</a></h2>'

  for cat_name,present_subdirs,total_weight,average_success_rate in aggr_pairs_info:
    #print cat_name,present_subdirs,total_weight,average_success_rate
    html+='<div class="span-3 prepend-0 colborder">'
    html+='<h3>%s</h3>'%cat_name
    html+='<div><span class="alt">Avg. Success rate:</span></div>'
    html+='<div><span class="alt">%2.1f%%</span></div>'%(average_success_rate*100)
    html+='</div>'    
    html+='<div class="span-9">'

    html+='<div><p><span class="caps alt">DQM Directories (%i comparisons):</span></p></div>' %total_weight
    html+='<div><p><span class="alt">name: succ. rate - rel. weight</span></p></div>'
    html+='<ul>'    
    for subdirname in sorted(present_subdirs.keys()):
      this_dir_dict=present_subdirs[subdirname]
      nsucc=this_dir_dict["nsucc"]
      weight=this_dir_dict["weight"]
      html+='<li><span class="caps">%s: %2.1f%% - %2.1f%%</span></li>'%(subdirname,100*float(nsucc)/weight,100*float(weight)/total_weight)
    html+='</ul>'    
    html+='</div>'
    
    html+='<div class="span-6 last">'
    html+=build_gauge(average_success_rate)
    html+='</div>'
    
    html+='<hr>'
  return html+'<br><a href="#top">Top...</a> </div><hr>'  
    
 #-------------------------------------------------------------------------------

def make_twiki_table(dir_dict,aggregation_rules):
  
  # decide the release
  meta= dir_dict.items()[0][1].meta
  releases=sorted([meta.release1,meta.release2])
  latest_release=releases[1].split("-")[0]
  
  
  aggr_pairs_info= get_aggr_pairs_info(dir_dict,aggregation_rules)
  
  # Now Let's build the HTML
  
  html= '<div class="span-20 colborder">'
  html+='<h2 class="alt"><a name="twiki_table">Twiki snipppet for release managers</a></h2>'
  html+='<div>| Release | Comparison |'
  for cat_name,present_subdirs,total_weight,average_success_rate in aggr_pairs_info:
    html+=cat_name
    html+=" | "
  html+='</div>'
  
  html+='<div>| %s |  %%ICON{arrowdot}%%  | '%latest_release

  # Now add all the line with small gauges

  for cat_name,present_subdirs,total_weight,average_success_rate in aggr_pairs_info:
    #print cat_name,present_subdirs,total_weight,average_success_rate
    html+=build_gauge(average_success_rate,small=True,escaped=True)
    html+=" | "    
  
  html+='</div> <a href="#top">Top...</a>'
  html+='<hr>'
  return html+'</div>'
  
#-------------------------------------------------------------------------------

def get_pie_tooltip(directory):
  tooltip="%s\nS:%2.1f%% N:%2.1f%% F:%2.1f%% Sk:%2.1f%%" %(directory.name,directory.get_success_rate(),directory.get_null_rate(),directory.get_fail_rate(),directory.get_skiped_rate())
  return tooltip

#-------------------------------------------------------------------------------

def make_barchart_summary(dir_dict,name="the_chart",title="DQM directory",the_aggr_pairs=[]):  
  
  aggr_pairs_info= get_aggr_pairs_info(dir_dict,the_aggr_pairs)       

  script="""
    <script type="text/javascript" src="https://www.google.com/jsapi"></script>
    <script type="text/javascript">
      google.load("visualization", "1", {packages:["corechart"]});
      google.setOnLoadCallback(drawChart);
      function drawChart() {
        var data = new google.visualization.DataTable();
        data.addColumn('string', 'DQM Directory');
        data.addColumn('number', 'Success Rate');
        """
  script+="data.addRows(%i);\n"%len(aggr_pairs_info)
  counter=0
  for subsystname,present_directories,weight,success_rate in aggr_pairs_info:
    #print subsystname,present_directories
    script+="data.setValue(%i, 0, '%s');\n"%(counter,subsystname)
    script+="data.setValue(%i, 1, %2.2f);\n"%(counter,success_rate)
    counter+=1
  script+="""
        var chart = new google.visualization.BarChart(document.getElementById('%s'));
        chart.draw(data, {width: 1024, height: %i, title: 'Success Rate',
                          vAxis: {title: '%s', titleTextStyle: {color: 'red'},textStyle: {fontSize: 14}}
                         });
      }
    </script>
    """%(name,40*counter,title)
  return script


#-------------------------------------------------------------------------------

def make_summary_table(indir,aggregation_rules,aggregation_rules_twiki, hashing_flag):
  """Create a table, with as rows the directories and as columns the samples.
  Each box in the table will contain a pie chart linking to the directory.
  """  
  #aggregation_rules={}
  #aggregation_rules_twiki={}

  chdir(indir)
  if os.path.isabs(indir):
      title = basename(indir)
  else:
      title=indir
  title=title.strip(".")
  title=title.strip("/")
  
  
  # Get the list of pickles
  sample_pkls=filter(lambda name: name.endswith(".pkl"),listdir("./"))
  
  # Load directories, build a list of all first level subdirs  
  dir_unpicklers=[]
  n_unpicklers=0
  for sample_pkl in sample_pkls:
    dir_unpickler=unpickler(sample_pkl)
    dir_unpickler.start()
    n_unpicklers+=1
    dir_unpicklers.append(dir_unpickler)
    if n_unpicklers>=1: #pickleing is very expensive. Do not overload cpu
      n_unpicklers=0
      for dir_unpickler in dir_unpicklers:
        dir_unpickler.join()
  
  dir_dict={}
  
  # create a fake global directory
  global_dir=Directory("global","")  
  for dir_unpickler in dir_unpicklers:
    dir_unpickler.join()
    directory=dir_unpickler.directory
    #directory.prune("Run summary")    
    #directory.calcStats()
    global_dir.meta=directory.meta
    dir_dict[dir_unpickler.filename.replace(".pkl","")]=directory
    global_dir.subdirs.append(directory)
  
  global_dir.calcStats()
  
  directories_barchart=make_barchart_summary(dir_dict,'dir_chart',"DQM Directory")
  categories_barchart=make_barchart_summary(dir_dict,'cat_chart','Category',aggregation_rules)
  
  page_html = get_page_header(additional_header=directories_barchart+categories_barchart)
  rel1=""
  rel2=""
  try:
    rel1,rel2=title.split("VS")
  except:
    rel1=global_dir.meta.release1.split("-")[0]
    rel2=global_dir.meta.release2.split("-")[0] 
    global_dir.meta.release1=rel1
    global_dir.meta.release2=rel2
    
  # union of all subdirs names
  all_subdirs=[]
  for directory in dir_dict.values():
    for subdir_name in directory.get_subdirs_names():
      all_subdirs.append(subdir_name)
  all_subdirs=sorted(list(set(all_subdirs)))
  
  # Get The title
  page_html+= '<div class="span-20">'+\
              '<h2><a name="top" href="https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon">RelMon</a> Global Report: %s</h2>'%title+\
              '</div>'+\
              '<div class="span-1">'+\
              '<h2><a href="%s">main...</a></h2>' %relmon_mainpage+\
              '</div>'+\
              '<hr>'
  page_html+='<div class="span-24"><p></p></div>\n'*3
  
  # Get The summary section
  page_html+= get_summary_section(global_dir,False)  

  # Make the anchor sections
  page_html+= '<div class="span-24">'
  page_html+= '<div class="span-20 colborder"><h2 class="alt">Sections:</h2>'+\
              '<ul>'+\
              '<li><a href="#summary_barchart">Summary Barchart</a></li>'+\
              '<li><a href="#categories">Categories</a></li>'+\
              '<li><a href="#detailed_barchart">Detailed Barchart</a></li>'+\
          '<li><a href="#summary_table">Summary Table</a></li>'+\
              '<li><a href="#rank_summary">Ranks Summary</a></li>'+\
              '<li><a href="#twiki_table">Twiki Table</a></li>'+\
              '</ul>'+\
              '</div><hr>'


  # Make the CategoriesBar chart
  page_html+='<div class="span-24"><h2 class="alt"><a name="summary_barchart">Summary Barchart</a></h2></div>'
  page_html+='<div id="cat_chart"></div> <a href="#top">Top...</a><hr>'

  # Make the gauges per categories
  page_html+=make_categories_summary(dir_dict,aggregation_rules)

  # Make the Directories chart
  page_html+='<div class="span-24"><h2 class="alt"><a name="detailed_barchart">Detailed Barchart</a></h2></div>'
  page_html+='<div id="dir_chart"></div> <a href="#top">Top...</a><hr>'
  
  # Barbarian vertical space. Suggestions are welcome
  for i in xrange(2):
    page_html+='<div class="span-24"><p></p></div>\n'

 
  # Prepare the table
  page_html+='<div class="span-24"><h2 class="alt"><a name="summary_table">Summary Table</a></h2></div>'

  for i in xrange(5):
    page_html+='<div class="span-24"><p></p></div>\n'
    
  page_html+="""
        <table border="1" >
          <tr>
          <td> </td>          
  """
  
  # First row with samples
  page_html+="""
          <td><div class="span-1"><p class="rotation" style="alt"><b>Summary</b></p></div></td>"""

  sorted_samples=sorted(dir_dict.keys())
  for sample in sorted_samples:
    sample_nick=sample
    ## For runs: put only the number after the _
    #if "_" in sample:
      #run_number=sample.split("_")[-1]      
      #if (not run_number.isalpha()) and len(run_number)>=6:
    #sample_nick=run_number
      
      
    page_html+="""
          <td><div class="span-1"><p class="rotation" style="">%s</p></div></td>"""%sample_nick
  page_html+="          </tr>\n"


 # FIRST ROW
 # Now the summaries  at the beginning of the table
  page_html+="<tr>"
  page_html+='<td  style="background-color:white;"><div class="span-1">'
  
  page_html+='<b>Summary</b></div></td>'
  page_html+='<td style="background-color:white;" class = "colborder" ><div class="span-1"><img src="%s" alt="%s"></div></td>'%(global_dir.get_summary_chart_ajax(55,55),get_pie_tooltip(global_dir))
  for sample in sorted_samples:
    col=dir_dict[sample]
    # check if the directory was a top one or not
    summary_page_name="RelMonSummary.html"
    if col.name!="":
      summary_page_name=hash_name(col.name, hashing_flag)+".html"
    img_link=col.get_summary_chart_ajax(55,55)
    page_html+='<td  style="background-color:white;"><div class="span-1">'
    page_html+='<a href="%s/%s"><img src="%s" title="%s"></a></div></td>' %(sample,summary_page_name,img_link,get_pie_tooltip(col))
  page_html+="</tr>"

  # Now the content
  for subdir_name in all_subdirs:  

    page_html+='          <tr>\n'
    page_html+='          <td style="background-color:white;">%s</td>\n' %subdir_name  

    row_summary=Directory("row_summary","")
    sample_counter=0
    n_samples=len(sorted_samples)    

    for sample in sorted_samples:
      subdirs_dict=directory.get_subdirs_dict()
      directory=dir_dict[sample]
      dir_is_there=subdirs_dict.has_key(subdir_name)
      if dir_is_there:
        row_summary.subdirs.append(subdirs_dict[subdir_name])

    # one first row for the summary!
    row_summary.calcStats()
    img_link=row_summary.get_summary_chart_ajax(55,55)
    page_html+='<td  style="background-color:white;"><div class="span-1">'
    page_html+='<img src="%s" title="%s"></div></td>' %(img_link,get_pie_tooltip(row_summary))

    for sample in sorted_samples:
      sample_counter+=1      

      directory=dir_dict[sample]
      subdirs_dict=directory.get_subdirs_dict()

      # Check if the directory is the top one
      summary_page=join(sample,"%s.html"%(hash_name(subdir_name, hashing_flag)))
      if directory.name!="":
        # We did not run on the topdir     
        summary_page=join(sample,"%s_%s.html"%(directory.name,hash_name(subdir_name,hashing_flag)))
      dir_is_there=subdirs_dict.has_key(subdir_name)

      img_link="https://chart.googleapis.com/chart?cht=p3&chco=C0C0C0&chs=50x50&chd=t:1"
      img_tooltip="N/A"
      if dir_is_there:
        #row_summary.subdirs.append(subdirs_dict[subdir_name])
        img_link=subdirs_dict[subdir_name].get_summary_chart_ajax(50,50)
        img_tooltip=get_pie_tooltip(subdirs_dict[subdir_name])

      page_html+='<td  style="background-color:white;"><div class="span-1">'
      if dir_is_there:
        page_html+='<a href="%s">'%(summary_page)
      page_html+='<img src="%s" title="%s" height=50 width=50>' %(img_link,img_tooltip)
      if dir_is_there:
        page_html+='</a>'
      page_html+='</div></td>' 

    page_html+="          </tr>\n"        



  page_html+='</table> <a href="#top">Top...</a><hr>'

  page_html+=get_rank_section(global_dir)

  page_html+=make_twiki_table(dir_dict,aggregation_rules_twiki)

  page_html+=get_page_footer()
  return page_html  


#-----------UPDATES------
def hash_name(file_name, flag):
    if flag: #if hashing flag is ON then return
        return hashlib.md5(file_name).hexdigest()[:10] #md5 hashed file name with length 10
    else:
        return file_name #return standart name
