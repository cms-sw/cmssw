from ROOT import *

def getPTHisto(f_name, dir_name, h_name, clone_suffix = "_cln"):
  f = TFile.Open(f_name)
  h0 = f.Get("%s/%s"%(dir_name,h_name)).Clone(h_name + clone_suffix)
  return h0

def getPtHistos(f_def, f_gem, dname):

  if "_pat8" in dname: f_def += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_def_pat8.root"
  if dname == "minbias_pt05_pat8" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt05_pat8.root"
  if dname == "minbias_pt06_pat8" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt06_pat8.root"
  if dname == "minbias_pt10_pat8" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt10_pat8.root"
  if dname == "minbias_pt15_pat8" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt15_pat8.root"
  if dname == "minbias_pt20_pat8" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt20_pat8.root"
  if dname == "minbias_pt30_pat8" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt30_pat8.root"
  if dname == "minbias_pt40_pat8" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt40_pat8.root"
  
  if "_pat2" in dname : f_def += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_def_pat2.root"
  if dname == "minbias_pt05_pat2" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt05_pat2.root"
  if dname == "minbias_pt06_pat2" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt06_pat2.root"
  if dname == "minbias_pt10_pat2" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt10_pat2.root"
  if dname == "minbias_pt15_pat2" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt15_pat2.root"
  if dname == "minbias_pt20_pat2" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt20_pat2.root"
  if dname == "minbias_pt30_pat2" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt30_pat2.root"
  if dname == "minbias_pt40_pat2" : f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt40_pat2.root"
  
  result_def = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_1b", "_def")
  result_def_2s = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_2s_1b", "_def")
  result_def_3s1b = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_3s1b_1b", "_def")
  result_def_2s1b = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_2s1b_1b", "_def")
  result_def_eta_all = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s", "_def")
  result_def_eta_all_3s1b = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_3s1b", "_def")
  result_def_eta_no1a = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_no1a", "_def")
  result_def_eta_no1a_3s1b = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_3s1b_no1a", "_def")
  result_gmtsing = getPTHisto(f_def, dir, "h_rt_gmt_ptmax_sing_1b", "_def")
  
  result_gem = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_3s1b_1b", "_gem")
  result_gem_2s1b = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_2s1b_1b", "_gem")
  result_gem_eta_all = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_3s1b", "_gem")
  result_gem_eta_no1a = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_3s1b_no1a", "_gem")
  ##result_gmtsing = getPTHisto(f_gem, dir, "h_rt_gmt_ptmax_sing_1b", "_def") 

if __name__ == "__main__":
  #print confirmation message - not supposed to work in standalone mode
  print "It works!"
