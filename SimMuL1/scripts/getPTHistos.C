void getPTHistos(TString dname, TString f_def, TString f_gem, TString gem_label)
{
  if (dname.Contains("_pat8"))      f_def += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_def_pat8.root";
  if (dname == "minbias_pt05_pat8") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt05_pat8.root";
  if (dname == "minbias_pt06_pat8") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt06_pat8.root";
  if (dname == "minbias_pt10_pat8") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt10_pat8.root";
  if (dname == "minbias_pt15_pat8") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt15_pat8.root";
  if (dname == "minbias_pt20_pat8") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt20_pat8.root";
  if (dname == "minbias_pt30_pat8") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt30_pat8.root";
  if (dname == "minbias_pt40_pat8") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt40_pat8.root";
  
  if (dname.Contains("_pat2"))      f_def += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_def_pat2.root";
  if (dname == "minbias_pt05_pat2") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt05_pat2.root";
  if (dname == "minbias_pt06_pat2") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt06_pat2.root";
  if (dname == "minbias_pt10_pat2") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt10_pat2.root";
  if (dname == "minbias_pt15_pat2") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt15_pat2.root";
  if (dname == "minbias_pt20_pat2") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt20_pat2.root";
  if (dname == "minbias_pt30_pat2") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt30_pat2.root";
  if (dname == "minbias_pt40_pat2") f_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt40_pat2.root";
  
  result_def = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_1b", "_def");
  result_def_2s = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_2s_1b", "_def");
  result_def_3s1b = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_3s1b_1b", "_def");
  result_def_2s1b = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_2s1b_1b", "_def");
  result_def_2s123 = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_2s123_1b", "_def");
  result_def_2s13 = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_2s13_1b", "_def");
  result_def_eta_all = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s", "_def");
  result_def_eta_all_3s1b = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_3s1b", "_def");
  result_def_eta_no1a = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_no1a", "_def");
  result_def_eta_no1a_3s1b = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_3s1b_no1a", "_def");
  result_gmtsing = getPTHisto(f_def, dir, "h_rt_gmt_ptmax_sing_1b", "_def");
  result_def_gmtsing = getPTHisto(f_def, dir, "h_rt_gmt_ptmax_sing_1b", "_def");
  result_def_gmtsing_no1a = getPTHisto(f_def, dir, "h_rt_gmt_ptmax_sing6_no1a", "_def");
 
  result_gem = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_3s1b_1b", "_gem");
  result_gem_2s1b = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_2s1b_1b", "_gem");
  result_gem_2s123 = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_2s123_1b", "_gem");
  result_gem_2s13 = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_2s13_1b", "_gem");
  result_gem_eta_all = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_3s1b", "_gem");
  result_gem_eta_no1a = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_3s1b_no1a", "_gem");
  //result_gmtsing = getPTHisto(f_gem, dir, "h_rt_gmt_ptmax_sing_1b", "_def");
  result_gem_gmtsing_no1a = getPTHisto(f_def, dir, "h_rt_gmt_ptmax_sing6_3s1b_no1a", "_def");

}
