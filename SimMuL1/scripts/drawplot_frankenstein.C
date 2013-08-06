
/*
.L drawplot_gmtrt.C

*/

TString ext = ".png";
TString filesDir = "files/";
TString plotDir = "plots/";

void getPTHistos(TString dname)
{
TString  f_def = filesDir;
TString  f_gem = filesDir;

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
result_def_gmtsing = getPTHisto(f_def, dir, "h_rt_gmt_ptmax_sing_1b", "_def");
result_def_gmtsing_no1a = getPTHisto(f_def, dir, "h_rt_gmt_ptmax_sing6_no1a", "_def");

result_gem = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_3s1b_1b", "_gem");
result_gem_2s1b = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_2s1b_1b", "_gem");
result_gem_2s123 = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_2s123_1b", "_gem");
result_gem_2s13 = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_2s13_1b", "_gem");
result_gem_eta_all = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_3s1b", "_gem");
result_gem_eta_no1a = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_3s1b_no1a", "_gem");
//result_gem_gmtsing = getPTHisto(f_gem, dir, "h_rt_gmt_ptmax_sing_1b", "_def");
result_gem_gmtsing_no1a = getPTHisto(f_def, dir, "h_rt_gmt_ptmax_sing6_3s1b_no1a", "_def");
}



void drawplot_frankenstein()
{

  gROOT->ProcessLine(".L drawplot_gmtrt.C");

  //gem_dir = "gem_vadim/";
  gem_dir = plotDir;
  gem_label = "gem98";

//gem_dir = "gem95/"; gem_label = "gem95";

//do_not_print = true;

//gROOT->SetBatch(true);

//gStyle->SetStatW(0.13);
//gStyle->SetStatH(0.08);
gStyle->SetStatW(0.07);
gStyle->SetStatH(0.06);

gStyle->SetOptStat(0);

gStyle->SetTitleStyle(0);
gStyle->SetTitleAlign(13);// coord in top left
gStyle->SetTitleX(0.);
gStyle->SetTitleY(1.);
gStyle->SetTitleW(1);
gStyle->SetTitleH(0.058);
gStyle->SetTitleBorderSize(0);

gStyle->SetPadLeftMargin(0.126);
gStyle->SetPadRightMargin(0.04);
gStyle->SetPadTopMargin(0.06);
gStyle->SetPadBottomMargin(0.13);

gStyle->SetMarkerStyle(1);

TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
gPad->SetLogx(1);
gPad->SetGridx(1);gPad->SetGridy(1);

TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
gPad->SetLogx(1);gPad->SetLogy(1);
gPad->SetGridx(1);gPad->SetGridy(1);


// ********** PAT2 **********

getPTHistos("minbias_pt06_pat2");
hh = (TH1D*)result_def_3s1b->Clone("gem_new");
for (int b = hh->FindBin(6.01); b <= hh->GetNbinsX(); ++b) hh->SetBinContent(b, 0);
hh_all = (TH1D*)result_def_eta_all_3s1b->Clone("gem_new_eta_all");
for (int b = hh_all->FindBin(6.01); b <= hh_all->GetNbinsX(); ++b) hh_all->SetBinContent(b, 0);
hh_no1a = (TH1D*)result_def_eta_no1a_3s1b->Clone("gem_new_eta_no1a");
for (int b = hh_no1a->FindBin(6.01); b <= hh_no1a->GetNbinsX(); ++b) hh_no1a->SetBinContent(b, 0);
hh_2s1b = (TH1D*)result_def_2s1b->Clone("gem_new_2s1b");
for (int b = hh_2s1b->FindBin(6.01); b <= hh_2s1b->GetNbinsX(); ++b) hh_2s1b->SetBinContent(b, 0);
hh_2s123 = (TH1D*)result_def_2s123->Clone("gem_new_2s123");
for (int b = hh_2s123->FindBin(6.01); b <= hh_2s123->GetNbinsX(); ++b) hh_2s123->SetBinContent(b, 0);
hh_2s13 = (TH1D*)result_def_2s13->Clone("gem_new_2s13");
for (int b = hh_2s13->FindBin(6.01); b <= hh_2s13->GetNbinsX(); ++b) hh_2s13->SetBinContent(b, 0);
hh_sing_no1a = (TH1D*)result_gem_gmtsing_no1a->Clone("hh_sing_no1a");
for (int b = hh_sing_no1a->FindBin(6.01); b <= hh_sing_no1a->GetNbinsX(); ++b) hh_sing_no1a->SetBinContent(b, 0);

h06 = (TH1D*)result_gem->Clone("gem_new_06");
for (int b = h06->FindBin(6.01); b < h06->FindBin(10.01); ++b) {hh->SetBinContent(b, h06->GetBinContent(b)); hh->SetBinError(b, h06->GetBinError(b));}
h06_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_06");
for (int b = h06_all->FindBin(6.01); b < h06_all->FindBin(10.01); ++b) {hh_all->SetBinContent(b, h06_all->GetBinContent(b)); hh_all->SetBinError(b, h06_all->GetBinError(b));}
h06_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_06");
for (int b = h06_no1a->FindBin(6.01); b < h06_no1a->FindBin(10.01); ++b) {hh_no1a->SetBinContent(b, h06_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h06_no1a->GetBinError(b));}
h06_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_06");
for (int b = h06_2s1b->FindBin(6.01); b < h06_2s1b->FindBin(10.01); ++b) {hh_2s1b->SetBinContent(b, h06_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h06_2s1b->GetBinError(b));}
h06_2s123 = (TH1D*)result_gem_2s123->Clone("gem_new_2s123_06");
for (int b = h06_2s123->FindBin(6.01); b < h06_2s123->FindBin(10.01); ++b) {hh_2s123->SetBinContent(b, h06_2s123->GetBinContent(b)); hh_2s123->SetBinError(b, h06_2s123->GetBinError(b));}
h06_2s13 = (TH1D*)result_gem_2s13->Clone("gem_new_2s13_06");
for (int b = h06_2s13->FindBin(6.01); b < h06_2s13->FindBin(10.01); ++b) {hh_2s13->SetBinContent(b, h06_2s13->GetBinContent(b)); hh_2s13->SetBinError(b, h06_2s13->GetBinError(b));}
h06_sing_no1a = (TH1D*)result_gem_gmtsing_no1a->Clone("gem_new_sing_06");
for (int b = h06_sing_no1a->FindBin(6.01); b < h06_sing_no1a->FindBin(10.01); ++b) {hh_sing_no1a->SetBinContent(b, h06_sing_no1a->GetBinContent(b)); hh_sing_no1a->SetBinError(b, h06_sing_no1a->GetBinError(b));}

getPTHistos("minbias_pt10_pat2");
h10 = (TH1D*)result_gem->Clone("gem10");
for (int b = h10->FindBin(10.01); b < h10->FindBin(15.01); ++b) {hh->SetBinContent(b, h10->GetBinContent(b)); hh->SetBinError(b, h10->GetBinError(b));}
h10_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_10");
for (int b = h10_all->FindBin(10.01); b < h10_all->FindBin(15.01); ++b) {hh_all->SetBinContent(b, h10_all->GetBinContent(b)); hh_all->SetBinError(b, h10_all->GetBinError(b));}
h10_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_10");
for (int b = h10_no1a->FindBin(10.01); b < h10_no1a->FindBin(15.01); ++b) {hh_no1a->SetBinContent(b, h10_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h10_no1a->GetBinError(b));}
h10_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_10");
for (int b = h10_2s1b->FindBin(10.01); b < h10_2s1b->FindBin(15.01); ++b) {hh_2s1b->SetBinContent(b, h10_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h10_2s1b->GetBinError(b));}
h10_2s123 = (TH1D*)result_gem_2s123->Clone("gem_new_2s123_10");
for (int b = h10_2s123->FindBin(10.01); b < h10_2s123->FindBin(15.01); ++b) {hh_2s123->SetBinContent(b, h10_2s123->GetBinContent(b)); hh_2s123->SetBinError(b, h10_2s123->GetBinError(b));}
h10_2s13 = (TH1D*)result_gem_2s13->Clone("gem_new_2s13_10");
for (int b = h10_2s13->FindBin(10.01); b < h10_2s13->FindBin(15.01); ++b) {hh_2s13->SetBinContent(b, h10_2s13->GetBinContent(b)); hh_2s13->SetBinError(b, h10_2s13->GetBinError(b));}
h10_sing_no1a = (TH1D*)result_gem_gmtsing_no1a->Clone("gem_new_sing_10");
for (int b = h10_sing_no1a->FindBin(10.01); b < h10_sing_no1a->FindBin(15.01); ++b) {hh_sing_no1a->SetBinContent(b, h10_sing_no1a->GetBinContent(b)); hh_sing_no1a->SetBinError(b, h10_sing_no1a->GetBinError(b));}

getPTHistos("minbias_pt15_pat2");
h15 = (TH1D*)result_gem->Clone("gem15");
for (int b = h15->FindBin(15.01); b < h15->FindBin(20.01); ++b) {hh->SetBinContent(b, h15->GetBinContent(b)); hh->SetBinError(b, h15->GetBinError(b));}
h15_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_15");
for (int b = h15_all->FindBin(15.01); b < h15_all->FindBin(20.01); ++b) {hh_all->SetBinContent(b, h15_all->GetBinContent(b)); hh_all->SetBinError(b, h15_all->GetBinError(b));}
h15_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_15");
for (int b = h15_no1a->FindBin(15.01); b < h15_no1a->FindBin(20.01); ++b) {hh_no1a->SetBinContent(b, h15_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h15_no1a->GetBinError(b));}
h15_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_15");
for (int b = h15_2s1b->FindBin(15.01); b < h15_2s1b->FindBin(20.01); ++b) {hh_2s1b->SetBinContent(b, h15_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h15_2s1b->GetBinError(b));}
h15_2s123 = (TH1D*)result_gem_2s123->Clone("gem_new_2s123_15");
for (int b = h15_2s123->FindBin(15.01); b < h15_2s123->FindBin(20.01); ++b) {hh_2s123->SetBinContent(b, h15_2s123->GetBinContent(b)); hh_2s123->SetBinError(b, h15_2s123->GetBinError(b));}
h15_2s13 = (TH1D*)result_gem_2s13->Clone("gem_new_2s13_15");
for (int b = h15_2s13->FindBin(15.01); b < h15_2s13->FindBin(20.01); ++b) {hh_2s13->SetBinContent(b, h15_2s13->GetBinContent(b)); hh_2s13->SetBinError(b, h15_2s13->GetBinError(b));}
h15_sing_no1a = (TH1D*)result_gem_gmtsing_no1a->Clone("gem_new_sing_15");
for (int b = h15_sing_no1a->FindBin(15.01); b < h15_sing_no1a->FindBin(20.01); ++b) {hh_sing_no1a->SetBinContent(b, h15_sing_no1a->GetBinContent(b)); hh_sing_no1a->SetBinError(b, h15_sing_no1a->GetBinError(b));}

getPTHistos("minbias_pt20_pat2");
h20 = (TH1D*)result_gem->Clone("gem20");
for (int b = h20->FindBin(20.01); b < h20->FindBin(30.01); ++b) {hh->SetBinContent(b, h20->GetBinContent(b)); hh->SetBinError(b, h20->GetBinError(b));}
h20_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_20");
for (int b = h20_all->FindBin(20.01); b < h20_all->FindBin(30.01); ++b) {hh_all->SetBinContent(b, h20_all->GetBinContent(b)); hh_all->SetBinError(b, h20_all->GetBinError(b));}
h20_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_20");
for (int b = h20_no1a->FindBin(20.01); b < h20_no1a->FindBin(30.01); ++b) {hh_no1a->SetBinContent(b, h20_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h20_no1a->GetBinError(b));}
h20_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_20");
for (int b = h20_2s1b->FindBin(20.01); b < h20_2s1b->FindBin(30.01); ++b) {hh_2s1b->SetBinContent(b, h20_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h20_2s1b->GetBinError(b));}
h20_2s123 = (TH1D*)result_gem_2s123->Clone("gem_new_2s123_20");
for (int b = h20_2s123->FindBin(20.01); b < h20_2s123->FindBin(30.01); ++b) {hh_2s123->SetBinContent(b, h20_2s123->GetBinContent(b)); hh_2s123->SetBinError(b, h20_2s123->GetBinError(b));}
h20_2s13 = (TH1D*)result_gem_2s13->Clone("gem_new_2s13_20");
for (int b = h20_2s13->FindBin(20.01); b < h20_2s13->FindBin(30.01); ++b) {hh_2s13->SetBinContent(b, h20_2s13->GetBinContent(b)); hh_2s13->SetBinError(b, h20_2s13->GetBinError(b));}
h20_sing_no1a = (TH1D*)result_gem_gmtsing_no1a->Clone("gem_new_sing_20");
for (int b = h20_sing_no1a->FindBin(20.01); b < h20_sing_no1a->FindBin(30.01); ++b) {hh_sing_no1a->SetBinContent(b, h20_sing_no1a->GetBinContent(b)); hh_sing_no1a->SetBinError(b, h20_sing_no1a->GetBinError(b));}

getPTHistos("minbias_pt30_pat2");
h30 = (TH1D*)result_gem->Clone("gem30");
for (int b = h30->FindBin(30.01); b <= h30->FindBin(40.01); ++b) {hh->SetBinContent(b, h30->GetBinContent(b)); hh->SetBinError(b, h30->GetBinError(b));}
h30_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_30");
for (int b = h30_all->FindBin(30.01); b < h30_all->FindBin(40.01); ++b) {hh_all->SetBinContent(b, h30_all->GetBinContent(b)); hh_all->SetBinError(b, h30_all->GetBinError(b));}
h30_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_30");
for (int b = h30_no1a->FindBin(30.01); b < h30_no1a->FindBin(40.01); ++b) {hh_no1a->SetBinContent(b, h30_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h30_no1a->GetBinError(b));}
h30_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_30");
for (int b = h30_2s1b->FindBin(30.01); b < h30_2s1b->FindBin(40.01); ++b) {hh_2s1b->SetBinContent(b, h30_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h30_2s1b->GetBinError(b));}
h30_2s123 = (TH1D*)result_gem_2s123->Clone("gem_new_2s123_30");
for (int b = h30_2s123->FindBin(30.01); b < h30_2s123->FindBin(40.01); ++b) {hh_2s123->SetBinContent(b, h30_2s123->GetBinContent(b)); hh_2s123->SetBinError(b, h30_2s123->GetBinError(b));}
h30_2s13 = (TH1D*)result_gem_2s13->Clone("gem_new_2s13_30");
for (int b = h30_2s13->FindBin(30.01); b < h30_2s13->FindBin(40.01); ++b) {hh_2s13->SetBinContent(b, h30_2s13->GetBinContent(b)); hh_2s13->SetBinError(b, h30_2s13->GetBinError(b));}
h30_sing_no1a = (TH1D*)result_gem_gmtsing_no1a->Clone("gem_new_sing_30");
for (int b = h30_sing_no1a->FindBin(30.01); b < h30_sing_no1a->FindBin(40.01); ++b) {hh_sing_no1a->SetBinContent(b, h30_sing_no1a->GetBinContent(b)); hh_sing_no1a->SetBinError(b, h30_sing_no1a->GetBinError(b));}

getPTHistos("minbias_pt40_pat2");
h40 = (TH1D*)result_gem->Clone("gem30");
for (int b = h40->FindBin(40.01); b <= h40->GetNbinsX(); ++b) {hh->SetBinContent(b, h40->GetBinContent(b)); hh->SetBinError(b, h40->GetBinError(b));}
h40_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_40");
for (int b = h40_all->FindBin(40.01); b < h40_all->GetNbinsX(); ++b) {hh_all->SetBinContent(b, h40_all->GetBinContent(b)); hh_all->SetBinError(b, h40_all->GetBinError(b));}
h40_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_40");
for (int b = h40_no1a->FindBin(40.01); b < h40_no1a->GetNbinsX(); ++b) {hh_no1a->SetBinContent(b, h40_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h40_no1a->GetBinError(b));}
h40_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_40");
for (int b = h40_2s1b->FindBin(40.01); b < h40_2s1b->GetNbinsX(); ++b) {hh_2s1b->SetBinContent(b, h40_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h40_2s1b->GetBinError(b));}
h40_2s123 = (TH1D*)result_gem_2s123->Clone("gem_new_2s123_40");
for (int b = h40_2s123->FindBin(40.01); b < h40_2s123->GetNbinsX(); ++b) {hh_2s123->SetBinContent(b, h40_2s123->GetBinContent(b)); hh_2s123->SetBinError(b, h40_2s123->GetBinError(b));}
h40_2s13 = (TH1D*)result_gem_2s13->Clone("gem_new_2s13_40");
for (int b = h40_2s13->FindBin(40.01); b < h40_2s13->GetNbinsX(); ++b) {hh_2s13->SetBinContent(b, h40_2s13->GetBinContent(b)); hh_2s13->SetBinError(b, h40_2s13->GetBinError(b));}
h40_sing_no1a = (TH1D*)result_gem_gmtsing_no1a->Clone("gem_new_sing_40");
for (int b = h40_sing_no1a->FindBin(40.01); b < h40_sing_no1a->GetNbinsX(); ++b) {hh_sing_no1a->SetBinContent(b, h40_sing_no1a->GetBinContent(b)); hh_sing_no1a->SetBinError(b, h40_sing_no1a->GetBinError(b));}

for (int b = 1; b <= hh->GetNbinsX(); ++b) if (hh->GetBinContent(b)==0) hh->SetBinError(b, 0.);
for (int b = 1; b <= hh_all->GetNbinsX(); ++b) if (hh_all->GetBinContent(b)==0) hh_all->SetBinError(b, 0.);
for (int b = 1; b <= hh_no1a->GetNbinsX(); ++b) if (hh_no1a->GetBinContent(b)==0) hh_no1a->SetBinError(b, 0.);
for (int b = 1; b <= hh_2s1b->GetNbinsX(); ++b) if (hh_2s1b->GetBinContent(b)==0) hh_2s1b->SetBinError(b, 0.);
for (int b = 1; b <= hh_2s123->GetNbinsX(); ++b) if (hh_2s123->GetBinContent(b)==0) hh_2s123->SetBinError(b, 0.);
for (int b = 1; b <= hh_2s13->GetNbinsX(); ++b) if (hh_2s13->GetBinContent(b)==0) hh_2s13->SetBinError(b, 0.);
for (int b = 1; b <= hh_sing_no1a->GetNbinsX(); ++b) if (hh_sing_no1a->GetBinContent(b)==0) hh_sing_no1a->SetBinError(b, 0.);


TString the_ttl = "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]";
TString the_ttl_no1a = "CSC L1 trigger rates in 1.2<|#eta|<2.14;p_{T}^{cut} [GeV/c];rate [kHz]";
TString the_ttl_all = "CSC L1 trigger rates in 1.<|#eta|<2.4;p_{T}^{cut} [GeV/c];rate [kHz]";

hh = setPTHisto(hh, the_ttl, kGreen+3, 1, 1);
hh_all = setPTHisto(hh_all, the_ttl_all, kGreen+3, 1, 1);
hh_no1a = setPTHisto(hh_no1a, the_ttl_no1a, kGreen+3, 1, 1);
hh_2s1b = setPTHisto(hh_2s1b, the_ttl, kGreen+3, 1, 1);
hh_2s123 = setPTHisto(hh_2s123, the_ttl, kGreen+3, 1, 1);
hh_2s13 = setPTHisto(hh_2s13, the_ttl, kGreen+3, 1, 1);
hh_sing_no1a = setPTHisto(hh_sing_no1a, the_ttl_no1a, kGreen+3, 1, 1);

result_def_gmtsing = setPTHisto(result_def_gmtsing, the_ttl, kAzure+1, 1, 1);
result_def_gmtsing_no1a = setPTHisto(result_def_gmtsing_no1a, the_ttl_no1a, kAzure+1, 1, 1);

result_def = setPTHisto(result_def, the_ttl, kAzure+9, 1, 1);
result_def_2s = setPTHisto(result_def_2s, the_ttl, kAzure+9, 1, 1);
result_def_3s1b = setPTHisto(result_def_3s1b, the_ttl, kAzure+9, 1, 1);
result_def_2s1b = setPTHisto(result_def_2s1b, the_ttl, kAzure+9, 1, 1);
result_def_2s123 = setPTHisto(result_def_2s123, the_ttl, kAzure+9, 1, 1);
result_def_2s13 = setPTHisto(result_def_2s13, the_ttl, kAzure+9, 1, 1);
result_def_eta_all = setPTHisto(result_def_eta_all, the_ttl_all, kAzure+9, 1, 1);
result_def_eta_all_3s1b = setPTHisto(result_def_eta_all_3s1b, the_ttl_all, kAzure+9, 1, 1);
result_def_eta_no1a = setPTHisto(result_def_eta_no1a, the_ttl_no1a, kAzure+9, 1, 1);
result_def_eta_no1a_3s1b = setPTHisto(result_def_eta_no1a_3s1b, the_ttl_no1a, kAzure+9, 1, 1);



hh->SetFillColor(kGreen+4);
hh_all->SetFillColor(kGreen+4);
hh_no1a->SetFillColor(kGreen+4);
hh_2s1b->SetFillColor(kGreen+4);
hh_2s123->SetFillColor(kGreen+4);
hh_2s13->SetFillColor(kGreen+4);
hh_sing_no1a->SetFillColor(kGreen+4);

result_def_2s__pat2 = (TH1D*) result_def_2s->Clone("result_def_2s__pat2");
result_def_3s__pat2 = (TH1D*) result_def->Clone("result_def_3s__pat2");
result_def_2s1b__pat2 = (TH1D*) result_def_2s1b->Clone("result_def_2s1b__pat2");
result_def_2s123__pat2 = (TH1D*) result_def_2s123->Clone("result_def_2s123__pat2");
result_def_2s13__pat2 = (TH1D*) result_def_2s13->Clone("result_def_2s13__pat2");
result_def_3s1b__pat2 = (TH1D*) result_def_3s1b->Clone("result_def_3s1b__pat2");
result_def_gmtsing__pat2 = (TH1D*) result_def_gmtsing->Clone("result_def_gmtsing__pat2");;
result_def_gmtsing_no1a__pat2 = (TH1D*) result_def_gmtsing_no1a->Clone("result_def_gmtsing_no1a__pat2");;

result_gem_2s1b__pat2 = (TH1D*) hh_2s1b->Clone("result_gem_2s1b__pat2");
result_gem_2s123__pat2 = (TH1D*) hh_2s123->Clone("result_gem_2s123__pat2");
result_gem_2s13__pat2 = (TH1D*) hh_2s13->Clone("result_gem_2s13__pat2");
result_gem_3s1b__pat2 = (TH1D*) hh->Clone("result_gem_3s1b__pat2");
result_gem_sing_no1a__pat2 = (TH1D*)hh_sing_no1a->Clone("result_gem_sing_no1a__pat2");


// --- def-3s   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def->Draw("e3");
hh->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def,"default emulator","f");
leg->AddEntry(result_def,"Tracks: with #geq3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh,"with GEM match","f");
leg->AddEntry(result_def,"Tracks: same, plus req. one stub from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh, result_def, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2__ratio" + ext);


// --- def-3s-3s1b   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_3s1b->Draw("e3");
hh->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s1b,"default emulator","f");
leg->AddEntry(hh,"with GEM match","f");
leg->AddEntry(result_def_3s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_3s1b,"with #geq3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_3s1b,"and require one stub to be from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh, result_def_3s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2__ratio" + ext);


// --- def-3s-2s1b   gem-3s-2s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_2s1b->Draw("e3");
hh_2s1b->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_2s1b,"default emulator","f");
leg->AddEntry(hh_2s1b,"with GEM match","f");
leg->AddEntry(result_def_2s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_2s1b,"with #geq2 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_2s1b,"and require one stub to be from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat2" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_2s1b, result_def_2s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat2__ratio" + ext);


// --- def-3s   gem-3s-2s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def->Draw("e3");
hh_2s1b->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def,"default emulator","f");
leg->AddEntry(result_def,"Tracks: with #geq3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh_2s1b,"with GEM match","f");
leg->AddEntry(hh_2s1b,"Tracks: with #geq2 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh_2s1b,"and require one stub to be from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat2" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_2s1b, result_def, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat2__ratio" + ext);


// --- def-3s-3s1b   gem-3s-2s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_3s1b->Draw("e3");
hh_2s1b->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s1b,"default emulator","f");
leg->AddEntry(result_def_3s1b,"Tracks req. #geq3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_3s1b,"and require one stub to be from ME1/b","");
leg->AddEntry(hh_2s1b,"with GEM match","f");
leg->AddEntry(hh_2s1b,"Tracks req. #geq2 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh_2s1b,"and require one stub to be from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat2" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_2s1b, result_def_3s1b, "", 0.,3.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat2__ratio" + ext);


// --- eta 1-2.4  def-3s   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_all->Draw("e3");
hh_all->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_all,"default emulator","f");
leg->AddEntry(result_def_eta_all,"Tracks: with #geq3 stubs in 1.<|#eta|<2.4","");
leg->AddEntry(hh_all,"with GEM match","f");
leg->AddEntry(result_def_eta_all,"Tracks: same, plus req. one stub from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_all, result_def_eta_all, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2__ratio" + ext);


// --- eta 1-2.4  def-3s-3s1b   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_all_3s1b->Draw("e3");
hh_all->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_all_3s1b,"default emulator","f");
leg->AddEntry(hh_all,"with GEM match","f");
leg->AddEntry(result_def_eta_all_3s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_eta_all_3s1b,"with #geq3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_eta_all_3s1b,"and require one stub to be from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_all, result_def_eta_all_3s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2__ratio" + ext);


// --- eta 1.2-2.1 def-3s   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_no1a->Draw("e3");
hh_no1a->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_no1a,"default emulator","f");
leg->AddEntry(result_def_eta_no1a,"Tracks: with #geq3 stubs in 1.2<|#eta|<2.4","");
leg->AddEntry(hh_no1a,"with GEM match","f");
leg->AddEntry(result_def_eta_no1a,"Tracks: same, plus req. one stub from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.2-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_no1a, result_def_eta_no1a, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.2-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2__ratio" + ext);


// --- eta 1.2-2.1  def-3s-3s1b   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_no1a_3s1b->Draw("e3");
hh_no1a->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_no1a_3s1b,"default emulator","f");
leg->AddEntry(hh_no1a,"with GEM match","f");
leg->AddEntry(result_def_eta_no1a_3s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_eta_no1a_3s1b,"with #geq3 stubs in 1.2<|#eta|<2.14","");
leg->AddEntry(result_def_eta_no1a_3s1b,"and require one stub to be from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.2-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_no1a, result_def_eta_no1a_3s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.2-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2__ratio" + ext);







// ********** PAT8 **********

getPTHistos("minbias_pt06_pat8");
hh = (TH1D*)result_def_3s1b->Clone("gem_new");
for (int b = hh->FindBin(6.01); b <= hh->GetNbinsX(); ++b) hh->SetBinContent(b, 0);
hh_all = (TH1D*)result_def_eta_all_3s1b->Clone("gem_new_eta_all");
for (int b = hh_all->FindBin(6.01); b <= hh_all->GetNbinsX(); ++b) hh_all->SetBinContent(b, 0);
hh_no1a = (TH1D*)result_def_eta_no1a_3s1b->Clone("gem_new_eta_no1a");
for (int b = hh_no1a->FindBin(6.01); b <= hh_no1a->GetNbinsX(); ++b) hh_no1a->SetBinContent(b, 0);
hh_2s1b = (TH1D*)result_def_2s1b->Clone("gem_new_2s1b");
for (int b = hh_2s1b->FindBin(6.01); b <= hh_2s1b->GetNbinsX(); ++b) hh_2s1b->SetBinContent(b, 0);

h06 = (TH1D*)result_gem->Clone("gem_new_06");
for (int b = h06->FindBin(6.01); b < h06->FindBin(10.01); ++b) {hh->SetBinContent(b, h06->GetBinContent(b)); hh->SetBinError(b, h06->GetBinError(b));}
h06_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_06");
for (int b = h06_all->FindBin(6.01); b < h06_all->FindBin(10.01); ++b) {hh_all->SetBinContent(b, h06_all->GetBinContent(b)); hh_all->SetBinError(b, h06_all->GetBinError(b));}
h06_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_06");
for (int b = h06_no1a->FindBin(6.01); b < h06_no1a->FindBin(10.01); ++b) {hh_no1a->SetBinContent(b, h06_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h06_no1a->GetBinError(b));}
h06_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_06");
for (int b = h06_2s1b->FindBin(6.01); b < h06_2s1b->FindBin(10.01); ++b) {hh_2s1b->SetBinContent(b, h06_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h06_2s1b->GetBinError(b));}

getPTHistos("minbias_pt10_pat8");
h10 = (TH1D*)result_gem->Clone("gem10");
for (int b = h10->FindBin(10.01); b < h10->FindBin(15.01); ++b) {hh->SetBinContent(b, h10->GetBinContent(b)); hh->SetBinError(b, h10->GetBinError(b));}
h10_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_10");
for (int b = h10_all->FindBin(10.01); b < h10_all->FindBin(15.01); ++b) {hh_all->SetBinContent(b, h10_all->GetBinContent(b)); hh_all->SetBinError(b, h10_all->GetBinError(b));}
h10_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_10");
for (int b = h10_no1a->FindBin(10.01); b < h10_no1a->FindBin(15.01); ++b) {hh_no1a->SetBinContent(b, h10_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h10_no1a->GetBinError(b));}
h10_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_10");
for (int b = h10_2s1b->FindBin(10.01); b < h10_2s1b->FindBin(15.01); ++b) {hh_2s1b->SetBinContent(b, h10_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h10_2s1b->GetBinError(b));}

getPTHistos("minbias_pt15_pat8");
h15 = (TH1D*)result_gem->Clone("gem15");
for (int b = h15->FindBin(15.01); b < h15->FindBin(20.01); ++b) {hh->SetBinContent(b, h15->GetBinContent(b)); hh->SetBinError(b, h15->GetBinError(b));}
h15_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_15");
for (int b = h15_all->FindBin(15.01); b < h15_all->FindBin(20.01); ++b) {hh_all->SetBinContent(b, h15_all->GetBinContent(b)); hh_all->SetBinError(b, h15_all->GetBinError(b));}
h15_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_15");
for (int b = h15_no1a->FindBin(15.01); b < h15_no1a->FindBin(20.01); ++b) {hh_no1a->SetBinContent(b, h15_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h15_no1a->GetBinError(b));}
h15_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_15");
for (int b = h15_2s1b->FindBin(15.01); b < h15_2s1b->FindBin(20.01); ++b) {hh_2s1b->SetBinContent(b, h15_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h15_2s1b->GetBinError(b));}

getPTHistos("minbias_pt20_pat8");
h20 = (TH1D*)result_gem->Clone("gem20");
for (int b = h20->FindBin(20.01); b < h20->FindBin(30.01); ++b) {hh->SetBinContent(b, h20->GetBinContent(b)); hh->SetBinError(b, h20->GetBinError(b));}
h20_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_20");
for (int b = h20_all->FindBin(20.01); b < h20_all->FindBin(30.01); ++b) {hh_all->SetBinContent(b, h20_all->GetBinContent(b)); hh_all->SetBinError(b, h20_all->GetBinError(b));}
h20_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_20");
for (int b = h20_no1a->FindBin(20.01); b < h20_no1a->FindBin(30.01); ++b) {hh_no1a->SetBinContent(b, h20_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h20_no1a->GetBinError(b));}
h20_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_20");
for (int b = h20_2s1b->FindBin(20.01); b < h20_2s1b->FindBin(30.01); ++b) {hh_2s1b->SetBinContent(b, h20_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h20_2s1b->GetBinError(b));}

getPTHistos("minbias_pt30_pat8");
h30 = (TH1D*)result_gem->Clone("gem30");
for (int b = h30->FindBin(30.01); b <= h30->FindBin(40.01); ++b) {hh->SetBinContent(b, h30->GetBinContent(b)); hh->SetBinError(b, h30->GetBinError(b));}
h30_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_30");
for (int b = h30_all->FindBin(30.01); b < h30_all->FindBin(40.01); ++b) {hh_all->SetBinContent(b, h30_all->GetBinContent(b)); hh_all->SetBinError(b, h30_all->GetBinError(b));}
h30_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_30");
for (int b = h30_no1a->FindBin(30.01); b < h30_no1a->FindBin(40.01); ++b) {hh_no1a->SetBinContent(b, h30_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h30_no1a->GetBinError(b));}
h30_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_30");
for (int b = h30_2s1b->FindBin(30.01); b < h30_2s1b->FindBin(40.01); ++b) {hh_2s1b->SetBinContent(b, h30_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h30_2s1b->GetBinError(b));}

getPTHistos("minbias_pt40_pat8");
h40 = (TH1D*)result_gem->Clone("gem30");
for (int b = h40->FindBin(40.01); b <= h40->GetNbinsX(); ++b) {hh->SetBinContent(b, h40->GetBinContent(b)); hh->SetBinError(b, h40->GetBinError(b));}
h40_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_40");
for (int b = h40_all->FindBin(40.01); b < h40_all->GetNbinsX(); ++b) {hh_all->SetBinContent(b, h40_all->GetBinContent(b)); hh_all->SetBinError(b, h40_all->GetBinError(b));}
h40_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_40");
for (int b = h40_no1a->FindBin(40.01); b < h40_no1a->GetNbinsX(); ++b) {hh_no1a->SetBinContent(b, h40_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h40_no1a->GetBinError(b));}
h40_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_40");
for (int b = h40_2s1b->FindBin(40.01); b < h40_2s1b->GetNbinsX(); ++b) {hh_2s1b->SetBinContent(b, h40_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h40_2s1b->GetBinError(b));}

for (int b = 1; b <= hh->GetNbinsX(); ++b) if (hh->GetBinContent(b)==0) hh->SetBinError(b, 0.);
for (int b = 1; b <= hh_all->GetNbinsX(); ++b) if (hh_all->GetBinContent(b)==0) hh_all->SetBinError(b, 0.);
for (int b = 1; b <= hh_no1a->GetNbinsX(); ++b) if (hh_no1a->GetBinContent(b)==0) hh_no1a->SetBinError(b, 0.);
for (int b = 1; b <= hh_2s1b->GetNbinsX(); ++b) if (hh_2s1b->GetBinContent(b)==0) hh_2s1b->SetBinError(b, 0.);


hh = setPTHisto(hh, the_ttl, kGreen+3, 1, 1);
hh_all = setPTHisto(hh_all, the_ttl, kGreen+3, 1, 1);
hh_no1a = setPTHisto(hh_no1a, the_ttl, kGreen+3, 1, 1);
hh_2s1b = setPTHisto(hh_2s1b, the_ttl, kGreen+3, 1, 1);

result_def_gmtsing = setPTHisto(result_def_gmtsing, the_ttl, kAzure+1, 1, 1);

result_def = setPTHisto(result_def, the_ttl, kAzure+9, 1, 1);
result_def_2s = setPTHisto(result_def_2s, the_ttl, kAzure+9, 1, 1);
result_def_3s1b = setPTHisto(result_def_3s1b, the_ttl, kAzure+9, 1, 1);
result_def_2s1b = setPTHisto(result_def_2s1b, the_ttl, kAzure+9, 1, 1);
result_def_eta_all = setPTHisto(result_def_eta_all, the_ttl, kAzure+9, 1, 1);
result_def_eta_all_3s1b = setPTHisto(result_def_eta_all_3s1b, the_ttl, kAzure+9, 1, 1);
result_def_eta_no1a = setPTHisto(result_def_eta_no1a, the_ttl, kAzure+9, 1, 1);
result_def_eta_no1a_3s1b = setPTHisto(result_def_eta_no1a_3s1b, the_ttl, kAzure+9, 1, 1);



hh->SetFillColor(kGreen+4);
hh_all->SetFillColor(kGreen+4);
hh_no1a->SetFillColor(kGreen+4);
hh_2s1b->SetFillColor(kGreen+4);

result_def_2s__pat8 = (TH1D*) result_def_2s->Clone("result_def_2s__pat8");
result_def_3s__pat8 = (TH1D*) result_def->Clone("result_def_3s__pat8");
result_def_2s1b__pat8 = (TH1D*) result_def_2s1b->Clone("result_def_2s1b__pat8");
result_def_3s1b__pat8 = (TH1D*) result_def_3s1b->Clone("result_def_3s1b__pat8");
result_def_gmtsing__pat8 = (TH1D*) result_def_gmtsing->Clone("result_def_gmtsing__pat8");;

result_gem_2s1b__pat8 = (TH1D*) hh_2s1b->Clone("result_gem_2s1b__pat8");
result_gem_3s1b__pat8 = (TH1D*) hh->Clone("result_gem_3s1b__pat8");


// --- def-3s   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def->Draw("e3");
hh->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def,"default emulator","f");
leg->AddEntry(result_def,"Tracks: with #geq3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh,"with GEM match","f");
leg->AddEntry(result_def,"Tracks: same, plus req. one stub from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh, result_def, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio" + ext);


// --- def-3s-3s1b   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_3s1b->Draw("e3");
hh->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s1b,"default emulator","f");
leg->AddEntry(hh,"with GEM match","f");
leg->AddEntry(result_def_3s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_3s1b,"with #geq3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_3s1b,"and require one stub to be from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh, result_def_3s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio" + ext);


// --- def-3s-2s1b   gem-3s-2s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_2s1b->Draw("e3");
hh_2s1b->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_2s1b,"default emulator","f");
leg->AddEntry(hh_2s1b,"with GEM match","f");
leg->AddEntry(result_def_2s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_2s1b,"with #geq2 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_2s1b,"and require one stub to be from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat8" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_2s1b, result_def_2s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat8__ratio" + ext);


// --- def-3s   gem-3s-2s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def->Draw("e3");
hh_2s1b->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def,"default emulator","f");
leg->AddEntry(result_def,"Tracks: with #geq3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh_2s1b,"with GEM match","f");
leg->AddEntry(hh_2s1b,"Tracks: with #geq2 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh_2s1b,"and require one stub to be from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat8" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_2s1b, result_def, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat8__ratio" + ext);


// --- def-3s-3s1b   gem-3s-2s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_3s1b->Draw("e3");
hh_2s1b->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s1b,"default emulator","f");
leg->AddEntry(result_def_3s1b,"Tracks req. #geq3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_3s1b,"and require one stub to be from ME1/b","");
leg->AddEntry(hh_2s1b,"with GEM match","f");
leg->AddEntry(hh_2s1b,"Tracks req. #geq2 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh_2s1b,"and require one stub to be from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat8" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_2s1b, result_def_3s1b, "", 0.,3.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat8__ratio" + ext);


// --- eta 1-2.4 def-3s   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_all->Draw("e3");
hh_all->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_all,"default emulator","f");
leg->AddEntry(result_def_eta_all,"Tracks: with #geq3 stubs in 1.<|#eta|<2.4","");
leg->AddEntry(hh_all,"with GEM match","f");
leg->AddEntry(result_def_eta_all,"Tracks: same, plus req. one stub from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_all, result_def_eta_all, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio" + ext);


// --- eta 1-2.4  def-3s-3s1b   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_all_3s1b->Draw("e3");
hh_all->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_all_3s1b,"default emulator","f");
leg->AddEntry(hh_all,"with GEM match","f");
leg->AddEntry(result_def_eta_all_3s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_eta_all_3s1b,"with #geq3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_eta_all_3s1b,"and require one stub to be from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_all, result_def_eta_all_3s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio" + ext);


// --- eta 1.2-2.1 def-3s   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_no1a->Draw("e3");
hh_no1a->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_no1a,"default emulator","f");
leg->AddEntry(result_def_eta_no1a,"Tracks: with #geq3 stubs in 1.2<|#eta|<2.4","");
leg->AddEntry(hh_no1a,"with GEM match","f");
leg->AddEntry(result_def_eta_no1a,"Tracks: same, plus req. one stub from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.2-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_no1a, result_def_eta_no1a, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.2-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio" + ext);


// --- eta 1.2-2.1  def-3s-3s1b   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_no1a_3s1b->Draw("e3");
hh_no1a->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_no1a_3s1b,"default emulator","f");
leg->AddEntry(hh_no1a,"with GEM match","f");
leg->AddEntry(result_def_eta_no1a_3s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_eta_no1a_3s1b,"with #geq3 stubs in 1.2<|#eta|<2.14","");
leg->AddEntry(result_def_eta_no1a_3s1b,"and require one stub to be from ME1/b","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.2-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8" + ext);


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_no1a, result_def_eta_no1a_3s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.2-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio" + ext);





//-------------------------- "Sequential" combinations -----------------------------

result_def_2s__pat2->SetFillColor(kAzure+2);
result_def_2s1b__pat2->SetFillColor(kAzure+5);
result_def_2s123__pat2->SetFillColor(kViolet+3);
result_def_2s13__pat2->SetFillColor(kViolet+1);
result_def_3s__pat2->SetFillColor(kAzure+3);
result_def_3s1b__pat2->SetFillColor(kAzure+6);

result_def_2s__pat8->SetFillColor(kViolet);
result_def_2s1b__pat8->SetFillColor(kViolet+3);
result_def_3s__pat8->SetFillColor(kViolet+1);
result_def_3s1b__pat8->SetFillColor(kViolet+4);

result_def_gmtsing__pat2->SetFillColor(kRed);
result_def_gmtsing__pat8->SetFillColor(kRed);

result_def_gmtsing_no1a__pat2->SetFillColor(kRed);

result_gem_2s1b__pat2->SetFillColor(kGreen+1);
result_gem_2s123__pat2->SetFillColor(kGreen-2);
result_gem_2s13__pat2->SetFillColor(kGreen-3);
result_gem_3s1b__pat2->SetFillColor(kGreen+3);
result_gem_2s1b__pat8->SetFillColor(kGreen-2);
result_gem_3s1b__pat8->SetFillColor(kGreen-3);

result_gem_sing_no1a__pat2->SetFillColor(kGreen+1);

/*
result_def_2s__pat2->GetYaxis()->SetRangeUser(0.01, 3000.);
result_def_2s__pat8->GetYaxis()->SetRangeUser(0.01, 3000.);
result_def_3s__pat2->GetYaxis()->SetRangeUser(0.01, 3000.);
result_def_3s__pat8->GetYaxis()->SetRangeUser(0.01, 3000.);
result_def_gmtsing__pat2->GetYaxis()->SetRangeUser(.1, 1000.);
result_def_gmtsing__pat8->GetYaxis()->SetRangeUser(.1, 1000.);
*/
result_def_2s__pat2->GetYaxis()->SetRangeUser(0.01, 8000.);
result_def_2s__pat8->GetYaxis()->SetRangeUser(0.01, 8000.);
result_def_3s__pat2->GetYaxis()->SetRangeUser(0.01, 8000.);
result_def_3s__pat8->GetYaxis()->SetRangeUser(0.01, 8000.);
result_gem_2s1b__pat2->GetYaxis()->SetRangeUser(0.01, 8000.);
result_gem_3s1b__pat2->GetYaxis()->SetRangeUser(0.01, 8000.);
result_def_gmtsing__pat2->GetYaxis()->SetRangeUser(.1, 5000.);
result_def_gmtsing__pat8->GetYaxis()->SetRangeUser(.1, 5000.);

result_def_gmtsing_no1a__pat2->GetYaxis()->SetRangeUser(1., 8000.);

bool get_ptshift = false;
TH1D* result_gem_3s1b__pat2_ptshift,  *result_gem_2s1b__pat2_ptshift;
TH1D* result_gem_3s1b__pat2_ptshiftX, *result_gem_2s1b__pat2_ptshiftX;
TFile *fsave;
TFile *fsaveX;
if (get_ptshift) {
fsave = new TFile(filesDir + "gem_3plus_ptshift.root");
result_gem_3s1b__pat2__ptshift = (TH1D*) fsave->Get("result_gem_3s1b__pat2")->Clone("result_gem_3s1b__pat2__ptshift");
result_gem_3s1b__pat2__ptshift->SetFillColor(kGreen-5);
result_gem_3s1b__pat2__ptshift->SetLineColor(kGreen-5);
result_gem_2s1b__pat2__ptshift = (TH1D*) fsave->Get("result_gem_2s1b__pat2")->Clone("result_gem_2s1b__pat2__ptshift");
result_gem_2s1b__pat2__ptshift->SetFillColor(kGreen-5);
result_gem_2s1b__pat2__ptshift->SetLineColor(kGreen-5);

fsaveX = new TFile(filesDir + "gem_3plus_ptshiftX.root");
result_gem_3s1b__pat2__ptshiftX = (TH1D*) fsaveX->Get("result_gem_3s1b__pat2")->Clone("result_gem_3s1b__pat2__ptshiftX");
result_gem_3s1b__pat2__ptshiftX->SetFillColor(kGreen-5);
result_gem_3s1b__pat2__ptshiftX->SetLineColor(kGreen-5);
result_gem_2s1b__pat2__ptshiftX = (TH1D*) fsaveX->Get("result_gem_2s1b__pat2")->Clone("result_gem_2s1b__pat2__ptshiftX");
result_gem_2s1b__pat2__ptshiftX->SetFillColor(kGreen-5);
result_gem_2s1b__pat2__ptshiftX->SetLineColor(kGreen-5);
 }

///----- 3 stubs

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_3s__pat2->Draw("e3");
result_def_3s1b__pat2->Draw("same e3");
result_gem_3s1b__pat2->Draw("same e3");

TLegend *leg = new TLegend(0.47,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s__pat2,"CSCTF tracks #geq3 stubs:","");
leg->AddEntry(result_def_3s__pat2,"any stubs","f");
leg->AddEntry(result_def_3s1b__pat2,"has ME1/b stub","f");
leg->AddEntry(result_gem_3s1b__pat2,"has (ME1/b + GEM) stub","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__3s__Frankenstein_pat2" + ext);

((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(result_gem_3s1b__pat2, result_def_3s1b__pat2, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__3s1b__Frankenstein_pat2__ratio" + ext);
hh_ratio = setHistoRatio(result_gem_3s1b__pat2, result_def_3s__pat2, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__3s__Frankenstein_pat2__ratio" + ext);



((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_3s__pat2->Draw("e3");
result_def_3s1b__pat2->Draw("same e3");
result_def_gmtsing__pat2->Draw("same e3");
result_gem_3s1b__pat2->Draw("same e3");

TLegend *leg = new TLegend(0.47,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_gmtsing__pat2,"GMT single Mu selection","f");
leg->AddEntry(result_def_3s__pat2,"CSCTF tracks #geq3 stubs:","");
leg->AddEntry(result_def_3s__pat2,"any stubs","f");
leg->AddEntry(result_def_3s1b__pat2,"has ME1/b stub","f");
leg->AddEntry(result_gem_3s1b__pat2,"has (ME1/b + GEM) stub","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__3s_GMT__Frankenstein_pat2" + ext);

((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(result_gem_3s1b__pat2, result_def_gmtsing__pat2, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__3s_GMT__Frankenstein_pat2__ratio" + ext);




if (get_ptshift) { //-----

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_gem_3s1b__pat2->Draw("e3");
result_gem_3s1b__pat2__ptshift->Draw("same e3");

TLegend *leg = new TLegend(0.47,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s__pat2,"CSCTF tracks #geq3 stubs:","");
leg->AddEntry(result_gem_3s1b__pat2,"has (ME1/b + GEM) stub","");
leg->AddEntry(result_gem_3s1b__pat2,"#Delta#phi(GEM.LCT) as for PT threshold","f");
leg->AddEntry(result_gem_3s1b__pat2__ptshift,"tighter #Delta#phi(GEM.LCT)","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__3s_tightGEM__Frankenstein_pat2" + ext);

((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(result_gem_3s1b__pat2__ptshift, result_gem_3s1b__pat2, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__3s_tightGEM__Frankenstein_pat2__ratio" + ext);


((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_gem_3s1b__pat2->Draw("e3");
result_gem_3s1b__pat2__ptshiftX->Draw("same e3");

TLegend *leg = new TLegend(0.47,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s__pat2,"CSCTF tracks #geq3 stubs:","");
leg->AddEntry(result_gem_3s1b__pat2,"has (ME1/b + GEM) stub","");
leg->AddEntry(result_gem_3s1b__pat2,"#Delta#phi(GEM.LCT) as for PT threshold","f");
leg->AddEntry(result_gem_3s1b__pat2__ptshiftX,"tighter #Delta#phi(GEM.LCT)","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__3s_xtightGEM__Frankenstein_pat2" + ext);

((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(result_gem_3s1b__pat2__ptshiftX, result_gem_3s1b__pat2, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__3s_xtightGEM__Frankenstein_pat2__ratio" + ext);

} //-----



((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_3s__pat8->Draw("e3");
result_def_3s1b__pat8->Draw("same e3");
result_gem_3s1b__pat8->Draw("same e3");

TLegend *leg = new TLegend(0.47,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s__pat8,"CSCTF tracks #geq3 stubs:","");
leg->AddEntry(result_def_3s__pat8,"any stubs","f");
leg->AddEntry(result_def_3s1b__pat8,"has ME1/b stub","f");
leg->AddEntry(result_gem_3s1b__pat8,"has (ME1/b + GEM) stub","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__3s__Frankenstein_pat8" + ext);



((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_3s__pat2->Draw("e3");
result_def_3s__pat8->Draw("same e3");
result_def_3s1b__pat8->Draw("same e3");
result_gem_3s1b__pat8->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s__pat2,"CSCTF tracks #geq3 stubs:","");
leg->AddEntry(result_def_3s__pat2,"any stubs","f");
leg->AddEntry(result_def_3s__pat8,"any stubs, tight patt.","f");
leg->AddEntry(result_def_3s1b__pat8,"has ME1/b stub, tight patt.","f");
leg->AddEntry(result_gem_3s1b__pat8,"has (ME1/b + GEM) stub, tight patt.","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__3s__Frankenstein" + ext);


/// ----- 2 stubs


((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_2s__pat2->Draw("e3");
result_def_2s1b__pat2->Draw("same e3");
result_gem_2s1b__pat2->Draw("same e3");

TLegend *leg = new TLegend(0.47,0.685,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_2s__pat2,"CSCTF tracks #geq2 stubs:","");
leg->AddEntry(result_def_2s__pat2,"any stubs","f");
leg->AddEntry(result_def_2s1b__pat2,"has ME1/b stub","f");
leg->AddEntry(result_gem_2s1b__pat2,"has (ME1/b + GEM) stub","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__2s__Frankenstein_pat2" + ext);

((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(result_gem_2s1b__pat2, result_def_2s1b__pat2, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__2s1b__Frankenstein_pat2__ratio" + ext);



((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_2s__pat2->Draw("e3");
result_def_2s1b__pat2->Draw("same e3");
result_def_gmtsing__pat2->Draw("same e3");
result_gem_2s1b__pat2->Draw("same e3");

TLegend *leg = new TLegend(0.47,0.685,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_gmtsing__pat2,"GMT single Mu selection","f");
leg->AddEntry(result_def_2s__pat2,"CSCTF tracks #geq2 stubs:","");
leg->AddEntry(result_def_2s__pat2,"any stubs","f");
leg->AddEntry(result_def_2s1b__pat2,"has ME1/b stub","f");
leg->AddEntry(result_gem_2s1b__pat2,"has (ME1/b + GEM) stub","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__2s_GMT__Frankenstein_pat2" + ext);

((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(result_gem_2s1b__pat2, result_def_gmtsing__pat2, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__2s_GMT__Frankenstein_pat2__ratio" + ext);





if (get_ptshift) { //-----

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_gem_2s1b__pat2->Draw("e3");
result_gem_2s1b__pat2__ptshift->Draw("same e3");

TLegend *leg = new TLegend(0.47,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_2s__pat2,"CSCTF tracks #geq2 stubs:","");
leg->AddEntry(result_gem_2s1b__pat2,"has (ME1/b + GEM) stub","");
leg->AddEntry(result_gem_2s1b__pat2,"#Delta#phi(GEM.LCT) as for PT threshold","f");
leg->AddEntry(result_gem_2s1b__pat2__ptshift,"tighter #Delta#phi(GEM.LCT)","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__2s_tightGEM__Frankenstein_pat2" + ext);

((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(result_gem_2s1b__pat2__ptshift, result_gem_2s1b__pat2, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__2s_tightGEM__Frankenstein_pat2__ratio" + ext);

fsave->Close();


((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_gem_2s1b__pat2->Draw("e3");
result_gem_2s1b__pat2__ptshiftX->Draw("same e3");

TLegend *leg = new TLegend(0.47,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_2s__pat2,"CSCTF tracks #geq2 stubs:","");
leg->AddEntry(result_gem_2s1b__pat2,"has (ME1/b + GEM) stub","");
leg->AddEntry(result_gem_2s1b__pat2,"#Delta#phi(GEM.LCT) as for PT threshold","f");
leg->AddEntry(result_gem_2s1b__pat2__ptshiftX,"tighter #Delta#phi(GEM.LCT)","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__2s_xtightGEM__Frankenstein_pat2" + ext);

((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(result_gem_2s1b__pat2__ptshiftX, result_gem_2s1b__pat2, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__2s_xtightGEM__Frankenstein_pat2__ratio" + ext);

fsaveX->Close();

} //-----



((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_2s__pat8->Draw("e3");
result_def_2s1b__pat8->Draw("same e3");
result_gem_2s1b__pat8->Draw("same e3");

TLegend *leg = new TLegend(0.47,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_2s__pat8,"CSCTF tracks #geq2 stubs:","");
leg->AddEntry(result_def_2s__pat8,"any stubs","f");
leg->AddEntry(result_def_2s1b__pat8,"has ME1/b stub","f");
leg->AddEntry(result_gem_2s1b__pat8,"has (ME1/b + GEM) stub","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__2s__Frankenstein_pat8" + ext);



((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_2s__pat2->Draw("e3");
result_def_2s__pat8->Draw("same e3");
result_def_2s1b__pat8->Draw("same e3");
result_gem_2s1b__pat8->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_2s__pat2,"CSCTF tracks #geq2 stubs:","");
leg->AddEntry(result_def_2s__pat2,"any stubs","f");
leg->AddEntry(result_def_2s__pat8,"any stubs, tight patt.","f");
leg->AddEntry(result_def_2s1b__pat8,"has ME1/b stub, tight patt.","f");
leg->AddEntry(result_gem_2s1b__pat8,"has (ME1/b + GEM) stub, tight patt.","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__2s__Frankenstein" + ext);


/// ----- GMT current "default" single trigger




((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_gmtsing__pat2->Draw("e3");
result_def_2s__pat2->Draw("same e3");
result_def_2s1b__pat2->Draw("same e3");
result_def_3s__pat2->Draw("same e3");
result_def_3s1b__pat2->Draw("same e3");
result_def_gmtsing__pat2->Draw("same e3");

TLegend *leg = new TLegend(0.49,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_gmtsing__pat2,"GMT single Mu selection","f");
leg->AddEntry(result_def_2s__pat2,"Tracks: #geq2 stubs","f");
leg->AddEntry(result_def_2s1b__pat2,"   #geq2 with ME1/b stubs","f");
leg->AddEntry(result_def_3s__pat2,"   #geq3 stubs","f");
leg->AddEntry(result_def_3s1b__pat2,"   #geq3 with ME1/b stubs","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__GMT__Frankenstein_pat2" + ext);



///----- 2 & 3 stubs


((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_3s__pat2->Draw("e3");
result_def_3s1b__pat2->Draw("same e3");
result_gem_2s1b__pat2->Draw("same e3");
result_gem_3s1b__pat2->Draw("same e3");


TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s__pat2,"CSCTF tracks with:","");
leg->AddEntry(result_def_3s__pat2,"3+ stubs","f");
leg->AddEntry(result_def_3s1b__pat2,"3+ stubs with ME1/b stub","f");
leg->AddEntry(result_gem_2s1b__pat2,"2+ stubs with (ME1/b + GEM) stub","f");
leg->AddEntry(result_gem_3s1b__pat2,"3+ stubs with (ME1/b + GEM) stub","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__2s3s__Frankenstein_pat2" + ext);


///----- 2 & 123 & 13 & 3 stubs


((TCanvas*)gROOT->FindObject("cAll100"))->cd();
//result_def_3s__pat2->Draw("e3");
//result_def_2s1b__pat2->Draw("same e2");
//result_def_2s123__pat2->Draw("same e2");
//result_def_2s13__pat2->Draw("same e2");
//result_def_3s1b__pat2->Draw("same e3");
result_gem_2s1b__pat2->Draw("e3");
result_gem_2s123__pat2->Draw("same e3");
result_gem_2s13__pat2->Draw("same e3");
result_gem_3s1b__pat2->Draw("same e3");


TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_gem_2s1b__pat2,"CSCTF tracks that have","");
leg->AddEntry(result_gem_2s1b__pat2,"(ME1/b + GEM) stub, with total ","");
leg->AddEntry(result_gem_2s1b__pat2,"2+ stubs","f");
leg->AddEntry(result_gem_2s123__pat2,"2+ stubs (no ME1-4 tracks)","f");
leg->AddEntry(result_gem_2s13__pat2,"2+ stubs (no ME1-2 and ME1-4)","f");
leg->AddEntry(result_gem_3s1b__pat2,"3+ stubs","f");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__2s3s123__Frankenstein_pat2" + ext);

((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(result_gem_2s123__pat2, result_gem_2s1b__pat2, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__2s3s123__Frankenstein_pat2__ratio" + ext);

((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(result_gem_2s13__pat2, result_gem_3s1b__pat2, "", 0.,1.4);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__sequential__3s2s13__Frankenstein_pat2__ratio" + ext);


///----- GMT 1.2 eta 2.14


((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_gmtsing_no1a__pat2->Draw("e3");
result_gem_sing_no1a__pat2->Draw("same e3");


TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("GMT Single tracks 1.2<|#eta|<2.14");
leg->AddEntry(result_def_gmtsing_no1a__pat2,"whole #eta range","f");
leg->AddEntry(result_gem_sing_no1a__pat2,"Whole range except in ME1/b:","f");
leg->AddEntry(result_gem_sing_no1a__pat2,"3+ stubs with (ME1/b + GEM)","");
leg->Draw();

drawPULabel();

gPad->Print(gem_dir + "rates__1.2-2.1_PU100__sequential__GMT2s1b__Frankenstein_pat2" + ext);

((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(result_gem_sing_no1a__pat2, result_def_gmtsing_no1a__pat2, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.2-2.1_PU100__sequential__GMT2s1b__Frankenstein_pat2__ratio" + ext);

return;

}
