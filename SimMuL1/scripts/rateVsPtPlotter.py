## ROOT modules
from ROOT import *
from getPTHistos import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT
ROOT.gROOT.SetBatch(1)

## functions
def drawEtaLabel(minEta, maxEta, x=0.17, y=0.35, font_size=0.):
  label = minEta + " < |#eta| < " + maxEta
  tex = TLatex(x, y,label)
  if font_size > 0.:
      tex.SetFontSize(font_size)
  tex.SetTextSize(0.05)
  tex.SetNDC()
  tex.Draw()
  return tex

def drawLumiLabel(x=0.17, y=0.35):
  tex = TLatex(x, y,"L = 4*10^{34} cm^{-2} s^{-1}")
  tex.SetTextSize(0.05)
  tex.SetNDC()
  tex.Draw()
  return tex

def drawL1Label(x=0.17, y=0.35):
  tex = TLatex(x, y,"L1 trigger in 2012 configuration")
  tex.SetTextSize(0.04)
  tex.SetNDC()
  tex.Draw()
  return tex

def produceRatePlot(h, i, j, m, col0, col1, col2, col3, miny, maxy, k, l, plots, ext):
  c = TCanvas("c","c",800,800)
  c.Clear()
  pad1 = TPad("pad1","top pad",0.0,0.25,1.0,1.0)
  pad1.Draw()
  pad2 = TPad("pad2","bottom pad",0,0.,1.0,.30)
  pad2.Draw()

  pad1.cd()
  pad1.SetLogx(1)
  pad1.SetLogy(1)
  pad1.SetGridx(1)
  pad1.SetGridy(1)
  pad1.SetFrameBorderMode(0)
  pad1.SetFillColor(kWhite)
  
  h.SetFillColor(col0)
  i.SetFillColor(col1)
  j.SetFillColor(col2)
  m.SetFillColor(col3)

  h.Draw("e3")
  i.Draw("same e3")
  j.Draw("same e3")
  m.Draw("same e3")
  h.Draw("same e3")
  h.GetYaxis().SetRangeUser(miny, maxy)
  h.GetXaxis().SetTitle("")
  
  leg = TLegend(0.45,0.7,.93,0.93,"","brNDC")
  leg.SetMargin(0.25)
  leg.SetBorderSize(0)
  leg.SetFillStyle(0)
  leg.SetTextSize(0.04)
  leg.SetFillStyle(1001)
  leg.SetFillColor(kWhite)
  leg.AddEntry(h, "L1 single muon trigger","f")
  leg.AddEntry(0, "(final 2012 configuration)","")
  ##leg.AddEntry((TObject*)0,          "L1 selections (#geq " + k + " stations):","")
  leg.AddEntry(i,"CSC #geq%d stubs (anywhere)"%(k),"f")
  leg.AddEntry(j,"CSC #geq%d stubs (one in ME1/b)"%(k),"f")
  leg.AddEntry(m,"GEM+CSC integrated trigger","f")
  leg.Draw()
  
  drawLumiLabel(0.17,.3)
  drawEtaLabel("1.64","2.14",0.17,.37)
  
  pad2.cd()
  pad2.SetLogx(1)
  pad2.SetLogy(1)
  pad2.SetGridx(1)
  pad2.SetGridy(1)
  pad2.SetFillColor(kWhite)
  pad2.SetFrameBorderMode(0)
  pad2.SetLeftMargin(0.126)
  pad2.SetRightMargin(0.04)
  pad2.SetTopMargin(0.06)
  pad2.SetBottomMargin(0.4)
  
  hh_ratio = setHistoRatio(m, j, "", 0.01,1.1,col2)
  hh_ratio.GetXaxis().SetTitle("L1 muon candidate p_{T}^{cut} [GeV/c]")
  hh_ratio.GetYaxis().SetNdivisions(3)
  hh_ratio.Draw("P")
  
  hh_ratio_gmt = setHistoRatio(m, h, "", 0.01,1.1,col0)
  hh_ratio_gmt.Draw("P same")
  
  leg = TLegend(0.15,0.45,.45,0.7,NULL,"brNDC")
  leg.SetMargin(0.1)
  leg.SetBorderSize(0)
  leg.SetTextSize(0.1)
  leg.SetFillStyle(1001)
  leg.SetFillColor(kWhite)
  leg.AddEntry(hh_ratio_gmt, "(GEM+CSC)/GMT","p")
  leg.AddEntry(hh_ratio,     "(GEM+CSC)/CSC #geq%d stubs"%(k),"p")
  leg.Draw("same")
  
  c.SaveAs(plots + "rates_vs_pt__PU100__def_%ds_%ds1b_%ds1bgem__%s%s"%(k,k,k,l,ext))


"""
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
result_def_eta_all = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s", "_def");
result_def_eta_all_3s1b = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_3s1b", "_def");
result_def_eta_no1a = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_no1a", "_def");
result_def_eta_no1a_3s1b = getPTHisto(f_def, dir, "h_rt_gmt_csc_ptmax_3s_3s1b_no1a", "_def");
result_gmtsing = getPTHisto(f_def, dir, "h_rt_gmt_ptmax_sing_1b", "_def");

result_gem = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_3s1b_1b", "_gem");
result_gem_2s1b = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_2s1b_1b", "_gem");
result_gem_eta_all = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_3s1b", "_gem");
result_gem_eta_no1a = getPTHisto(f_gem, dir, "h_rt_gmt_csc_ptmax_3s_3s1b_no1a", "_gem");
//result_gmtsing = getPTHisto(f_gem, dir, "h_rt_gmt_ptmax_sing_1b", "_def");
}


void drawplot_frankenstein_ptshiftX()
{
  gROOT.ProcessLine(".L drawplot_gmtrt.C");

  //gem_dir = "gemPTX/";
  gem_dir = "plots/rate_vs_pt_shiftX/";
  gem_label = "gem98";

//gem_dir = "gem95/"; gem_label = "gem95";

//do_not_print = true;

gROOT.SetBatch(true);

//gStyle.SetStatW(0.13);
//gStyle.SetStatH(0.08);
gStyle.SetStatW(0.07);
gStyle.SetStatH(0.06);

gStyle.SetOptStat(0);

gStyle.SetTitleStyle(0);
gStyle.SetTitleAlign(13);// coord in top left
gStyle.SetTitleX(0.);
gStyle.SetTitleY(1.);
gStyle.SetTitleW(1);
gStyle.SetTitleH(0.058);
gStyle.SetTitleBorderSize(0);

gStyle.SetPadLeftMargin(0.126);
gStyle.SetPadRightMargin(0.04);
gStyle.SetPadTopMargin(0.06);
gStyle.SetPadBottomMargin(0.13);

gStyle.SetMarkerStyle(1);

TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
gPad.SetLogx(1);
gPad.SetGridx(1);gPad.SetGridy(1);

TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
gPad.SetLogx(1);gPad.SetLogy(1);
gPad.SetGridx(1);gPad.SetGridy(1);


// ********** PAT2 **********

getPTHistos("minbias_pt10_pat2");
hh = (TH1D*)result_def_3s1b.Clone("gem_new");
for (int b = hh.FindBin(7.01); b <= hh.GetNbinsX(); ++b) hh.SetBinContent(b, 0);
hh_all = (TH1D*)result_def_eta_all_3s1b.Clone("gem_new_eta_all");
for (int b = hh_all.FindBin(7.01); b <= hh_all.GetNbinsX(); ++b) hh_all.SetBinContent(b, 0);
hh_no1a = (TH1D*)result_def_eta_no1a_3s1b.Clone("gem_new_eta_no1a");
for (int b = hh_no1a.FindBin(7.01); b <= hh_no1a.GetNbinsX(); ++b) hh_no1a.SetBinContent(b, 0);
hh_2s1b = (TH1D*)result_def_2s1b.Clone("gem_new_2s1b");
for (int b = hh_2s1b.FindBin(7.01); b <= hh_2s1b.GetNbinsX(); ++b) hh_2s1b.SetBinContent(b, 0);

h06 = (TH1D*)result_gem.Clone("gem_new_06");
for (int b = h06.FindBin(7.01); b < h06.FindBin(8.01); ++b) {hh.SetBinContent(b, h06.GetBinContent(b)); hh.SetBinError(b, h06.GetBinError(b));}
h06_all = (TH1D*)result_gem_eta_all.Clone("gem_new_eta_all_06");
for (int b = h06_all.FindBin(7.01); b < h06_all.FindBin(8.01); ++b) {hh_all.SetBinContent(b, h06_all.GetBinContent(b)); hh_all.SetBinError(b, h06_all.GetBinError(b));}
h06_no1a = (TH1D*)result_gem_eta_no1a.Clone("gem_new_eta_no1a_06");
for (int b = h06_no1a.FindBin(7.01); b < h06_no1a.FindBin(8.01); ++b) {hh_no1a.SetBinContent(b, h06_no1a.GetBinContent(b)); hh_no1a.SetBinError(b, h06_no1a.GetBinError(b));}
h06_2s1b = (TH1D*)result_gem_2s1b.Clone("gem_new_2s1b_06");
for (int b = h06_2s1b.FindBin(7.01); b < h06_2s1b.FindBin(8.01); ++b) {hh_2s1b.SetBinContent(b, h06_2s1b.GetBinContent(b)); hh_2s1b.SetBinError(b, h06_2s1b.GetBinError(b));}

getPTHistos("minbias_pt15_pat2");
h10 = (TH1D*)result_gem.Clone("gem10");
for (int b = h10.FindBin(8.01); b < h10.FindBin(10.01); ++b) {hh.SetBinContent(b, h10.GetBinContent(b)); hh.SetBinError(b, h10.GetBinError(b));}
h10_all = (TH1D*)result_gem_eta_all.Clone("gem_new_eta_all_10");
for (int b = h10_all.FindBin(8.01); b < h10_all.FindBin(10.01); ++b) {hh_all.SetBinContent(b, h10_all.GetBinContent(b)); hh_all.SetBinError(b, h10_all.GetBinError(b));}
h10_no1a = (TH1D*)result_gem_eta_no1a.Clone("gem_new_eta_no1a_10");
for (int b = h10_no1a.FindBin(8.01); b < h10_no1a.FindBin(10.01); ++b) {hh_no1a.SetBinContent(b, h10_no1a.GetBinContent(b)); hh_no1a.SetBinError(b, h10_no1a.GetBinError(b));}
h10_2s1b = (TH1D*)result_gem_2s1b.Clone("gem_new_2s1b_10");
for (int b = h10_2s1b.FindBin(8.01); b < h10_2s1b.FindBin(10.01); ++b) {hh_2s1b.SetBinContent(b, h10_2s1b.GetBinContent(b)); hh_2s1b.SetBinError(b, h10_2s1b.GetBinError(b));}

getPTHistos("minbias_pt20_pat2");
h15 = (TH1D*)result_gem.Clone("gem15");
for (int b = h15.FindBin(10.01); b < h15.FindBin(15.01); ++b) {hh.SetBinContent(b, h15.GetBinContent(b)); hh.SetBinError(b, h15.GetBinError(b));}
h15_all = (TH1D*)result_gem_eta_all.Clone("gem_new_eta_all_15");
for (int b = h15_all.FindBin(10.01); b < h15_all.FindBin(15.01); ++b) {hh_all.SetBinContent(b, h15_all.GetBinContent(b)); hh_all.SetBinError(b, h15_all.GetBinError(b));}
h15_no1a = (TH1D*)result_gem_eta_no1a.Clone("gem_new_eta_no1a_15");
for (int b = h15_no1a.FindBin(10.01); b < h15_no1a.FindBin(15.01); ++b) {hh_no1a.SetBinContent(b, h15_no1a.GetBinContent(b)); hh_no1a.SetBinError(b, h15_no1a.GetBinError(b));}
h15_2s1b = (TH1D*)result_gem_2s1b.Clone("gem_new_2s1b_15");
for (int b = h15_2s1b.FindBin(10.01); b < h15_2s1b.FindBin(15.01); ++b) {hh_2s1b.SetBinContent(b, h15_2s1b.GetBinContent(b)); hh_2s1b.SetBinError(b, h15_2s1b.GetBinError(b));}

getPTHistos("minbias_pt30_pat2");
h20 = (TH1D*)result_gem.Clone("gem20");
for (int b = h20.FindBin(15.01); b < h20.FindBin(20.01); ++b) {hh.SetBinContent(b, h20.GetBinContent(b)); hh.SetBinError(b, h20.GetBinError(b));}
h20_all = (TH1D*)result_gem_eta_all.Clone("gem_new_eta_all_20");
for (int b = h20_all.FindBin(15.01); b < h20_all.FindBin(20.01); ++b) {hh_all.SetBinContent(b, h20_all.GetBinContent(b)); hh_all.SetBinError(b, h20_all.GetBinError(b));}
h20_no1a = (TH1D*)result_gem_eta_no1a.Clone("gem_new_eta_no1a_20");
for (int b = h20_no1a.FindBin(15.01); b < h20_no1a.FindBin(20.01); ++b) {hh_no1a.SetBinContent(b, h20_no1a.GetBinContent(b)); hh_no1a.SetBinError(b, h20_no1a.GetBinError(b));}
h20_2s1b = (TH1D*)result_gem_2s1b.Clone("gem_new_2s1b_20");
for (int b = h20_2s1b.FindBin(15.01); b < h20_2s1b.FindBin(20.01); ++b) {hh_2s1b.SetBinContent(b, h20_2s1b.GetBinContent(b)); hh_2s1b.SetBinError(b, h20_2s1b.GetBinError(b));}

getPTHistos("minbias_pt40_pat2");
h30 = (TH1D*)result_gem.Clone("gem30");
for (int b = h30.FindBin(20.01); b <= h30.GetNbinsX(); ++b) {hh.SetBinContent(b, h30.GetBinContent(b)); hh.SetBinError(b, h30.GetBinError(b));}
h30_all = (TH1D*)result_gem_eta_all.Clone("gem_new_eta_all_30");
for (int b = h30_all.FindBin(20.01); b < h30_all.GetNbinsX(); ++b) {hh_all.SetBinContent(b, h30_all.GetBinContent(b)); hh_all.SetBinError(b, h30_all.GetBinError(b));}
h30_no1a = (TH1D*)result_gem_eta_no1a.Clone("gem_new_eta_no1a_30");
for (int b = h30_no1a.FindBin(20.01); b < h30_no1a.GetNbinsX(); ++b) {hh_no1a.SetBinContent(b, h30_no1a.GetBinContent(b)); hh_no1a.SetBinError(b, h30_no1a.GetBinError(b));}
h30_2s1b = (TH1D*)result_gem_2s1b.Clone("gem_new_2s1b_30");
for (int b = h30_2s1b.FindBin(20.01); b < h30_2s1b.GetNbinsX(); ++b) {hh_2s1b.SetBinContent(b, h30_2s1b.GetBinContent(b)); hh_2s1b.SetBinError(b, h30_2s1b.GetBinError(b));}

for (int b = 1; b <= hh.GetNbinsX(); ++b) if (hh.GetBinContent(b)==0) hh.SetBinError(b, 0.);
for (int b = 1; b <= hh_all.GetNbinsX(); ++b) if (hh_all.GetBinContent(b)==0) hh_all.SetBinError(b, 0.);
for (int b = 1; b <= hh_no1a.GetNbinsX(); ++b) if (hh_no1a.GetBinContent(b)==0) hh_no1a.SetBinError(b, 0.);
for (int b = 1; b <= hh_2s1b.GetNbinsX(); ++b) if (hh_2s1b.GetBinContent(b)==0) hh_2s1b.SetBinError(b, 0.);


//TString the_ttl = "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]";

//  TString the_ttl = "         L1 Single Muon Trigger                             CMS Simulation Preliminary;L1 candidate muon p_{T}^{cut} [GeV/c];rate [kHz]";
  TString the_ttl = "                                                    CMS Simulation Preliminary;L1 muon candidate p_{T}^{cut} [GeV/c];Trigger rate [kHz]";



hh = setPTHisto(hh, the_ttl, kGreen+3, 1, 1);
hh_all = setPTHisto(hh_all, the_ttl, kGreen+3, 1, 1);
hh_no1a = setPTHisto(hh_no1a, the_ttl, kGreen+3, 1, 1);
hh_2s1b = setPTHisto(hh_2s1b, the_ttl, kGreen+3, 1, 1);

result_gmtsing = setPTHisto(result_gmtsing, the_ttl, kAzure+1, 1, 1);

result_def = setPTHisto(result_def, the_ttl, kAzure+9, 1, 1);
result_def_2s = setPTHisto(result_def_2s, the_ttl, kAzure+9, 1, 1);
result_def_3s1b = setPTHisto(result_def_3s1b, the_ttl, kAzure+9, 1, 1);
result_def_2s1b = setPTHisto(result_def_2s1b, the_ttl, kAzure+9, 1, 1);
result_def_eta_all = setPTHisto(result_def_eta_all, the_ttl, kAzure+9, 1, 1);
result_def_eta_all_3s1b = setPTHisto(result_def_eta_all_3s1b, the_ttl, kAzure+9, 1, 1);
result_def_eta_no1a = setPTHisto(result_def_eta_no1a, the_ttl, kAzure+9, 1, 1);
result_def_eta_no1a_3s1b = setPTHisto(result_def_eta_no1a_3s1b, the_ttl, kAzure+9, 1, 1);

hh.SetFillColor(kGreen+4);
hh_all.SetFillColor(kGreen+4);
hh_no1a.SetFillColor(kGreen+4);
hh_2s1b.SetFillColor(kGreen+4);

result_def_2s__pat2 = (TH1D*) result_def_2s.Clone("result_def_2s__pat2");
result_def_3s__pat2 = (TH1D*) result_def.Clone("result_def_3s__pat2");
result_def_2s1b__pat2 = (TH1D*) result_def_2s1b.Clone("result_def_2s1b__pat2");
result_def_3s1b__pat2 = (TH1D*) result_def_3s1b.Clone("result_def_3s1b__pat2");
result_gmtsing__pat2 = (TH1D*) result_gmtsing.Clone("result_gmtsing__pat2");;

result_gem_2s1b__pat2 = (TH1D*) hh_2s1b.Clone("result_gem_2s1b__pat2");
result_gem_3s1b__pat2 = (TH1D*) hh.Clone("result_gem_3s1b__pat2");


// ********** PAT8 **********

getPTHistos("minbias_pt10_pat8");
hh = (TH1D*)result_def_3s1b.Clone("gem_new");
for (int b = hh.FindBin(7.01); b <= hh.GetNbinsX(); ++b) hh.SetBinContent(b, 0);
hh_all = (TH1D*)result_def_eta_all_3s1b.Clone("gem_new_eta_all");
for (int b = hh_all.FindBin(7.01); b <= hh_all.GetNbinsX(); ++b) hh_all.SetBinContent(b, 0);
hh_no1a = (TH1D*)result_def_eta_no1a_3s1b.Clone("gem_new_eta_no1a");
for (int b = hh_no1a.FindBin(7.01); b <= hh_no1a.GetNbinsX(); ++b) hh_no1a.SetBinContent(b, 0);
hh_2s1b = (TH1D*)result_def_2s1b.Clone("gem_new_2s1b");
for (int b = hh_2s1b.FindBin(7.01); b <= hh_2s1b.GetNbinsX(); ++b) hh_2s1b.SetBinContent(b, 0);

h06 = (TH1D*)result_gem.Clone("gem_new_06");
for (int b = h06.FindBin(7.01); b < h06.FindBin(8.01); ++b) {hh.SetBinContent(b, h06.GetBinContent(b)); hh.SetBinError(b, h06.GetBinError(b));}
h06_all = (TH1D*)result_gem_eta_all.Clone("gem_new_eta_all_06");
for (int b = h06_all.FindBin(7.01); b < h06_all.FindBin(8.01); ++b) {hh_all.SetBinContent(b, h06_all.GetBinContent(b)); hh_all.SetBinError(b, h06_all.GetBinError(b));}
h06_no1a = (TH1D*)result_gem_eta_no1a.Clone("gem_new_eta_no1a_06");
for (int b = h06_no1a.FindBin(7.01); b < h06_no1a.FindBin(8.01); ++b) {hh_no1a.SetBinContent(b, h06_no1a.GetBinContent(b)); hh_no1a.SetBinError(b, h06_no1a.GetBinError(b));}
h06_2s1b = (TH1D*)result_gem_2s1b.Clone("gem_new_2s1b_06");
for (int b = h06_2s1b.FindBin(7.01); b < h06_2s1b.FindBin(8.01); ++b) {hh_2s1b.SetBinContent(b, h06_2s1b.GetBinContent(b)); hh_2s1b.SetBinError(b, h06_2s1b.GetBinError(b));}

getPTHistos("minbias_pt15_pat8");
h10 = (TH1D*)result_gem.Clone("gem10");
for (int b = h10.FindBin(8.01); b < h10.FindBin(10.01); ++b) {hh.SetBinContent(b, h10.GetBinContent(b)); hh.SetBinError(b, h10.GetBinError(b));}
h10_all = (TH1D*)result_gem_eta_all.Clone("gem_new_eta_all_10");
for (int b = h10_all.FindBin(8.01); b < h10_all.FindBin(10.01); ++b) {hh_all.SetBinContent(b, h10_all.GetBinContent(b)); hh_all.SetBinError(b, h10_all.GetBinError(b));}
h10_no1a = (TH1D*)result_gem_eta_no1a.Clone("gem_new_eta_no1a_10");
for (int b = h10_no1a.FindBin(8.01); b < h10_no1a.FindBin(10.01); ++b) {hh_no1a.SetBinContent(b, h10_no1a.GetBinContent(b)); hh_no1a.SetBinError(b, h10_no1a.GetBinError(b));}
h10_2s1b = (TH1D*)result_gem_2s1b.Clone("gem_new_2s1b_10");
for (int b = h10_2s1b.FindBin(8.01); b < h10_2s1b.FindBin(10.01); ++b) {hh_2s1b.SetBinContent(b, h10_2s1b.GetBinContent(b)); hh_2s1b.SetBinError(b, h10_2s1b.GetBinError(b));}

getPTHistos("minbias_pt20_pat8");
h15 = (TH1D*)result_gem.Clone("gem15");
for (int b = h15.FindBin(10.01); b < h15.FindBin(15.01); ++b) {hh.SetBinContent(b, h15.GetBinContent(b)); hh.SetBinError(b, h15.GetBinError(b));}
h15_all = (TH1D*)result_gem_eta_all.Clone("gem_new_eta_all_15");
for (int b = h15_all.FindBin(10.01); b < h15_all.FindBin(15.01); ++b) {hh_all.SetBinContent(b, h15_all.GetBinContent(b)); hh_all.SetBinError(b, h15_all.GetBinError(b));}
h15_no1a = (TH1D*)result_gem_eta_no1a.Clone("gem_new_eta_no1a_15");
for (int b = h15_no1a.FindBin(10.01); b < h15_no1a.FindBin(15.01); ++b) {hh_no1a.SetBinContent(b, h15_no1a.GetBinContent(b)); hh_no1a.SetBinError(b, h15_no1a.GetBinError(b));}
h15_2s1b = (TH1D*)result_gem_2s1b.Clone("gem_new_2s1b_15");
for (int b = h15_2s1b.FindBin(10.01); b < h15_2s1b.FindBin(15.01); ++b) {hh_2s1b.SetBinContent(b, h15_2s1b.GetBinContent(b)); hh_2s1b.SetBinError(b, h15_2s1b.GetBinError(b));}

getPTHistos("minbias_pt30_pat8");
h20 = (TH1D*)result_gem.Clone("gem20");
for (int b = h20.FindBin(15.01); b < h20.FindBin(20.01); ++b) {hh.SetBinContent(b, h20.GetBinContent(b)); hh.SetBinError(b, h20.GetBinError(b));}
h20_all = (TH1D*)result_gem_eta_all.Clone("gem_new_eta_all_20");
for (int b = h20_all.FindBin(15.01); b < h20_all.FindBin(20.01); ++b) {hh_all.SetBinContent(b, h20_all.GetBinContent(b)); hh_all.SetBinError(b, h20_all.GetBinError(b));}
h20_no1a = (TH1D*)result_gem_eta_no1a.Clone("gem_new_eta_no1a_20");
for (int b = h20_no1a.FindBin(15.01); b < h20_no1a.FindBin(20.01); ++b) {hh_no1a.SetBinContent(b, h20_no1a.GetBinContent(b)); hh_no1a.SetBinError(b, h20_no1a.GetBinError(b));}
h20_2s1b = (TH1D*)result_gem_2s1b.Clone("gem_new_2s1b_20");
for (int b = h20_2s1b.FindBin(15.01); b < h20_2s1b.FindBin(20.01); ++b) {hh_2s1b.SetBinContent(b, h20_2s1b.GetBinContent(b)); hh_2s1b.SetBinError(b, h20_2s1b.GetBinError(b));}

getPTHistos("minbias_pt40_pat8");
h30 = (TH1D*)result_gem.Clone("gem30");
for (int b = h30.FindBin(20.01); b <= h30.GetNbinsX(); ++b) {hh.SetBinContent(b, h30.GetBinContent(b)); hh.SetBinError(b, h30.GetBinError(b));}
h30_all = (TH1D*)result_gem_eta_all.Clone("gem_new_eta_all_30");
for (int b = h30_all.FindBin(20.01); b < h30_all.GetNbinsX(); ++b) {hh_all.SetBinContent(b, h30_all.GetBinContent(b)); hh_all.SetBinError(b, h30_all.GetBinError(b));}
h30_no1a = (TH1D*)result_gem_eta_no1a.Clone("gem_new_eta_no1a_30");
for (int b = h30_no1a.FindBin(20.01); b < h30_no1a.GetNbinsX(); ++b) {hh_no1a.SetBinContent(b, h30_no1a.GetBinContent(b)); hh_no1a.SetBinError(b, h30_no1a.GetBinError(b));}
h30_2s1b = (TH1D*)result_gem_2s1b.Clone("gem_new_2s1b_30");
for (int b = h30_2s1b.FindBin(20.01); b < h30_2s1b.GetNbinsX(); ++b) {hh_2s1b.SetBinContent(b, h30_2s1b.GetBinContent(b)); hh_2s1b.SetBinError(b, h30_2s1b.GetBinError(b));}

for (int b = 1; b <= hh.GetNbinsX(); ++b) if (hh.GetBinContent(b)==0) hh.SetBinError(b, 0.);
for (int b = 1; b <= hh_all.GetNbinsX(); ++b) if (hh_all.GetBinContent(b)==0) hh_all.SetBinError(b, 0.);
for (int b = 1; b <= hh_no1a.GetNbinsX(); ++b) if (hh_no1a.GetBinContent(b)==0) hh_no1a.SetBinError(b, 0.);
for (int b = 1; b <= hh_2s1b.GetNbinsX(); ++b) if (hh_2s1b.GetBinContent(b)==0) hh_2s1b.SetBinError(b, 0.);


hh = setPTHisto(hh, the_ttl, kGreen+3, 1, 1);
hh_all = setPTHisto(hh_all, the_ttl, kGreen+3, 1, 1);
hh_no1a = setPTHisto(hh_no1a, the_ttl, kGreen+3, 1, 1);
hh_2s1b = setPTHisto(hh_2s1b, the_ttl, kGreen+3, 1, 1);

result_gmtsing = setPTHisto(result_gmtsing, the_ttl, kAzure+1, 1, 1);

result_def = setPTHisto(result_def, the_ttl, kAzure+9, 1, 1);
result_def_2s = setPTHisto(result_def_2s, the_ttl, kAzure+9, 1, 1);
result_def_3s1b = setPTHisto(result_def_3s1b, the_ttl, kAzure+9, 1, 1);
result_def_2s1b = setPTHisto(result_def_2s1b, the_ttl, kAzure+9, 1, 1);
result_def_eta_all = setPTHisto(result_def_eta_all, the_ttl, kAzure+9, 1, 1);
result_def_eta_all_3s1b = setPTHisto(result_def_eta_all_3s1b, the_ttl, kAzure+9, 1, 1);
result_def_eta_no1a = setPTHisto(result_def_eta_no1a, the_ttl, kAzure+9, 1, 1);
result_def_eta_no1a_3s1b = setPTHisto(result_def_eta_no1a_3s1b, the_ttl, kAzure+9, 1, 1);

hh.SetFillColor(kGreen+4);
hh_all.SetFillColor(kGreen+4);
hh_no1a.SetFillColor(kGreen+4);
hh_2s1b.SetFillColor(kGreen+4);

result_def_2s__pat8 = (TH1D*) result_def_2s.Clone("result_def_2s__pat8");
result_def_3s__pat8 = (TH1D*) result_def.Clone("result_def_3s__pat8");
result_def_2s1b__pat8 = (TH1D*) result_def_2s1b.Clone("result_def_2s1b__pat8");
result_def_3s1b__pat8 = (TH1D*) result_def_3s1b.Clone("result_def_3s1b__pat8");
result_gmtsing__pat8 = (TH1D*) result_gmtsing.Clone("result_gmtsing__pat8");;

result_gem_2s1b__pat8 = (TH1D*) hh_2s1b.Clone("result_gem_2s1b__pat8");
result_gem_3s1b__pat8 = (TH1D*) hh.Clone("result_gem_3s1b__pat8");

//-------------------------- "Sequential" combinations -----------------------------

result_def_2s__pat2.SetFillColor(kAzure+2);
result_def_2s1b__pat2.SetFillColor(kAzure+5);
result_def_3s__pat2.SetFillColor(kAzure+3);
result_def_3s1b__pat2.SetFillColor(kAzure+6);

result_def_2s__pat8.SetFillColor(kViolet);
result_def_2s1b__pat8.SetFillColor(kViolet+3);
result_def_3s__pat8.SetFillColor(kViolet+1);
result_def_3s1b__pat8.SetFillColor(kViolet+4);

result_gmtsing__pat2.SetFillColor(kRed);
result_gmtsing__pat8.SetFillColor(kRed);

result_gem_2s1b__pat2.SetFillColor(kGreen+1);
result_gem_3s1b__pat2.SetFillColor(kGreen+3);
result_gem_2s1b__pat8.SetFillColor(kGreen-2);
result_gem_3s1b__pat8.SetFillColor(kGreen-3);

/*
result_def_2s__pat2.GetYaxis().SetRangeUser(0.01, 3000.);
result_def_2s__pat8.GetYaxis().SetRangeUser(0.01, 3000.);
result_def_3s__pat2.GetYaxis().SetRangeUser(0.01, 3000.);
result_def_3s__pat8.GetYaxis().SetRangeUser(0.01, 3000.);
result_gmtsing__pat2.GetYaxis().SetRangeUser(.1, 1000.);
result_gmtsing__pat8.GetYaxis().SetRangeUser(.1, 1000.);
*/
result_def_2s__pat2.GetYaxis().SetRangeUser(0.01, 8000.);
result_def_2s__pat8.GetYaxis().SetRangeUser(0.01, 8000.);
result_def_3s__pat2.GetYaxis().SetRangeUser(0.01, 8000.);
result_def_3s__pat8.GetYaxis().SetRangeUser(0.01, 8000.);
result_gem_2s1b__pat2.GetYaxis().SetRangeUser(0.01, 8000.);
result_gem_3s1b__pat2.GetYaxis().SetRangeUser(0.01, 8000.);
result_gmtsing__pat2.GetYaxis().SetRangeUser(.1, 5000.);
result_gmtsing__pat8.GetYaxis().SetRangeUser(.1, 5000.);



  Color_t col0 = kRed;
  Color_t col1 = kViolet+1;
  Color_t col2 = kAzure+2;
  Color_t col3 = kGreen-2;

  TString plots = "plots/rate_vs_pt_shiftX/";
  TString ext = ".png";



  produceRatePlot(result_gmtsing, result_def_2s__pat2, result_def_2s1b__pat2, result_gem_2s1b__pat2, 
		  col0, col1, col2, col3, 0.1, 10000, "2", "loose", plots, ext);
  ##produceRatePlot(result_gmtsing, result_def_2s, result_def_2s1b, result_gem_2s1b, 
  ##  		  col0, col1, col2, col3, 0.1, 10000, "2", "tight", plots, ext);
  produceRatePlot(result_gmtsing, result_def_3s__pat2, result_def_3s1b__pat2, result_gem_3s1b__pat2, 
		  col0, col1, col2, col3, 0.01, 10000, "3", "loose", plots, ext);
  ## produceRatePlot(result_gmtsing, result_def_3s, result_def_3s1b, result_gem_3s1b, 
  ##  		  col0, col1, col2, col3, 0.01, 10000, "3", "tight", plots, ext);
"""
