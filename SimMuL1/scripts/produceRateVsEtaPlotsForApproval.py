from helpers import *

## ROOT modules
from ROOT import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT
ROOT.gROOT.SetBatch(1)

gem_dir = "files/"
gem_label = "gem98"
dir = "SimMuL1StrictAll"

def drawLumiLabel2(x=0.2, y=0.4):
  tex = TLatex(x, y,"L = 4*10^{34} cm^{-2} s^{-1}")
  tex.SetTextSize(0.05)
  tex.SetNDC()
  tex.Draw("same")
  return tex

def drawPULabel(x=0.17, y=0.15, font_size=0.):
  tex = TLatex(x, y,"L=4*10^{34} (25ns PU100)")
  if (font_size > 0.):
      tex.SetFontSize(font_size)
  tex.SetNDC()
  tex.Draw("same")
  return tex

def setHistoEta(f_name, name, cname, title, lcolor, lstyle, lwidth):
  f = TFile.Open(f_name)
##  print "opening ",f
  h0 = GetH(f,dir,name)
  h = h0.Clone(name+cname)
  h.SetTitle(title)
  h.Sumw2()
  scale(h)
  h.SetLineColor(lcolor)
  ##h.SetFillColor(lcolor)
  h.SetLineStyle(lstyle)
  h.SetLineWidth(lwidth)
  h.SetTitle(title)
  ##h.GetXaxis().SetRangeUser(1.2, 2.4)
  h.GetYaxis().SetRangeUser(0.1,2500)
  h.GetXaxis().SetTitleSize(0.055)
  h.GetXaxis().SetTitleOffset(1.05)
  h.GetXaxis().SetLabelSize(0.045)
  h.GetXaxis().SetLabelOffset(0.003)
  h.GetXaxis().SetTitleFont(62)
  h.GetXaxis().SetLabelFont(62)
  h.GetXaxis().SetMoreLogLabels(1)
  h.GetYaxis().SetTitleSize(0.055)
  h.GetYaxis().SetTitleOffset(0.9)
  h.GetYaxis().SetLabelSize(0.045)
  h.GetYaxis().SetTitleFont(62)
  h.GetYaxis().SetLabelFont(62)
  ##h.GetYaxis().SetLabelOffset(0.015)
  return h

def setHistoRatio(num, denom, title = "", ymin=0.4, ymax=1.6, color = kRed+3):
  ratio = num.Clone("%s--%s_ratio"%(num.GetName(),denom.GetName()))
  ratio.Divide(num, denom, 1., 1.)
  ratio.SetTitle(title)
  ratio.GetYaxis().SetRangeUser(ymin, ymax)
  ratio.GetYaxis().SetTitle("Ratio")
  ratio.GetYaxis().SetTitleSize(.14)
  ##ratio.GetYaxis().SetTitleSize(.1)
  ratio.GetYaxis().SetTitleOffset(0.4)
  ratio.GetYaxis().SetLabelSize(.11)
  ##ratio.GetXaxis().SetMoreLogLabels(1)
  ##ratio.GetXaxis().SetTitle("track #eta")
  ratio.GetXaxis().SetLabelSize(.11)
  ratio.GetXaxis().SetTitleSize(.14)
  ratio.GetXaxis().SetTitleOffset(1.) 
  ratio.SetLineWidth(2)
  ratio.SetFillColor(color)
  ratio.SetLineColor(color)
  ratio.SetMarkerColor(color)
  ratio.SetMarkerStyle(20)
  ratio.SetLineColor(color)
  ratio.SetMarkerColor(color)
  ##ratio.Draw("e3")
  return ratio

def addRatioPlotLegend(h,k):
  leg = TLegend(0.17,0.35,.47,0.5,"","brNDC")
  leg.SetMargin(0.1)
  leg.SetBorderSize(0)
  leg.SetTextSize(0.1)
  leg.SetFillStyle(1001)
  leg.SetFillColor(kWhite)
  leg.AddEntry(h, "(GEM+CSC)/CSC #geq%d stubs (one in ME1/b)"%(k),"P");
  leg.Draw("same")
  return leg

def addRatePlotLegend(h, i, j, k, l):
  leg = TLegend(0.16,0.67,.8,0.9,"L1 Selections (#geq%d stubs, L1 candidate p_{T}#geq%d GeV/c):"%(k,l),"brNDC");
  leg.SetMargin(0.20)
  leg.SetBorderSize(0)
  leg.SetTextSize(0.04)
  leg.SetFillStyle(1001)
  leg.SetFillColor(kWhite)
  leg.AddEntry(h,"CSC #geq%d stubs (anywhere)"%(k),"l");
  leg.AddEntry(i,"CSC #geq%d stubs (one in ME1/b)"%(k),"l");
  leg.AddEntry(j,"GEM+CSC integrated trigger","l");
  leg.Draw("same")
  ## carbage collection in PyROOT is something to pay attention to!
  SetOwnership(leg, False)
  return leg

def addRatePlot(h, i, j, col1, col2, col3, sty1, sty2, sty3, sty4, miny, maxy):
  h.SetFillColor(col1);
  i.SetFillColor(col2);
  j.SetFillColor(col3);
  
  h.SetFillStyle(sty1);
  i.SetFillStyle(sty2);
  j.SetFillStyle(sty3);
  
  ## Slava's proposal
  h.SetFillStyle(0);
  i.SetFillStyle(0);
  j.SetFillStyle(0);
  
  h.SetLineStyle(1);
  i.SetLineStyle(4);
  j.SetLineStyle(2);
  
  h.SetLineWidth(2);
  i.SetLineWidth(2);
  j.SetLineWidth(2);

  h.GetYaxis().SetRangeUser(miny,maxy);
  i.GetYaxis().SetRangeUser(miny,maxy);
  j.GetYaxis().SetRangeUser(miny,maxy);

  i.Draw("hist e1");
  j.Draw("same hist e1");
  h.Draw("same hist e1");

def setPad1Attributes(pad1):
  pad1.SetGridx(1)
  pad1.SetGridy(1)
  pad1.SetFrameBorderMode(0)
  pad1.SetFillColor(kWhite)
  pad1.SetTopMargin(0.06)
  pad1.SetBottomMargin(0.13)

def setPad2Attributes(pad2):
  pad2.SetLogy(1)
  pad2.SetGridx(1)
  pad2.SetGridy(1)
  pad2.SetFillColor(kWhite)
  pad2.SetFrameBorderMode(0)
  pad2.SetTopMargin(0.06)
  pad2.SetBottomMargin(0.3)

def produceRateVsEtaPlot(h, i, j, col1, col2, col3, sty1, sty2, sty3, sty4, 
                         miny, maxy, k, l, plots, ext):
  c = TCanvas("c","c",800,800)
  c.Clear()

  pad1 = TPad("pad1","top pad",0.0,0.25,1.0,1.0)
  pad1.Draw("")
  pad2 = TPad("pad2","bottom pad",0.,0.,1.0,.30)
  pad2.Draw("same")

  pad1.cd()
  setPad1Attributes(pad1)
  addRatePlot(h,i,j,col1,col2,col3,sty1,sty2,sty3,3355,miny,maxy)
  leg = addRatePlotLegend(h, i, j, k, l)
  tex = drawLumiLabel2()
  
  pad2.cd()
  setPad2Attributes(pad2)
  gem_ratio = setHistoRatio(j, i, "", 0.01,2.0, col2)
  gem_ratio.GetYaxis().SetNdivisions(3)  
  gem_ratio.Draw("Pe")
  leg = addRatioPlotLegend(gem_ratio,k)

  c.SaveAs(plots + "rates_vs_eta__minpt%d__PU100__def_%ds_%ds1b_%ds1bgem%s"%(l,k,k,k,ext))

def produceRateVsEtaPlotsForApproval(ext, plots):
  gStyle.SetOptStat(0)
  gStyle.SetTitleStyle(0)
  ## ##gStyle.SetPadTopMargin(0.08)
  ## gStyle.SetTitleH(0.06)

  ## input files
  f_def =      gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_def_pat2.root"
  f_g98_pt10 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt10_pat2.root"
  f_g98_pt15 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt15_pat2.root"
  f_g98_pt20 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt20_pat2.root"
  f_g98_pt30 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt30_pat2.root"
  f_g98_pt40 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt40_pat2.root"

  ## general stuff
  hdir = "SimMuL1StrictAll"

  ## colors - same colors as for rate vs pt plots!!
  col1 = kViolet+1
  col2 = kAzure+2
  col3 = kGreen-2

  ## styles
  sty1 = 3345
  sty2 = 3003
  sty3 = 2002

  ## Declaration of histograms
  ttl = "                                              CMS Simulation Preliminary;L1 muon candidate #eta;Trigger rate [kHz]";

  pt = 10
  h_rt_tf10_2s = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_2s"%(pt), "_hAll100", ttl, col1, 1, 2)
  h_rt_tf10_2s1b = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_2s_2s1b"%(pt), "_hAll100", ttl, col2, 1, 2)
  h_rt_tf10_gpt10_2s1b = setHistoEta(f_g98_pt10, "h_rt_gmt_csc_ptmax%d_eta_2s_2s1b"%(pt), "_hAll100", ttl, col3, 1, 2)
  h_rt_tf10_3s = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_3s"%(pt), "_hAll100", ttl, col1, 1, 2)
  h_rt_tf10_3s1b = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_3s_3s1b"%(pt), "_hAll100", ttl, col2, 1, 2)
  h_rt_tf10_gpt10_3s1b = setHistoEta(f_g98_pt10, "h_rt_gmt_csc_ptmax%d_eta_3s_3s1b"%(pt), "_hAll100", ttl, col3, 7, 2)
  
  pt = 20
  h_rt_tf20_2s = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_2s"%(pt), "_hAll100", ttl, col1, 1, 2)
  h_rt_tf20_2s1b = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_2s_2s1b"%(pt), "_hAll100", ttl, col2, 1, 2)
  h_rt_tf20_gpt20_2s1b = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax%d_eta_2s_2s1b"%(pt), "_hAll100", ttl, col3, 1, 2)
  h_rt_tf20_3s = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_3s"%(pt), "_hAll100", ttl, col1, 1, 2)
  h_rt_tf20_3s1b = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_3s_3s1b"%(pt), "_hAll100", ttl, col2, 1, 2)
  h_rt_tf20_gpt20_3s1b = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax%d_eta_3s_3s1b"%(pt), "_hAll100", ttl, col3, 1, 2)

  pt = 30
  h_rt_tf30_2s = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_2s"%(pt), "_hAll100", ttl, col1, 1, 2)
  h_rt_tf30_2s1b = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_2s_2s1b"%(pt), "_hAll100", ttl, col2, 1, 2)
  h_rt_tf30_gpt30_2s1b = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax%d_eta_2s_2s1b"%(pt), "_hAll100", ttl, col3, 1, 2)
  h_rt_tf30_3s = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_3s"%(pt), "_hAll100", ttl, col1, 1, 2)
  h_rt_tf30_3s1b = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_3s_3s1b"%(pt), "_hAll100", ttl, col2, 1, 2)
  h_rt_tf30_gpt30_3s1b = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax%d_eta_3s_3s1b"%(pt), "_hAll100", ttl, col3, 1, 2)

  ## Style
  gStyle.SetStatW(0.07)
  gStyle.SetStatH(0.06)
  gStyle.SetOptStat(0)
  gStyle.SetTitleStyle(0)
  gStyle.SetTitleAlign(13)## coord in top left
  gStyle.SetTitleX(0.)
  gStyle.SetTitleY(1.)
  gStyle.SetTitleW(1)
  gStyle.SetTitleH(0.058)
  gStyle.SetTitleBorderSize(0)

  ## ------------ +2 stubs, L1 candidate muon pt=10GeV ----------------##
  produceRateVsEtaPlot(h_rt_tf10_2s,h_rt_tf10_2s1b,h_rt_tf10_gpt10_2s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,0.01,80,2,10,plots,ext)
  ## ------------ +2 stubs, L1 candidate muon pt=20GeV ----------------##
  produceRateVsEtaPlot(h_rt_tf20_2s,h_rt_tf20_2s1b,h_rt_tf20_gpt20_2s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,0.01,40,2,20,plots,ext)
  ## ------------ +2 stubs, L1 candidate muon pt=30GeV ----------------##
  produceRateVsEtaPlot(h_rt_tf30_2s,h_rt_tf30_2s1b,h_rt_tf30_gpt30_2s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,0.01,30,2,30,plots,ext)
  ## ------------ +3 stubs, L1 candidate muon pt=10GeV ----------------##
  produceRateVsEtaPlot(h_rt_tf10_3s,h_rt_tf10_3s1b,h_rt_tf10_gpt10_3s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,0.01,25,3,10,plots,ext)
  ## ------------ +3 stubs, L1 candidate muon pt=20GeV ----------------##
  produceRateVsEtaPlot(h_rt_tf20_3s,h_rt_tf20_3s1b,h_rt_tf20_gpt20_3s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,0.01,10,3,20,plots,ext)
  ## ------------ +3 stubs, L1 candidate muon pt=30GeV ----------------##
  produceRateVsEtaPlot(h_rt_tf30_3s,h_rt_tf30_3s1b,h_rt_tf30_gpt30_3s1b,
                       col1,col2,col3,sty1,sty2,sty3,3355,0.01, 6,3,30,plots,ext)
  
if __name__ == "__main__":
  produceRateVsEtaPlotsForApproval(".pdf", "plots/rate_vs_eta/")

"""
void gem_rate_draw()
{
  -- KEEP THIS FRAGMENT FOR THE TIME BEING!! --
  -- NEED TO FIGURE OUT WHAT TO DO WITH ME1A --

  // // Including the region ME1/1a

  //  ((TCanvas*)gROOT->FindObject("cAll100"))->cd();

  //  maxy = 30.;//10;

  //  h_rt_tf20_3s1ab->GetYaxis()->SetRangeUser(miny,maxy);
  //  h_rt_tf20_gpt20_3s1ab->GetYaxis()->SetRangeUser(miny,maxy);


  //  ((TCanvas*)gROOT->FindObject("cAll100"))->cd();
  //  h_rt_tf20_3s->Draw("hist e1");
  //  h_rt_tf20_gpt20_3s1ab->Draw("hist e1 same");
  //  h_rt_tf20_3s->Draw("hist e1 same");
  //  h_rt_tf20_3s1ab->Draw("hist e1 same");

  //  TLegend *leg = new TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
  //  leg->SetBorderSize(0);
  //  leg->SetFillStyle(0);
  //  leg->AddEntry(h_rt_tf20_3s,"Tracks: p_{T}>=20, 3+ stubs","");
  //  leg->AddEntry(h_rt_tf20_3s,"anywhere","l");
  //  leg->AddEntry(h_rt_tf20_3s1ab,"with ME1 in 1.6<|#eta|","l");
  //  leg->AddEntry(h_rt_tf20_gpt20_3s1ab,"with (ME1+GEM) in 1.6<|#eta|","l");
  //  leg->Draw();

  //  TLatex *  tex = new TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
  //  tex->SetNDC();
  //  tex->Draw();

  //  Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab.png").Data());


  // TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
  // gPad->SetGridx(1);gPad->SetGridy(1);

  // gem_ratio = setHistoRatio(h_rt_tf20_gpt20_3s1ab, h_rt_tf20_3s1ab, "", 0.,1.8);
  // gem_ratio->Draw("e1");

  //  Print(cAll100r, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab__ratio.png").Data());



  vs_eta_minpt = "30";
  ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate [kHz]";

  
  TH1D* h_rt_tf30_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, kAzure+2, 1, 2);
  TH1D* h_rt_tf30_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, kAzure+5, 1, 2);
  TH1D* h_rt_tf30_gpt30_2s1b   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, kGreen+1, 7, 2);
  TH1D* h_rt_tf30_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, kAzure+3, 1, 2);
  TH1D* h_rt_tf30_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kAzure+6, 1, 2);
  TH1D* h_rt_tf30_3s1ab   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1ab", "_hAll100", ttl, kAzure+2, 1, 2);
  TH1D* h_rt_tf30_gpt30_3s1b   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kGreen+3, 7, 2);
  TH1D* h_rt_tf30_gpt30_3s1ab   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1ab", "_hAll100", ttl, kGreen, 7, 2);





  // //==========================  Including the   ME1a

  // ((TCanvas*)gROOT->FindObject("cAll100"))->cd();

  // h_rt_tf30_3s->GetYaxis()->SetRangeUser(miny,maxy);
  // h_rt_tf30_3s1ab->GetYaxis()->SetRangeUser(miny,maxy);
  h_rt_tf30_gpt30_3s1ab->GetYaxis()->SetRangeUser(miny,maxy);

  // ((TCanvas*)gROOT->FindObject("cAll100"))->cd();
  // h_rt_tf30_3s->Draw("hist e1");
  // h_rt_tf30_gpt30_3s1ab->Draw("hist e1 same");
  // h_rt_tf30_3s->Draw("hist e1 same");
  // h_rt_tf30_3s1ab->Draw("hist e1 same");

  // TLegend *leg = new TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
  // leg->SetBorderSize(0);
  // leg->SetFillStyle(0);
  // leg->AddEntry(h_rt_tf30_3s,"Tracks: p_{T}>=30, 3+ stubs","");
  // leg->AddEntry(h_rt_tf30_3s,"anywhere","l");
  // leg->AddEntry(h_rt_tf30_3s1ab,"with ME1 in 1.6<|#eta|","l");
  // leg->AddEntry(h_rt_tf30_gpt30_3s1ab,"with (ME1+GEM) in 1.6<|#eta|<2.14","l");
  // leg->Draw();
 
  // TLatex *  tex = new TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
  // tex->SetNDC();
  // tex->Draw();
 
  // Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab.png").Data());

  // TCanvas* cAll100r2 = new TCanvas("cAll100r2","cAll100r2",800,300) ;
  // gPad->SetGridx(1);gPad->SetGridy(1);
 
  // gem_ratio = setHistoRatio(h_rt_tf30_gpt30_3s1ab, h_rt_tf30_3s1ab, "", 0.,1.8);
  // gem_ratio->Draw("e1");

  // Print(cAll100r2, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab__ratio.png").Data());



  // //==========================  Comparison with/withous Stub in ME1a ==========================//

  //  ((TCanvas*)gROOT->FindObject("cAll100"))->cd();

  //  h_rt_tf30_3s1b->Draw("hist e1");
  //  h_rt_tf30_3s1ab->Draw("hist e1 same");

  //  TLegend *leg = new TLegend(0.2,0.65,.80,0.90,NULL,"brNDC");
  //  leg->SetBorderSize(0);
  //  leg->SetFillStyle(0);
  //  leg->AddEntry(h_rt_tf30_3s1b,"Tracks: p_{T}>=30, 3+ stubs","");
  //  leg->AddEntry(h_rt_tf30_3s1b,"with ME1 in 1.6<|#eta|<2.14","l");
  //  leg->AddEntry(h_rt_tf30_3s1ab,"with ME1 in 1.6<|#eta|<2.4","l");
  //  leg->Draw();
  
  
  //  Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab_compstubME1a.png").Data());
  
  
  //  TCanvas* cAll100r2 = new TCanvas("cAll100r2","cAll100r2",800,300) ;
  //  gPad->SetGridx(1);gPad->SetGridy(1);
  
  //  gem_ratio = setHistoRatio2(h_rt_tf30_3s1ab, h_rt_tf30_3s1b, "", 0.,1.8);
  //  gem_ratio->Draw("e1");
  
  //  Print(cAll100r2, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab__ratio_compstubME1a.png").Data());


  //==========================  Comparison with/withous Stub in ME1a + GEMS ==========================//
  
  ((TCanvas*)gROOT->FindObject("cAll100"))->cd();
  
  h_rt_tf30_gpt30_3s1b->Draw("hist e1");
  h_rt_tf30_gpt30_3s1ab->Draw("hist e1 same");
  
  TLegend *leg = new TLegend(0.2,0.65,.80,0.90,NULL,"brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(h_rt_tf30_3s1b,"Tracks: p_{T}>=30, 3+ stubs","");
  leg->AddEntry(h_rt_tf30_3s1b,"with ME1 in 1.6<|#eta|<2.14","l");
  leg->AddEntry(h_rt_tf30_3s1ab,"with ME1 in 1.6<|#eta|<2.4","l");
  leg->Draw();
  
  
  Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab_compstubME1a.png").Data());
}
"""
