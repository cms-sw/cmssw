from triggerPlothelpers import *

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
  ratio.GetYaxis().SetTitle("ratio: (with GEM)/default")
  ratio.GetYaxis().SetTitle("ratio")
  ##ratio.GetYaxis().SetTitle("(ME1/b + GEM) / ME1/b")
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

def addRatioPlotLegend(h):
  leg = TLegend(0.17,0.4,.47,0.5,"","brNDC")
  leg.SetMargin(0.1)
  leg.SetBorderSize(0)
  leg.SetTextSize(0.1)
  leg.SetFillStyle(1001)
  leg.SetFillColor(kWhite)
  leg.AddEntry(h, "GEM+CSC/CSC tight","P")
  leg.Draw("same")

def addRatePlotLegend(h, i, j, k, l):
  leg = TLegend(0.16,0.67,.8,0.9,"L1 Selections (#geq" + k + " stations, L1 candidate p_{T}#geq" + l + " GeV/c):","brNDC")
  leg.SetMargin(0.20)
  leg.SetBorderSize(0)
  leg.SetTextSize(0.04)
  leg.SetFillStyle(1001)
  leg.SetFillColor(kWhite)
  leg.AddEntry(h,"CSC, loose","f")
  leg.AddEntry(i,"CSC, tight","f")
  leg.AddEntry(j,"GEM+CSC Integrated Trigger","f")
  leg.Draw("same")
       
def addRatePlots(h, i, j, col1, col2, col3, sty1, sty2, sty3, sty4, miny, maxy):
  h.SetFillColor(col1)
  i.SetFillColor(col2)
  j.SetFillColor(col3)

  h.SetFillStyle(sty1)
  i.SetFillStyle(sty2)
  j.SetFillStyle(sty3)
  
  h.GetYaxis().SetRangeUser(miny,maxy)
  i.GetYaxis().SetRangeUser(miny,maxy)
  j.GetYaxis().SetRangeUser(miny,maxy)
  
  i_clone = i.Clone("i_clone")
  j_clone = j.Clone("j_clone")
  i_clone2 = i.Clone("i_clone2")
  j_clone2 = j.Clone("j_clone2")

  for ii in range(0,15):
    i_clone2.SetBinContent(ii,0)
    j_clone2.SetBinContent(ii,0)
    i_clone2.SetBinError(ii,0)
    j_clone2.SetBinError(ii,0)
  for ii in range(26,35):
    i_clone2.SetBinContent(ii,0)
    j_clone2.SetBinContent(ii,0)
    i_clone2.SetBinError(ii,0)
    j_clone2.SetBinError(ii,0)
  for ii in range(15,26):
    i_clone.SetBinContent(ii,0)
    j_clone.SetBinContent(ii,0)
    i_clone.SetBinError(ii,0)
    j_clone.SetBinError(ii,0)
 
  j_clone.SetFillStyle(sty4)

  i_clone.Draw("hist e1 same")
  j_clone.Draw("hist e1 same")
  h.Draw("hist e1 same")
  i_clone2.Draw("hist e1 same")
  j_clone2.Draw("hist e1 same")

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

def produceRateVsEtaPlot(h, i, j, col1, col2, col3, sty1, sty2, sty3, sty4, miny, maxy, k, l, plots, ext):
  c = TCanvas("c","c",800,800)
  c.Clear()
  pad1 = TPad("pad1","top pad",0.0,0.25,1.0,1.0)
  pad1.Draw("")
  pad2 = TPad("pad2","bottom pad",0,0.,1.0,.30)
  pad2.Draw("same")

  pad1.cd()
  setPad1Attributes(pad1)
  addRatePlots(h,i,j,col1,col2,col3,sty1,sty2,sty3,3355,miny,maxy)
  addRatePlotLegend(h, i, j, k,l)
  drawLumiLabel2()

  pad2.cd()
  setPad2Attributes(pad2)
  gem_ratio = setHistoRatio(j, i, "", 0.01,2.0, col2)
  gem_ratio.Draw("Pe")
  addRatioPlotLegend(gem_ratio)
  
  c.SaveAs(plots + "rates_vs_eta__minpt" + l + "__PU100__def_" + k + "s_" + k + "s1b_" + k + "s1bgem" + ext)

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
  sty2 = 2003
  sty3 = 2002

  ## Declaration of histograms
  ttl = "        L1 Single Muon Trigger                   CMS Simulation;L1 muon candidate #eta;rate [kHz]"

  vs_eta_minpt = "10"
  h_rt_tf10_2s = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, col1, 1, 2)
  h_rt_tf10_2s1b = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col2, 1, 2)
  h_rt_tf10_gpt10_2s1b = setHistoEta(f_g98_pt10, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col3, 1, 2)
  h_rt_tf10_3s = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, col1, 1, 2)
  h_rt_tf10_3s1b = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col2, 1, 2)
  h_rt_tf10_gpt10_3s1b = setHistoEta(f_g98_pt10, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col3, 7, 2)
  
  vs_eta_minpt = "20"
  h_rt_tf20_2s = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, col1, 1, 2)
  h_rt_tf20_2s1b = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col2, 1, 2)
  h_rt_tf20_gpt20_2s1b = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col3, 1, 2)
  h_rt_tf20_3s = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, col1, 1, 2)
  h_rt_tf20_3s1b = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col2, 1, 2)
  h_rt_tf20_gpt20_3s1b = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col3, 1, 2)

  vs_eta_minpt = "30"
  h_rt_tf30_2s = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, col1, 1, 2)
  h_rt_tf30_2s1b = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col2, 1, 2)
  h_rt_tf30_gpt30_2s1b = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col3, 1, 2)
  h_rt_tf30_3s = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, col1, 1, 2)
  h_rt_tf30_3s1b = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col2, 1, 2)
  h_rt_tf30_gpt30_3s1b = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col3, 1, 2)

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
		       col1,col2,col3,sty1,sty2,sty3,3355,0.01,80,"2","10",plots,ext)
  ## ------------ +2 stubs, L1 candidate muon pt=20GeV ----------------##
  produceRateVsEtaPlot(h_rt_tf20_2s,h_rt_tf20_2s1b,h_rt_tf20_gpt20_2s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,0.01,40,"2","20",plots,ext)
  ## ------------ +2 stubs, L1 candidate muon pt=30GeV ----------------##
  produceRateVsEtaPlot(h_rt_tf30_2s,h_rt_tf30_2s1b,h_rt_tf30_gpt30_2s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,0.01,30,"2","30",plots,ext)
  ## ------------ +3 stubs, L1 candidate muon pt=10GeV ----------------##
  produceRateVsEtaPlot(h_rt_tf10_3s,h_rt_tf10_3s1b,h_rt_tf10_gpt10_3s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,0.01,25,"3","10",plots,ext)
  ## ------------ +3 stubs, L1 candidate muon pt=20GeV ----------------##
  produceRateVsEtaPlot(h_rt_tf20_3s,h_rt_tf20_3s1b,h_rt_tf20_gpt20_3s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,0.01,10,"3","20",plots,ext)
  ## ------------ +3 stubs, L1 candidate muon pt=30GeV ----------------##
  produceRateVsEtaPlot(h_rt_tf30_3s,h_rt_tf30_3s1b,h_rt_tf30_gpt30_3s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,0.01, 6,"3","30",plots,ext)
  
if __name__ == "__main__":
##  produceRateVsEtaPlotsForApproval(".C", "plots/rate_vs_eta/")
  produceRateVsEtaPlotsForApproval(".pdf", "plots/rate_vs_eta/")
