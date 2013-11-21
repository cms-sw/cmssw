from triggerPlotHelpers import *

## ROOT modules
from ROOT import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT
ROOT.gROOT.SetBatch(1)

#_______________________________________________________________________________
def drawLumiLabel2(x=0.2, y=0.4):
  """Label for the luminosity"""
  tex = TLatex(x, y,"L = 4*10^{34} cm^{-2} s^{-1}")
  tex.SetTextSize(0.05)
  tex.SetNDC()
  tex.Draw("same")
  return tex

#_______________________________________________________________________________
def drawPULabel(x=0.17, y=0.15, font_size=0.):
  """Label for luminosity, pile-up and bx"""
  tex = TLatex(x, y,"L=4*10^{34} (25ns PU100)")
  if (font_size > 0.):
      tex.SetFontSize(font_size)
  tex.SetNDC()
  tex.Draw("same")
  return tex

#_______________________________________________________________________________
def setHistoEta(f_name, name, cname, title, lcolor, lstyle, lwidth):
  """Style for rate plot"""
  f = TFile.Open(f_name)
  dir = f.Get("SimMuL1StrictAll")
  h0 = getH(dir,name)
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

#_______________________________________________________________________________
def setHistoRatio(num, denom, title = "", ymin=0.4, ymax=1.6, color = kRed+3):
  """Style for ratio plot"""
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

#_______________________________________________________________________________
def addRatePlotLegend(h, i, j, k, l):
  """Add legend to the trigger rate plot"""
  leg = TLegend(0.16,0.67,.8,0.9,
                "L1 Selections (L1 muon candidate p_{T}#geq%d GeV/c):"%(k),"brNDC");
  leg.SetMargin(0.15)
  leg.SetBorderSize(0)
  leg.SetTextSize(0.04)
  leg.SetFillStyle(1001)
  leg.SetFillColor(kWhite)
  leg.AddEntry(h,"CSC #geq%d stubs (anywhere)"%(l),"l");
  leg.AddEntry(i,"CSC #geq%d stubs (one in Station 1)"%(l),"l");
  leg.AddEntry(j,"GEM+CSC integrated trigger with #geq%d stubs"%(l),"l");
#  leg.AddEntry(0,"with #geq%d stubs"%(k),"");
  leg.Draw("same")
  ## carbage collection in PyROOT is something to pay attention to
  ## if you leave out this line, no legend will be drawn!
  SetOwnership(leg, False)
  return leg

#_______________________________________________________________________________
def addRatioPlotLegend(h,k):
  """Add legend to the ratio plot"""
  leg = TLegend(0.15,0.09,.45,0.65,"","brNDC");
  #leg = TLegend(0.17,0.35,.47,0.5,"","brNDC")
  leg.SetMargin(0.1)
  leg.SetBorderSize(0)
  leg.SetTextSize(0.1)
  leg.SetFillStyle(0);
  ##  leg.SetFillStyle(1001)
  ##  leg.SetFillColor(kWhite)
  leg.AddEntry(h, "(GEM+CSC)/CSC #geq%d stubs (one in Station 1)"%(k),"P");
  leg.Draw("same")
  return leg

#_______________________________________________________________________________
def setPad1Attributes(pad1):
  """Attributes for the top pad"""
  pad1.SetGridx(1)
  pad1.SetGridy(1)
  pad1.SetFrameBorderMode(0)
  pad1.SetFillColor(kWhite)
  pad1.SetTopMargin(0.06)
  pad1.SetBottomMargin(0.13)

#_______________________________________________________________________________
def setPad2Attributes(pad2):
  """Attributes for the bottom pad"""
  pad2.SetLogy(1)
  pad2.SetGridx(1)
  pad2.SetGridy(1)
  pad2.SetFillColor(kWhite)
  pad2.SetFrameBorderMode(0)
  pad2.SetTopMargin(0.06)
  pad2.SetBottomMargin(0.3)

#_______________________________________________________________________________
def produceRateVsEtaPlotForApproval(f_def, f_gem, pt_threshold = 20, n_stubs = 3):
  """Build the rate and ratio plot on a canvas"""

  colors = [kViolet+1,kAzure+2,kGreen-2]
  styles = [3345,3003,1001]

  miny = {
#    5 :  { 2 : 0.01, 3 : 0.01 },
#    6 :  { 2 : 0.01, 3 : 0.01 },
    10 : { 2 : 0.01, 3 : 0.01 },
#    15 : { 2 : 0.01, 3 : 0.01 },
    20 : { 2 : 0.01, 3 : 0.01 },
    30 : { 2 : 0.01, 3 : 0.01 },
#    40 : { 2 : 0.01, 3 : 0.01 }
    }

  maxy = {
#    5 : { 2 : 5, 3 : 3 },
#    6 : { 2 : 5, 3 : 3 },
    10 : { 2 : 80, 3 : 25 },
#    15 : { 2 : 50, 3 : 20 },
    20 : { 2 : 33, 3 : 8 },
    30 : { 2 : 30, 3 : 6 },
#    40 : { 2 : 10, 3 : 2 }
    }
  
  ## Declaration of histograms
  ttl = "                               CMS Phase-2 Simulation Preliminary;L1 muon candidate #eta;Trigger rate [kHz]";

  h = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_%ds"%(pt_threshold,n_stubs), "_hAll100", ttl, colors[0], 1, 2)
  i = setHistoEta(f_def, "h_rt_gmt_csc_ptmax%d_eta_%ds_%ds1b"%(pt_threshold,n_stubs,n_stubs), "_hAll100", ttl, colors[1], 1, 2)
  j = setHistoEta(f_gem, "h_rt_gmt_csc_ptmax%d_eta_%ds_%ds1b"%(pt_threshold,n_stubs,n_stubs), "_hAll100", ttl, colors[2], 1, 2)

  c = TCanvas("c","c",800,800)
  c.Clear()

  pad1 = TPad("pad1","top pad",0.0,0.25,1.0,1.0)
  pad1.Draw("")
  pad2 = TPad("pad2","bottom pad",0.,0.,1.0,.30)
  pad2.Draw("same")

  pad1.cd()
  setPad1Attributes(pad1)

  h.SetFillColor(colors[0]);
  i.SetFillColor(colors[1]);
  j.SetFillColor(colors[2]);
  
  ## Slava's proposal
  h.SetFillStyle(0);
  i.SetFillStyle(0);
  j.SetFillStyle(0);
  
  h.SetLineStyle(1);
  i.SetLineStyle(1);
  j.SetLineStyle(1);
  
  h.SetLineWidth(2);
  i.SetLineWidth(2);
  j.SetLineWidth(2);
  
  miny = miny[pt_threshold][n_stubs]
  maxy = maxy[pt_threshold][n_stubs]

  h.GetYaxis().SetRangeUser(miny,maxy);
  i.GetYaxis().SetRangeUser(miny,maxy);
  j.GetYaxis().SetRangeUser(miny,maxy);

  i.Draw("hist e1");
  j.Draw("same hist e1");
  h.Draw("same hist e1");

  leg = addRatePlotLegend(h, i, j, pt_threshold, n_stubs)
  tex = drawLumiLabel2()

  mini = miny
  if n_stubs==2:
    maxi = 20
  elif n_stubs==3:
    maxi = 5

  l1 = TLine(1.6,mini,1.6,maxi)
  l1.SetLineStyle(7)
  l1.SetLineWidth(2)
  l1.Draw()

  l2 = TLine(2.15,mini,2.15,maxi)
  l2.SetLineStyle(7)
  l2.SetLineWidth(2)
  l2.Draw()

  if n_stubs==2:
    ypos = .55
  elif n_stubs==3:
    ypos = .57
      
  tex2 = TLatex(.515,ypos,"GE-1/1 region")
  tex2.SetTextSize(0.05)
  tex2.SetNDC()
  tex2.Draw()
  
  pad2.cd()
  setPad2Attributes(pad2)
  gem_ratio = setHistoRatio(j, i, "", 0.01,2.0, colors[1])
  gem_ratio.GetYaxis().SetNdivisions(3)  
  gem_ratio.Draw("Pe")
  leg = addRatioPlotLegend(gem_ratio,n_stubs)

  c.SaveAs(output_dir + "rates_vs_eta__minpt%d__PU100__def_%ds_%ds1b_%ds1bgem"%(pt_threshold,n_stubs,n_stubs,n_stubs) + ext)

#_______________________________________________________________________________
def produceRateVsEtaPlotsForApproval(): 
  """Produce the rate & ratio vs eta plots"""

  produceRateVsEtaPlotForApproval(f_def, f_g98_pt10, 10, 2)
  produceRateVsEtaPlotForApproval(f_def, f_g98_pt10, 10, 3)
#  produceRateVsEtaPlotForApproval(f_def, f_g98_pt15, 15, 2)
#  produceRateVsEtaPlotForApproval(f_def, f_g98_pt15, 15, 3)
  produceRateVsEtaPlotForApproval(f_def, f_g98_pt20, 20, 2)
  produceRateVsEtaPlotForApproval(f_def, f_g98_pt20, 20, 3)
  produceRateVsEtaPlotForApproval(f_def, f_g98_pt30, 30, 2)
  produceRateVsEtaPlotForApproval(f_def, f_g98_pt30, 30, 3)
#  produceRateVsEtaPlotForApproval(f_def, f_g98_pt40, 40, 2)
#  produceRateVsEtaPlotForApproval(f_def, f_g98_pt40, 40, 3)

  
#_______________________________________________________________________________
if __name__ == "__main__":

  input_dir = "files/"
  output_dir = "plots_cmssw_601_postls1/rate_vs_eta/"
  ext = ".png"

  ## input files
  f_def =      input_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_def_pat2.root"
  f_g98_pt10 = input_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt10_pat2.root"
  f_g98_pt15 = input_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt15_pat2.root"
  f_g98_pt20 = input_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt20_pat2.root"
  f_g98_pt30 = input_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt30_pat2.root"
  f_g98_pt40 = input_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt40_pat2.root"

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

  produceRateVsEtaPlotsForApproval() 


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
