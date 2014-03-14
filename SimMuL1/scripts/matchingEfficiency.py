## custom modules
from effFunctions import *
from cuts import *
from tdrStyle import *
from GEMCSCdPhiDict import *
from math import *

## ROOT modules
from ROOT import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT
ROOT.gROOT.SetBatch(1)

gStyle.SetStatW(0.07)
gStyle.SetStatH(0.06)

gStyle.SetOptStat(0)

gStyle.SetTitleStyle(0)
gStyle.SetTitleAlign(13) ## coord in top left
gStyle.SetTitleX(0.)
gStyle.SetTitleY(1.)
gStyle.SetTitleW(1)
gStyle.SetTitleH(0.058)
gStyle.SetTitleBorderSize(0)

gStyle.SetPadLeftMargin(0.126)
gStyle.SetPadRightMargin(0.04)
gStyle.SetPadTopMargin(0.06)
gStyle.SetPadBottomMargin(0.13)

gStyle.SetMarkerStyle(1)

#_______________________________________________________________________________
def getTree(fileName):
    """Get tree for given filename"""

    analyzer = "GEMCSCAnalyzer"
    trk_eff = "trk_eff_st1"

    file = TFile.Open(fileName)
    if not file:
        sys.exit('Input ROOT file %s is missing.' %(fileName))

    dir = file.Get(analyzer)
    if not dir:
        sys.exit('Directory %s does not exist.' %(dir))
        
    tree = dir.Get(trk_eff)
    if not tree:
        sys.exit('Tree %s does not exist.' %(tree))

    return tree


#_______________________________________________________________________________
def getDphi(eff,pt,evenOdd):
    """Return the delta phi cut value given: (1) an efficiency, (2) a pt value and (3) choice for even/odd chambers"""

    return dphi_lct_pad["%s"%(eff)]["%s"%(pt)]["%s"%(evenOdd)]

#_______________________________________________________________________________
def padMatchingEffVsGenMuonPhiForPosAndNegMuons(
    filesDir, plotDir, pt, doOverlaps, ext):

    """
    This functions makes the matching effciency vs generated muon phi
    for positive and negative muons. These plots were used in the approval round of 
    September 9th 2013.
    """

    ok_eta = TCut("abs(eta)>1.64 && abs(eta)<2.12")
    if (doOverlaps):
        cut1 = ok_pad1_overlap
        cut2 = ok_pad2_overlap
        overlapStr = "_overlap"
    else:
        cut1 = ok_pad1
        cut2 = ok_pad2
        overlapStr = ""
        
    t = getTree("%sgem_csc_delta_pt%d_pad4.root"%(filesDir,pt))

    ## latest instructions by Vadim on 21-08-2013
    ok_pad1_or_pad2 = OR(ok_pad1,ok_pad2)
    ok_eta_and_Qn = AND(ok_eta,ok_Qn)
    ok_eta_and_Qp = AND(ok_eta,ok_Qp)

    ## variables for the plot
    title = " " * 9 + "GEM pad matching" + " " * 16 + "CMS Simulation Preliminary"
    xTitle = "Generated muon #phi [deg]"
    yTitle = "Reconstruction efficiency"
    toPlot = "fmod(phi*180./TMath::Pi(), 360/18.)"
    h_bins = "(40,-10,10)"
    nBins = int(h_bins[1:-1].split(',')[0])
    minBin = int(h_bins[1:-1].split(',')[1])
    maxBin = int(h_bins[1:-1].split(',')[2])

    c = TCanvas("c","c",800,600)
    c.cd()
    base  = TH1F("base","",nBins,minBin,maxBin)
    base.SetMinimum(0.0)
    base.SetMaximum(1.1)
    base.Draw("")
    base.GetXaxis().SetLabelSize(0.05)
    base.GetYaxis().SetLabelSize(0.05)
    base.SetTitle("%s;%s;%s"%(title,xTitle,yTitle))
    hgn = draw_geff(t, "%s;%s;%s"%(title,xTitle,yTitle), h_bins, 
                    toPlot, ok_eta_and_Qn, ok_pad1_or_pad2,"same",     kRed)
    hgp = draw_geff(t, "%s;%s;%s"%(title,xTitle,yTitle), h_bins, 
                    toPlot, ok_eta_and_Qp, ok_pad1_or_pad2,"same", kBlue)
      
    maxi = 1.1
    mini = 0.0
    """
    hgn.SetMinimum(mini)
    hgn.SetMaximum(maxi)
    hgn.GetXaxis().SetLabelSize(0.05)
    hgn.GetYaxis().SetLabelSize(0.05)
    hgp.GetXaxis().SetLabelSize(0.05)
    hgp.GetYaxis().SetLabelSize(0.05)
    """
    
    l1 = TLine(-5,mini,-5,maxi)
    l1.SetLineStyle(2)
    l1.Draw()
    l2 = TLine(5,mini,5,maxi)
    l2.SetLineStyle(2)
    l2.Draw()
    
    leg = TLegend(0.25,0.23,.75,0.5, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.06)
    leg.AddEntry(0,"muon p_{T} = %d GeV/c"%(pt),"") 
    leg.AddEntry(hgp, "Postive muons","l")
    leg.AddEntry(hgn, "Negative muons","l")
    leg.Draw()

    ## Print additional information
    """
    tex2 = TLatex(.67,.8,"   L1 Trigger")
    tex2.SetTextSize(0.05)
    tex2.SetNDC()
    tex2.Draw()
    """
  
    tex = TLatex(.7,.2,"1.64<|#eta|<2.12")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw()

#    gPad.Print("%sgem_pad_eff_for_LCT_vs_phi_pt20%s%s"%(plotDir,overlapStr,ext))
    c.Print("%sgem_pad_eff_for_LCT_vs_phi_pt%d%s%s"%(plotDir,pt,overlapStr,ext))



#_______________________________________________________________________________
def padMatchingEffVsHalfStripForOddEven(filesDir, plotDir, pt, doOverlaps, ext):
    """efficiency vs half-strip  - separate odd-even""" 
 
    gStyle.SetTitleStyle(0);
    gStyle.SetTitleAlign(13); ##coord in top left
    gStyle.SetTitleX(0.);
    gStyle.SetTitleY(1.);
    gStyle.SetTitleW(1);
    gStyle.SetTitleH(0.058);
    gStyle.SetTitleBorderSize(0);
    
    gStyle.SetPadLeftMargin(0.126);
    gStyle.SetPadRightMargin(0.04);
    gStyle.SetPadTopMargin(0.06);
    gStyle.SetPadBottomMargin(0.13);
    gStyle.SetOptStat(0);
    gStyle.SetMarkerStyle(1);
    

    ok_eta = TCut("TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.12")
    if (doOverlaps):
        cut1 = ok_pad1_overlap
        cut2 = ok_pad2_overlap
        overlapStr = "_overlap"
    else:
        cut1 = ok_pad1
        cut2 = ok_pad2
        overlapStr = ""

    t = getTree("%sgem_csc_delta_pt%d_pad4.root"%(filesDir,pt))
    ho = draw_geff(t, "         GEM pad matching               CMS Simulation Preliminary;LCT half-strip number;Efficiency", 
                  "h_odd", "(130,0.5,130.5)", "hs_lct_odd", AND(ok_lct1,ok_eta), cut1, "", kRed)
    he = draw_geff(t, "         GEM pad matching               CMS Simulation Preliminary;LCT half-strip number;Efficiency", 
                  "h_evn", "(130,0.5,130.5)", "hs_lct_even", AND(ok_lct1,ok_eta), cut2, "same")
    ho.SetMinimum(0.)
    ho.GetXaxis().SetLabelSize(0.05)
    ho.GetYaxis().SetLabelSize(0.05)
    

    leg = TLegend(0.25,0.23,.75,0.5, "", "brNDC");
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.06)
    leg.AddEntry(0,"muon p_{T} = %s GeV/c"%(pt),"")
    leg.AddEntry(he, "\"Close\" chamber pairs","l")
    leg.AddEntry(ho, "\"Far\" chamber pairs","l")
    leg.Draw();
    
    tex = TLatex(.66,.73,"1.64<|#eta|<2.12")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw()
    
    ## this has to be fixed
    c.Print("%sgem_pad_eff_for_LCT_vs_HS_pt%d%s%s"%(plotDir,pt,overlapStr,ext))


#_______________________________________________________________________________
def padMatchingEffVsLctEtaForOddEven(filesDir, plotDir, pt, doOverlaps, ext):
    """efficiency vs LCT eta  - separate odd-even""" 
 
    gStyle.SetTitleStyle(0);
    gStyle.SetTitleAlign(13); ##coord in top left
    gStyle.SetTitleX(0.);
    gStyle.SetTitleY(1.);
    gStyle.SetTitleW(1);
    gStyle.SetTitleH(0.058);
    gStyle.SetTitleBorderSize(0);
    
    gStyle.SetPadLeftMargin(0.126);
    gStyle.SetPadRightMargin(0.04);
    gStyle.SetPadTopMargin(0.06);
    gStyle.SetPadBottomMargin(0.13);
    gStyle.SetOptStat(0);
    gStyle.SetMarkerStyle(1);
    
    ok_eta = TCut("TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.12")
    if (doOverlaps):
        cut1 = ok_pad1_overlap
        cut2 = ok_pad2_overlap
        overlapStr = "_overlap"
    else:
        cut1 = ok_pad1
        cut2 = ok_pad2
        overlapStr = ""

    t = getTree("%sgem_csc_delta_pt%d_pad4.root"%(filesDir,pt))
    topTitle = "         GEM pad matching               CMS Simulation Preliminary"
    xTitle = "LCT |#eta|LCT |#eta|"
    yTitle = "Efficiency"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)
    ho = draw_geff(t, title, "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_lct_odd)", AND(ok_lct1,ok_eta), cut1, "", kRed)
    he = draw_geff(t, title, "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta_lct_even)", AND(ok_lct1,ok_eta), cut2, "same")
    ho.SetMinimum(0.)
    ho.GetXaxis().SetLabelSize(0.05)
    ho.GetYaxis().SetLabelSize(0.05)
    
    leg = TLegend(0.25,0.23,.75,0.5, "", "brNDC");
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.06)
    leg.AddEntry(0,"muon p_{T} = %s GeV/c"%(pt),"")
    leg.AddEntry(he, "\"Close\" chamber pairs","l")
    leg.AddEntry(ho, "\"Far\" chamber pairs","l")
    leg.Draw();
    
    tex = TLatex(.66,.73,"1.64<|#eta|<2.12")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw()
    
    ## this has to be fixed
    c.Print("%sgem_pad_eff_for_LCT_vs_LCT_pt%d%s%s"%(plotDir,pt,overlapStr,ext))

#_______________________________________________________________________________
def padMatchingEffVsSimTrackEtaForOddEven(filesDir, plotDir, pt, doOverlaps, ext):
    """efficiency vs simtrack eta  - separate odd-even""" 
 
    gStyle.SetTitleStyle(0);
    gStyle.SetTitleAlign(13); ##coord in top left
    gStyle.SetTitleX(0.);
    gStyle.SetTitleY(1.);
    gStyle.SetTitleW(1);
    gStyle.SetTitleH(0.058);
    gStyle.SetTitleBorderSize(0);
    
    gStyle.SetPadLeftMargin(0.126);
    gStyle.SetPadRightMargin(0.04);
    gStyle.SetPadTopMargin(0.06);
    gStyle.SetPadBottomMargin(0.13);
    gStyle.SetOptStat(0);
    gStyle.SetMarkerStyle(1);
    
    ok_eta = TCut("TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.12")
    if (doOverlaps):
        cut1 = ok_pad1_overlap
        cut2 = ok_pad2_overlap
        overlapStr = "_overlap"
    else:
        cut1 = ok_pad1
        cut2 = ok_pad2
        overlapStr = ""

    t = getTree("%sgem_csc_delta_pt%d_pad4.root"%(filesDir,pt))

    topTitle = "         GEM pad matching               CMS Simulation Preliminary"
    xTitle = "SimTrack |#eta|"
    yTitle = "Efficiency"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)
    
    ho = draw_geff(t, title, "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", AND(ok_lct1,ok_eta), cut1, "", kRed)
    he = draw_geff(t, title, "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", AND(ok_lct1,ok_eta), cut2, "same")
    ho.SetMinimum(0.)
    ho.GetXaxis().SetLabelSize(0.05)
    ho.GetYaxis().SetLabelSize(0.05)
    
    leg = TLegend(0.25,0.23,.75,0.5, "", "brNDC");
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.06)
    leg.AddEntry(0,"muon p_{T} = %s GeV/c"%(pt),"")
    leg.AddEntry(he, "\"Close\" chamber pairs","l")
    leg.AddEntry(ho, "\"Far\" chamber pairs","l")
    leg.Draw();
    
    tex = TLatex(.66,.73,"1.64<|#eta|<2.12")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw()
    
    ## this has to be fixed
    c.Print("%sgem_pad_eff_for_LCT_vs_TrkEta_pt%d%s%s"%(plotDir,pt,overlapStr,ext))


#_______________________________________________________________________________
def cscMatchingEfficiencyToStripsAndWires(filesDir, input_file, plotDir, ext):

    gStyle.SetTitleStyle(0);
    gStyle.SetTitleAlign(13); ##coord in top left
    gStyle.SetTitleX(0.);
    gStyle.SetTitleY(1.);
    gStyle.SetTitleW(1);
    gStyle.SetTitleH(0.058);
    gStyle.SetTitleBorderSize(0);
    
    gStyle.SetPadLeftMargin(0.126);
    gStyle.SetPadRightMargin(0.04);
    gStyle.SetPadTopMargin(0.06);
    gStyle.SetPadBottomMargin(0.13);
    gStyle.SetOptStat(0);
    gStyle.SetMarkerStyle(1);
    
    ok_eta = TCut("TMath::Abs(eta)>1.5 && TMath::Abs(eta)<2.5")

    t = getTree("%s%s"%(filesDir,input_file))

    ## variables for the plot
    topTitle = " " * 11 + "CSC Digi matching" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Generated muon #eta"
    yTitle = "Efficiency"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)
    toPlot = "TMath::Abs(eta)"
    h_bins = "(100,1.5,2.5)"
    nBins = int(h_bins[1:-1].split(',')[0])
    minBin = float(h_bins[1:-1].split(',')[1])
    maxBin = float(h_bins[1:-1].split(',')[2])

    c = TCanvas("c","c",700,450)
    c.Clear()
    base  = TH1F("base",title,nBins,minBin,maxBin)
    base.SetMinimum(0.0)
    base.SetMaximum(1.02)
    base.Draw("")
    base.GetXaxis().SetLabelSize(0.05)
    base.GetYaxis().SetLabelSize(0.05)
    base.GetYaxis().SetRangeUser(yrange[0],yrange[1])

    h1 = draw_geff(t, title, h_bins, toPlot, ok_sh1, ok_w1, "same", kRed)
    h2 = draw_geff(t, title, h_bins, toPlot, ok_sh1, ok_st1, "same")
   
    leg = TLegend(0.45,0.2,.75,0.35, "", "brNDC");
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.06)
    leg.AddEntry(h1, "Wires","l")
    leg.AddEntry(h2, "Strips","l")
    leg.Draw("same");
    
    tex = TLatex(.45,.4,"1.5<|#eta|<2.5")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw("same")

    c.Print("%scsc_digi_matching_efficiency%s"%(plotDir,ext))


#_______________________________________________________________________________
def cscMatchingEfficiencyToStripsAndWires_2(filesDir, input_file, plotDir, ext):

    gStyle.SetTitleStyle(0);
    gStyle.SetTitleAlign(13); ##coord in top left
    gStyle.SetTitleX(0.);
    gStyle.SetTitleY(1.);
    gStyle.SetTitleW(1);
    gStyle.SetTitleH(0.058);
    gStyle.SetTitleBorderSize(0);
    
    gStyle.SetPadLeftMargin(0.126);
    gStyle.SetPadRightMargin(0.04);
    gStyle.SetPadTopMargin(0.06);
    gStyle.SetPadBottomMargin(0.13);
    gStyle.SetOptStat(0);
    gStyle.SetMarkerStyle(1);
    
    ok_eta = TCut("TMath::Abs(eta)>1.5 && TMath::Abs(eta)<2.5")

    t = getTree("%s%s"%(filesDir, input_file))

    ## variables for the plot
    topTitle = " " * 11 + "CSC Digi matching" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Generated muon #eta"
    yTitle = "Efficiency"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)
    toPlot = "TMath::Abs(eta)"
    h_bins = "(100,1.5,2.5)"
    nBins = int(h_bins[1:-1].split(',')[0])
    minBin = float(h_bins[1:-1].split(',')[1])
    maxBin = float(h_bins[1:-1].split(',')[2])

    c = TCanvas("c","c",700,450)
    c.Clear()
    base  = TH1F("base",title,nBins,minBin,maxBin)
    base.SetMinimum(0.0)
    base.SetMaximum(1.02)
    base.Draw("")
    base.GetXaxis().SetLabelSize(0.05)
    base.GetYaxis().SetLabelSize(0.05)

    h1 = draw_geff(t, title, h_bins, toPlot, ok_sh1, OR(ok_w1,ok_st1), "same", kRed)
    h2 = draw_geff(t, title, h_bins, toPlot, ok_sh1, AND(ok_w1,ok_st1), "same")
   
    leg = TLegend(0.45,0.2,.75,0.35, "", "brNDC");
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.06)
    leg.AddEntry(h1, "Wires OR strips","l")
    leg.AddEntry(h2, "Wires AND strips","l")
    leg.Draw("same");
    
    tex = TLatex(.45,.4,"1.5<|#eta|<2.5")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw("same")

    c.Print("%scsc_combined_digi_matching_efficiency%s"%(plotDir,ext))



#_______________________________________________________________________________
def cscMatchingEfficiencyToAlctClct(filesDir, input_file, plotDir, ext):

    gStyle.SetTitleStyle(0);
    gStyle.SetTitleAlign(13); ##coord in top left
    gStyle.SetTitleX(0.);
    gStyle.SetTitleY(1.);
    gStyle.SetTitleW(1);
    gStyle.SetTitleH(0.058);
    gStyle.SetTitleBorderSize(0);
    
    gStyle.SetPadLeftMargin(0.126);
    gStyle.SetPadRightMargin(0.04);
    gStyle.SetPadTopMargin(0.06);
    gStyle.SetPadBottomMargin(0.13);
    gStyle.SetOptStat(0);
    gStyle.SetMarkerStyle(1);
    
    ok_eta = TCut("TMath::Abs(eta)>1.5 && TMath::Abs(eta)<2.5")

    t = getTree("%s%s"%(filesDir, input_file))

    ## variables for the plot
    topTitle = " " * 11 + "CSC Stub matching" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Generated muon #eta"
    yTitle = "Efficiency"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)
    toPlot = "TMath::Abs(eta)"
    h_bins = "(100,1.5,2.5)"
    nBins = int(h_bins[1:-1].split(',')[0])
    minBin = float(h_bins[1:-1].split(',')[1])
    maxBin = float(h_bins[1:-1].split(',')[2])

    c = TCanvas("c","c",700,450)
    c.Clear()
    base  = TH1F("base",title,nBins,minBin,maxBin)
    base.SetMinimum(0.0)
    base.SetMaximum(1.02)
    base.Draw("")
    base.GetXaxis().SetLabelSize(0.05)
    base.GetYaxis().SetLabelSize(0.05)

    h1 = draw_geff(t, title, h_bins, toPlot, AND(ok_sh1,ok_w1), ok_alct1, "same", kRed)
    h2 = draw_geff(t, title, h_bins, toPlot, AND(ok_sh1,ok_st1), ok_clct1, "same")
   
    leg = TLegend(0.45,0.2,.75,0.35, "", "brNDC");
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.06)
    leg.AddEntry(h1, "ALCT","l")
    leg.AddEntry(h2, "CLCT","l")
    leg.Draw("same");
    
    tex = TLatex(.45,.4,"1.64<|#eta|<2.12")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw("same")

    c.Print("%scsc_stub_matching_efficiency%s"%(plotDir,ext))


#_______________________________________________________________________________
def cscMatchingEfficiencyToAlctClct_2(filesDir, input_file, plotDir, ext):

    gStyle.SetTitleStyle(0);
    gStyle.SetTitleAlign(13); ##coord in top left
    gStyle.SetTitleX(0.);
    gStyle.SetTitleY(1.);
    gStyle.SetTitleW(1);
    gStyle.SetTitleH(0.058);
    gStyle.SetTitleBorderSize(0);
    
    gStyle.SetPadLeftMargin(0.126);
    gStyle.SetPadRightMargin(0.04);
    gStyle.SetPadTopMargin(0.06);
    gStyle.SetPadBottomMargin(0.13);
    gStyle.SetOptStat(0);
    gStyle.SetMarkerStyle(1);
    
    ok_eta = TCut("TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.12")

    t = getTree("%s%s"%(filesDir,input_file))

    ## variables for the plot
    topTitle = " " * 11 + "CSC Stub matching" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Generated muon #eta"
    yTitle = "Efficiency"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)
    toPlot = "TMath::Abs(eta)"
    h_bins = "(100,1.5,2.5)"
    nBins = int(h_bins[1:-1].split(',')[0])
    minBin = float(h_bins[1:-1].split(',')[1])
    maxBin = float(h_bins[1:-1].split(',')[2])

    c = TCanvas("c","c",700,450)
    c.Clear()
    base  = TH1F("base",title,nBins,minBin,maxBin)
    base.SetMinimum(0.0)
    base.SetMaximum(1.02)
    base.Draw("")
    base.GetXaxis().SetLabelSize(0.05)
    base.GetYaxis().SetLabelSize(0.05)
    #,ok_w1,ok_st1,  AND(ok_sh1,ok_w1,ok_st1)
    h1 = draw_geff(t, title, h_bins, toPlot, ok_sh1, OR(ok_alct1,ok_clct1), "same", kRed)
    h2 = draw_geff(t, title, h_bins, toPlot, ok_sh1, AND(ok_alct1,ok_clct1), "same")
   
    leg = TLegend(0.45,0.2,.75,0.35, "", "brNDC");
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.06)
    leg.AddEntry(h1, "ALCT OR CLCT","l")
    leg.AddEntry(h2, "ALCT AND CLCT","l")
    leg.Draw("same");
    
    tex = TLatex(.45,.4,"1.64<|#eta|<2.12")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw("same")

    c.Print("%scsc_combined_stub_matching_efficiency%s"%(plotDir,ext))


#_______________________________________________________________________________
def cscMatchingEfficiencyToLct(filesDir, input_file, plotDir, ext):

    gStyle.SetTitleStyle(0);
    gStyle.SetTitleAlign(13); ##coord in top left
    gStyle.SetTitleX(0.);
    gStyle.SetTitleY(1.);
    gStyle.SetTitleW(1);
    gStyle.SetTitleH(0.058);
    gStyle.SetTitleBorderSize(0);
    
    gStyle.SetPadLeftMargin(0.126);
    gStyle.SetPadRightMargin(0.04);
    gStyle.SetPadTopMargin(0.06);
    gStyle.SetPadBottomMargin(0.13);
    gStyle.SetOptStat(0);
    gStyle.SetMarkerStyle(1);
    
    ok_eta = TCut("TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.12")

    t = getTree("%s%s"%(filesDir, input_file))

    ## variables for the plot
    topTitle = " " * 11 + "CSC Stub matching" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Generated muon #eta"
    yTitle = "Efficiency"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)
    toPlot = "TMath::Abs(eta)"
    h_bins = "(100,1.5,2.5)"
    nBins = int(h_bins[1:-1].split(',')[0])
    minBin = float(h_bins[1:-1].split(',')[1])
    maxBin = float(h_bins[1:-1].split(',')[2])

    c = TCanvas("c","c",700,450)
    c.Clear()
    base  = TH1F("base",title,nBins,minBin,maxBin)
    base.SetMinimum(0.0)
    base.SetMaximum(1.02)
    base.Draw("")
    base.GetXaxis().SetLabelSize(0.05)
    base.GetYaxis().SetLabelSize(0.05)
    #AND(ok_sh1, ok_st1, ok_w1)
    h1 = draw_geff(t, title, h_bins, toPlot, AND(ok_sh1, ok_st1, ok_w1, ok_alct1, ok_clct1), ok_lct1, "same", kRed)
    h2 = draw_geff(t, title, h_bins, toPlot, ok_sh1, ok_lct1, "same", kBlue)

##    h2 = draw_geff(t, title, h_bins, toPlot, AND(ok_sh2, ok_st2, ok_w2, ok_alct2, ok_clct2), ok_lct2, "same", kBlue)
   
    leg = TLegend(0.10,0.2,.75,0.35, "", "brNDC");
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.04)
    leg.AddEntry(h1, "LCT,denominator requiring simtrack with any ALCT and any CLCT","l")
    leg.AddEntry(h2, "LCT","l")
    leg.Draw("same");
    
    tex = TLatex(.25,.4,"PU140,1.64<|#eta|<2.12")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw("same")

    c.Print("%scsc_lct_matching_efficiency%s"%(plotDir,ext))

#____________________________________________________________________
def PadmatchingtoSimTrEfficiency(filesDir, input_file, plotDir, ext):

    gStyle.SetTitleStyle(0);
    gStyle.SetTitleAlign(13); ##coord in top left
    gStyle.SetTitleX(0.);
    gStyle.SetTitleY(1.);
    gStyle.SetTitleW(1);
    gStyle.SetTitleH(0.058);
    gStyle.SetTitleBorderSize(0);
    
    gStyle.SetPadLeftMargin(0.126);
    gStyle.SetPadRightMargin(0.04);
    gStyle.SetPadTopMargin(0.06);
    gStyle.SetPadBottomMargin(0.13);
    gStyle.SetOptStat(0);
    gStyle.SetMarkerStyle(1);
    
    ok_eta = TCut("TMath::Abs(eta)>1.5 && TMath::Abs(eta)<2.5")

    t = getTree("%s%s"%(filesDir, input_file))

    ## variables for the plot
    topTitle = " " * 11 + "GEM Pad matching Efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Generated muon #eta"
    yTitle = "Efficiency"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)
    toPlot = "TMath::Abs(eta)"
    h_bins = "(70,1.5,2.2)"
    nBins = int(h_bins[1:-1].split(',')[0])
    minBin = float(h_bins[1:-1].split(',')[1])
    maxBin = float(h_bins[1:-1].split(',')[2])

    c = TCanvas("c","c",700,450)
    c.Clear()
    base  = TH1F("base",title,nBins,minBin,maxBin)
    base.SetMinimum(0.0)
    base.SetMaximum(1.02)
    base.Draw("")
    base.GetXaxis().SetLabelSize(0.05)
    base.GetYaxis().SetLabelSize(0.05)

     
    Cut1_den = ok_gsh1 
    Cut1_num = ok_pad1
    Cut2_den = ok_gsh2 
    Cut2_num = ok_pad2 
    Cut3_den = OR(ok_gsh1,ok_gsh2) 
    Cut3_num = OR(ok_pad1,ok_pad2) 
    h1 = draw_geff(t, title, h_bins, toPlot, Cut1_den, Cut1_num, "same", kRed)
    h2 = draw_geff(t, title, h_bins, toPlot, Cut2_den, Cut2_num, "same", kBlue)
#    h3 = draw_geff(t, title, h_bins, toPlot, Cut3_den, Cut3_num, "same", kGreen)

    
    leg = TLegend(0.45,0.2,.75,0.35, "", "brNDC");
    leg.SetBorderSize(0)
#    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.AddEntry(h1, "Pad matching for odd GEMDetId","l")
    leg.AddEntry(h2, "Pad matching for even GEMDetId","l")
#    leg.AddEntry(h3, "Pad1 or Pad2","l")
    leg.Draw("same");
   
    
    tex = TLatex(.25,.4,"PU140,1.64<|#eta|<2.12")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw("same")

    c.Print("%sgempad_matching_efficiency%s"%(plotDir,ext))

#____________________________________________________________________
def PadDigimatchingtoSimTrEfficiency(filesDir, input_file, plotDir, ext):

    gStyle.SetTitleStyle(0);
    gStyle.SetTitleAlign(13); ##coord in top left
    gStyle.SetTitleX(0.);
    gStyle.SetTitleY(1.);
    gStyle.SetTitleW(1);
    gStyle.SetTitleH(0.058);
    gStyle.SetTitleBorderSize(0);
    
    gStyle.SetPadLeftMargin(0.126);
    gStyle.SetPadRightMargin(0.04);
    gStyle.SetPadTopMargin(0.06);
    gStyle.SetPadBottomMargin(0.13);
    gStyle.SetOptStat(0);
    gStyle.SetMarkerStyle(1);
    
    ok_eta = TCut("TMath::Abs(eta)>1.5 && TMath::Abs(eta)<2.5")

    t = getTree("%s%s"%(filesDir, input_file))

    ## variables for the plot
    topTitle = " " * 11 + "GEM Pad Digi matching Efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Generated muon #eta"
    yTitle = "Efficiency"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)
    toPlot = "TMath::Abs(eta)"
    h_bins = "(70,1.5,2.2)"
    nBins = int(h_bins[1:-1].split(',')[0])
    minBin = float(h_bins[1:-1].split(',')[1])
    maxBin = float(h_bins[1:-1].split(',')[2])

    c = TCanvas("c","c",700,450)
    c.Clear()
    base  = TH1F("base",title,nBins,minBin,maxBin)
    base.SetMinimum(0.0)
    base.SetMaximum(1.02)
    base.Draw("")
    base.GetXaxis().SetLabelSize(0.05)
    base.GetYaxis().SetLabelSize(0.05)

     
    Cut1_den = ok_gsh1 
    Cut1_num = ok_gdg1
    Cut2_den = ok_gsh2
    Cut2_num = ok_gdg2
#    Cut3_den = OR(Cut1_den,Cut2_den) 
#    Cut3_num = OR(Cut1_num,Cut2_num) 
    h1 = draw_geff(t, title, h_bins, toPlot, Cut1_den, Cut1_num, "same", kRed)
    h2 = draw_geff(t, title, h_bins, toPlot, Cut2_den, Cut2_num, "same", kBlue)
#    h3 = draw_geff(t, title, h_bins, toPlot, Cut3_den, Cut3_num, "same", kGreen)

    
    leg = TLegend(0.25,0.2,.75,0.35, "", "brNDC");
    leg.SetBorderSize(0)
#    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.AddEntry(h1, "GEM Digi matching for odd GEMDetId ","l")
    leg.AddEntry(h2, "GEM Digi matching for even GEMDetId","l")
#    leg.AddEntry(h3, "Pad1 or Pad2","l")
    leg.Draw("same");
   
    
    tex = TLatex(.25,.4,"PU140 ,1.64<|#eta|<2.12")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw("same")

    c.Print("%sgempad_Digi_matching_efficiency%s"%(plotDir,ext))


#____________________________________________________________________
def PadmatchingwithLCTstoSimTrEfficiency(filesDir, input_file, plotDir, ext):

    gStyle.SetTitleStyle(0);
    gStyle.SetTitleAlign(13); ##coord in top left
    gStyle.SetTitleX(0.);
    gStyle.SetTitleY(1.);
    gStyle.SetTitleW(1);
    gStyle.SetTitleH(0.058);
    gStyle.SetTitleBorderSize(0);
    
    gStyle.SetPadLeftMargin(0.126);
    gStyle.SetPadRightMargin(0.04);
    gStyle.SetPadTopMargin(0.06);
    gStyle.SetPadBottomMargin(0.13);
    gStyle.SetOptStat(0);
    gStyle.SetMarkerStyle(1);
    
    ok_eta = TCut("TMath::Abs(eta)>1.5 && TMath::Abs(eta)<2.5")

    t = getTree("%s%s"%(filesDir, input_file))

    ## variables for the plot
    topTitle = " " * 11 + "GEM Pad matching with LCT Efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Generated muon #eta"
    yTitle = "Efficiency"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)
    toPlot = "TMath::Abs(eta)"
    h_bins = "(70,1.5,2.2)"
    nBins = int(h_bins[1:-1].split(',')[0])
    minBin = float(h_bins[1:-1].split(',')[1])
    maxBin = float(h_bins[1:-1].split(',')[2])

    c = TCanvas("c","c",700,450)
    c.Clear()
    base  = TH1F("base",title,nBins,minBin,maxBin)
    base.SetMinimum(0.0)
    base.SetMaximum(1.02)
    base.Draw("")
    base.GetXaxis().SetLabelSize(0.05)
    base.GetYaxis().SetLabelSize(0.05)
    base.GetYaxis().SetRangeUser(yrange[0],yrange[1])

     
    Cut1_den = ok_lct1 
    Cut1_num = ok_pad1_lct1 
    Cut2_den = ok_lct2 
    Cut2_num = ok_pad2_lct2
#    Cut3_den = OR(Cut1_den,Cut2_den) 
#    Cut3_num = OR(Cut1_num,Cut2_num) 
    h1 = draw_geff(t, title, h_bins, toPlot, Cut1_den, Cut1_num, "same", kRed)
    h2 = draw_geff(t, title, h_bins, toPlot, Cut2_den, Cut2_num, "same", kBlue)
#    h3 = draw_geff(t, title, h_bins, toPlot, Cut3_den, Cut3_num, "same", kGreen)

    
    leg = TLegend(0.25,0.2,.75,0.35, "require simtrack with any lct ", "brNDC");
    leg.SetBorderSize(0)
#    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.AddEntry(h1, "Pad matching for odd GEMDetId","l")
    leg.AddEntry(h2, "Pad matching for even GEMDetId","l")
#    leg.AddEntry(h3, "Pad1 or Pad2","l")
    leg.Draw("same");
   
    
    tex = TLatex(.25,.4,"PU140,1.64<|#eta|<2.12")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw("same")

    c.Print("%sgempad_matching_lcts_efficiency%s"%(plotDir,ext))

#____________________________________________________________________
def CoPadmatchingtoSimTrEfficiency(filesDir, input_file, plotDir, ext):

    gStyle.SetTitleStyle(0);
    gStyle.SetTitleAlign(13); ##coord in top left
    gStyle.SetTitleX(0.);
    gStyle.SetTitleY(1.);
    gStyle.SetTitleW(1);
    gStyle.SetTitleH(0.058);
    gStyle.SetTitleBorderSize(0);
    
    gStyle.SetPadLeftMargin(0.126);
    gStyle.SetPadRightMargin(0.04);
    gStyle.SetPadTopMargin(0.06);
    gStyle.SetPadBottomMargin(0.13);
    gStyle.SetOptStat(0);
    gStyle.SetMarkerStyle(1);
    
    ok_eta = TCut("TMath::Abs(eta)>1.5 && TMath::Abs(eta)<2.5")

    t = getTree("%s%s"%(filesDir, input_file))

    ## variables for the plot
    topTitle = " " * 11 + "GEM CoPad matching Efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Generated muon #eta"
    yTitle = "Efficiency"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)
    toPlot = "TMath::Abs(eta)"
    h_bins = "(70,1.5,2.2)"
    nBins = int(h_bins[1:-1].split(',')[0])
    minBin = float(h_bins[1:-1].split(',')[1])
    maxBin = float(h_bins[1:-1].split(',')[2])
    yrange = [0.4,1.02]
    xrange = [1.5,2.2]

    c = TCanvas("c","c",800,600)
    c.Clear()
    base  = TH1F("base",title,nBins,minBin,maxBin)
    base.SetMinimum(0.0)
    base.SetMaximum(1.02)
    base.Draw("")
    base.GetXaxis().SetLabelSize(0.05)
    base.GetYaxis().SetLabelSize(0.05)
    base.GetYaxis().SetRangeUser(yrange[0],yrange[1])
     
    Cut1_den = ok_gsh1 
    Cut1_num = ok_copad1
    Cut2_den = ok_gsh2 
    Cut2_num = ok_copad2
    Cut3_den = OR(ok_gsh1,ok_gsh2) 
    Cut3_num = OR(ok_copad1,ok_copad2) 
    h1 = draw_geff(t, title, h_bins, toPlot, Cut1_den, Cut1_num, "same", kRed, 1, 1)
    h2 = draw_geff(t, title, h_bins, toPlot, Cut2_den, Cut2_num, "same", kBlue, 1, 1)
#    h3 = draw_geff(t, title, h_bins, toPlot, Cut3_den, Cut3_num, "same", kGreen)

    
    leg = TLegend(0.15,0.75,0.75,0.90, "", "brNDC");
    leg.SetBorderSize(0)
#    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.AddEntry(h1, "CoPad matching for odd GEMDetId","l")
    leg.AddEntry(h2, "CoPad matching for even GEMDetId","l")
#    leg.AddEntry(h3, "Pad1 or Pad2","l")
    leg.Draw("same");
   
    
    tex = TLatex(.25,.7,"PU140,1.64<|#eta|<2.12")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw("same")

    c.Print("%sgem_copad_matching_efficiency%s"%(plotDir,ext))

#_______________________________________________________________________________
def makePlots(ext):
    input_dir = "files_09_10_2013"
    output_dir = "plots_09_10_2013/track_matching_eff/"

    padMatchingEffVsGenMuonPhiForPosAndNegMuons(input_dir,output_dir, 20, True, ext)    
    padMatchingEffVsHalfStripForOddEven(input_dir,output_dir, 20, True, ext)
    padMatchingEffVsLctEtaForOddEven(input_dir,output_dir, 20, True, ext)
    padMatchingEffVsSimTrackEtaForOddEven(input_dir,output_dir, 20, True, ext)


if __name__ == "__main__":
    """
    makePlots(".pdf")
    makePlots(".png")
    makePlots(".eps")
    """
    input_dir = "files/"
    output_dir = "csc_gem_matching/"

#    input_file = "gem-csc_stub_ana_pt20_PU100_moreStats.root"
    input_file = "out_GEMCSC_Ana.root"
    ext = ".png"
    
    cscMatchingEfficiencyToStripsAndWires(input_dir, input_file, output_dir, ".png")
    cscMatchingEfficiencyToStripsAndWires_2(input_dir, input_file, output_dir, ".png")
    cscMatchingEfficiencyToAlctClct(input_dir, input_file, output_dir, ".png")
    cscMatchingEfficiencyToAlctClct_2(input_dir, input_file, output_dir, ".png")
    cscMatchingEfficiencyToLct(input_dir, input_file, output_dir, ".png")
    PadmatchingtoSimTrEfficiency(input_dir, input_file, output_dir, ext)
    PadmatchingwithLCTstoSimTrEfficiency(input_dir, input_file, output_dir, ext)
    PadDigimatchingtoSimTrEfficiency(input_dir, input_file, output_dir, ext)
    
    CoPadmatchingtoSimTrEfficiency(input_dir, input_file, output_dir, ext)

    


   ### DO NOT REMOVE THE STUFF IN COMMENTS ###
   ### WE MIGHT NEED IT LATER ON ###

    """ 
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;z SimTrack |#eta|;Efficiency", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_gemsh_odd)", ok_gsh1, ok_gdg1, "P", kRed)
    h1 = draw_eff(gt, "Efficiency for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Efficiency", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_pad1 || ok_pad2, "P", kViolet)
    h2 = draw_eff(gt, "Efficiency for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Efficiency", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_2pad1 || ok_2pad2, "P same", kViolet-6)
    TLegend *leg = new TLegend(0.42,0.23,.96,0.4, NULL, "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h1, "at least one pad","l")
    leg.AddEntry(he, "two pads in two GEMs","l")
    leg.Draw()
    cEff.Print("gem_pad_eff_for_Trk_vsTrkEta_pt40%s"%(ext))
    
    draw_eff(gt, "Efficiency for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Efficiency", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_gsh1 || ok_gsh2, "P", kViolet)
    draw_eff(gt, "Efficiency for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Efficiency", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_g2sh1 || ok_g2sh2 , "P", kOrange)
    draw_eff(gt, "Efficiency for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Efficiency", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_copad1 || ok_copad2 , "P same", kRed)
    
    
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",
             "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_lct_odd)", ok_lct1, ok_pad1_overlap, kRed, 5)
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",
             "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta_lct_even)", ok_lct2, ok_pad2_overlap, kBlue, 5)
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber",130,0.5,130.5)
    h.SetTitle("Efficiency for track with LCT to have GEM pad in chamber")
    h.GetXaxis().SetTitle("LCT |#eta|")
    h.GetYaxis().SetTitle("Efficiency")
    h.SetStats(0)
    h.Draw()
    ho.Draw("same")
    he.Draw("same")
    leg = TLegend(0.50,0.23,.9,0.4,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(.04)
    leg.AddEntry(0, "p_{T} = %s GeV/c"%(pt), "")
    leg.AddEntry(0, "a pad spans 4 strips", "")
    leg.AddEntry(ho,"odd chambers","l")
    leg.AddEntry(he,"even chambers","l")
    leg.Draw()
    c.SaveAs("gem_pad_eff_for_LCT_vsLCTEta_pt40_overlap%s"%(ext))
    
    
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Efficiency", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct1, ok_pad1_overlap, "P", kRed)
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Efficiency", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct2, ok_pad2_overlap, "P same")
    leg.Draw()
    cEff.Print("gem_pad_eff_for_LCT_vsTrkEta_pt40_overlap%s"%(ext))
    
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;z SimTrack |#eta|;Efficiency", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct1 && Ep, ok_pad1_overlap, "P", kRed)
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;z SimTrack |#eta|;Efficiency", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct2 && Ep, ok_pad2_overlap, "P same")
    """

    
"""
def eff_hs_1(filesDir, plotDir, f_name, ext):
    
    t = getTree("%s%s"%(filesDir, f_name));
#    dphi = getDphi("%s"%(eff),"%s"%(pt[i]),"%s"%(oddEven))
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_gsh1_lct1_eta , ok_pad1, kRed)
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_gsh2_lct2_eta, ok_pad2)
    
    c.SaveAs("%stest%s"%(plotDir, ext))


def eff_hs_2(filesDir, plotDir, f_name, ext):

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1_eta, ok_gsh1, kRed)
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2_eta, ok_gsh2)
    c.SaveAs("%stest%s"%(plotDir, ext))
    
def eff_hs_3(filesDir, plotDir, f_name, ext):

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1_eta_Qn, ok_pad1_dphi1, kRed)
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2_eta_Qn, ok_pad2_dphi2)
    c.SaveAs("%stest%s"%(plotDir, ext))

def eff_hs_4(filesDir, plotDir, f_name, ext):

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,-0.2,0.2)", "fmod(phi+TMath::Pi()/36., TMath::Pi()/18.)", ok_eta, ok_lct1, kRed)
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(130,-0.2,0.2)", "fmod(phi+TMath::Pi()/36., TMath::Pi()/18.)",ok_eta, ok_lct2)
    c.SaveAs("%stest%s"%(plotDir, ext))


def eff_hs_5(filesDir, plotDir, f_name, ext):

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(384,0.,384.)", "strip_gemsh_odd", ok_gsh1_eta, ok_pad1, kRed);
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(384,0.,384.)", "strip_gemsh_even", ok_gsh2_eta, ok_pad2);
    c.SaveAs("%stest%s"%(plotDir, ext))


def eff_hs_6(filesDir, plotDir, f_name, ext):

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(384,0.,384.)", "strip_gemsh_odd", ok_gsh1_eta, ok_gdg1, kRed);
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(384,0.,384.)", "strip_gemsh_even", ok_gsh2_eta, ok_gdg2);
    c.SaveAs("%stest%s"%(plotDir, ext))

def eff_hs_7(filesDir, plotDir, f_name, ext):

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with GEM digi to have GEM pad in chamber;digi strip;Eff.", "h_odd", "(384,0.5,384.5)", "strip_gemdg_odd", ok_gdg1_eta, ok_pad1,kRed);
    he = draw_eff(t, "Eff. for track with GEM digi to have GEM pad in chamber;digi strip;Eff.", "h_evn", "(384,0.5,384.5)", "strip_gemdg_even", ok_gdg2_eta, ok_pad2);
    c.SaveAs("%stest%s"%(plotDir, ext))


def eff_hs_8(filesDir, plotDir, f_name, ext):

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;|#eta|;Eff.", "hname", "(90,1.5,2.2)", "TMath::Abs(eta)", ok_lct2, ok_pad2, kRed)
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;|#eta|;Eff.", "hname", "(90,1.5,2.2)", "TMath::Abs(eta)", ok_lct1, ok_pad1)
    c.SaveAs("%stest%s"%(plotDir, ext))

def eff_hs_9(filesDir, plotDir, f_name, ext):

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;|#phi|;Eff.", "hname", "(128,0,3.2)", "TMath::Abs(phi)", 
                  AND(ok_lct2,AND(ok_eta,ok_pt)), ok_pad2,kRed)
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;|#phi|;Eff.", "hname", "(128,0,3.2)", "TMath::Abs(phi)", 
                  AND(ok_lct2,AND(ok_eta,ok_pt)), ok_pad1)
    c.SaveAs("%stest%s"%(plotDir, ext))


def eff_hs_10(filesDir, plotDir, f_name, ext):

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;p_{T} [GeV/c];Eff.", "h_odd", "(50,0.,50.)", "pt", ok_lct1_eta, ok_pad1, kRed)
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;p_{T} [GeV/c];Eff.", "h_evn", "(50,0.,50.)", "pt", ok_lct2_eta, ok_pad2)
    c.SaveAs("%stest%s"%(plotDir, ext))
"""

"""
def eff_hs_11(filesDir, plotDir, f_name, ext):
    Comment to be added here
    
    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |\#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,-0.2,0.2)", "fmod(phi+TMath::Pi()/36., TMath::Pi()/18.)", ok_eta, ok_lct1 || ok_lct2, "", kRed);
    hg = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,-0.2,0.2)", "fmod(phi+TMath::Pi()/36., TMath::Pi()/18.)", ok_eta, ok_pad1 || ok_pad2, "same");
    hgp = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,-0.2,0.2)", "fmod(phi+TMath::Pi()/36., TMath::Pi()/18.)", ok_eta&&Qpos, ok_pad1 || ok_pad2, "same", kGreen);
    c.SaveAs("%stest%s"%(plotDir, ext))
"""

## gt = getTree("gem_csc_delta_pt40_pad4.root");


## draw_eff(gt, "Eff. ;p_{T}, GeV/c;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_sh1 && ok_eta , ok_digi1)
## draw_eff(gt, "Eff. ;p_{T}, GeV/c;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_digi1 && ok_eta , ok_lct1)
## draw_eff(gt, "Eff. ;p_{T}, GeV/c;Eff.", "hname", "(50,0.,50.)", "pt", ok_sh1 && ok_eta , ok_lct1)


## draw_eff(gt, "Eff. of |CLCT pattern bend| selection for matched LCTs;p_{T}, GeV/c;Eff.", "hname2", "(50,0.,50.)", "pt", (ok_lct1||ok_lct2) && ok_eta, "TMath::Abs(bend_lct_odd)<2 || TMath::Abs(bend_lct_even)<2")
## draw_eff(gt, "Eff. of |CLCT pattern bend| selection for matched LCTs;p_{T}, GeV/c;Eff.", "hname1", "(50,0.,50.)", "pt", (ok_lct1||ok_lct2) && ok_eta, "TMath::Abs(bend_lct_odd)<1 || TMath::Abs(bend_lct_even)<1","same",kBlack)
## draw_eff(gt, "Eff. of |CLCT bend|<3 selection for matched LCTs;p_{T}, GeV/c;Eff.", "hname3", "(50,0.,50.)", "pt", (ok_lct1||ok_lct2) && ok_eta, "TMath::Abs(bend_lct_odd)<3 || TMath::Abs(bend_lct_even)<3","same",kGreen+2)
## draw_eff(gt, "Eff. of |CLCT bend|<4 selection for matched LCTs;p_{T}, GeV/c;Eff.", "hname4", "(50,0.,50.)", "pt", (ok_lct1||ok_lct2) && ok_eta, "TMath::Abs(bend_lct_odd)<4 || TMath::Abs(bend_lct_even)<4","same",kRed)


## // efficiency vs half-strip  - separate odd-even
## TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<1.9"
## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1 && ok_eta , ok_pad1, "", kRed)
## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2 && ok_eta , ok_pad2, "same")

## // efficiency vs half-strip  - including overlaps in odd&even
## TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<1.9"
## TCut ok_pad1_overlap = ok_pad1 || (ok_lct2 && ok_pad2);
## TCut ok_pad2_overlap = ok_pad2 || (ok_lct1 && ok_pad1);
## TTree *t = getTree("gem_csc_delta_pt20_pad4.root");

## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1 && ok_eta , ok_pad1_overlap, "", kRed)
## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2 && ok_eta , ok_pad2_overlap, "same")


## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |\#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_lct_odd)", ok_lct1, ok_pad1, "", kRed)
## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |\#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta_lct_even)", ok_lct2, ok_pad2, "same")

## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;trk |\#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct1, ok_pad1, "", kRed)
## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;trk |\#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct2, ok_pad2, "same")


## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_env", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")

## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd", "(50,0.,50.)", "pt", ok_lct1 && ok_eta && ok_pad1, ok_dphi1)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn", "(50,0.,50.)", "pt", ok_lct2 && ok_eta && ok_pad2, ok_dphi2, "same")





## // 98% pt10
## TCut ok_dphi1 = "TMath::Abs(dphi_pad_odd) < 0.01076"
## TCut ok_dphi2 = "TMath::Abs(dphi_pad_even) < 0.004863"
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd_10", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1, "", kRed)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn_10", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")
## // 98% pt30
## TCut ok_dphi1 = "TMath::Abs(dphi_pad_odd) < 0.00571"
## TCut ok_dphi2 = "TMath::Abs(dphi_pad_even) < 0.00306"
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd_20", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1, "same", kRed)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn_20", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")
## // 98% pt30
## TCut ok_dphi1 = "TMath::Abs(dphi_pad_odd) < 0.00426"
## TCut ok_dphi2 = "TMath::Abs(dphi_pad_even) < 0.00256"
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd_30", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1, "same", kRed)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn_30", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")
## // 98% pt40
## TCut ok_dphi1 = "TMath::Abs(dphi_pad_odd) < 0.00351"
## TCut ok_dphi2 = "TMath::Abs(dphi_pad_even) < 0.00231"
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd_40", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1, "same", kRed)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn_40", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")


## |#Delta#phi_{odd}(LCT,Pad)| < 5.5 mrad
## |#Delta#phi_{even}(LCT,Pad)| < 3.1 mrad
## |#Delta\#eta(LCT,Pad)| < 0.08

## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_sh1 && ok_eta , ok_lct1 && ok_pad1 && ok_dphi1)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_sh2 && ok_eta , ok_lct2 && ok_pad2 && ok_dphi2, "same")




## draw_eff(gt, "title;pt;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1)
## draw_eff(gt, "title;pt;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")
## draw_eff(gt, "title;|\#eta|;Eff.", "hname", "(45,1.5,2.2)", "TMath::Abs(eta)", ok_sh1 , ok_lct1 && ok_pad1 )
## draw_eff(gt, "title;|#phi|;Eff.", "hname", "(128,0,3.2)", "TMath::Abs(phi)", ok_sh2 , ok_lct2 && ok_pad2 )




## gt.Draw("TMath::Abs(eta)", ok_sh1, "");
## gt.Draw("TMath::Abs(eta)", ok_sh1 && ok_lct1, "same");
## gt.Draw("TMath::Abs(eta)", ok_sh1 && ok_lct1 && ok_pad1, "same");

## gt.Draw("TMath::Abs(phi)", ok_sh1 && ok_lct1 && ok_pad1, "");
## gt.Draw("TMath::Abs(phi)", ok_sh1, "");
## gt.Draw("TMath::Abs(eta)", ok_sh1 && ok_lct1, "same");

## gt.Draw("TMath::Abs(phi)", ok_sh1, "");
## gt.Draw("TMath::Abs(phi)", ok_sh1 && ok_lct1, "same");
## gt.Draw("TMath::Abs(phi)", ok_sh1 && ok_lct1 && ok_pad1, "same");


## gt.Draw("pt", ok_sh1, "");
## gt.Draw("pt", ok_sh1 && ok_lct1, "same");
## gt.Draw("pt", ok_sh1 && ok_lct1 && ok_pad1, "same");
## gt.Draw("pt", ok_sh1, "");
## gt.Draw("pt", ok_sh1 && ok_lct1 && ok_pad1, "same");

## gt.Draw("pt>>h1", ok_sh1, "");
## dn=(TH1F*)h1.Clone("dn")
## gt.Draw("pt>>h2", ok_sh1 && ok_lct1 && ok_pad1, "same");
## nm=(TH1F*)h2.Clone("nm")

## dn.Draw()
## nm.Draw("same")

## nm.Divide(dn)
## nm.Draw()

## gt.Draw("pt>>h2", ok_sh1 && ok_lct1, "same");
## nmlct=(TH1F*)h2.Clone("nmlct")
## nmlct.Divide(dn)

## nm.Draw()
## nmlct.Draw("same")



