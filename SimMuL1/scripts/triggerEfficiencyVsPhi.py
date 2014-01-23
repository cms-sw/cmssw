from ROOT import * 
from triggerPlotHelpers import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT
ROOT.gROOT.SetBatch(1)

## global variables
gMinEta = 1.45
gMaxEta = 2.5

#_______________________________________________________________________________
def setEffHisto(num_name, den_name, dir, nrebin, lcolor, lstyle, lwidth,
                htitle, xtitle, ytitle, x_range, y_range):
    """Set efficiency histogram"""

    hd = getH(dir, den_name)
    hn = getH(dir, num_name)
    
    hd.Sumw2()
    hn.Sumw2()

    myRebin(hd, nrebin)
    myRebin(hn, nrebin)
    heff = hn.Clone(num_name+"_eff")

    hd.Sumw2()
    heff.Sumw2()
    heff.Divide(heff,hd)
    heff.SetLineColor(lcolor)
    heff.SetLineStyle(lstyle)
    heff.SetLineWidth(lwidth)
    heff.SetTitle(htitle)
    heff.GetXaxis().SetTitle(xtitle)
    heff.GetYaxis().SetTitle(ytitle)
    heff.GetXaxis().SetRangeUser(x_range[0],x_range[1])
    heff.GetYaxis().SetRangeUser(y_range[0],y_range[1])
    heff.GetXaxis().SetTitleSize(0.05)
    heff.GetXaxis().SetTitleOffset(1.)
    heff.GetYaxis().SetTitleSize(0.05)
    heff.GetYaxis().SetTitleOffset(1.)
    heff.GetXaxis().SetLabelOffset(0.015)
    heff.GetYaxis().SetLabelOffset(0.015)
    heff.GetXaxis().SetLabelSize(0.05)
    heff.GetYaxis().SetLabelSize(0.05)
    return heff


#_______________________________________________________________________________
def getEffHisto(fname, hdir, num_name, den_name, nrebin, lcolor, lstyle, lwidth, 
                title, x_range, y_range):
    fh = TFile.Open(fname)
    
    hd0 = fh.Get(hdir + "/" + den_name)
    hn0 = fh.Get(hdir + "/" + num_name)
    
    hd = hd0.Clone(den_name+"_cln_"+fname)
    hn = hn0.Clone(num_name+"_cln_"+fname)
    hd.Sumw2()
    hn.Sumw2()
    
    myRebin(hd, nrebin)
    myRebin(hn, nrebin)
    
    heff = hn.Clone(num_name+"_eff_"+fname)
    
    hd.Sumw2()
    heff.Sumw2()
    
    heff.Divide(heff,hd)
    
    heff.SetLineColor(lcolor)
    heff.SetLineStyle(lstyle)
    heff.SetLineWidth(lwidth)
    
    heff.SetTitle(title)
    ##heff.GetXaxis().SetTitle(xtitle)
    ##heff.GetYaxis().SetTitle(ytitle)
    heff.GetXaxis().SetRangeUser(x_range[0],x_range[1])
    heff.GetYaxis().SetRangeUser(y_range[0],y_range[1])
    
    heff.GetXaxis().SetTitleSize(0.07)
    heff.GetXaxis().SetTitleOffset(0.7)
    heff.GetYaxis().SetLabelOffset(0.015)
    
    heff.GetXaxis().SetLabelSize(0.05)
    heff.GetYaxis().SetLabelSize(0.05)
    
    h1 = hn0
    h2 = hd0
    he = heff
    
    ##fh.Close()
    return heff



#_______________________________________________________________________________
def simTrackToAlctMatchingEfficiencyVsPhiME1():

    dir = getRootDirectory(input_dir, file_name)

    etareb = 1
    yrange = [0.6,1.2]
    xrange = [-3.14,3.14]    

    h_eff_phi_me1_after_alct = setEffHisto("h_phi_me1_after_alct","h_phi_initial",dir, etareb, kRed, 0, 2, "eff(#phi): ME1 stub studies","#phi","",xrange,yrange)
    h_eff_phi_me1_after_alct_okAlct = setEffHisto("h_phi_me1_after_alct_okAlct","h_phi_initial",dir, etareb, kRed+2, 0, 2, "eff(#phi): ME1 stub studies","#phi","",xrange,yrange)
    h_eff_phi_me1_after_clct = setEffHisto("h_phi_me1_after_clct","h_phi_initial",dir, etareb, kBlue, 0, 2, "eff(#phi): ME1 stub studies","#phi","",xrange,yrange)
    h_eff_phi_me1_after_clct_okClct = setEffHisto("h_phi_me1_after_clct_okClct","h_phi_initial",dir, etareb, kBlue+4, 0, 2, "eff(#phi): ME1 stub studies","#phi","",xrange,yrange)
    h_eff_phi_me1_after_lct = setEffHisto("h_phi_me1_after_lct","h_phi_initial",dir, etareb, kGreen+1, 0, 2, "eff(#phi): ME1 stub studies","#phi","",xrange,yrange)
    h_eff_phi_me1_after_lct_okAlctClct = setEffHisto("h_phi_me1_after_lct_okAlctClct","h_phi_initial",dir, etareb, kOrange, 0, 2, "eff(#phi): ME1 stub studies","#phi","",xrange,yrange)
#    h_eff_phi_after_mplct = setEffHisto("h_phi_me1_after_mplct","h_phi_initial",dir, etareb, kGreen+2, 2, 2, "eff(#phi): ME1 stub studies","#phi","",xrange,yrange)

    c = TCanvas("h_eff_eta_me1_after_alct","h_eff_eta_me1_after_alct",1000,600 )     
    c.cd()
    h_eff_phi_me1_after_alct.Draw("hist")
    h_eff_phi_me1_after_alct_okAlct.Draw("hist")
    h_eff_phi_me1_after_clct.Draw("same hist")
    h_eff_phi_me1_after_clct_okClct.Draw("same hist")
    h_eff_phi_me1_after_lct.Draw("same hist")
    h_eff_phi_me1_after_lct_okAlctClct.Draw("same hist")
 #    h_eff_phi_me1_after_mplct.Draw("same hist")

    leg = TLegend(0.2,0.2,0.926,0.4,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    leg.SetNColumns(3)
    leg.SetHeader("Efficiency for #mu with p_{T}>20 crossing a ME1 chamber with")

    leg.AddEntry(h_eff_phi_me1_after_alct,"any ALCT","pl")
    leg.AddEntry(h_eff_phi_me1_after_alct_okAlct,"Correct ALCT","pl")
    leg.AddEntry(h_eff_phi_me1_after_clct,"any CLCT","pl")
    leg.AddEntry(h_eff_phi_me1_after_clct_okClct,"Correct CLCT","pl")
    leg.AddEntry(h_eff_phi_me1_after_lct,"any LCT","pl")
    leg.AddEntry(h_eff_phi_me1_after_lct_okAlctClct,"LCT with correct ALCT and CLCT","pl")
    leg.Draw()

    c.Print("%ssimTrackToALctMatchingEfficiencyVsPhiME1%s"%(output_dir, ext))

    
#_______________________________________________________________________________
def simTrackToClctMatchingEfficiencyVsWGME1a():

    dir = getRootDirectory(input_dir, file_name)
    
    xTitle = "Strip"
    yTitle = "Efficiency"
    topTitle = "CLCT Construction efficiency dependence on Strip in ME1a"
    fullTitle = "%s;%s;%s"%(topTitle,xTitle,yTitle)

    etareb = 1
    yrange = [0.0,1.04]
    xrange = [-0.5,47.5]    

    h_eff_strip_me1a_after_clct = setEffHisto("h_strip_me1a_after_clct","h_strip_me1a_initial",dir, etareb, kRed, 0, 2, "","","",xrange,yrange)
    h_eff_strip_me1a_after_clct_okClct = setEffHisto("h_strip_me1a_after_clct_okClct","h_strip_me1a_initial",dir, etareb, kBlue, 0, 2, "","","",xrange,yrange)

    h_eff_strip_me1a_after_clct.SetTitle(topTitle)
    h_eff_strip_me1a_after_clct.SetXTitle(xTitle)

    c = TCanvas("c","c",1000,600)     
    c.cd()

    h_eff_strip_me1a_after_clct.Draw("hist")
    h_eff_strip_me1a_after_clct_okClct.Draw("same hist")

    leg = TLegend(0.6,0.2,0.926,0.4,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    #leg.SetNColumns(3)
#    leg.SetHeader("Figure out title")
    leg.AddEntry(h_eff_strip_me1a_after_clct,"any CLCT, pretrig3,trig4","pl")
    leg.AddEntry(h_eff_strip_me1a_after_clct_okClct,"Good CLCT, pretrig3,trig4","pl")
    leg.Draw()

    tex = TLatex(0.15,0.2," PU400, minimum number of hits for simtrack is the same as requirement for pretrigger")
    tex.SetTextSize(0.025)
    tex.SetTextFont(72)
    tex.SetNDC()
    tex.Draw("same")

    c.Print("%ssimTrackToCLctMatchingEfficiencyVsStripME1a%s"%(output_dir, ext))
	
#_______________________________________________________________________________
def simTrackToClctMatchingEfficiencyVsWGME1b():

    dir = getRootDirectory(input_dir, file_name)
    
    xTitle = "Strip"
    yTitle = "Efficiency"
    topTitle = "CLCT Construction efficiency dependence on Strip in ME1b"
    fullTitle = "%s;%s;%s"%(topTitle,xTitle,yTitle)

    etareb = 1
    yrange = [0.0,1.04]
    xrange = [-0.5,63.5]    

    h_eff_strip_me1b_after_clct = setEffHisto("h_strip_me1b_after_clct","h_strip_me1b_initial",dir, etareb, kRed, 0, 2, "","","",xrange,yrange)
    h_eff_strip_me1b_after_clct_okClct = setEffHisto("h_strip_me1b_after_clct_okClct","h_strip_me1b_initial",dir, etareb, kBlue, 0, 2, "","","",xrange,yrange)

    h_eff_strip_me1b_after_clct.SetTitle(topTitle)
    h_eff_strip_me1b_after_clct.SetXTitle(xTitle)

    c = TCanvas("c","c",1000,600)     
    c.cd()

    h_eff_strip_me1b_after_clct.Draw("hist")
    h_eff_strip_me1b_after_clct_okClct.Draw("same hist")

    leg = TLegend(0.6,0.2,0.926,0.4,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    #leg.SetNColumns(3)
#    leg.SetHeader("Figure out title")
    leg.AddEntry(h_eff_strip_me1b_after_clct,"any CLCT,pretrig3,trig4","pl")
    leg.AddEntry(h_eff_strip_me1b_after_clct_okClct,"Good CLCT,pretrig3,trig4","pl")
    leg.Draw()

    tex = TLatex(0.15,0.2," PU400, minimum number of hits for simtrack is the same as requirement for pretrigger")
    tex.SetTextSize(0.025)
    tex.SetTextFont(72)
    tex.SetNDC()
    tex.Draw("same")

    c.Print("%ssimTrackToCLctMatchingEfficiencyVsStripME1b%s"%(output_dir, ext))




#_______________________________________________________________________________
if __name__ == "__main__":

    ## some global style settings
    gStyle.SetTitleStyle(0)
    gStyle.SetTitleAlign(13) ##// coord in top left
    gStyle.SetTitleX(0.)
    gStyle.SetTitleY(1.)
    gStyle.SetTitleW(1)
    gStyle.SetTitleH(0.058)
    gStyle.SetTitleBorderSize(0)
    
    gStyle.SetPadLeftMargin(0.126)
    gStyle.SetPadRightMargin(0.04)
    gStyle.SetPadTopMargin(0.06)
    gStyle.SetPadBottomMargin(0.13)
    gStyle.SetOptStat(0)
    gStyle.SetMarkerStyle(1)

    ## global variables
    input_dir = "files/"
    file_name = "Sven_trigger_eff_PU400_pretrig3_trig4_lct2_CLCTVSStrip.root"
    output_dir = "plots_CLCTConstruction_PU400_Trig4_Strip/"
# output_dir = "plots_MinHits_Eta/"
    ext = ".png"

    simTrackToClctMatchingEfficiencyVsWGME1a()
    simTrackToClctMatchingEfficiencyVsWGME1b()


#    simTrackToAlctMatchingEfficiencyVsPhiME1()
#   simTrackToClctMatchingEfficiencyVsPhiME1()
#   simTrackToLctMatchingEfficiencyVsPhiME1()
