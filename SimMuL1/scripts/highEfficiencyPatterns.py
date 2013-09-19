##
## This script plots the high efficiency patterns for 4 different working points 
## Todo: Integrate the draf_eff() function in effFunctions.py
## Note: use TEfficiency class for these plots
## 

## custom modules
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

gStyle.SetMarkerStyle(1)

#_______________________________________________________________________________
def draw_eff(t,title, h_name, h_bins, to_draw, denom_cut, extra_num_cut, 
             color = kBlue, marker_st = 20):
    """Make an efficiency plot"""
    
    ## total numerator selection cut
    num_cut = TCut("%s && %s" %(denom_cut.GetTitle(), extra_num_cut.GetTitle()))

    t.Draw(to_draw + ">>num_" + h_name + h_bins, num_cut, "goff")
    num = TH1F(gDirectory.Get("num_" + h_name).Clone("num_" + h_name))
    t.Draw(to_draw + ">>denom_" + h_name + h_bins, denom_cut, "goff")
    den = TH1F(gDirectory.Get("denom_" + h_name).Clone("denom_" + h_name))

    useTEfficiency = True
    if useTEfficiency:
        eff = TEfficiency(num, den)
    else:
        eff = TGraphAsymmErrors(num, den)

    eff.SetTitle(title)
    eff.SetLineWidth(2)
    eff.SetLineColor(color)
    eff.SetMarkerStyle(marker_st)
    eff.SetMarkerColor(color)
    eff.SetMarkerSize(.5)
    return eff


#_______________________________________________________________________________
def getTree(fileName):
    """Get tree for given filename"""

    analyzer = "GEMCSCAnalyzer"
    trk_eff = "trk_eff"

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
def highEfficiencyPatterns(filesDir, plotDir, eff, oddEven, ext):
    """Produce plot with GEM high efficiency patterns"""
    
    pt = ["pt10","pt20","pt30","pt40"]    
    pt_labels = ["10","20","30","40"]
#    pt_labels = ["10 GeV/c","20 GeV/c","30 GeV/c","40 GeV/c"]
    dphis = [0.,0.,0.,0.]

    marker_colors = [kRed, kViolet+1, kAzure+2, kGreen-2]
    marker_styles = [20,21,23,22]

    t = getTree("%sgem_csc_eff_pt2pt50_pad4.root"%(filesDir));

    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","          GEM-CSC bending Angle                       CMS Simulation Preliminary;Generated muon p_{T} [GeV/c];",50,0.,50.)
    superscript = "p_{T}>p_{T}^{min}"
    subscript = "0"
##    h.GetYaxis().SetTitle("    |#Delta#phi_{{}^{(GEM,CSC)}}|<|#Delta#phi_{0}^{WP}| Cut Efficiency");
##    h.GetYaxis().SetTitle("    |#Delta#phi_{{}^{(GEM,CSC)}}|<|#Delta#phi_{%s}^{%s}| Cut Efficiency"%(subscript,superscript));
    h.GetYaxis().SetTitle("    |#Delta#phi_{{}^{(GEM,CSC)}}|<#Delta#phi_{%s} Cut Efficiency"%(subscript));
    h.GetYaxis().SetTitleOffset(.9)
    h.SetStats(0)

    ## label sizes more leglible
    h.GetXaxis().SetLabelSize(.05)
    h.GetXaxis().SetLabelOffset(.005)
    h.GetYaxis().SetLabelSize(.05)
    h.GetYaxis().SetLabelOffset(.005)

    h.Draw("")
    histoList = []
    for i in range(len(pt)):
        dphi = getDphi("%d"%(eff),"%s"%(pt[i]),"%s"%(oddEven))
        dphis[i] = dphi
        if oddEven=="even":
            ok_dphi = TCut("TMath::Abs(dphi_pad_even) < %f"%(dphi))
            denom_cut = ok_pad2_lct2_eta
            closeFar = "Close"
        else:
            ok_dphi = TCut("TMath::Abs(dphi_pad_odd) < %f"%(dphi))
            denom_cut = ok_pad1_lct1_eta
            closeFar = "Far"

        h2 = draw_eff(t, "", "h2", "(50,0.,50.)", "pt", 
                      denom_cut, ok_dphi, marker_colors[i], marker_styles[i])
        histoList.append(h2)
        h2.SetMarkerSize(1)
        h2.Draw("same")
    
    ## add legend
    leg = TLegend(0.52,0.17,.93,0.57, "High efficiency patterns:", "brNDC")
    for n in range(len(pt)):
        superscript = "\"%s\""%(pt_labels[n])
        superscript = "p_{T}>%s"%(pt_labels[n])
        subscript = "0"
        #leg.AddEntry(histoList[n], "#Delta#phi_{%s}^{%s} = %.1f mrad"%(subscript,superscript,dphis[n]*1000), "p")
        leg.AddEntry(histoList[n], "#Delta#phi_{%s} = %.1f mrad"%(subscript,dphis[n]*1000), "p")
        #leg.AddEntry(histoList[n], "WP = %s"%(pt_labels[n]), "p")


    leg.SetBorderSize(0)
    leg.SetMargin(0.1)
    leg.SetFillStyle(0)
    leg.SetFillStyle(1001)
    leg.SetFillColor(kWhite)
    leg.SetTextSize(0.05)
    leg.Draw("same")

    ## Adding additional information - top right
    """
    tex2 = TLatex(.75,.82,'   L1 Trigger')
    tex2.SetTextSize(0.05)
    tex2.SetNDC()
    tex2.Draw("same")
    """

    tex3 = TLatex(.735,.82,'1.64<|#eta|<2.14')
    tex3.SetTextSize(0.05)
    tex3.SetNDC()
    tex3.Draw("same")

    ## hardcore nitpicking over here!
    if closeFar == "Close":
        xpos = 0.57
    else:
        xpos = 0.611

    tex = TLatex(xpos,.75,'"%s" chamber pairs'%(closeFar))
    tex.Draw("same")
    tex.SetTextSize(0.05)
    tex.SetNDC()

    ## save the file
    c.SaveAs("%sGEM_highEffPatterns_%s_%s%s"%(plotDir, eff,oddEven,ext))

if __name__ == "__main__":
    highEfficiencyPatterns("files/", "plots/highEffPatterns/", 98, "even", ".pdf")
    highEfficiencyPatterns("files/", "plots/highEffPatterns/", 98, "odd", ".pdf")
    highEfficiencyPatterns("files/", "plots/highEffPatterns/", 98, "even", ".png")
    highEfficiencyPatterns("files/", "plots/highEffPatterns/", 98, "odd", ".png")
    highEfficiencyPatterns("files/", "plots/highEffPatterns/", 98, "even", ".eps")
    highEfficiencyPatterns("files/", "plots/highEffPatterns/", 98, "odd", ".eps")
