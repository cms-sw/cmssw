from ROOT import *


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
def draw_geff(t, title, h_bins, to_draw, den_cut, extra_num_cut, 
              opt = "", color = kBlue, marker_st = 1, marker_sz = 1.):
    """Make an efficiency plot"""
    
    ## total numerator selection cut 
    ## the extra brackets around the extra_num_cut are necessary !!
    num_cut = TCut("%s && (%s)" %(den_cut.GetTitle(), extra_num_cut.GetTitle()))
    
    ## PyROOT works a little different than ROOT when you are plotting 
    ## histograms directly from tree. Hence, this work-around
    nBins = int(h_bins[1:-1].split(',')[0])
    minBin = int(h_bins[1:-1].split(',')[1])
    maxBin = int(h_bins[1:-1].split(',')[2])
    num = TH1F("num", "", nBins, minBin, maxBin) 
    den = TH1F("den", "", nBins, minBin, maxBin)

    t.Draw(to_draw + ">>num", num_cut, "goff")
    t.Draw(to_draw + ">>den", den_cut, "goff")

    doConsistencyCheck = False
    if doConsistencyCheck:
        for i in range(0,nBins):
            print i, num.GetBinContent(i), den.GetBinContent(i)
            if num.GetBinContent(i) > den.GetBinContent(i):
                print ">>>Error: passed entries > total entries" 

    eff = TEfficiency(num, den)

    """
    eff = TEfficiency(num, den)
    eff.Draw()
    eff.Paint("")
    eff = eff.GetPaintedGraph()
    """

    if not "same" in opt:
        num.Reset()
        num.GetYaxis().SetRangeUser(0.0,1.1)
        num.SetStats(0)
        num.SetTitle(title)
        num.Draw()
        
    eff.SetLineWidth(2)
    eff.SetLineColor(color)
    eff.Draw(opt + " same")
    eff.SetMarkerStyle(marker_st)
    eff.SetMarkerColor(color)
    eff.SetMarkerSize(marker_sz)

    SetOwnership(eff, False)
    return eff
    


    
