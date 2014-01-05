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
def cscStationOccupanciesVsEta():
    """Plot occupancy efficiencies in CSC stations 1-4"""

    dir = getRootDirectory(input_dir, file_name)
    
    etareb = 1
    yrange = [0.,1.04]
    xrange = [1.4,2.55]

    c = TCanvas("c","c",1280,720) 
    c.Divide(2,2,0.0001,0.0001)

    h_eff_eta_me1_initial = setEffHisto("h_eta_me1_initial","h_eta_initial",dir, etareb, kBlue, 1, 2, "Sim track with 2 < p_{T} < 50 to have #geq 4 sim hits in ME1","Sim track #eta","Efficiency",xrange,yrange)
    h_eff_eta_me2_initial = setEffHisto("h_eta_me2_initial","h_eta_initial",dir, etareb, kBlue, 1, 2, "Sim track with 2 < p_{T} < 50 to have #geq 4 sim hits in ME2","Sim track #eta","Efficiency",xrange,yrange)
    h_eff_eta_me3_initial = setEffHisto("h_eta_me3_initial","h_eta_initial",dir, etareb, kBlue, 1, 2, "Sim track with 2 < p_{T} < 50 to have #geq 4 sim hits in ME3","Sim track #eta","Efficiency",xrange,yrange)
    h_eff_eta_me4_initial = setEffHisto("h_eta_me4_initial","h_eta_initial",dir, etareb, kBlue, 1, 2, "Sim track with 2 < p_{T} < 50 to have #geq 4 sim hits in ME4","Sim track #eta","Efficiency",xrange,yrange)
    
    c.cd(1) ; h_eff_eta_me1_initial.Draw("hist")
    c.cd(2) ; h_eff_eta_me2_initial.Draw("hist")
    c.cd(3) ; h_eff_eta_me3_initial.Draw("hist")
    c.cd(4) ; h_eff_eta_me4_initial.Draw("hist")
    
    c.SaveAs("%sc_eff_eta_simh_by_st%s"%(output_dir,ext))

#_______________________________________________________________________________
def cscStationOccupanciesMatchedVsEta_2():
    """Plots the efficiency that a simtrack has at least 4 layers of simhits in CSC stations 1-3"""

    dir = getRootDirectory(input_dir, file_name)
    
    etareb = 1
    yrange = [0.,1.04]
    xrange = [1.4,2.55]

    c = TCanvas("c","c",1280,720) 

    h_eta_initial_1st = setEffHisto("h_eta_initial_1st","h_eta_initial",dir, etareb, kBlue, 1, 2, "Sim track with 2 < p_{T} < 50 to have #geq 4 sim hits in #geq N CSC stations","Sim track #eta","Efficiency",xrange,yrange)
    h_eta_initial_2st = setEffHisto("h_eta_initial_2st","h_eta_initial",dir, etareb, kRed+1, 1, 2, "","Sim track #eta","",xrange,yrange)
    h_eta_initial_3st = setEffHisto("h_eta_initial_3st","h_eta_initial",dir, etareb, kGreen+1, 1, 2, "","Sim track #eta","",xrange,yrange)

    h_eta_initial_1st.Draw("hist")
    h_eta_initial_2st.Draw("hist same")
    h_eta_initial_3st.Draw("hist same")

    l_eff_eta_simh = TLegend(0.2,0.2,1.0,0.6,"","brNDC")
    l_eff_eta_simh.SetBorderSize(0)
    l_eff_eta_simh.SetFillStyle(0)
    l_eff_eta_simh.SetHeader("Efficiency for sim track with 2 < p_{T} < 50 to have #geq 4 sim hits in")
    l_eff_eta_simh.AddEntry(h_eta_initial_1st,"#geq 1 CSC station","pl")
    l_eff_eta_simh.AddEntry(h_eta_initial_2st,"#geq 2 CSC stations","pl")
    l_eff_eta_simh.AddEntry(h_eta_initial_3st,"#geq 3 CSC stations","pl")
    l_eff_eta_simh.Draw()

    c.SaveAs("%sc_eff_eta_simh%s"%(output_dir,ext))

#_______________________________________________________________________________
def cscStationOccupanciesMatchedVsEta_3():
    """Plots the efficiency that a simtrack has at least 4 layers of simhits in CSC stations 1-3"""
    
    dir = getRootDirectory(input_dir, file_name)
    
    etareb = 1
    yrange = [0.,1.04]
    xrange = [1.4,2.55]

    c = TCanvas("c","c",1280,720) 

    h_eta_me1_initial_ = setEffHisto("h_eta_me1_initial","h_eta_initial0",dir, etareb, kBlue, 1, 2, "Sim track with 2 < p_{T} < 50 to have #geq 4 sim hits in ME1 plus more stations","Sim track #eta","Efficiency",xrange,yrange)
    h_eta_me1_initial_2st = setEffHisto("h_eta_me1_initial_2st","h_eta_initial0",dir, etareb, kRed+1, 1, 2, "","Sim track #eta","",xrange,yrange)
    h_eta_me1_initial_3st = setEffHisto("h_eta_me1_initial_3st","h_eta_initial0",dir, etareb, kGreen+1, 1, 2, "","Sim track #eta","",xrange,yrange)

    h_eta_me1_initial_.Draw("hist")
    h_eta_me1_initial_2st.Draw("hist same")
    h_eta_me1_initial_3st.Draw("hist same")

    l_eff_eta_simh_me1 = TLegend(0.2,0.2,1.0,0.6,"","brNDC")
    l_eff_eta_simh_me1.SetBorderSize(0)
    l_eff_eta_simh_me1.SetFillStyle(0)
    l_eff_eta_simh_me1.SetHeader("Efficiency for sim track with 2 < p_{T} < 50 to have #geq 4 sim hits in")
    l_eff_eta_simh_me1.AddEntry(h_eta_me1_initial_,"ME1","pl")
    l_eff_eta_simh_me1.AddEntry(h_eta_me1_initial_2st,"ME1 + #geq 1 CSC station","pl")
    l_eff_eta_simh_me1.AddEntry(h_eta_me1_initial_3st,"ME1 + #geq 2 CSC stations","pl")
    l_eff_eta_simh_me1.Draw()

    c.SaveAs("%sc_eff_eta_simh_me1%s"%(output_dir,ext))

#_______________________________________________________________________________
def cscStationOccupanciesMatchedMpcVsEta():
    """Plot occupancy efficiencies with MPC matches in CSC stations 1-4"""

    #### FIXME - these plots look very strange ####
    
    dir = getRootDirectory(input_dir, file_name)
    
    etareb = 1
    yrange = [0.,1.04]
    xrange = [1.4,2.55]

    c = TCanvas("c","c",1280,720) 
    c.Divide(2,2,0.0001,0.0001)
    
    h_eff_eta_me1_mpc = setEffHisto("h_eta_me1_mpc","h_eta_initial",dir, etareb, kBlue, 1, 2, "Matching sim track with 2 < p_{T} < 50 to MPC stub in ME1","Sim track #eta","Efficiency",xrange,yrange)
    h_eff_eta_me2_mpc = setEffHisto("h_eta_me2_mpc","h_eta_initial",dir, etareb, kBlue, 1, 2, "Matching sim track with 2 < p_{T} < 50 to MPC stub in ME2","Sim track #eta","Efficiency",xrange,yrange)
    h_eff_eta_me3_mpc = setEffHisto("h_eta_me3_mpc","h_eta_initial",dir, etareb, kBlue, 1, 2, "Matching sim track with 2 < p_{T} < 50 to MPC stub in ME3","Sim track #eta","Efficiency",xrange,yrange)
    h_eff_eta_me4_mpc = setEffHisto("h_eta_me4_mpc","h_eta_initial",dir, etareb, kBlue, 1, 2, "Matching sim track with 2 < p_{T} < 50 to MPC stub in ME4","Sim track #eta","Efficiency",xrange,yrange)

    c.cd(1) ; h_eff_eta_me1_mpc.Draw("hist")
    c.cd(2) ; h_eff_eta_me2_mpc.Draw("hist")
    c.cd(3) ; h_eff_eta_me3_mpc.Draw("hist")
    c.cd(4) ; h_eff_eta_me4_mpc.Draw("hist")
    
    c.SaveAs("%sc_eff_eta_mpc_by_st%s"%(output_dir,ext))


#_______________________________________________________________________________
def cscStationOccupanciesMatchedMpcVsEta_2():
    """Plot occupancy efficiencies with MPC matches in CSC stations 1-3"""

    #### FIXME - these plots look very strange ####

    dir = getRootDirectory(input_dir, file_name)
    
    etareb = 1
    yrange = [0.,1.04]
    xrange = [1.4,2.55]

    c = TCanvas("c","c",1280,720) 
    c.cd()

    h_eta_mpc_1st = setEffHisto("h_eta_mpc_1st","h_eta_initial0",dir, etareb, kBlue, 1, 2, "Matching sim track with 2 < p_{T} < 50 to MPC stub(s) in #geq N CSC stations","Sim track #eta","Efficiency",xrange,yrange)
    h_eta_mpc_2st = setEffHisto("h_eta_mpc_2st","h_eta_initial0",dir, etareb, kBlue, 9, 2, "","Sim track #eta","Efficiency",xrange,yrange)
    h_eta_mpc_3st = setEffHisto("h_eta_mpc_3st","h_eta_initial0",dir, etareb, kBlue, 2, 2, "","Sim track #eta","Efficiency",xrange,yrange)

    h_eta_mpc_1st.Draw("hist")
    h_eta_mpc_2st.Draw("hist same")
    h_eta_mpc_3st.Draw("hist same")

    l_eff_eta_mpc = TLegend(0.2,0.2,1.0,0.6,"","brNDC")
    l_eff_eta_mpc.SetBorderSize(0)
    l_eff_eta_mpc.SetFillStyle(0)
    l_eff_eta_mpc.SetHeader("Efficiency for sim track with 2 < p_{T} < 50 to match MPC stub in")
    l_eff_eta_mpc.AddEntry(h_eta_mpc_1st,"#geq 1 CSC station","pl")
    l_eff_eta_mpc.AddEntry(h_eta_mpc_2st,"#geq 2 CSC stations","pl")
    l_eff_eta_mpc.AddEntry(h_eta_mpc_3st,"#geq 3 CSC stations","pl")
    l_eff_eta_mpc.Draw()

    c.SaveAs("%sc_eff_eta_mpc%s"%(output_dir,ext))


#_______________________________________________________________________________
def cscStationOccupanciesMatchedMpcVsEta_3():
    """Plot occupancy efficiencies with MPC matches in CSC stations 1-3"""

    dir = getRootDirectory(input_dir, file_name)
    
    etareb = 1
    yrange = [0.,1.04]
    xrange = [1.4,2.55]

    c = TCanvas("c","c",1280,720) 
    c.cd()

    h_eta_mpc_1st_r = setEffHisto("h_eta_mpc_1st","h_eta_initial_1st",dir, etareb, kBlue, 1, 2, "Matching sim track with 2 < p_{T} < 50 to MPC stub if it has #geq 4 sim hits in #geq N CSC stations","Sim track #eta","Efficiency",xrange,yrange)
    h_eta_mpc_2st_r = setEffHisto("h_eta_mpc_2st","h_eta_initial_2st",dir, etareb, kRed+1, 1, 2, "","Sim track #eta","Efficiency",xrange,yrange)
    h_eta_mpc_3st_r = setEffHisto("h_eta_mpc_3st","h_eta_initial_3st",dir, etareb, kGreen+1, 1, 2, "","Sim track #eta","Efficiency",xrange,yrange)

    h_eta_mpc_1st_r.Draw("hist")
    h_eta_mpc_2st_r.Draw("hist same")
    h_eta_mpc_3st_r.Draw("hist same")

    l_eff_eta_mpc_relative = TLegend(0.2,0.2,1.0,0.6,"","brNDC")
    l_eff_eta_mpc_relative.SetBorderSize(0)
    l_eff_eta_mpc_relative.SetFillStyle(0)
    l_eff_eta_mpc_relative.SetHeader("Efficiency for sim track with 2 < p_{T} < 50 to match MPC stub if it has #geq 4 sim hits in")
    l_eff_eta_mpc_relative.AddEntry(h_eta_mpc_1st_r,"#geq 1 CSC station","pl")
    l_eff_eta_mpc_relative.AddEntry(h_eta_mpc_2st_r,"#geq 2 CSC stations","pl")
    l_eff_eta_mpc_relative.AddEntry(h_eta_mpc_3st_r,"#geq 3 CSC stations","pl")
    l_eff_eta_mpc_relative.Draw()

    c.SaveAs("%sc_eff_eta_mpc_relative%s"%(output_dir,ext))

#_______________________________________________________________________________
def tfCandidateTriggerEfficiencyVsEta():

    dir = getRootDirectory(input_dir, file_name)
    
    etareb = 1
    yrange = [0.,1.04]
    xrange = [1.4,2.5]

    c = TCanvas("c","c",1000,600 ) 

    h_eff_eta_after_tfcand_ok_plus_q1   = setEffHisto("h_eta_after_tfcand_ok_plus_q1","h_eta_after_mpc_ok_plus",dir, etareb, kBlue, 1,1, "eff(#eta): TF quality studies (denom: 2MPCs)","#eta","",xrange,yrange)
    h_eff_eta_after_tfcand_ok_plus_q2   = setEffHisto("h_eta_after_tfcand_ok_plus_q2","h_eta_after_mpc_ok_plus",dir, etareb, kCyan+2, 1,1, "","","",xrange,yrange)
    h_eff_eta_after_tfcand_ok_plus_q3   = setEffHisto("h_eta_after_tfcand_ok_plus_q3","h_eta_after_mpc_ok_plus",dir, etareb, kMagenta+1, 1,1, "","","",xrange,yrange)

    h_eff_eta_after_tfcand_ok_plus_pt10_q1   = setEffHisto("h_eta_after_tfcand_ok_plus_pt10_q1","h_eta_after_mpc_ok_plus",dir, etareb, kBlue, 2,2, "eff(#eta): TF quality studies (denom: 2MPCs)","#eta","",xrange,yrange)
    h_eff_eta_after_tfcand_ok_plus_pt10_q2   = setEffHisto("h_eta_after_tfcand_ok_plus_pt10_q2","h_eta_after_mpc_ok_plus",dir, etareb, kCyan+2, 2,2, "","","",xrange,yrange)
    h_eff_eta_after_tfcand_ok_plus_pt10_q3   = setEffHisto("h_eta_after_tfcand_ok_plus_pt10_q3","h_eta_after_mpc_ok_plus",dir, etareb, kMagenta+1, 2,2, "","","",xrange,yrange)

    h_eff_eta_after_tfcand_ok_plus_q1.Draw("hist")
    h_eff_eta_after_tfcand_ok_plus_pt10_q1.Draw("same hist")
    h_eff_eta_after_tfcand_ok_plus_q2.Draw("same hist")
    h_eff_eta_after_tfcand_ok_plus_pt10_q2.Draw("same hist")
    h_eff_eta_after_tfcand_ok_plus_q3.Draw("same hist")
    h_eff_eta_after_tfcand_ok_plus_pt10_q3.Draw("same hist")

    leg = TLegend(0.347,0.19,0.926,0.45,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("TF track with matched stubs in 2st and ")
    leg.AddEntry(h_eff_eta_after_tfcand_ok_plus_q1,"Q#geq1","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_ok_plus_pt10_q1,"Q#geq1, p_{T}^{TF}>10","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_ok_plus_q2,"Q#geq2","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_ok_plus_pt10_q2,"Q#geq2, p_{T}^{TF}>10","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_ok_plus_q3,"Q=3","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_ok_plus_pt10_q3,"Q=3, p_{T}^{TF}>10","pl")
    leg.Draw()

    c.SaveAs("%sh_eff_eta_tf_q%s"%(output_dir,ext))


#_______________________________________________________________________________
def cscStubMatchingEfficiencyVsEtaSummary():
    """Summary of the CSC stub matching efficiencies"""

    dir = getRootDirectory(input_dir, file_name)
    
    etareb = 1
    yrange = [0.,1.04]
    xrange = [1.4,2.5]

    c = TCanvas("c","c",1000,600 ) 

    h_eff_eta_me1_after_lct = setEffHisto("h_eta_me1_after_lct","h_eta_initial",dir, etareb, kRed, 2, 2, "eff(#eta): ME1 stub studies","#eta","",xrange,yrange)
    h_eff_eta_me1_after_alct = setEffHisto("h_eta_me1_after_alct","h_eta_initial",dir, etareb, kBlue+1, 2, 2, "","","",xrange,yrange)
    h_eff_eta_me1_after_clct = setEffHisto("h_eta_me1_after_clct","h_eta_initial",dir, etareb, kGreen+1, 2, 2, "","","",xrange,yrange)
    h_eff_eta_me1_after_alct_okAlct = setEffHisto("h_eta_me1_after_alct_okAlct","h_eta_initial",dir, etareb, kBlue+1, 1, 2, "","","",xrange,yrange)
    h_eff_eta_me1_after_clct_okClct = setEffHisto("h_eta_me1_after_clct_okClct","h_eta_initial",dir, etareb, kGreen+1, 1, 2, "","","",xrange,yrange)
    h_eff_eta_me1_after_alctclct = setEffHisto("h_eta_me1_after_alctclct","h_eta_initial",dir, etareb, kYellow+2, 2, 2, "","","",xrange,yrange)
    h_eff_eta_me1_after_alctclct_okAlctClct = setEffHisto("h_eta_me1_after_alctclct_okAlctClct","h_eta_initial",dir, etareb, kYellow+2, 1,2, "","","",xrange,yrange)
    h_eff_eta_me1_after_lct_okClctAlct = setEffHisto("h_eta_me1_after_lct_okAlctClct","h_eta_initial",dir, etareb, kRed, 1,2, "","","",xrange,yrange)

    h_eff_eta_me1_after_lct.Draw("hist")
    h_eff_eta_me1_after_alct.Draw("same hist")
    h_eff_eta_me1_after_alct_okAlct.Draw("same hist")
    h_eff_eta_me1_after_clct.Draw("same hist")
    h_eff_eta_me1_after_clct_okClct.Draw("same hist")
    h_eff_eta_me1_after_alctclct.Draw("same hist")
    h_eff_eta_me1_after_alctclct_okAlctClct.Draw("same hist")
    h_eff_eta_me1_after_lct.Draw("same hist")
    h_eff_eta_me1_after_lct_okClctAlct.Draw("same hist")

    """
    h_eff_eta_me1_after_lct_okClctAlct.Fit("pol0","R0","",1.63,2.38)
    eff11 = (h_eff_eta_me1_after_lct_okClctAlct.GetFunction("pol0")).GetParameter(0)
    print eff11
    h_eff_eta_me1_after_lct_okClctAlct.Fit("pol0","R0","",1.63,2.05)
    eff1b = (h_eff_eta_me1_after_lct_okClctAlct.GetFunction("pol0")).GetParameter(0)
    h_eff_eta_me1_after_lct_okClctAlct.Fit("pol0","R0","",2.05,2.38)
    eff1a = (h_eff_eta_me1_after_lct_okClctAlct.GetFunction("pol0")).GetParameter(0)
    print eff11, eff1b, eff1a
    """

    leg = TLegend(0.347,0.222,0.926,0.535,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("Efficiency for #mu with p_{T}>20 crossing a ME1 chamber with")
    leg.AddEntry(h_eff_eta_me1_after_alct,"any ALCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_alct_okAlct,"correct ALCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_clct,"any CLCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_clct_okClct,"correct CLCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_alctclct,"any ALCT and CLCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_alctclct_okAlctClct,"correct ALCT and CLCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_lct,"any LCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_lct_okClctAlct,"correct LCT","pl")
    leg.Draw()

    """
    if (interactive):
    char ghn[111]
    sprintf(ghn,"h_eff_eta_me1_after_lct_okClctAlct_%s",dname)
    gh = (TH1F*)h_eff_eta_me1_after_lct_okClctAlct.Clone(ghn)
    gh.GetYaxis().SetRangeUser(0.8,1.02)

    gh.SetTitle("LCT finding efficiency in ME1 for #mu with p_{T}>10")
    gh.GetXaxis().SetRangeUser(0.8,2.5)
    gh.GetYaxis().SetRangeUser(0.,1.05)
    gh.GetXaxis().SetTitle("#eta")
    gh.GetYaxis().SetTitle("Eff.")
    gh.GetXaxis().SetTitleSize(0.07)
    gh.GetXaxis().SetTitleOffset(0.7)
    gh.GetYaxis().SetLabelOffset(0.015)
    """

    c.SaveAs("%sh_eff_eta_me1_steps_stubs%s"%(output_dir,ext))


#_______________________________________________________________________________
def cscTFMatchingEfficiencyVsEtaME1():
    """CSC TF matching efficiencies in ME1"""

    dir = getRootDirectory(input_dir, file_name)
    
    etareb = 1
    yrange = [0.,1.04]
    xrange = [1.4,2.5]

    c = TCanvas("c","c",1000,600 ) 

    h_eff_eta_me1_after_lct_okClctAlct         = setEffHisto("h_eta_me1_after_lct_okAlctClct","h_eta_initial",dir, etareb, kRed, 1,2, "eff(#eta): ME1 TF studies", "#eta","",xrange,yrange)
    h_eff_eta_me1_after_mplct_okClctAlct_plus = setEffHisto("h_eta_me1_after_mplct_okAlctClct_plus","h_eta_initial",dir, etareb, kBlack, 1,2, "","","",xrange,yrange)
    ##h_eff_eta_after_tfcand_ok_plus      = setEffHisto("h_eta_after_tfcand_ok_plus","h_eta_initial",dir, etareb, kBlue, 1,2, "","","",xrange,yrange)
    ##h_eff_eta_after_tfcand_ok_plus_pt10 = setEffHisto("h_eta_after_tfcand_ok_plus_pt10","h_eta_initial",dir, etareb, kBlue, 2,2, "","","",xrange,yrange)
    h_eff_eta_me1_after_tf_ok_plus	  = setEffHisto("h_eta_me1_after_tf_ok_plus","h_eta_initial",dir, etareb, kBlue, 1,2, "","","",xrange,yrange)
    h_eff_eta_me1_after_tf_ok_plus_pt10 = setEffHisto("h_eta_me1_after_tf_ok_plus_pt10","h_eta_initial",dir, etareb, kBlue, 2,2, "","","",xrange,yrange)

    h_eff_eta_me1_after_lct_okClctAlct.Draw("hist")
    h_eff_eta_me1_after_mplct_okClctAlct_plus.Draw("same hist")
    h_eff_eta_me1_after_tf_ok_plus.Draw("same hist")
    h_eff_eta_me1_after_tf_ok_plus_pt10.Draw("same hist")

    leg = TLegend(0.347,0.222,0.926,0.535,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Eff. for #mu with p_{T}>20 crossing ME1+one more station with")
    leg.AddEntry(h_eff_eta_me1_after_lct_okClctAlct,"LCT matched in ME1","pl")
    leg.AddEntry(h_eff_eta_me1_after_mplct_okClctAlct_plus,"MPC matched in ME1+one","pl")
    leg.AddEntry(h_eff_eta_me1_after_tf_ok_plus,"TF track with matched stubs in ME1+one","pl")
    leg.AddEntry(h_eff_eta_me1_after_tf_ok_plus_pt10,"p_{T}^{TF}>10 TF track with matched stubs in ME1+one","pl")
    leg.Draw()

    c.SaveAs("%sh_eff_eta_me1_tf%s"%(output_dir,ext))


#_______________________________________________________________________________
def cscTFCandMatchingEfficiencyVsEta():
    """CSC TF Cand matching efficiencies"""

    dir = getRootDirectory(input_dir, file_name)
    
    etareb = 1
    yrange = [0.,1.04]
    xrange = [1.4,2.5]

    c = TCanvas("c","c",1000,600 ) 

    h_eff_eta_after_mpc_ok_plus = setEffHisto("h_eta_after_mpc_ok_plus","h_eta_initial",dir, etareb, kBlack, 1,2, "eff(#eta): TF studies","#eta","",xrange,yrange)
    h_eff_eta_after_tfcand_ok_plus      = setEffHisto("h_eta_after_tfcand_ok_plus","h_eta_initial",dir, etareb, kBlue, 1,2, "","","",xrange,yrange)
    h_eff_eta_after_tfcand_ok_plus_pt10 = setEffHisto("h_eta_after_tfcand_ok_plus_pt10","h_eta_initial",dir, etareb, kBlue, 2,2, "","","",xrange,yrange)

    h_eff_eta_after_mpc_ok_plus.Draw("hist")
    h_eff_eta_after_tfcand_ok_plus.Draw("same hist")
    h_eff_eta_after_tfcand_ok_plus_pt10.Draw("same hist")

    leg = TLegend(0.347,0.19,0.926,0.45,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Eff. for #mu with p_{T}>20 crossing #geq2 stations with")
    leg.AddEntry(h_eff_eta_after_mpc_ok_plus,"MPC matched in 2stations","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_ok_plus,"TF track with matched stubs in 2st","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_ok_plus_pt10,"p_{T}^{TF}>10 TF track with matched stubs in 2st","pl")
    leg.Draw()

    c.SaveAs("%sh_eff_eta_tf%s"%(output_dir,ext))


#_______________________________________________________________________________
def cscTFCandMatchingEfficiencyVsEta_2():
    """CSC TF Cand matching efficiencies"""

    dir = getRootDirectory(input_dir, file_name)
    
    etareb = 1
    yrange = [0.,1.04]
    xrange = [1.4,2.5]

    c = TCanvas("c","c",1000,600 ) 

    h_eff_eta_after_mpc_ok_plus = setEffHisto("h_eta_after_mpc_ok_plus","h_eta_initial",dir, etareb, kBlack, 1,2, "eff(#eta): TF studies (3TF stubs in ME1a)","#eta","",xrange,yrange)
    h_eff_eta_after_mpc_ok_plus_3st = setEffHisto("h_eta_after_mpc_ok_plus_3st","h_eta_initial",dir, etareb, kOrange+2, 1,2, "","","",xrange,yrange)
    ##h_eff_eta_after_mpc_ok_plus_3st1a = setEffHisto("h_eta_after_mpc_ok_plus_3st1a","h_eta_initial",dir, etareb, kBlack-4, 1,2, "","","",xrange,yrange)
    h_eff_eta_after_tfcand_ok_plus_3st1a      = setEffHisto("h_eta_after_tfcand_ok_plus_3st1a","h_eta_initial",dir, etareb, kBlue, 1,2, "","","",xrange,yrange)
    h_eff_eta_after_tfcand_ok_plus_pt10_3st1a = setEffHisto("h_eta_after_tfcand_ok_plus_pt10_3st1a","h_eta_initial",dir, etareb, kBlue, 2,2, "","","",xrange,yrange)

    h_eff_eta_after_mpc_ok_plus.Draw("hist")
    h_eff_eta_after_mpc_ok_plus_3st.Draw("same hist")
    ##h_eff_eta_after_mpc_ok_plus_3st1a.Draw("same hist")
    h_eff_eta_after_tfcand_ok_plus_3st1a.Draw("same hist")
    h_eff_eta_after_tfcand_ok_plus_pt10_3st1a.Draw("same hist")

    leg = TLegend(0.347,0.19,0.926,0.45,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Eff. for #mu with p_{T}>20 crossing #geq2 stations with")
    leg.AddEntry(h_eff_eta_after_mpc_ok_plus,"MPC matched in 2stations","pl")
    leg.AddEntry(h_eff_eta_after_mpc_ok_plus_3st,"MPC matched in 3stations","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_ok_plus_3st1a,"TF track with matched stubs in 2st","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_ok_plus_pt10_3st1a,"p_{T}^{TF}>10 TF track with matched stubs in 2st","pl")
    leg.Draw()

    c.SaveAs("%sh_eff_eta_tf_3st1a%s"%(output_dir,ext))

#_______________________________________________________________________________
def cscTFCandMatchingEfficiencyVsEta_3():
    """CSC TF Cand matching efficiencies"""

    dir = getRootDirectory(input_dir, file_name)
    
    etareb = 1
    yrange = [0.,1.04]
    xrange = [1.4,2.5]

    c = TCanvas("c","c",1000,600 ) 

    h_eff_eta_after_mpc = setEffHisto("h_eta_after_mpc","h_eta_initial",dir, etareb, kBlack, 1,2, "eff(#eta): TF studies (3TF stubs in ME1a)","#eta","",xrange,yrange)
    h_eff_eta_after_mpc_st1 = setEffHisto("h_eta_after_mpc_st1","h_eta_initial",dir, etareb, kOrange+2, 1,2, "","","",xrange,yrange)
    ##h_eff_eta_after_mpc_ok_plus_3st1a = setEffHisto("h_eta_after_mpc_ok_plus_3st1a","h_eta_initial",dir, etareb, kBlack-4, 1,2, "","","",xrange,yrange)
    h_eff_eta_after_tfcand_org_st1 = setEffHisto("h_eta_after_tfcand_org_st1","h_eta_initial",dir, etareb, kBlue, 1,2, "","","",xrange,yrange)
    h_eff_eta_after_tfcand_pt10 = setEffHisto("h_eta_after_tfcand_pt10","h_eta_initial",dir, etareb, kRed+1, 2,2, "","","",xrange,yrange)
    h_eff_eta_after_tfcand_my_st1 = setEffHisto("h_eta_after_tfcand_my_st1","h_eta_initial",dir, etareb, kGreen+1, 2,2, "","","",xrange,yrange)
    h_eff_eta_after_tfcand_my_st1_pt10 = setEffHisto("h_eta_after_tfcand_my_st1_pt10","h_eta_initial",dir, etareb, kPink+1, 2,2, "","","",xrange,yrange)

    h_eff_eta_after_mpc.Draw("hist")
    h_eff_eta_after_mpc_st1.Draw("same hist")
    ##h_eff_eta_after_mpc_st1_good.Draw("same hist")
    h_eff_eta_after_tfcand_org_st1.Draw("same hist")
    ##h_eff_eta_after_tfcand_comm_st1.Draw("same hist")
    h_eff_eta_after_tfcand_pt10.Draw("same hist")
    h_eff_eta_after_tfcand_my_st1.Draw("same hist")
    h_eff_eta_after_tfcand_my_st1_pt10.Draw("same hist")
    ##h_eff_eta_after_xtra_dr_pt10.Draw("same hist")

    leg = TLegend(0.2518248,0.263986,0.830292,0.5332168,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Efficiency after")
    leg.AddEntry(h_eff_eta_after_mpc,"match to MPC","pl")
    leg.AddEntry(h_eff_eta_after_mpc_st1,"match to MPC (at least one in ME1)","pl")
    ##leg.AddEntry(h_eff_eta_after_mpc_st1_good,"match to MPC (at least one good in ME1)","pl")
    leg.AddEntry(0,"match to MPC & TF track:","")
    leg.AddEntry(h_eff_eta_after_tfcand_pt10," (a) with additional TF cut p_{T}>10","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_org_st1," (b) at least 1 original TF stub in ME1","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_my_st1," (c) at least 1 matched TF stub in ME1","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_my_st1_pt10," (d) at least 1 matched TF stub in ME1 and TF p_{T}>10","pl")
    ##leg.AddEntry(h_eff_eta_after_tfcand_comm_st1," at least 1 my=original stub in St1","pl")
    ##leg.AddEntry(h_eff_eta_after_xtra_dr,"(dashed lines correspond to p_{T}>10)","")
    leg.Draw()

    c.SaveAs("%sh_eff_eta_steps_full10_tf%s"%(output_dir,ext))


#_______________________________________________________________________________
def cscMpcTFCandGmtTriggerEfficiencyVsEta():
    """cscMpcTFCandGmtTriggerEfficiency"""

    dir = getRootDirectory(input_dir, file_name)
    
    etareb = 1
    yrange = [0.,1.04]
    xrange = [1.4,2.5]

    c = TCanvas("c","c",1000,600 ) 
    
    h_eff_eta_after_mpc = setEffHisto("h_eta_after_mpc","h_eta_initial",dir, etareb, kBlack, 1,2, "eff(#eta): TF studies","#eta","",xrange,yrange)
    h_eff_eta_after_tfcand_pt10 = setEffHisto("h_eta_after_tfcand_pt10","h_eta_initial",dir, etareb, kBlue, 1,2, "","","",xrange,yrange)
    h_eff_eta_after_gmtreg_pt10 = setEffHisto("h_eta_after_gmtreg_pt10","h_eta_initial",dir, etareb, kRed+1, 2,2, "","","",xrange,yrange)
    h_eff_eta_after_gmt_pt10 = setEffHisto("h_eta_after_gmt_pt10","h_eta_initial",dir, etareb, kGreen+1, 1,2, "","","",xrange,yrange)

    h_eff_eta_after_mpc.Draw("hist")
    h_eff_eta_after_tfcand_pt10.Draw("same hist")
    h_eff_eta_after_gmtreg_pt10.Draw("same hist")
    h_eff_eta_after_gmt_pt10.Draw("same hist")
    ##h_eff_eta_after_xtra_dr_pt10.Draw("same hist")
    
    leg = TLegend(0.2518248,0.263986,0.830292,0.5332168,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    
    leg.SetHeader("With additional TF cut p_{T}>10 Efficiency after")
    leg.AddEntry(h_eff_eta_after_mpc,"match to MPC","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_pt10,"match to MPC & TF track","pl")
    leg.AddEntry(h_eff_eta_after_gmtreg_pt10,"match to MPC & TF trk & CSC GMT trk","pl")
    leg.AddEntry(h_eff_eta_after_gmt_pt10,"match to MPC & TF trk & GMT trk","pl")
    ##leg.AddEntry(h_eff_eta_after_xtra_dr,"(dashed lines correspond to p_{T}>10)","")
    leg.Draw()

    c.SaveAs("%sh_eff_eta_steps_full10%s"%(output_dir,ext))
    

#_______________________________________________________________________________
def drawplot_etastep(fname, pu, dname="tf"):
    
    ## directory for plots is made as 
    ## pdir = $PWD/pic[_{dname}]_{pu}

    ##gStyle.SetStatW(0.13)
    ##gStyle.SetStatH(0.08)
    gStyle.SetStatW(0.07)
    gStyle.SetStatH(0.06)
    gStyle.SetOptStat(0)
    gStyle.SetTitleStyle(0)


    """
    ## directory inside of root file:
    if (strcmp(pu,"StrictDirect")==0) sprintf(dir,"SimMuL1Strict")
    if (strcmp(pu,"StrictChamber")==0) sprintf(dir,"SimMuL1StrictAll")
    if (strcmp(pu,"NaturalDirect")==0) sprintf(dir,"SimMuL1")
    if (strcmp(pu,"NaturalChamber")==0) sprintf(dir,"SimMuL1All")
    if (strcmp(pu,"StrictDeltaY")==0) sprintf(dir,"SimMuL1StrictDY")
    if (strcmp(pu,"NaturalDeltaY")==0) sprintf(dir,"SimMuL1DY")
    if (strcmp(pu,"StrictChamber0")==0) sprintf(dir,"SimMuL1StrictAll0")
    """

    etareb=1
    ptreb=2


    ##eff_eta_after_mpc_ok_plus = setEffHisto("h_eta_after_mpc_ok_plus","h_eta_initial",dir, etareb, kRed, 1,2, "asdasd","xxx","yyy",xrange,yrange)
    ##eff_eta_after_mpc_ok_plus .Draw("hist")

    """
    c_eff_DR2_tf = TCanvas("c_eff_DR2_tf","c_eff_DR2_tf",1000,600 ) 

    xrangeDR[2]={0.,3.5}

    h_eff_DR_2SimTr_after_tfcand_ok_plus = setEffHisto("h_DR_2SimTr_after_tfcand_ok_plus","h_DR_2SimTr_after_mpc_ok_plus",dir, 3, kBlue, 1,2, "eff(#DeltaR(Tr1,Tr2)): for Tr1: p_{T}^{MC}>20, denom: 2 MPCs, 1.2<#eta<2.1","#DeltaR(Tr1,Tr2)","",xrangeDR,yrange)

    h_eff_DR_2SimTr_after_tfcand_ok_plus.SetLineColor(0)
    h_eff_DR_2SimTr_after_tfcand_ok_plus.Draw("hist")


    TGraphAsymmErrors *gg = TGraphAsymmErrors()
    gg.BayesDivide((const TH1*)h1,(const TH1*)h2)
    gg.Draw("p")

    Print(c_eff_DR2_tf,"h_eff_DR_2SimTr_tf" + ext)
    """


    """
    h_eta_initial = getH(dir,"h_eta_initial")
    h_eta_after_mpc = getH(dir,"h_eta_after_mpc")
    h_eta_after_mpc_st1 = getH(dir,"h_eta_after_mpc_st1")
    h_eta_after_mpc_st1_good = getH(dir,"h_eta_after_mpc_st1_good")
    h_eta_after_mpc_ok = getH(dir,"h_eta_after_mpc_ok")
    h_eta_after_mpc_ok_plus = getH(dir,"h_eta_after_mpc_ok_plus")

    h_eta_after_tftrack = getH(dir,"h_eta_after_tftrack")

    h_eta_after_tfcand = getH(dir,"h_eta_after_tfcand")
    h_eta_after_tfcand_q1 = getH(dir,"h_eta_after_tfcand_q1")
    h_eta_after_tfcand_q2 = getH(dir,"h_eta_after_tfcand_q2")
    h_eta_after_tfcand_q3 = getH(dir,"h_eta_after_tfcand_q3")
    h_eta_after_tfcand_ok = getH(dir,"h_eta_after_tfcand_ok")
    h_eta_after_tfcand_ok_plus = getH(dir,"h_eta_after_tfcand_ok_plus")
    h_eta_after_tfcand_ok_pt10 = getH(dir,"h_eta_after_tfcand_ok_pt10")
    h_eta_after_tfcand_ok_plus_pt10 = getH(dir,"h_eta_after_tfcand_ok_plus_pt10")

    h_eta_after_tfcand_all = getH(dir,"h_eta_after_tfcand_all")
    h_eta_after_tfcand_all_pt10 = getH(dir,"h_eta_after_tfcand_all_pt10")

    h_eta_after_gmtreg     = getH(dir,"h_eta_after_gmtreg")
    h_eta_after_gmtreg_all = getH(dir,"h_eta_after_gmtreg_all")
    h_eta_after_gmtreg_dr  = getH(dir,"h_eta_after_gmtreg_dr")
    h_eta_after_gmt        = getH(dir,"h_eta_after_gmt")
    h_eta_after_gmt_all    = getH(dir,"h_eta_after_gmt_all")
    h_eta_after_xtra       = getH(dir,"h_eta_after_xtra")
    h_eta_after_xtra_all   = getH(dir,"h_eta_after_xtra_all")
    h_eta_after_xtra_dr    = getH(dir,"h_eta_after_xtra_dr")

    h_eta_after_tfcand_pt10 = getH(dir,"h_eta_after_tfcand_pt10")
    h_eta_after_tfcand_my_st1 = getH(dir,"h_eta_after_tfcand_my_st1")
    h_eta_after_tfcand_org_st1 = getH(dir,"h_eta_after_tfcand_org_st1")
    h_eta_after_tfcand_comm_st1 = getH(dir,"h_eta_after_tfcand_comm_st1")
    h_eta_after_tfcand_my_st1_pt10 = getH(dir,"h_eta_after_tfcand_my_st1_pt10")
    h_eta_after_gmtreg_dr_pt10  = getH(dir,"h_eta_after_gmtreg_dr_pt10")
    h_eta_after_gmtreg_pt10= getH(dir,"h_eta_after_gmtreg_pt10")
    h_eta_after_gmt_pt10        = getH(dir,"h_eta_after_gmt_pt10")
    h_eta_after_xtra_dr_pt10    = getH(dir,"h_eta_after_xtra_dr_pt10")

    ## = getH(dir,"")

    h_eta_me1_after_alct = getH(dir,"h_eta_me1_after_alct")
    h_eta_me1_after_alct_okAlct = getH(dir,"h_eta_me1_after_alct_okAlct")
    h_eta_me1_after_clct = getH(dir,"h_eta_me1_after_clct")
    h_eta_me1_after_clct_okClct = getH(dir,"h_eta_me1_after_clct_okClct")
    h_eta_me1_after_alctclct = getH(dir,"h_eta_me1_after_alctclct")
    h_eta_me1_after_alctclct_okAlct = getH(dir,"h_eta_me1_after_alctclct_okAlct")
    h_eta_me1_after_alctclct_okClct = getH(dir,"h_eta_me1_after_alctclct_okClct")
    h_eta_me1_after_alctclct_okAlctClct = getH(dir,"h_eta_me1_after_alctclct_okAlctClct")

    h_eta_me1_after_lct = getH(dir,"h_eta_me1_after_lct")
    h_eta_me1_after_lct_okAlct = getH(dir,"h_eta_me1_after_lct_okAlct")
    h_eta_me1_after_lct_okAlctClct = getH(dir,"h_eta_me1_after_lct_okAlctClct")
    h_eta_me1_after_lct_okClct = getH(dir,"h_eta_me1_after_lct_okClct")
    h_eta_me1_after_lct_okClctAlct = getH(dir,"h_eta_me1_after_lct_okClctAlct")
    h_eta_me1_after_mplct_okAlctClct = getH(dir,"h_eta_me1_after_mplct_okAlctClct")
    h_eta_me1_after_mplct_okAlctClct_plus = getH(dir,"h_eta_me1_after_mplct_okAlctClct_plus")
    h_eta_me1_after_tf_ok = getH(dir,"h_eta_me1_after_tf_ok")
    h_eta_me1_after_tf_ok_pt10 = getH(dir,"h_eta_me1_after_tf_ok_pt10")
    h_eta_me1_after_tf_ok_plus = getH(dir,"h_eta_me1_after_tf_ok_plus")
    h_eta_me1_after_tf_ok_plus_pt10 = getH(dir,"h_eta_me1_after_tf_ok_plus_pt10")



    myRebin(h_eta_initial,etareb)
    myRebin(h_eta_after_mpc,etareb)
    myRebin(h_eta_after_mpc_st1,etareb)
    myRebin(h_eta_after_mpc_st1_good,etareb)
    myRebin(h_eta_after_mpc_ok,etareb)
    myRebin(h_eta_after_mpc_ok_plus,etareb)
    myRebin(h_eta_after_tftrack,etareb)

    myRebin(h_eta_after_tfcand,etareb)
    myRebin(h_eta_after_tfcand_q1,etareb)
    myRebin(h_eta_after_tfcand_q2,etareb)
    myRebin(h_eta_after_tfcand_q3,etareb)
    myRebin(h_eta_after_tfcand_ok,etareb)
    myRebin(h_eta_after_tfcand_ok_plus,etareb)
    myRebin(h_eta_after_tfcand_ok_pt10,etareb)
    myRebin(h_eta_after_tfcand_ok_plus_pt10,etareb)

    myRebin(h_eta_after_tfcand_all,etareb)
    myRebin(h_eta_after_tfcand_all_pt10,etareb)

    myRebin(h_eta_after_gmtreg    ,etareb)
    myRebin(h_eta_after_gmtreg_all,etareb)
    myRebin(h_eta_after_gmtreg_dr ,etareb)
    myRebin(h_eta_after_gmt       ,etareb)
    myRebin(h_eta_after_gmt_all   ,etareb)
    myRebin(h_eta_after_xtra      ,etareb)
    myRebin(h_eta_after_xtra_all  ,etareb)
    myRebin(h_eta_after_xtra_dr   ,etareb)

    myRebin(h_eta_after_tfcand_pt10,etareb)
    myRebin(h_eta_after_tfcand_my_st1,etareb)
    myRebin(h_eta_after_tfcand_org_st1,etareb)
    myRebin(h_eta_after_tfcand_comm_st1,etareb)
    myRebin(h_eta_after_tfcand_my_st1_pt10,etareb)
    myRebin(h_eta_after_gmtreg_pt10    ,etareb)
    myRebin(h_eta_after_gmtreg_dr_pt10 ,etareb)
    myRebin(h_eta_after_gmt_pt10       ,etareb)
    myRebin(h_eta_after_xtra_dr_pt10   ,etareb)

    myRebin(h_eta_me1_after_alct,etareb)
    myRebin(h_eta_me1_after_alct_okAlct,etareb)
    myRebin(h_eta_me1_after_clct,etareb)
    myRebin(h_eta_me1_after_clct_okClct,etareb)
    myRebin(h_eta_me1_after_alctclct,etareb)
    myRebin(h_eta_me1_after_alctclct_okAlct,etareb)
    myRebin(h_eta_me1_after_alctclct_okClct,etareb)
    myRebin(h_eta_me1_after_alctclct_okAlctClct,etareb)

    myRebin(h_eta_me1_after_lct ,etareb)
    myRebin(h_eta_me1_after_lct_okAlct ,etareb)
    myRebin(h_eta_me1_after_lct_okAlctClct ,etareb)
    myRebin(h_eta_me1_after_lct_okClct ,etareb)
    myRebin(h_eta_me1_after_lct_okClctAlct ,etareb)
    myRebin(h_eta_me1_after_mplct_okAlctClct ,etareb)
    myRebin(h_eta_me1_after_mplct_okAlctClct_plus ,etareb)
    myRebin(h_eta_me1_after_tf_ok ,etareb)
    myRebin(h_eta_me1_after_tf_ok_pt10 ,etareb)
    myRebin(h_eta_me1_after_tf_ok_plus ,etareb)
    myRebin(h_eta_me1_after_tf_ok_plus_pt10 ,etareb)

    h_eff_eta_after_mpc =  h_eta_after_mpc.Clone("h_eff_eta_after_mpc")
    h_eff_eta_after_mpc_st1 =  h_eta_after_mpc_st1.Clone("h_eff_eta_after_mpc_st1")
    h_eff_eta_after_mpc_st1_good =  h_eta_after_mpc_st1_good.Clone("h_eff_eta_after_mpc_st1_good")
    h_eff_eta_after_mpc_ok =  h_eta_after_mpc_ok.Clone("h_eff_eta_after_mpc_ok")
    h_eff_eta_after_mpc_ok_plus =  h_eta_after_mpc_ok_plus.Clone("h_eff_eta_after_mpc_ok_plus")
    h_eff_eta_after_tftrack =  h_eta_after_tftrack.Clone("h_eff_eta_after_tftrack")

    h_eff_eta_after_tfcand =  h_eta_after_tfcand.Clone("h_eff_eta_after_tfcand")
    h_eff_eta_after_tfcand_q1 =  h_eta_after_tfcand_q1.Clone("h_eff_eta_after_tfcand_q1")
    h_eff_eta_after_tfcand_q2 =  h_eta_after_tfcand_q2.Clone("h_eff_eta_after_tfcand_q2")
    h_eff_eta_after_tfcand_q3 =  h_eta_after_tfcand_q3.Clone("h_eff_eta_after_tfcand_q3")
    h_eff_eta_after_tfcand_ok =  h_eta_after_tfcand_ok.Clone("h_eff_eta_after_tfcand_ok")
    h_eff_eta_after_tfcand_ok_plus =  h_eta_after_tfcand_ok_plus.Clone("h_eff_eta_after_tfcand_ok_plus")
    h_eff_eta_after_tfcand_ok_pt10 =  h_eta_after_tfcand_ok_pt10.Clone("h_eff_eta_after_tfcand_ok_pt10")
    h_eff_eta_after_tfcand_ok_plus_pt10 =  h_eta_after_tfcand_ok_plus_pt10.Clone("h_eff_eta_after_tfcand_ok_plus_pt10")

    h_eff_eta_after_tfcand_all =  h_eta_after_tfcand_all.Clone("h_eff_eta_after_tfcand_all")
    h_eff_eta_after_tfcand_all_pt10 =  h_eta_after_tfcand_all_pt10.Clone("h_eff_eta_after_tfcand_all_pt10")

    h_eff_eta_after_gmtreg =  h_eta_after_gmtreg.Clone("h_eff_eta_after_gmtreg")
    h_eff_eta_after_gmtreg_all =  h_eta_after_gmtreg_all.Clone("h_eff_eta_after_gmtreg_all")
    h_eff_eta_after_gmtreg_dr =  h_eta_after_gmtreg_dr.Clone("h_eff_eta_after_gmtreg_dr")
    h_eff_eta_after_gmt =  h_eta_after_gmt.Clone("h_eff_eta_after_gmt")
    h_eff_eta_after_gmt_all =  h_eta_after_gmt_all.Clone("h_eff_eta_after_gmt_all")
    h_eff_eta_after_xtra =  h_eta_after_xtra.Clone("h_eff_eta_after_xtra")
    h_eff_eta_after_xtra_all =  h_eta_after_xtra_all.Clone("h_eff_eta_after_xtra_all")
    h_eff_eta_after_xtra_dr =  h_eta_after_xtra_dr.Clone("h_eff_eta_after_xtra_dr")

    h_eff_eta_after_tfcand_pt10 =  h_eta_after_tfcand_pt10.Clone("h_eff_eta_after_tfcand_pt10")
    h_eff_eta_after_tfcand_my_st1 =  h_eta_after_tfcand_my_st1.Clone("h_eff_eta_after_tfcand_my_st1")
    h_eff_eta_after_tfcand_org_st1 =  h_eta_after_tfcand_org_st1.Clone("h_eff_eta_after_tfcand_org_st1")
    h_eff_eta_after_tfcand_comm_st1 =  h_eta_after_tfcand_comm_st1.Clone("h_eff_eta_after_tfcand_comm_st1")
    h_eff_eta_after_tfcand_my_st1_pt10 =  h_eta_after_tfcand_my_st1_pt10.Clone("h_eff_eta_after_tfcand_my_st1_pt10")
    h_eff_eta_after_gmtreg_pt10 =  h_eta_after_gmtreg_pt10.Clone("h_eff_eta_after_gmtreg_pt10")
    h_eff_eta_after_gmtreg_dr_pt10 =  h_eta_after_gmtreg_dr_pt10.Clone("h_eff_eta_after_gmtreg_dr_pt10")
    h_eff_eta_after_gmt_pt10 =  h_eta_after_gmt_pt10.Clone("h_eff_eta_after_gmt_pt10")
    h_eff_eta_after_xtra_dr_pt10 =  h_eta_after_xtra_dr_pt10.Clone("h_eff_eta_after_xtra_dr_pt10")

    h_eff_eta_me1_after_alct =  h_eta_me1_after_alct.Clone("h_eff_eta_me1_after_alct")
    h_eff_eta_me1_after_alct_okAlct =  h_eta_me1_after_alct_okAlct.Clone("h_eff_eta_me1_after_alct_okAlct")
    h_eff_eta_me1_after_clct =  h_eta_me1_after_clct.Clone("h_eff_eta_me1_after_clct")
    h_eff_eta_me1_after_clct_okClct =  h_eta_me1_after_clct_okClct.Clone("h_eff_eta_me1_after_clct_okClct")
    h_eff_eta_me1_after_alctclct =  h_eta_me1_after_alctclct.Clone("h_eff_eta_me1_after_alctclct")
    h_eff_eta_me1_after_alctclct_okAlct =  h_eta_me1_after_alctclct_okAlct.Clone("h_eff_eta_me1_after_alctclct_okAlct")
    h_eff_eta_me1_after_alctclct_okClct =  h_eta_me1_after_alctclct_okClct.Clone("h_eff_eta_me1_after_alctclct_okClct")
    h_eff_eta_me1_after_alctclct_okAlctClct =  h_eta_me1_after_alctclct_okAlctClct.Clone("h_eff_eta_me1_after_alctclct_okAlctClct")

    h_eff_eta_me1_after_lct =  h_eta_me1_after_lct.Clone("h_eff_eta_me1_after_lct")
    h_eff_eta_me1_after_lct_okAlct =  h_eta_me1_after_lct_okAlct.Clone("h_eff_eta_me1_after_lct_okAlct")
    h_eff_eta_me1_after_lct_okAlctClct =  h_eta_me1_after_lct_okAlctClct.Clone("h_eff_eta_me1_after_lct_okAlctClct")
    h_eff_eta_me1_after_lct_okClct =  h_eta_me1_after_lct_okClct.Clone("h_eff_eta_me1_after_lct_okClct")
    h_eff_eta_me1_after_lct_okClctAlct =  h_eta_me1_after_lct_okClctAlct.Clone("h_eff_eta_me1_after_lct_okClctAlct")
    h_eff_eta_me1_after_mplct_okAlctClct =  h_eta_me1_after_mplct_okAlctClct.Clone("h_eff_eta_me1_after_mplct_okAlctClct")
    h_eff_eta_me1_after_mplct_okAlctClct_plus =  h_eta_me1_after_mplct_okAlctClct_plus.Clone("h_eff_eta_me1_after_mplct_okAlctClct_plus")
    h_eff_eta_me1_after_tf_ok =  h_eta_me1_after_tf_ok.Clone("h_eff_eta_me1_after_tf_ok")
    h_eff_eta_me1_after_tf_ok_pt10 =  h_eta_me1_after_tf_ok_pt10.Clone("h_eff_eta_me1_after_tf_ok_pt10")
    h_eff_eta_me1_after_tf_ok_plus =  h_eta_me1_after_tf_ok_plus.Clone("h_eff_eta_me1_after_tf_ok_plus")
    h_eff_eta_me1_after_tf_ok_plus_pt10 =  h_eta_me1_after_tf_ok_plus_pt10.Clone("h_eff_eta_me1_after_tf_ok_plus_pt10")


    h_eta_initial.Sumw2()
    h_eff_eta_after_mpc_st1.Sumw2()
    h_eff_eta_after_tfcand_my_st1.Sumw2()
    h_eff_eta_me1_after_alctclct_okAlctClct.Sumw2()
    h_eff_eta_me1_after_lct_okClctAlct.Sumw2()

    h_eff_eta_after_mpc.Divide(h_eff_eta_after_mpc,h_eta_initial)
    h_eff_eta_after_mpc_st1.Divide(h_eff_eta_after_mpc_st1,h_eta_initial,1,1,"B")
    h_eff_eta_after_mpc_st1_good.Divide(h_eff_eta_after_mpc_st1_good,h_eta_initial,1,1,"B")
    h_eff_eta_after_mpc_ok.Divide(h_eff_eta_after_mpc_ok,h_eta_initial)
    h_eff_eta_after_mpc_ok_plus.Divide(h_eff_eta_after_mpc_ok_plus,h_eta_initial)

    h_eff_eta_after_tftrack.Divide(h_eff_eta_after_tftrack,h_eta_initial)
    h_eff_eta_after_tfcand.Divide(h_eff_eta_after_tfcand,h_eta_initial)
    h_eff_eta_after_tfcand_q1.Divide(h_eff_eta_after_tfcand_q1,h_eta_initial)
    h_eff_eta_after_tfcand_q2.Divide(h_eff_eta_after_tfcand_q2,h_eta_initial)
    h_eff_eta_after_tfcand_q3.Divide(h_eff_eta_after_tfcand_q3,h_eta_initial)
    h_eff_eta_after_tfcand_ok.Divide(h_eff_eta_after_tfcand_ok,h_eta_initial)
    h_eff_eta_after_tfcand_ok_plus.Divide(h_eff_eta_after_tfcand_ok_plus,h_eta_initial)
    h_eff_eta_after_tfcand_ok_pt10.Divide(h_eff_eta_after_tfcand_ok_pt10,h_eta_initial)
    h_eff_eta_after_tfcand_ok_plus_pt10.Divide(h_eff_eta_after_tfcand_ok_plus_pt10,h_eta_initial)
    h_eff_eta_after_tfcand_all     .Divide(h_eff_eta_after_tfcand_all,h_eta_initial)
    h_eff_eta_after_tfcand_all_pt10.Divide(h_eff_eta_after_tfcand_all_pt10,h_eta_initial)

    h_eff_eta_after_gmtreg    .Divide(h_eff_eta_after_gmtreg,h_eta_initial)
    h_eff_eta_after_gmtreg_all.Divide(h_eff_eta_after_gmtreg_all,h_eta_initial)
    h_eff_eta_after_gmtreg_dr .Divide(h_eff_eta_after_gmtreg_dr,h_eta_initial)
    h_eff_eta_after_gmt       .Divide(h_eff_eta_after_gmt,h_eta_initial)
    h_eff_eta_after_gmt_all   .Divide(h_eff_eta_after_gmt_all,h_eta_initial)
    h_eff_eta_after_xtra      .Divide(h_eff_eta_after_xtra,h_eta_initial)
    h_eff_eta_after_xtra_all  .Divide(h_eff_eta_after_xtra_all,h_eta_initial)
    h_eff_eta_after_xtra_dr   .Divide(h_eff_eta_after_xtra_dr,h_eta_initial)

    h_eff_eta_after_tfcand_pt10    .Divide(h_eff_eta_after_tfcand_pt10,h_eta_initial,1,1,"B")
    h_eff_eta_after_tfcand_my_st1 .Divide(h_eff_eta_after_tfcand_my_st1,h_eta_initial,1,1,"B")
    h_eff_eta_after_tfcand_org_st1 .Divide(h_eff_eta_after_tfcand_org_st1,h_eta_initial,1,1,"B")
    h_eff_eta_after_tfcand_comm_st1 .Divide(h_eff_eta_after_tfcand_comm_st1,h_eta_initial,1,1,"B")
    h_eff_eta_after_tfcand_my_st1_pt10 .Divide(h_eff_eta_after_tfcand_my_st1_pt10,h_eta_initial,1,1,"B")
    h_eff_eta_after_gmtreg_pt10    .Divide(h_eff_eta_after_gmtreg_pt10,h_eta_initial)
    h_eff_eta_after_gmtreg_dr_pt10 .Divide(h_eff_eta_after_gmtreg_dr_pt10,h_eta_initial)
    h_eff_eta_after_gmt_pt10       .Divide(h_eff_eta_after_gmt_pt10,h_eta_initial)
    h_eff_eta_after_xtra_dr_pt10   .Divide(h_eff_eta_after_xtra_dr_pt10,h_eta_initial)


    h_eff_eta_me1_after_alct.Divide(h_eff_eta_me1_after_alct,h_eta_initial)
    h_eff_eta_me1_after_alct_okAlct.Divide(h_eff_eta_me1_after_alct_okAlct,h_eta_initial)
    h_eff_eta_me1_after_clct.Divide(h_eff_eta_me1_after_clct,h_eta_initial)
    h_eff_eta_me1_after_clct_okClct.Divide(h_eff_eta_me1_after_clct_okClct,h_eta_initial)
    h_eff_eta_me1_after_alctclct.Divide(h_eff_eta_me1_after_alctclct,h_eta_initial)
    h_eff_eta_me1_after_alctclct_okAlct.Divide(h_eff_eta_me1_after_alctclct_okAlct,h_eta_initial)
    h_eff_eta_me1_after_alctclct_okClct.Divide(h_eff_eta_me1_after_alctclct_okClct,h_eta_initial)
    h_eff_eta_me1_after_alctclct_okAlctClct.Divide(h_eff_eta_me1_after_alctclct_okAlctClct,h_eta_initial)

    h_eff_eta_me1_after_lct .Divide(h_eff_eta_me1_after_lct,h_eta_initial)
    h_eff_eta_me1_after_lct_okAlct .Divide(h_eff_eta_me1_after_lct_okAlct,h_eta_initial)
    h_eff_eta_me1_after_lct_okAlctClct.Divide(h_eff_eta_me1_after_lct_okAlctClct,h_eta_initial)
    h_eff_eta_me1_after_lct_okClct .Divide(h_eff_eta_me1_after_lct_okClct,h_eta_initial)
    h_eff_eta_me1_after_lct_okClctAlct.Divide(h_eff_eta_me1_after_lct_okClctAlct,h_eta_initial)
    h_eff_eta_me1_after_mplct_okAlctClct.Divide(h_eff_eta_me1_after_mplct_okAlctClct,h_eta_initial)
    h_eff_eta_me1_after_mplct_okAlctClct_plus.Divide(h_eff_eta_me1_after_mplct_okAlctClct_plus,h_eta_initial)
    h_eff_eta_me1_after_tf_ok      .Divide(h_eff_eta_me1_after_tf_ok,h_eta_initial)
    h_eff_eta_me1_after_tf_ok_pt10 .Divide(h_eff_eta_me1_after_tf_ok_pt10,h_eta_initial)
    h_eff_eta_me1_after_tf_ok_plus      .Divide(h_eff_eta_me1_after_tf_ok_plus,h_eta_initial)
    h_eff_eta_me1_after_tf_ok_plus_pt10 .Divide(h_eff_eta_me1_after_tf_ok_plus_pt10,h_eta_initial)



    ##h_eff_eta_after_mpc.SetFillColor(7)
    ##h_eff_eta_after_tftrack.SetFillColor(8)
    ##
    ##h_eff_eta_after_tfcand     .SetFillColor(1)
    ##h_eff_eta_after_tfcand_q1.SetFillColor(2)
    ##h_eff_eta_after_tfcand_q2.SetFillColor(3)
    ##h_eff_eta_after_tfcand_q3.SetFillColor(4)
    ##
    ##h_eff_eta_after_tfcand_all     .SetFillColor(5)


    c2 = TCanvas("h_eff_eta","h_eff_eta",900,900 ) 
    c2.Divide(2,2)
    c2.cd(1)
    h_eff_eta_after_mpc.GetXaxis().SetRangeUser(0.9,2.5)
    h_eff_eta_after_mpc.Draw("hist")
    ##h_eff_eta_after_mpc_ok.Draw("same hist")
    h_eff_eta_after_mpc_ok_plus.Draw("same hist")
    ##h_eff_eta_after_tftrack.Draw("same hist")
    h_eff_eta_after_tfcand.Draw("same hist")
    ##h_eff_eta_after_tfcand_ok.Draw("same hist")
    h_eff_eta_after_tfcand_ok_plus.Draw("same hist")
    h_eff_eta_after_tfcand_ok_plus_pt10.Draw("same hist")

    c2.cd(2)
    h_eff_eta_after_tfcand.GetXaxis().SetRangeUser(0.9,2.5)
    h_eff_eta_after_tfcand     .Draw("hist")
    h_eff_eta_after_tfcand_q1.Draw("same hist")
    h_eff_eta_after_tfcand_q2.Draw("same hist")
    h_eff_eta_after_tfcand_q3.Draw("same hist")
    c2.cd(3)
    h_eff_eta_after_tfcand_all     .GetXaxis().SetRangeUser(0.9,2.5)
    h_eff_eta_after_tfcand_all     .Draw("hist")
    h_eff_eta_after_tfcand.Draw("same hist")
    c2.cd(2)
    Print(c2,"h_eff_eta" + ext)


    h_eff_eta_after_mpc.SetLineColor(kBlack)
    h_eff_eta_after_mpc_st1.SetLineColor(kBlack+2)
    h_eff_eta_after_mpc_st1_good.SetLineColor(kBlack+4)
    h_eff_eta_after_tftrack.SetLineColor(kViolet-2)
    h_eff_eta_after_tfcand.SetLineColor(kBlue)
    h_eff_eta_after_gmtreg.SetLineColor(kMagenta-2)
    h_eff_eta_after_gmtreg_dr.SetLineColor(kOrange-3)
    h_eff_eta_after_gmt.SetLineColor(kGreen+1)
    h_eff_eta_after_xtra.SetLineColor(kPink-4)
    h_eff_eta_after_xtra_dr.SetLineColor(kRed)

    h_eff_eta_after_tfcand_all     .SetLineColor(kBlue)
    h_eff_eta_after_gmtreg_all.SetLineColor(kMagenta-2)
    h_eff_eta_after_gmt_all.SetLineColor(kGreen+1)
    h_eff_eta_after_xtra_all.SetLineColor(kPink-4)


    h_eff_eta_after_tfcand_pt10    .SetLineColor(kBlue)
    h_eff_eta_after_tfcand_my_st1 .SetLineColor(kBlue)
    h_eff_eta_after_tfcand_org_st1 .SetLineColor(kBlue)
    h_eff_eta_after_tfcand_comm_st1 .SetLineColor(kBlue)
    h_eff_eta_after_tfcand_my_st1_pt10 .SetLineColor(30)
    h_eff_eta_after_gmtreg_pt10    .SetLineColor(kMagenta-2)
    h_eff_eta_after_gmtreg_dr_pt10 .SetLineColor(kOrange-3)
    h_eff_eta_after_gmt_pt10       .SetLineColor(kGreen+1)
    h_eff_eta_after_xtra_dr_pt10   .SetLineColor(kRed)

    h_eff_eta_after_mpc_st1_good.SetLineStyle(7)

    h_eff_eta_after_tfcand_pt10    .SetLineStyle(7)
    h_eff_eta_after_tfcand_my_st1  .SetLineStyle(7)
    h_eff_eta_after_tfcand_org_st1 .SetLineStyle(3)
    h_eff_eta_after_tfcand_comm_st1.SetLineStyle(2)
    h_eff_eta_after_tfcand_my_st1_pt10.SetLineStyle(7)
    h_eff_eta_after_gmtreg_pt10    .SetLineStyle(7)
    h_eff_eta_after_gmtreg_dr_pt10 .SetLineStyle(7)
    h_eff_eta_after_gmt_pt10       .SetLineStyle(7)
    h_eff_eta_after_xtra_dr_pt10   .SetLineStyle(7)

    h_eff_eta_after_mpc.SetTitle("L1 CSC trigger efficiency dependence on #eta")
    h_eff_eta_after_mpc.GetXaxis().SetRangeUser(0.85,2.5)
    h_eff_eta_after_mpc.GetXaxis().SetTitle("#eta")
    h_eff_eta_after_mpc.SetMinimum(0)
    """


    """
    c22 = TCanvas("h_eff_eta_steps","h_eff_eta_steps",1200,900 ) 
    c22.Divide(1,2)
    c22.cd(1)
    h_eff_eta_after_mpc.Draw("hist")
    ##h_eff_eta_after_tftrack.Draw("same")
    h_eff_eta_after_tfcand.Draw("same")
    ##h_eff_eta_after_gmtreg.Draw("same")
    h_eff_eta_after_gmt.Draw("same")
    ##h_eff_eta_after_xtra.Draw("same")
    h_eff_eta_after_xtra_dr.Draw("same")
    leg = TLegend(0.7815507,0.1702982,0.9846086,0.5740635,"","brNDC")
    ##leg.SetTextFont(12)
    leg.SetBorderSize(0)
    leg.SetTextFont(12)
    leg.SetLineColor(1)
    leg.SetLineStyle(1)
    leg.SetLineWidth(1)
    leg.SetFillColor(19)
    leg.SetFillStyle(0)
    leg.AddEntry(h_eff_eta_after_mpc,"h_eff_eta_after_mpc","pl")
    ##leg.AddEntry(h_eff_eta_after_tftrack,"h_eff_eta_after_tftrack","pl")
    leg.AddEntry(h_eff_eta_after_tfcand,"h_eff_eta_after_tfcand","pl")
    ##leg.AddEntry(h_eff_eta_after_gmtreg,"h_eff_eta_after_gmtreg","pl")
    leg.AddEntry(h_eff_eta_after_gmt,"h_eff_eta_after_gmt","pl")
    ##leg.AddEntry(h_eff_eta_after_xtra,"h_eff_eta_after_xtra","pl")
    leg.AddEntry(h_eff_eta_after_xtra_dr,"h_eff_eta_after_xtra_dr","pl")
    leg.Draw()
    c22.cd(2)
    h_eff_eta_after_mpc.GetXaxis().SetRangeUser(0.9,2.5)
    h_eff_eta_after_mpc.Draw("hist")
    ##h_eff_eta_after_tftrack.Draw("same")
    h_eff_eta_after_tfcand_all.Draw("same")
    ##h_eff_eta_after_gmtreg_all.Draw("same")
    h_eff_eta_after_gmt_all.Draw("same")
    ##h_eff_eta_after_xtra_all.Draw("same")
    h_eff_eta_after_xtra_dr.Draw("same")
    leg = TLegend(0.7815507,0.1702982,0.9846086,0.5740635,"","brNDC")
    ##leg.SetTextFont(12)
    leg.SetBorderSize(0)
    leg.SetTextFont(12)
    leg.SetLineColor(1)
    leg.SetLineStyle(1)
    leg.SetLineWidth(1)
    leg.SetFillColor(19)
    leg.SetFillStyle(0)
    leg.AddEntry(h_eff_eta_after_mpc,"h_eff_eta_after_mpc","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_all,"h_eff_eta_after_tfcand_all","pl")
    ##leg.AddEntry(h_eff_eta_after_gmtreg_all,"h_eff_eta_after_gmtreg_all","pl")
    leg.AddEntry(h_eff_eta_after_gmt_all,"h_eff_eta_after_gmt_all","pl")
    ##leg.AddEntry(h_eff_eta_after_xtra_all,"h_eff_eta_after_xtra_all","pl")
    leg.AddEntry(h_eff_eta_after_xtra_dr,"h_eff_eta_after_xtra_dr","pl")
    leg.Draw("hist")
    Print(c22,"h_eff_eta_step.eps")
    """

    """  ## commented on 10/21/09
    c22x = TCanvas("h_eff_eta_steps_full","h_eff_eta_steps_full",1000,600 ) 

    h_eff_eta_after_mpc.Draw("hist")
    ####h_eff_eta_after_tftrack.Draw("same")
    h_eff_eta_after_tfcand.Draw("same")
    h_eff_eta_after_gmtreg.Draw("same")
    ##h_eff_eta_after_gmtreg_dr.Draw("same")
    h_eff_eta_after_gmt.Draw("same")
    ####h_eff_eta_after_xtra.Draw("same")
    ##h_eff_eta_after_xtra_dr.Draw("same")

    ##h_eff_eta_after_tfcand_pt10    .Draw("same")
    ##h_eff_eta_after_gmtreg_pt10    .Draw("same")
    ####h_eff_eta_after_gmtreg_dr_pt10 .Draw("same")
    ##h_eff_eta_after_gmt_pt10       .Draw("same")
    ##h_eff_eta_after_xtra_dr_pt10   .Draw("same")

    leg = TLegend(0.2518248,0.263986,0.830292,0.5332168,"","brNDC")
    ##leg.SetTextFont(12)
    leg.SetBorderSize(0)
    ##leg.SetTextFont(12)
    leg.SetFillStyle(0)

    leg.SetHeader("Efficiency after")
    leg.AddEntry(h_eff_eta_after_mpc,"match to MPC","pl")
    ##leg.AddEntry(h_eff_eta_after_tftrack,"h_eff_eta_after_tftrack","pl")
    leg.AddEntry(h_eff_eta_after_tfcand,"match to MPC & TF track","pl")
    leg.AddEntry(h_eff_eta_after_gmtreg,"match to MPC & TF trk & CSC GMT trk","pl")
    ##leg.AddEntry(h_eff_eta_after_gmtreg_dr,"#Delta R match to CSC GMT trk","pl")
    leg.AddEntry(h_eff_eta_after_gmt,"match to MPC & TF trk & GMT trk","pl")
    ##leg.AddEntry(h_eff_eta_after_xtra,"h_eff_eta_after_xtra","pl")
    ##leg.AddEntry(h_eff_eta_after_xtra_dr,"#Delta R match to GMT","pl")
    ##leg.AddEntry(h_eff_eta_after_xtra_dr,"(dashed lines correspond to p_{T}>10)","")

    leg.Draw()
    Print(c22x,"h_eff_eta_steps_full.eps")
    Print(c22x,"h_eff_eta_steps_full" + ext)
    """
        


    """


    """

    """  ## commented on 10/21/09
    c22x10tfs = TCanvas("h_eff_eta_steps_full10tfs","h_eff_eta_steps_full10tfs",1000,600 ) 

    h_eff_eta_after_tfcand_pt10    .SetLineStyle(1)

    h_eff_eta_after_mpc.Draw("hist")
    h_eff_eta_after_mpc_st1.Draw("same hist")
    ##h_eff_eta_after_mpc_st1_good.Draw("same hist")
    ##h_eff_eta_after_tfcand_org_st1 .Draw("same hist")
    ##h_eff_eta_after_tfcand_comm_st1 .Draw("same hist")
    h_eff_eta_after_tfcand_pt10    .Draw("same hist")
    h_eff_eta_after_tfcand_my_st1 .Draw("same hist")
    ##h_eff_eta_after_tfcand_my_st1_pt10 .Draw("same hist")
    ##h_eff_eta_after_xtra_dr_pt10   .Draw("same hist")

    leg = TLegend(0.2518248,0.263986,0.830292,0.5332168,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    leg.SetHeader("Efficiency after")
    leg.AddEntry(NUL,"match to MPC:","")
    leg.AddEntry(h_eff_eta_after_mpc," at least one","pl")
    leg.AddEntry(h_eff_eta_after_mpc_st1," at least one in ME1","pl")
    ##leg.AddEntry(h_eff_eta_after_mpc_st1_good,"match to MPC (at least one good in ME1)","pl")
    leg.AddEntry(NUL,"match to MPC & TF track:","")
    leg.AddEntry(h_eff_eta_after_tfcand_pt10," with additional TF cut p_{T}>10","pl")
    ##leg.AddEntry(h_eff_eta_after_tfcand_org_st1," (b) at least 1 original TF stub in ME1","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_my_st1," at least 1 matched TF stub in ME1","pl")
    ##leg.AddEntry(h_eff_eta_after_tfcand_my_st1_pt10," (d) at least 1 matched TF stub in ME1 and TF p_{T}>10","pl")
    ##leg.AddEntry(h_eff_eta_after_tfcand_comm_st1," at least 1 my=original stub in St1","pl")
    ##leg.AddEntry(h_eff_eta_after_xtra_dr,"(dashed lines correspond to p_{T}>10)","")

    leg.Draw()
    Print(c22x10tfs,"h_eff_eta_steps_full10_tfs.eps")
    Print(c22x10tfs,"h_eff_eta_steps_full10_tfs" + ext)
    Print(c22x10tfs,"h_eff_eta_steps_full10_tfs.pdf")
    """


    do_h_eff_eta_steps_xchk1 = 0
    if (do_h_eff_eta_steps_xchk1):
        c222xx1 = TCanvas("h_eff_eta_steps_xchk1","h_eff_eta_steps_xchk1",1000,600 ) 

        h_eff_eta_after_gmtreg.SetTitle("CSC GMT efficiency dependence on #eta (cross-check)")
        h_eff_eta_after_gmtreg.GetXaxis().SetRangeUser(0.9,2.5)
        h_eff_eta_after_gmtreg.GetXaxis().SetTitle("#eta")
        h_eff_eta_after_gmtreg.SetMinimum(0)
        h_eff_eta_after_gmtreg.Draw("hist")
        h_eff_eta_after_gmtreg_dr.Draw("same  hist")

        ##h_eff_eta_after_gmtreg_pt10    .Draw("same hist")
        ##h_eff_eta_after_gmtreg_dr_pt10 .Draw("same hist")

        leg = TLegend(0.2518248,0.263986,0.830292,0.5332168,"","brNDC")
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)

        leg.SetHeader("Efficiency after")
        leg.AddEntry(h_eff_eta_after_gmtreg,"match to MPC & TF trk & CSC GMT trk","pl")
        leg.AddEntry(h_eff_eta_after_gmtreg_dr,"#Delta R match to CSC GMT trk","pl")
        ##leg.AddEntry(h_eff_eta_after_xtra_dr,"(dashed lines correspond to p_{T}>10)","")

        leg.Draw()
        Print(c222xx1,"h_eff_eta_steps_xchk1.eps")
        Print(c222xx1,"h_eff_eta_steps_xchk1" + ext)


    """
    c222xx2 = TCanvas("h_eff_eta_steps_xchk2","h_eff_eta_steps_xchk2",1000,600 ) 

    h_eff_eta_after_gmt.SetTitle("GMT efficiency dependence on #eta (cross-check)")
    h_eff_eta_after_gmt.GetXaxis().SetRangeUser(0.9,2.5)
    h_eff_eta_after_gmt.GetXaxis().SetTitle("#eta")
    h_eff_eta_after_gmt.SetMinimum(0)
    h_eff_eta_after_gmt.SetMaximum(1.05)

    h_eff_eta_after_gmt.Draw()
    h_eff_eta_after_xtra_dr.Draw("same hist")

    ##h_eff_eta_after_gmt_pt10       .Draw("same hist")
    ##h_eff_eta_after_xtra_dr_pt10   .Draw("same hist")

    leg = TLegend(0.2518248,0.263986,0.830292,0.5332168,"","brNDC")
    ##leg.SetTextFont(12)
    leg.SetBorderSize(0)
    ##leg.SetTextFont(12)
    leg.SetFillStyle(0)

    leg.SetHeader("Efficiency after")
    leg.AddEntry(h_eff_eta_after_gmt,"match to MPC & TF trk & GMT trk","pl")
    leg.AddEntry(h_eff_eta_after_xtra_dr,"#Delta R match to GMT","pl")
    ##leg.AddEntry(h_eff_eta_after_xtra_dr,"(dashed lines correspond to p_{T}>10)","")

    leg.Draw()
    Print(c222xx2,"h_eff_eta_steps_xchk2.eps")
    Print(c222xx2,"h_eff_eta_steps_xchk2" + ext)
    return

    """



    """
    h_eff_eta_me1_after_lct .SetLineColor(kRed)
    h_eff_eta_me1_after_lct_okAlct .SetLineColor(kRed-6)
    h_eff_eta_me1_after_lct_okAlctClct.SetLineColor(kMagenta-6)
    h_eff_eta_me1_after_lct_okClct .SetLineColor(kRed-6)
    h_eff_eta_me1_after_lct_okClctAlct.SetLineColor(kMagenta-6)
    h_eff_eta_me1_after_mplct_okAlctClct.SetLineColor(kBlack)
    h_eff_eta_me1_after_mplct_okAlctClct_plus.SetLineColor(kBlack)
    h_eff_eta_me1_after_tf_ok   .SetLineColor(kBlue)   
    h_eff_eta_me1_after_tf_ok_pt10 .SetLineColor(kBlue+2)
    h_eff_eta_me1_after_tf_ok_plus   .SetLineColor(kBlue)   
    h_eff_eta_me1_after_tf_ok_plus_pt10 .SetLineColor(kBlue+2)

    h_eff_eta_after_tfcand_all_pt10.SetLineColor(kCyan+2)

    h_eff_eta_me1_after_lct.SetLineWidth(2)
    h_eff_eta_me1_after_lct_okAlct.SetLineWidth(2)
    h_eff_eta_me1_after_lct_okAlctClct.SetLineWidth(2)
    h_eff_eta_me1_after_lct_okClct.SetLineWidth(2)
    h_eff_eta_me1_after_lct_okClctAlct.SetLineWidth(2)

    ##h_eff_eta_me1_after_lct.SetLineStyle(7)
    h_eff_eta_me1_after_lct_okAlct.SetLineStyle(3)
    h_eff_eta_me1_after_lct_okAlctClct.SetLineStyle(7)
    h_eff_eta_me1_after_lct_okClct.SetLineStyle(3)
    h_eff_eta_me1_after_lct_okClctAlct.SetLineStyle(7)
    h_eff_eta_me1_after_mplct_okAlctClct_plus.SetLineStyle(7)

    h_eff_eta_me1_after_tf_ok_pt10.SetLineStyle(7)
    h_eff_eta_me1_after_tf_ok_plus_pt10.SetLineStyle(7)
    h_eff_eta_after_tfcand_all_pt10.SetLineStyle(3)

    h_eff_eta_me1_after_lct.SetTitle("efficiency dependence on #eta: ME1 studies")
    h_eff_eta_me1_after_lct.GetXaxis().SetRangeUser(0.86,2.5)
    h_eff_eta_me1_after_lct.GetXaxis().SetTitle("#eta")
    h_eff_eta_me1_after_lct.SetMinimum(0)
    h_eff_eta_me1_after_lct.GetYaxis().SetRangeUser(0.,1.05)

    

    cme1n1 = TCanvas("h_eff_eta_me1_after_lct","h_eff_eta_me1_after_lct",1000,600 ) 


    h_eff_eta_me1_after_lct.Draw("hist")
    ##h_eff_eta_me1_after_lct_okAlct.Draw("same hist")
    ##h_eff_eta_me1_after_lct_okAlctClct.Draw("same hist")
    h_eff_eta_me1_after_lct_okClct.Draw("same hist")
    h_eff_eta_me1_after_lct_okClctAlct.Draw("same hist")
    h_eff_eta_me1_after_mplct_okAlctClct.Draw("same hist")
    h_eff_eta_me1_after_mplct_okAlctClct_plus.Draw("same hist")
    h_eff_eta_me1_after_tf_ok      .Draw("same hist")
    h_eff_eta_me1_after_tf_ok_pt10 .Draw("same hist")
    h_eff_eta_after_tfcand_all_pt10.Draw("same hist")

    leg = TLegend(0.347,0.222,0.926,0.535,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    leg.SetHeader("Efficiency for muon with p_{T}>10")
    ##leg.AddEntry(NUL,"match to LCT:","")
    leg.AddEntry(h_eff_eta_me1_after_lct,"any LCT in ME1 chamber crossed by #mu","pl")
    leg.AddEntry(h_eff_eta_me1_after_lct_okAlct,"  + track's ALCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_lct_okAlctClct,"  + track's CLCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_lct_okClct,"  + correct CLCT picked","pl")
    leg.AddEntry(h_eff_eta_me1_after_lct_okClctAlct,"  + correct ALCT mached to CLCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_mplct_okAlctClct,"  + pass MPC selection","pl")
    leg.AddEntry(h_eff_eta_me1_after_mplct_okAlctClct,"  + has two MPC matched","pl")
    leg.AddEntry(h_eff_eta_me1_after_tf_ok,"  + stub used in any TF track","pl")
    leg.AddEntry(h_eff_eta_me1_after_tf_ok_pt10,"  + stub used in TF track with p_{T}>10 GeV","pl")
    leg.AddEntry(h_eff_eta_after_tfcand_all_pt10,"there is TF track p_{T}>10 in #Delta R<0.3","pl")
    ##leg.AddEntry(," ","pl")
    leg.Draw()
    Print(cme1n1,"h_eff_eta_me1_steps_all.eps")
    Print(cme1n1,"h_eff_eta_me1_steps_all" + ext)
    Print(cme1n1,"h_eff_eta_me1_steps_all.pdf")



    cme1n2 = TCanvas("h_eff_eta_me1_after_lct2","h_eff_eta_me1_after_lct2",800,600 ) 
    h_eff_eta_me1_after_mplct_okAlctClct.GetXaxis().SetRangeUser(0.86,2.5)
    h_eff_eta_me1_after_mplct_okAlctClct.GetXaxis().SetTitle("#eta")
    h_eff_eta_me1_after_mplct_okAlctClct.SetMinimum(0)
    h_eff_eta_me1_after_mplct_okAlctClct.GetYaxis().SetRangeUser(0.,1.05)
    h_eff_eta_me1_after_mplct_okAlctClct.SetTitle("Efficiency in ME1 for #mu with p_{T}>10")
    h_eff_eta_me1_after_mplct_okAlctClct.GetYaxis().SetTitle("Eff.")
    h_eff_eta_me1_after_mplct_okAlctClct.GetXaxis().SetTitleSize(0.07)
    h_eff_eta_me1_after_mplct_okAlctClct.GetXaxis().SetTitleOffset(0.7)
    h_eff_eta_me1_after_mplct_okAlctClct.GetYaxis().SetLabelOffset(0.015)
    gStyle.SetTitleW(1)
    gStyle.SetTitleH(0.08)
    gStyle.SetTitleStyle(0)
    h_eff_eta_me1_after_mplct_okAlctClct.Draw("hist")
    h_eff_eta_me1_after_mplct_okAlctClct_plus.Draw("same hist")
    h_eff_eta_me1_after_tf_ok      .Draw("same hist")
    h_eff_eta_me1_after_tf_ok_pt10 .Draw("same hist")


    leg = TLegend(0.34,0.16,0.87,0.39,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("SimMuon has matched")
    leg.AddEntry(h_eff_eta_me1_after_mplct_okAlctClct,"MPC LCT stub in ME1","pl")
    leg.AddEntry(h_eff_eta_me1_after_mplct_okAlctClct_plus," + one more MPC LCT stub in ME234 + ","pl")
    leg.AddEntry(h_eff_eta_me1_after_tf_ok," + this stub was used by a TF track","pl")
    leg.AddEntry(h_eff_eta_me1_after_tf_ok_pt10,"  + this TF track has p_{T}(TF)>10 GeV","pl")
    leg.Draw()
    Print(cme1n2,"h_eff_eta_me1_tf" + ext)



    cme1n2_plus = TCanvas("h_eff_eta_me1_after_tfplus","h_eff_eta_me1_after_tfplus",800,600 ) 
    h_eff_eta_me1_after_mplct_okAlctClct_plus.GetXaxis().SetRangeUser(0.86,2.5)
    h_eff_eta_me1_after_mplct_okAlctClct_plus.GetXaxis().SetTitle("#eta")
    h_eff_eta_me1_after_mplct_okAlctClct_plus.SetMinimum(0)
    h_eff_eta_me1_after_mplct_okAlctClct_plus.GetYaxis().SetRangeUser(0.,1.05)
    h_eff_eta_me1_after_mplct_okAlctClct_plus.SetTitle("Efficiency in ME1 for #mu with p_{T}>10")
    h_eff_eta_me1_after_mplct_okAlctClct_plus.GetYaxis().SetTitle("Eff.")
    h_eff_eta_me1_after_mplct_okAlctClct_plus.GetXaxis().SetTitleSize(0.07)
    h_eff_eta_me1_after_mplct_okAlctClct_plus.GetXaxis().SetTitleOffset(0.7)
    h_eff_eta_me1_after_mplct_okAlctClct_plus.GetYaxis().SetLabelOffset(0.015)
    gStyle.SetTitleW(1)
    gStyle.SetTitleH(0.08)
    gStyle.SetTitleStyle(0)
    h_eff_eta_me1_after_mplct_okAlctClct_plus.Draw("hist")
    h_eff_eta_me1_after_tf_ok_plus      .Draw("same hist")
    h_eff_eta_me1_after_tf_ok_plus_pt10 .Draw("same hist")
    h_eff_eta_me1_after_lct_okClctAlct.Draw("same hist")

    leg = TLegend(0.34,0.16,0.87,0.39,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("SimMuon has matched")
    leg.AddEntry(h_eff_eta_me1_after_mplct_okAlctClct_plus,"MPC LCT stubs in ME1 and some other station","pl")
    leg.AddEntry(h_eff_eta_me1_after_tf_ok_plus," + these stubs were used by a TF track","pl")
    leg.AddEntry(h_eff_eta_me1_after_tf_ok_plus_pt10,"  + this TF track has p_{T}(TF)>10 GeV","pl")
    leg.Draw()
    Print(cme1n2_plus,"h_eff_eta_me1_tfplus" + ext)
    """


#_______________________________________________________________________________
def simTrackToAlctMatchingEfficiencyVsEtaME1():

    dir = getRootDirectory(input_dir, file_name)

    etareb = 1
    yrange = [0.8,1.04]
    xrange = [1.4,2.5]    
    xTitle = "SimTrack eta"
    yTitle = "Efficiency"
    topTitle = " " * 11 + "CSC Stub matching" + " " * 35 + "CMS Simulation Preliminary"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)

    h_eff_eta_me1_after_alct = setEffHisto("h_eta_me1_after_alct","h_eta_initial",dir, etareb, kBlue+1, 2, 2, title,xTitle,yTitle,xrange,yrange)
    h_eff_eta_me1_after_alct_okAlct = setEffHisto("h_eta_me1_after_alct_okAlct","h_eta_initial",dir, etareb, kBlue+1, 1, 2, "","","",xrange,yrange)

    c = TCanvas("h_eff_eta_me1_after_alct","h_eff_eta_me1_after_alct",1000,600 )     
    c.cd()
    h_eff_eta_me1_after_alct.Draw("hist")
    h_eff_eta_me1_after_alct.Draw("same hist")
    h_eff_eta_me1_after_alct_okAlct.Draw("same hist")

    leg = TLegend(0.347,0.222,0.926,0.535,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("Efficiency for #mu with p_{T}>20 crossing a ME1 chamber with")
    leg.AddEntry(h_eff_eta_me1_after_alct,"any ALCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_alct_okAlct,"correct ALCT","pl")
    leg.Draw()
    c.Print("%ssimTrackToAlctMatchingEfficiencyVsEtaME1%s"%(output_dir, ext))


#_______________________________________________________________________________
def simTrackToClctMatchingEfficiencyVsEtaME1():

    dir = getRootDirectory(input_dir, file_name)

    etareb = 1
    yrange = [0.8,1.04]
    xrange = [1.4,2.5]    
    xTitle = "SimTrack eta"
    yTitle = "Efficiency"
    topTitle = " " * 11 + "CSC Stub matching" + " " * 35 + "CMS Simulation Preliminary"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)

    h_eff_eta_me1_after_clct = setEffHisto("h_eta_me1_after_clct","h_eta_initial",dir, etareb, kBlue+1, 2, 2, title,xTitle,yTitle,xrange,yrange)
    h_eff_eta_me1_after_clct_okClct = setEffHisto("h_eta_me1_after_clct_okClct","h_eta_initial",dir, etareb, kBlue+1, 1, 2, "","","",xrange,yrange)

    c = TCanvas("h_eff_eta_me1_after_clct","h_eff_eta_me1_after_clct",1000,600 )     
    c.cd()
    h_eff_eta_me1_after_clct.Draw("hist")
    h_eff_eta_me1_after_clct.Draw("same hist")
    h_eff_eta_me1_after_clct_okClct.Draw("same hist")

    leg = TLegend(0.347,0.222,0.926,0.535,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("Efficiency for #mu with p_{T}>20 crossing a ME1 chamber with")
    leg.AddEntry(h_eff_eta_me1_after_clct,"any CLCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_clct_okClct,"correct CLCT","pl")
    leg.Draw()
    c.Print("%ssimTrackToClctMatchingEfficiencyVsEtaME1%s"%(output_dir, ext))

#_______________________________________________________________________________
def simTrackToAlctClctMatchingEfficiencyVsEtaME1():

    dir = getRootDirectory(input_dir, file_name)

    etareb = 1
    yrange = [0.8,1.04]
    xrange = [1.4,2.5]    
    xTitle = "SimTrack eta"
    yTitle = "Efficiency"
    topTitle = " " * 11 + "CSC Stub matching" + " " * 35 + "CMS Simulation Preliminary"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)

    h_eff_eta_me1_after_clct = setEffHisto("h_eta_me1_after_clct","h_eta_initial",dir, etareb, kBlue+1, 2, 2, title,xTitle,yTitle,xrange,yrange)
    h_eff_eta_me1_after_clct_okClct = setEffHisto("h_eta_me1_after_clct_okClct","h_eta_initial",dir, etareb, kBlue+1, 1, 2, "","","",xrange,yrange)
    h_eff_eta_me1_after_alct = setEffHisto("h_eta_me1_after_alct","h_eta_initial",dir, etareb, kRed+1, 2, 2, title,xTitle,yTitle,xrange,yrange)
    h_eff_eta_me1_after_alct_okAlct = setEffHisto("h_eta_me1_after_alct_okAlct","h_eta_initial",dir, etareb, kRed+1, 1, 2, "","","",xrange,yrange)
    h_eff_eta_me1_after_alctclct = setEffHisto("h_eta_me1_after_alctclct","h_eta_initial",dir, etareb, kGreen+1, 2, 2, title,xTitle,yTitle,xrange,yrange)
    h_eff_eta_me1_after_alctclct_okAlct = setEffHisto("h_eta_me1_after_alctclct_okAlct","h_eta_initial",dir, etareb, kGreen+1, 1, 2, title,xTitle,yTitle,xrange,yrange)
    h_eff_eta_me1_after_alctclct_okClct = setEffHisto("h_eta_me1_after_alctclct_okClct","h_eta_initial",dir, etareb, kOrange+1, 2, 2, title,xTitle,yTitle,xrange,yrange)
    h_eff_eta_me1_after_alctclct_okAlctClct = setEffHisto("h_eta_me1_after_alctclct_okAlctClct","h_eta_initial",dir, etareb, kOrange+1, 1, 2, title,xTitle,yTitle,xrange,yrange)

    c = TCanvas("","",1000,600 )     
    c.cd()
#    h_eff_eta_me1_after_clct.Draw("hist")
#    h_eff_eta_me1_after_clct.Draw("same hist")
#    h_eff_eta_me1_after_clct_okClct.Draw("same hist")
#    h_eff_eta_me1_after_alct.Draw("same hist")
#    h_eff_eta_me1_after_alct_okAlct.Draw("same hist")
    h_eff_eta_me1_after_alctclct.Draw("hist")
    h_eff_eta_me1_after_alctclct_okAlct.Draw("same hist")
    h_eff_eta_me1_after_alctclct_okClct.Draw("same hist")
    h_eff_eta_me1_after_alctclct_okAlctClct.Draw("same hist")

    leg = TLegend(0.347,0.15,0.926,0.45,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("Efficiency for #mu with p_{T}>20 crossing a ME1 chamber with")
#    leg.AddEntry(h_eff_eta_me1_after_clct,"any CLCT","pl")
#    leg.AddEntry(h_eff_eta_me1_after_clct_okClct,"correct CLCT","pl")
#    leg.AddEntry(h_eff_eta_me1_after_alct,"any ALCT","pl")
#    leg.AddEntry(h_eff_eta_me1_after_alct_okAlct,"correct ALCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_alctclct,"any ALCT & any CLCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_alctclct_okAlct,"correct ALCT & any CLCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_alctclct_okClct,"any ALCT & correct CLCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_alctclct_okAlctClct,"correct ALCT & correct CLCT","pl")

    leg.Draw()
    c.Print("%ssimTrackToAlctClctMatchingEfficiencyVsEtaME1%s"%(output_dir, ext))


#_______________________________________________________________________________
def simTrackToLctMatchingEfficiencyVsEtaME1():

    dir = getRootDirectory(input_dir, file_name)

    etareb = 1
    yrange = [0.8,1.04]
    xrange = [1.4,2.5]    
    xTitle = "SimTrack eta"
    yTitle = "Efficiency"
    topTitle = " " * 11 + "CSC Stub matching" + " " * 35 + "CMS Simulation Preliminary"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)

    h_eff_eta_me1_after_lct = setEffHisto("h_eta_me1_after_lct","h_eta_initial",dir, etareb, kRed, 2, 2, topTitle,xTitle,yTitle,xrange,yrange)
    h_eff_eta_me1_after_lct_okAlct = setEffHisto("h_eta_me1_after_lct_okAlct","h_eta_initial",dir, etareb, kRed+1, 2, 2, "eff(#eta): ME1 stub studies","#eta","",xrange,yrange)
    h_eff_eta_me1_after_lct_okClct = setEffHisto("h_eta_me1_after_lct_okClct","h_eta_initial",dir, etareb, kRed+2, 2, 2, "eff(#eta): ME1 stub studies","#eta","",xrange,yrange)
    h_eff_eta_me1_after_lct_okAlctClct = setEffHisto("h_eta_me1_after_lct_okAlctClct","h_eta_initial",dir, etareb, kRed+3, 2, 2, "eff(#eta): ME1 stub studies","#eta","",xrange,yrange)
    h_eff_eta_me1_after_lct_okClctAlct = setEffHisto("h_eta_me1_after_lct_okClctAlct","h_eta_initial",dir, etareb, kRed+4, 1,2, "","","",xrange,yrange)
    
    c = TCanvas("h_eff_eta_me1_after_alct","h_eff_eta_me1_after_alct",1000,600 )     
    c.cd()
    h_eff_eta_me1_after_lct.Draw("hist")
    h_eff_eta_me1_after_lct_okAlct.Draw("same hist")
    h_eff_eta_me1_after_lct_okClct.Draw("same hist")
    h_eff_eta_me1_after_lct_okAlctClct.Draw("same hist")
    h_eff_eta_me1_after_lct_okClctAlct.Draw("same hist")

    leg = TLegend(0.347,0.222,0.926,0.535,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    leg.SetNColumns(2)
    leg.SetHeader("Efficiency for #mu with p_{T}>20 crossing a ME1 chamber with")
    leg.AddEntry(h_eff_eta_me1_after_lct,"any LCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_lct_okAlct,"any LCT with correct ALCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_lct_okClct,"any LCT with correct CLCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_lct_okClctAlct,"correct LCT, CLCT-to-ALCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_lct_okAlctClct,"correct LCT, ALCT-to-CLCT","pl")
    leg.Draw()

    c.Print("%ssimTrackToLctMatchingEfficiencyVsEtaME1%s"%(output_dir, ext))

#_______________________________________________________________________________
def simTrackToMpcLctMatchingEfficiencyVsEtaME1():

    dir = getRootDirectory(input_dir, file_name)

    etareb = 1
    yrange = [0.8,1.04]
    xrange = [1.4,2.5]    
    xTitle = "SimTrack eta"
    yTitle = "Efficiency"
    topTitle = " " * 11 + "CSC Stub matching" + " " * 35 + "CMS Simulation Preliminary"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)

    h_eff_eta_me1_after_mplct_okAlctClct = setEffHisto("h_eta_me1_after_mplct_okAlctClct","h_eta_initial",dir, etareb, kBlue+1, 2, 2, title,xTitle,yTitle,xrange,yrange)
    h_eff_eta_me1_after_mplct_okAlctClct_plus = setEffHisto("h_eta_me1_after_mplct_okAlctClct_plus","h_eta_initial",dir, etareb, kBlue+1, 1, 2, "","","",xrange,yrange)

    c = TCanvas("h_eff_eta_me1_after_alct","h_eff_eta_me1_after_alct",1000,600 )     
    c.cd()
    h_eff_eta_me1_after_mplct_okAlctClct.Draw("hist")
    h_eff_eta_me1_after_mplct_okAlctClct_plus.Draw("same hist")

    leg = TLegend(0.347,0.222,0.926,0.535,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("Efficiency for #mu with p_{T}>20 crossing a ME1 chamber with")
    leg.AddEntry(h_eff_eta_me1_after_mplct_okAlctClct,"Good MPC LCT","pl")
    leg.AddEntry(h_eff_eta_me1_after_mplct_okAlctClct_plus,"  + has two MPC matched","pl")
    leg.Draw()
    c.Print("%ssimTrackToMpcLctMatchingEfficiencyVsEtaME1%s"%(output_dir, ext))

#_______________________________________________________________________________
def simTrackToMpcLctTfMatchingEfficiencyVsEtaME1():
    pass

#_______________________________________________________________________________
def simTrackToTFTrackMatchingEfficiencyVsEtaME1():

    dir = getRootDirectory(input_dir, file_name)

    etareb = 1
    yrange = [0.6,1.05]
    xrange = [1.4,2.5]    

    xTitle = "SimTrack eta"
    yTitle = "Efficiency"
    topTitle = " " * 11 + "CSC TFTrack matching" + " " * 35 + "CMS Simulation Preliminary"
    title = "%s;%s;%s"%(topTitle,xTitle,yTitle)

    h_eff_eta_me1_after_tf_ok = setEffHisto("h_eta_me1_after_tf_ok","h_eta_initial",dir, etareb, kBlue+1, 2, 2, title,xTitle,yTitle,xrange,yrange)
    h_eff_eta_me1_after_tf_ok_pt10 = setEffHisto("h_eta_me1_after_tf_ok_pt10","h_eta_initial",dir, etareb, kBlue+1, 1, 2, "","","",xrange,yrange)
    h_eff_eta_me1_after_tf_ok_plus = setEffHisto("h_eta_me1_after_tf_ok_plus","h_eta_initial",dir, etareb, kGreen+1, 2, 2, title,xTitle,yTitle,xrange,yrange)
    h_eff_eta_me1_after_tf_ok_plus_pt10 = setEffHisto("h_eta_me1_after_tf_ok_plus_pt10","h_eta_initial",dir, etareb, kGreen+1, 1, 2, "","","",xrange,yrange)

    c = TCanvas("c","c",1000,600 )     
    c.cd()

    h_eff_eta_me1_after_tf_ok.Draw("hist")
    h_eff_eta_me1_after_tf_ok_pt10.Draw("same hist")
    h_eff_eta_me1_after_tf_ok_plus.Draw("same hist")
    h_eff_eta_me1_after_tf_ok_plus_pt10.Draw("same hist")

    leg = TLegend(0.1,0.222,0.926,0.535,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("Efficiency for #mu with p_{T}>20 crossing a ME1 chamber with")
    leg.AddEntry(h_eff_eta_me1_after_tf_ok,"a TF Track","pl")
    leg.AddEntry(h_eff_eta_me1_after_tf_ok_pt10,"a TF Track with p_{T}>10","pl")
    leg.AddEntry(h_eff_eta_me1_after_tf_ok_plus,"a TF Track with a stub in ME1 and at least another station","pl")
    leg.AddEntry(h_eff_eta_me1_after_tf_ok_plus_pt10,"a TF Track with p_{T}>10, and with a stub in ME1 and another station","pl")
    leg.Draw()

    c.Print("%ssimTrackToTFTrackMatchingEfficiencyVsEtaME1%s"%(output_dir, ext))


#_______________________________________________________________________________
def globalMuonTriggerEfficiencyVsEtaME1():
    pass

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
def simTrackToAlctMatchingEfficiencyVsWGME11():

    dir = getRootDirectory(input_dir, file_name)
    
    xTitle = "WG"
    yTitle = "Efficiency"
    topTitle = "Correct ALCT efficiency dependence on WG in ME11"
    fullTitle = "%s;%s;%s"%(topTitle,xTitle,yTitle)

    etareb = 1
    yrange = [0.6,1.2]
    xrange = [0,100]    

    h_eff_wg_me11_after_alct_okAlct = setEffHisto("h_wg_me11_after_alct_okAlct","h_wg_me11_initial",dir, etareb, kRed, 0, 2, "","","",xrange,yrange)
    h_eff_wg_me11_after_alctclct_okAlctClct = setEffHisto("h_wg_me11_after_alctclct_okAlctClct","h_wg_me11_initial",dir, etareb, kBlue, 0, 2, "","","",xrange,yrange)
    h_eff_wg_me11_after_lct_okAlctClct = setEffHisto("h_wg_me11_after_lct_okAlctClct","h_wg_me11_initial",dir, etareb, kBlack, 0, 2, "","","",xrange,yrange)

    h_eff_wg_me11_after_alct_okAlct.SetTitle(topTitle)

    c = TCanvas("c","c",1000,600)     
    c.cd()

    h_eff_wg_me11_after_alct_okAlct.Draw("hist")
    h_eff_wg_me11_after_alctclct_okAlctClct.Draw("same hist")
    h_eff_wg_me11_after_lct_okAlctClct.Draw("same hist")

    leg = TLegend(0.2,0.2,0.926,0.4,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    #leg.SetNColumns(3)
    leg.SetHeader("Figure out title")
    leg.AddEntry(h_eff_wg_me11_after_alct_okAlct,"Good ALCT","pl")
    leg.AddEntry(h_eff_wg_me11_after_alctclct_okAlctClct,"Good ALCT & CLCT","pl")
    leg.AddEntry(h_eff_wg_me11_after_lct_okAlctClct,"Good LCT","pl")
    leg.Draw()

    c.Print("%ssimTrackToALctMatchingEfficiencyVsWGME11%s"%(output_dir, ext))

    
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
    file_name = "hp_dimu_CMSSW_6_2_0_SLHC5_upgrade2019_pu000_w3_gem98_pt2-50_PU0_pt0_new_eff.test.pt20.root"
    output_dir = "plots_pt20/"
    ext = ".png"

    simTrackToAlctMatchingEfficiencyVsEtaME1()
    simTrackToClctMatchingEfficiencyVsEtaME1()
    simTrackToAlctClctMatchingEfficiencyVsEtaME1()
    simTrackToLctMatchingEfficiencyVsEtaME1()
    simTrackToMpcLctMatchingEfficiencyVsEtaME1()
    simTrackToTFTrackMatchingEfficiencyVsEtaME1()
    
    simTrackToAlctMatchingEfficiencyVsWGME11()

    cscStationOccupanciesVsEta()
    cscStationOccupanciesMatchedVsEta_2()
    cscStationOccupanciesMatchedVsEta_3()
    
    cscStationOccupanciesMatchedMpcVsEta()
    cscStationOccupanciesMatchedMpcVsEta_2()
    cscStationOccupanciesMatchedMpcVsEta_3()

    tfCandidateTriggerEfficiencyVsEta()
    cscStubMatchingEfficiencyVsEtaSummary()
    cscTFMatchingEfficiencyVsEtaME1()
    cscTFCandMatchingEfficiencyVsEta()
    cscTFCandMatchingEfficiencyVsEta_2()
    cscTFCandMatchingEfficiencyVsEta_3()
    cscMpcTFCandGmtTriggerEfficiencyVsEta()
    
    """
    simTrackToAlctMatchingEfficiencyVsPhiME1()
    simTrackToClctMatchingEfficiencyVsPhiME1()
    simTrackToLctMatchingEfficiencyVsPhiME1()
    """
