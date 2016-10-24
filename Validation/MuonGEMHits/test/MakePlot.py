#!/usr/bin/env python

"""
Copied from GEMCode/GEMValidation
"""

from ROOT import *

import os
import sys
#sys.argv.append( '-b' )
import ROOT
ROOT.gROOT.SetBatch(1)

import optparse
def getEtaRange( station ) :
  etaRange = [1.55,2.15,1.65,2.45]
  if ( station ==1 or station==2 ) :
    return etaRange[ (station-1)*2], etaRange[ (station-1)*2+1 ]
  else :
    print "Something is wrong"
    return 1.5,2.6
    
def draw_occ(target_dir, h, ext =".png", opt = ""):
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(1110)
  c = TCanvas(h.GetTitle(),h.GetName(),600,600)
  c_title = c.GetTitle()
  c.Clear()
  if not h:
    sys.exit('h does not exist')
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw(opt)
  c.SaveAs(target_dir + c_title + ext)

def draw_bx(target_dir, h , ext = ".png", opt = ""):
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(1110)
  c = TCanvas(h.GetTitle(),h.GetName(),600,600)
  c_title = c.GetTitle()
  gPad.SetLogy()
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw(opt)
  h.SetMinimum(1.)
  c.SaveAs(target_dir + c_title + ext)

def draw_simple_zr(target_dir, h, ext =".png",opt="colz"):
  gStyle.SetOptStat(0)
  c = TCanvas(h.GetName(),h.GetName(),600,600)
  c.Divide(2,1)
  c_title = c.GetTitle()
  #c.Clear()
  if not h:
    sys.exit('h does not exist')
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  c.cd(1)
  gPad.SetRightMargin(0)
  gPad.SetBorderMode(0)
  h1 = h.Clone()
  h1.SetAxisRange(564,572)
  h1.Draw("col")
  c.cd(2)
  h2 = h.Clone()
  gPad.SetLeftMargin(0)
  gPad.SetBorderMode(0)
  h2.SetAxisRange(793,800)
  h2.Draw(opt)
  #c.cd()
  #c.Draw()
  #c.Update()
  c.SaveAs(target_dir + c_title + ext)


def draw_col_nostat(target_dir, h, ext =".png", opt = "colz"):
  gStyle.SetOptStat(0)
  c = TCanvas(h.GetTitle(),h.GetName(),800,600)
  c_title = c.GetTitle()
  c.Clear()
  if not h:
    sys.exit('h does not exist')
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw(opt)
  c.SaveAs(target_dir + c_title + ext)

def draw_col_eff(target_dir, h, ext =".png", opt = "colz"):
  gStyle.SetOptStat(0)
  c = TCanvas(h.GetTitle(),h.GetName(),800,600)
  c_title = c.GetTitle()
  c.Clear()
  if not h:
    sys.exit('h does not exist')
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.SetMaximum(1.2)
  h.Draw(opt)
  c.SaveAs(target_dir + c_title + ext)



def draw_col(target_dir, h, ext =".png", opt = "col"):
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(1110)
  c = TCanvas(h.GetTitle(),h.GetName(),600,600)
  c_title = c.GetTitle()
  c.Clear()
  if not h:
    sys.exit('h does not exist')
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw(opt)
  c.SaveAs(target_dir + c_title + ext)

def draw_col_userRange( target_dir, h, ext =".png", opt = "col"):
  gStyle.SetStatStyle(0)
  c = TCanvas(h.GetTitle(),h.GetName(),1600,1600)
  c_title = c.GetTitle()
  c.Clear()
  if not h:
    sys.exit('h does not exist')
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.SetLabelSize(0.02,"Y")
  h.SetLabelOffset(0,"Y")
  axis_title = h.GetXaxis().GetTitle()
  axis_title = axis_title+ "/"+str(h.GetXaxis().GetBinWidth(1))
  h.GetXaxis().SetTitle( axis_title)
  h.SetAxisRange(14.5,15.5)
  h.Draw(opt)
  c.SaveAs(target_dir + c_title + ext)
  

def draw_col_overflow(target_dir, h, ext =".png", opt = "col"):
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(110200)
  c = TCanvas(h.GetTitle(),h.GetName(),600,600)
  c_title = c.GetTitle()
  c.Clear()
  if not h:
    sys.exit('h does not exist')
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw(opt)
  c.SaveAs(target_dir + c_title + ext)

def draw_eff(target_dir, h, ext = ".png", opt = ""):
  c = TCanvas(h.GetTitle(), h.GetName(),600,600)
  c_title = c.GetTitle()
  c.Clear()
  if not h: 
    sys.exit('h does not exist')
  gPad.SetGrid(1)
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(0)
  gStyle.SetOptFit(0)
  h.GetYaxis().SetRangeUser(0,1.05)
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.SetMarkerStyle(1)
  h.SetMarkerColor(kBlue)
  h.SetMarkerSize(1)
  h.Draw(opt);
  xmin=h.GetXaxis().GetXmin()
  xmax=h.GetXaxis().GetXmax()
  if ( h.GetName().find("eta") != -1) :
    if ( h.GetName().find("st1") != -1) :
      xmin,xmax = getEtaRange(1)
    elif ( h.GetName().find("st2") != -1 ) :
      xmin,xmax = getEtaRange(2)
    else :
      print "Use default setting."

  f1 = TF1("fit1","pol0", xmin, xmax )
  r = h.Fit("fit1","RQS")
  ptstats = TPaveStats(0.25,0.35,0.75,0.55,"brNDC")
  ptstats.SetName("stats")
  ptstats.SetBorderSize(0)
  ptstats.SetLineWidth(0)
  ptstats.SetFillColor(0)
  ptstats.SetTextAlign(11)
  ptstats.SetTextFont(42)
  ptstats.SetTextSize(.05)
  ptstats.SetTextColor(kRed)
  ptstats.SetOptStat(0)
  ptstats.SetOptFit(1111)
  chi2 = int(r.Chi2())
  ndf = int(r.Ndf())
   ## prob = r.Prob()
  round(2.675, 2)
  p0 = f1.GetParameter(0)
  p0e = f1.GetParError(0)
  ptstats.AddText("#chi^{2} / ndf: %d/%d" %(chi2,ndf))
  ## ptstats.AddText("Fit probability: %f %" %(prob))
  ptstats.AddText("Efficiency: %f #pm %f %%"%(p0,p0e))
  ptstats.Draw("same")
  pt = TPaveText(0.09899329,0.9178322,0.8993289,0.9737762,"blNDC")
  pt.SetName("title")
  pt.SetBorderSize(1)
  pt.SetFillColor(0)
  pt.SetFillStyle(0)
  pt.SetTextFont(42)
  pt.AddText(h.GetTitle())
  pt.Draw("same")
  c.SaveAs(target_dir + c_title + ext)


def draw_plot( file, tDir,oDir ) :
  c = TCanvas("c","c",600,600)
  dqm_file = TFile( file)
  d1 = dqm_file.Get(tDir)
  key_list =[]

  try :
    tlist = d1.GetListOfKeys()
  except :
    print oDir
    if ( oDir.find("Digi") != -1 ):
      tDir = "DQMData/Run 1/MuonGEMDigisV/Run summary/GEMDigiTask"
      d1 = dqm_file.Get(tDir)
      tlist = d1.GetListOfKeys()
    elif ( oDir.find("RecHit") != -1 ):
      tDir = "DQMData/Run 1/MuonGEMRecHitsV/Run summary/GEMRecHitTask"
      d1 = dqm_file.Get(tDir)
      tlist = d1.GetListOfKeys()
    else :
      print "error"
      exit(-1)
  for x in tlist :
    key_list.append(x.GetName())
  for hist in key_list :
    if hist.find("track_") != -1 :
      draw_occ( oDir,d1.Get(hist)) 
    elif (hist.find("dcEta") !=-1 ) :
      draw_col_nostat( oDir,d1.Get(hist))
    elif (hist.find("simple_zr")!= -1 ) :
      draw_simple_zr( oDir,d1.Get(hist))
    elif (hist.find("eff_DigiHit") != -1 ) :
      draw_col_eff( oDir, d1.Get(hist))
    elif (hist.find("lx") !=-1 or hist.find("ly") != -1 or hist.find("dphi") != -1 or hist.find("_phi_dist") != -1 ) :
      draw_occ( oDir,d1.Get(hist))
    elif ( hist.find("bx") != -1 ) :
      draw_bx( oDir, d1.Get(hist)  )
    elif ( hist.find("xy") !=-1 or hist.find("zr") !=-1 or hist.find("roll_vs_strip")!= -1 or hist.find("phipad")!=-1 or hist.find("phistrip") != -1 or hist.find("sp")!=-1 or hist.find("sub")!=-1 ) :
      draw_col( oDir, d1.Get(hist) )
    elif ( hist.find("phiz") != -1 ) :
      draw_col_overflow( oDir, d1.Get(hist) )
    elif ( hist.find("eff") != -1 ) :
      draw_eff( oDir, d1.Get(hist) )
      #print "found "
    elif ( hist.find("geo_phi") != -1) :
      draw_col_userRange( oDir, d1.Get(hist))
    else :
      draw_occ( oDir, d1.Get(hist) )

 

if __name__ == '__main__' :
  usage = ": %prog [option] DQM_filename.root\negs) ./%prog -a DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root"
  parser = optparse.OptionParser(usage=usage)
  #parser.add_option("-i",dest='dqmfile',help='Input DQM filename',default="DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root")
  parser.add_option("-o",dest='directory',help='Name of output directory(Default : temp_plot)',default="temp_plot")
  parser.add_option("-a",action='store_true',dest='all',help='Enable all step option.(-s -d -r)',default=False)
  parser.add_option("-s",action='store_true',dest='simhit',help='Run simhit plotter',default=False)
  parser.add_option("-d",action='store_true',dest='digi',help='Run digi plotter',default=False)
  parser.add_option("-r",action='store_true',dest='reco',help='Run reco plotter',default=False)
  options, args = parser.parse_args()

  if len(sys.argv) ==1 :
    parser.print_help()
    exit()
  # If no argument, default name will be used.
  if len(args)==0 :
    print "Input file name is None."
    print "Use default name."
    args.append("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root")

  if len(args) != 1 : 
    print "Can not understand input argument"
    parser.print_help()
  
  steps= []
  if ( options.all ) :
    options.simhit=True
    options.digi=True
    options.reco=True

  if ( options.simhit) :
    steps.append("GEMHits")
  if ( options.digi) :
    steps.append("GEMDigis")
  if ( options.reco) :
    steps.append("GEMRecHits")

  for step in steps :
    tDir = "DQMData/Run 1/Muon%sV/Run summary/%sTask"%(step,step)
    oDir = options.directory+"_%s"%(step)+'/'
    os.system("mkdir -p "+oDir )
    draw_plot(args[0],tDir,oDir)  
    #print args[0],tDir, oDir
   
