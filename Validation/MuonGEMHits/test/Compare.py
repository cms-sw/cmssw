#!/usr/bin/env python3

"""
Copied from GEMCode/GEMValidation
"""

from ROOT import *

import os
import sys
import ROOT
ROOT.gROOT.SetBatch(1)

import optparse
    
output = TFile("output.root","RECREATE")


def draw_occ(target_dir, h1,h2, ext =".png", opt = ""):
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(1110)
  c = TCanvas(h1.GetTitle(),h1.GetName(),600,600)
  c_title = c.GetTitle()
  c.Clear()
  if not h1 or not h2:
    sys.exit('h1 or h2 does not exist')
  h1.SetLineWidth(2)
  h1.SetLineColor(kBlue)
  h1.SetMarkerColor(kBlue)
  h1.SetTitle(args[0])
  h1.Draw(opt)
  h2.SetLineWidth(2)
  h2.SetLineColor(kRed)
  h2.SetMarkerColor(kRed)
  h2.SetTitle(args[1])
  h2.Draw("same"+opt)
  gPad.SetTitle(c_title)
  leg = gPad.BuildLegend()
  h1.SetTitle(c_title)
  c.Update()
  c.SaveAs(target_dir + c_title + ext)
  

def draw_diff_strip(target_dir, h1,h2, ext =".png", opt = ""):
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(0)
  c = TCanvas("c1",("strip_diff_%s")%(h1.GetName()),600,600)
  c_title = c.GetTitle()
  c.Clear()
  if not h1 or not h2:
    sys.exit('h1 or h2 does not exist')
  xbin = h1.GetXaxis().GetNbins()
  xmin = h1.GetXaxis().GetXmin()
  xmax = h1.GetXaxis().GetXmax()
  title = ("Difference of strip phi between %s and %s")%(h1.GetName(),h2.GetName())
  h = TH1F("strip_diff",title,xbin,xmin,xmax)
  for x in range( xbin ) :
    value1 = h1.GetBinContent( x + 1 )  
    value2 = h2.GetBinContent( x + 1 )  
    h.SetBinContent( x+1, value1 - value2)
  h.Draw(opt)
  gPad.SetTitle(c_title)
  #leg = gPad.BuildLegend().SetFillColor(kWhite)
  c.Update()
  c.SaveAs(target_dir + c_title + ext)
  output.ReOpen("UPDATE")
  h.Write()

def draw_plot( file1, file2, tDir,oDir ) :
  c = TCanvas("c","c",600,600)
  dqm_file1 = TFile( file1)
  dqm_file2 = TFile( file2)
  d1 = dqm_file1.Get(tDir)
  d2 = dqm_file2.Get(tDir)
  key_list =[]

  tlist1 = d1.GetListOfKeys()
  for x in tlist1 :
    key_list.append(x.GetName())

  for hist in key_list :
    if ( hist.find("_phiz_") != -1 ) :
      draw_occ( oDir,d1.Get(hist), d2.Get(hist),".png","col");
    elif ( hist.find("strip_phi_dist") != -1 ) :
      draw_diff_strip( oDir, d1.Get(hist), d2.Get(hist) )
    elif ( hist.find("sp") != -1) :
      draw_occ( oDir, d1.Get(hist), d2.Get(hist))

if __name__ == '__main__' :
  usage = ": %prog DQM_file1.root file2.root \negs) ./%prog -o c_temp_plot DQM_v6.root DQM_v7.root"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option("-o",dest='directory',help='Name of output directory(Default : c_temp_plot)',default="c_temp_plot")
  options, args = parser.parse_args()

  if len(args)==0 :
    print "Input file name is None."
    print "Use default name.[ DQM_v6.root and DQM_v7.root]"
    args.append("DQM_v6.root")
    args.append("DQM_v7.root")

  tDir = "DQMData/Run 1/MuonGEMDigisV/Run summary/GEMDigisTask"
  oDir = options.directory+"_GEMDigis/"
  os.system("mkdir -p "+oDir )
  draw_plot(args[0],args[1],tDir,oDir) 
