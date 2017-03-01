#!/usr/bin/env python

"""
Copied from GEMCode/GEMValidation
"""

from ROOT import *

import os
import sys
import ROOT
ROOT.gROOT.SetBatch(1)

import optparse

range_min =0
range_max =0 

def draw_col_userRange( target_dir, h, min, max,ext =".png", opt = "colz"):
  gStyle.SetOptStat(0)
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
  h.SetAxisRange(float(min),float(max),"X")
  h.Draw(opt)
  c.SaveAs(target_dir + c_title + ext)
  

def draw_plot( file, tDir,oDir,min,max ) :
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
    else :
      print "error"
      exit(-1)
  for x in tlist :
    key_list.append(x.GetName())
  for hist in key_list :
    if ( hist.find("geo_phi") != -1) :
      draw_col_userRange( oDir, d1.Get(hist),min,max)

if __name__ == '__main__' :
  usage = ": %prog [option] DQM_filename.root\negs) ./%prog --min 14.5 --max 15.5 DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option("-o",dest='directory',help='Name of output directory(Default : ./)',default="./")
  parser.add_option("--min",dest='range_min',help='Minimum of phi degree',default=14.5)
  parser.add_option("--max",dest='range_max',help='Maximum of phi degree',default=15.5)
  options, args = parser.parse_args()

  print options.range_min, options.range_max
  min = options.range_min
  max = options.range_max
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
  steps.append("GEMDigis")

  for step in steps :
    tDir = "DQMData/Run 1/Muon%sV/Run summary/%sTask"%(step,step)
    oDir = options.directory
    os.system("mkdir -p "+oDir )
    draw_plot(args[0],tDir,oDir,min,max)  
