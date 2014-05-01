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
		if hist.find("eff") != -1 or hist.find("track_") != -1 :
			continue
		if (hist.find("lx") !=-1 or hist.find("ly") != -1 ) :
			continue
		if ( hist.find("bx") != -1 ) :
			draw_bx( oDir, d1.Get(hist)  )
		elif ( hist.find("xy") !=-1 or hist.find("zr") !=-1 or hist.find("roll_vs_strip")!= -1 or hist.find("phipad")!=-1 or hist.find("phistrip") != -1 ) :
			draw_col( oDir, d1.Get(hist) )
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
	 
