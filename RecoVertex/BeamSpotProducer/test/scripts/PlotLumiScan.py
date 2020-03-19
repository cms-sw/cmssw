#!/usr/bin/env python
#____________________________________________________________
#
#  PlotLumiScan
#
# A very simple way to make lumiscan plots from beamfit.txt files
# Needed files: A txt file specifying lumi section ranges eg. RunLumiScan.txt
#   All the beam fit txt files in <data_dir> created after running AnalyzeLumiScan.py
#
# Geng-yuan Jeng
# Geng-yuan.Jeng@cern.ch
#
# Fermilab, 2009
#
#____________________________________________________________

from __future__ import print_function
from builtins import range
import sys,os
import string
from array import array
from math import sqrt
from ROOT import gROOT,gPad,TH1F,TLegend,TFile,TStyle,TAxis

def get_list_files(directory,pattern="txt"):
    dir = []
    dir = os.listdir(directory)
    dir.sort(cmp)
    lfiles = []

    for f in dir:
        if f.find(pattern) != -1:
	    #print f
            lfiles.append(f)
    return lfiles;


def plot_trending(type,label,x,xe):
    h = TH1F(type+'_lumi',type+'_lumi',len(x),0.5,0.5+len(x))
    h.SetStats(0)
    h.GetXaxis().SetTitle("Lumisection")
    h.GetYaxis().SetTitle(type[:len(type)-1]+"_{0} (cm)")
    h.GetYaxis().SetLabelSize(0.03)
    h.SetTitleOffset(1.1)
    h.SetOption("e1")
    
    for i in range(len(x)):
        h.SetBinContent(i+1, x[i])
        h.SetBinError(i+1, xe[i])
        h.GetXaxis().SetBinLabel(i+1,label[i])
    return h

def main():
    
    if len(sys.argv) < 4:
        print("\n [Usage] python PlotLumiScan.py <LumiScanLists.txt> <data dir> <verbose:True/False>")
        sys.exit()

    lumilistfile = sys.argv[1]
    runinfofile = open(lumilistfile,"r")
    runinfolist = runinfofile.readlines()
    runsinfo = {}

    for line in runinfolist:
        npos=0
        for i in line.split():
            npos+=1
            if npos == 1:
                run="Run"+str(i)+"/"
            else:
                runsinfo.setdefault(run,[]).append(int(i))
##    print runsinfo

    infiledir = sys.argv[2]
    if infiledir.endswith("/") != 1:
        infiledir+="/"
    verbose = sys.argv[3]
    files = get_list_files(infiledir,"txt")
    nfiles = len(files)-1
    labels = []
    x0=[]
    y0=[]
    z0=[]
    sigZ=[]
    x0Err=[]
    y0Err=[]
    z0Err=[]
    sigZErr=[]

    ## Read files and put values into data containers
    ## Labels:
    
    lumilist = runsinfo.get(infiledir)

    for j in range((len(lumilist)+1)/2):
        labelName=str(lumilist[j*2])+"-"+str(lumilist[j*2+1])
        labels.append(labelName)
##    print labels

    for f in files:
        readfile = open(infiledir+f)
        for line in readfile:
            if line.find("X") != -1 and not "BeamWidth" in line and not "Emittance" in line:
                count=0
                for val in line.split():
                    count+=1
                    if count > 1:
                        x0.append(float(val))
            if line.find("Cov(0,j)") != -1:
                count=0
                for val in line.split():
                    count+=1
                    if count == 2:
                        valErr=sqrt(float(val))
                        x0Err.append(valErr)
            if line.find("Y") != -1 and not "BeamWidth" in line and not "Emittance" in line:
                count=0
                for val in line.split():
                    count+=1
                    if count > 1:
                        y0.append(float(val))
            if line.find("Cov(1,j)") != -1:
                count=0
                for val in line.split():
                    count+=1
                    if count == 3:
                        valErr=sqrt(float(val))
                        y0Err.append(valErr)
            if line.find("Z") != -1 and not "sigma" in line:
                count=0
                for val in line.split():
                    count+=1
                    if count > 1:
                        z0.append(float(val))
            if line.find("Cov(2,j)") != -1:
                count=0
                for val in line.split():
                    count+=1
                    if count == 4:
                        valErr=sqrt(float(val))
                        z0Err.append(valErr)
            if line.find("sigmaZ") != -1:
                count=0
                for val in line.split():
                    count+=1
                    if count > 1:
                        sigZ.append(float(val))
            if line.find("Cov(3,j)") != -1:
                count=0
                for val in line.split():
                    count+=1
                    if count == 5:
                        valErr=sqrt(float(val))
                        sigZErr.append(valErr)

    if verbose == "True":
        for i in range(len(x0)):
            print("     x0 = "+str(x0[i])+" +/- %1.8f (stats) [cm]" % (x0Err[i]))
            print("     y0 = "+str(y0[i])+" +/- %1.8f (stats) [cm]" % (y0Err[i]))
            print("     z0 = "+str(z0[i])+" +/- %1.6f (stats) [cm]" % (z0Err[i]))
            print("sigmaZ0 = "+str(sigZ[i])+" +/- %1.6f (stats) [cm]" % (sigZErr[i]))

    ## Make plots and save to root file
    rootFile = TFile("Summary.root","RECREATE");
    gROOT.SetStyle("Plain")
    
    hx0_lumi=plot_trending("x0",labels,x0,x0Err)
    hx0_lumi.SetTitle("x coordinate of beam spot vs. lumi")
    
    hy0_lumi=plot_trending("y0",labels,y0,y0Err)
    hy0_lumi.SetTitle("y coordinate of beam spot vs. lumi")

    hz0_lumi=plot_trending("z0",labels,z0,z0Err)
    hz0_lumi.SetTitle("z coordinate of beam spot vs. lumi")

    hsigZ_lumi=plot_trending("sigmaZ0",labels,sigZ,sigZErr)
    hsigZ_lumi.SetTitle("sigma z_{0} of beam spot vs. lumi")
    
    rootFile.Write();
    rootFile.Close();
    

#_________________________________    
if __name__ =='__main__':
    sys.exit(main())


