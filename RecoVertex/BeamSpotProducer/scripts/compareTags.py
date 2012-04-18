#!/usr/bin/env python
#____________________________________________________________
#
#
# Script to compare values between tags
#
# Lorenzo Uplegger
# uplegger@fnal.gov
#
# Fermilab, 2011
#
#____________________________________________________________

"""
   plotBeamSpotDB

   A very simple script to plot the beam spot data stored in condDB

   usage: %prog -t <tag name>
   -x, --data1      = DATA  : File, directory, payload or tag with values
   -y, --data2      = DATA  : File, directory, payload or tag with values to compare
   -b, --begin      = BEGIN : First IOV. Options: run number, or run:lumi
   -e, --end        = END   : Last IOV. Options: run number, or run:lumi
   -g, --graph              : Create a TGraphError instead of a TH1 object
   -T, --Time               : Create plots with time axis.
   -P, --Print              : Create PNG plots from canvas.
   -s, --suffix     = SUFFIX: Suffix will be added to plots filename.
   -o, --output     = OUTPUT: Filename of ROOT file with plots.
   -w, --wait               : Pause script after plotting a new histograms.
   -f, --fill               : Plot vs. Fill #
   
   Lorenzo Uplegger (uplegger@fnal.gov)
   Fermilab 2010
   
"""


import os, string, re, sys, math
import commands, time
from BeamSpotObj import BeamSpot
from IOVObj import IOV
from CommonMethods import *

try:
    import ROOT
except:
    print "\nCannot load PYROOT, make sure you have setup ROOT in the path"
    print "and pyroot library is also defined in the variable PYTHONPATH, try:\n"
    if (os.getenv("PYTHONPATH")):
        print " setenv PYTHONPATH ${PYTHONPATH}:$ROOTSYS/lib\n"
    else:
        print " setenv PYTHONPATH $ROOTSYS/lib\n"
        sys.exit()

from ROOT import TFile, TGraphErrors, TGaxis, TDatime
from ROOT import TCanvas, TH1F

# ROOT STYLE
#############################
def SetStyle():

    # canvas
    ROOT.gStyle.SetCanvasBorderMode(0)
    ROOT.gStyle.SetCanvasColor(0)
    ROOT.gStyle.SetCanvasDefH(600)
    ROOT.gStyle.SetCanvasDefW(600)
    ROOT.gStyle.SetCanvasDefX(0)
    ROOT.gStyle.SetCanvasDefY(0)
    # pad
    ROOT.gStyle.SetPadBorderMode(0)
    ROOT.gStyle.SetPadColor(0)
    ROOT.gStyle.SetPadGridX(False)
    ROOT.gStyle.SetPadGridY(False)
    ROOT.gStyle.SetGridColor(0)
    ROOT.gStyle.SetGridStyle(3)
    ROOT.gStyle.SetGridWidth(1)
                  
    ROOT.gStyle.SetFrameBorderMode(0)
    ROOT.gStyle.SetFrameFillColor(0)
    ROOT.gStyle.SetTitleColor(1)
    ROOT.gStyle.SetStatColor(0)

    # set the paper & margin sizes
    ROOT.gStyle.SetPaperSize(20,26)
    ROOT.gStyle.SetPadTopMargin(0.04)
    ROOT.gStyle.SetPadRightMargin(0.04)
    ROOT.gStyle.SetPadBottomMargin(0.14)
    ROOT.gStyle.SetPadLeftMargin(0.11)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)

    ROOT.gStyle.SetTextFont(42) #132
    ROOT.gStyle.SetTextSize(0.09)
    ROOT.gStyle.SetLabelFont(42,"xyz")
    ROOT.gStyle.SetTitleFont(42,"xyz")
    ROOT.gStyle.SetLabelSize(0.035,"xyz")
    ROOT.gStyle.SetTitleSize(0.045,"xyz")
    ROOT.gStyle.SetTitleOffset(1.1,"y")

    # use bold lines and markers
    ROOT.gStyle.SetMarkerStyle(8)
    ROOT.gStyle.SetHistLineWidth(2)
    ROOT.gStyle.SetLineWidth(1)
    #ROOT.gStyle.SetLineStyleString(2,"[12 12]") // postscript dashes

    ROOT.gStyle.SetMarkerSize(0.6)
    
    # do not display any of the standard histogram decorations
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetOptStat(0) #("m")
    ROOT.gStyle.SetOptFit(0)
    
    #ROOT.gStyle.SetPalette(1,0)
    ROOT.gStyle.cd()
    ROOT.gROOT.ForceStyle()


#########################################
if __name__ == '__main__':

    # COMMAND LINE OPTIONS
    #################################
    option,args = parse(__doc__);
    if not args and not option: exit()
    
    data1 = option.data1;
    data2 = option.data2;
    
    if((data1 and data2) == False): 
	print "Provide the 2 sources to compare options --data1 and -data2";
	exit();

    firstLumi = "0:0"
    lastLumi = "4999999999:4999999999"
    
    if option.begin:
        firstLumi = option.begin
    if option.end:
        lastLumi = option.end

    IOVbase = "lumibase"

    tag1 = '';
    tag2 = '';
    DB1  = '';
    DB2  = '';
    auth = "/afs/cern.ch/cms/DB/conddb"
    if(data1.find(".db") != -1):
        DB1 = "sqlite_file:"+data1;
    elif(data1.find(".") == -1):
        DB1 = "frontier://PromptProd/CMS_COND_31X_BEAMSPOT"
    elif(data1.find(".txt") != -1):
        tag1 = data1[data1.rfind("BeamSpotObjects"):data1.rfind("_offline")];
        print tag1

    if(data2.find(".db") != -1):
        DB2 = "sqlite_file:"+data1;
    elif(data2.find(".") == -1):
        DB2 = "frontier://PromptProd/CMS_COND_31X_BEAMSPOT"
    elif(data2.find(".txt") != -1):
        tag2 = data2[data2.rfind("BeamSpotObjects"):data2.rfind("_offline")];
        print tag2

    
    beamSpots1 = []
    readBeamSpotFile(data1,beamSpots1,IOVbase,firstLumi,lastLumi)

    beamSpots2 = []
    readBeamSpotFile(data2,beamSpots2,IOVbase,firstLumi,lastLumi)

    #RunFillList = {}
    #readRunFillFile("Collisions11_runVSfill.txt",RunFillList)
    #print "Size of Run-Fill List is ",str(len(RunFillList))
    #
    # TEMPORARY

    if option.fill: 
        RunFillList = {}
        
        tmpfile = open("Collisions12_runVSfill.txt")
        #tmpfile.seek(0)
        
        for line in tmpfile:

            run = line.split()[0]
            fill = line.split()[1]
            RunFillList[int(run)] = fill
            #RunFillList[int(run)] = []
            #RunFillList[int(run)].append(fill)
            #print str(line),"   fill = ",fill,"   run = ",run

        tmpfile.close()

        print "Size of Run-Fill List is ",str(len(RunFillList))
            
        RunFillMap = RunFillList.keys();
        RunFillMap.sort();
        print "Size of Run-Fill Map is ",str(len(RunFillMap))

    
    #jBeam = BeamSpot()
    nLumis = 0;
    runsAndLumis = {};
    for beamSpot in beamSpots1 + beamSpots2:
        if(not int(beamSpot.Run) in runsAndLumis):
            runsAndLumis[int(beamSpot.Run)] = [];
        for lumi in range(int(beamSpot.IOVfirst),int(beamSpot.IOVlast)+1):
            if(not lumi in runsAndLumis[int(beamSpot.Run)]):
                runsAndLumis[int(beamSpot.Run)].append(lumi);
        nLumis += int(beamSpot.IOVlast)-int(beamSpot.IOVfirst)+1;
    print nLumis

    nLumis = 0;

    sortedRuns = runsAndLumis.keys();
    sortedRuns.sort();
    binMap = {};
    bin = 1;
    for run in sortedRuns:
        nLumis += len(runsAndLumis[run]);
        runsAndLumis[run].sort();
        binMap[run] = {}
        for lumi in runsAndLumis[run]:
            binMap[run][lumi] = bin;
            print str(run),":",str(lumi),":",str(bin)
            bin += 1;
        #print run
        #print runsAndLumis[run];
    print "Total number of lumis:", nLumis;
    #print binMap;
#        for beamSpot2 in beamSpots2:
#            if (int(beamSpot1.Run) == int(beamSpot2.Run) and int(beamSpot2.IOVfirst) <= int(beamSpot1.IOVfirst) and int(beamSpot1.IOVfirst) <= int(beamSpot2.IOVlast)):
#                print beamSpot1.Run, beamSpot1.IOVfirst, beamSpot1.IOVlast
#                print beamSpot2.Run, beamSpot2.IOVfirst, beamSpot2.IOVlast
#                print







    # MAKE PLOTS
    ###################################    
    TGaxis.SetMaxDigits(8)

    graphlist = []
    graphnamelist = ['X','Y','Z','SigmaZ','dxdz','dydz','beamWidthX', 'beamWidthY']
    graphtitlelist = ['beam spot X','beam spot Y','beam spot Z','beam spot #sigma_Z','beam spot dX/dZ','beam spot dY/dZ','beam width X','beam width Y']
    graphXaxis = 'Run number'
    if IOVbase == 'runbase':
        graphXaxis = "Run number"
    if IOVbase == 'lumibase':
#        graphXaxis = 'Lumi section'
        graphXaxis = 'Run number' # only plotting first entry per run
    if IOVbase == 'timebase' or option.Time:
        graphXaxis = "Time"
        #dh = ROOT.TDatime(2010,06,01,00,00,00)
        ROOT.gStyle.SetTimeOffset(0) #dh.Convert())
    if option.fill:
        graphXaxis = 'Fill number'
                    
    graphYaxis = ['beam spot X [cm]','beam spot Y [cm]','beam spot Z [cm]', 'beam spot #sigma_{Z} [cm]', 'beam spot dX/dZ', 'beam spot dY/dZ','beam width X [cm]', 'beam width Y [cm]']

    cvlist = []
    counter = 1;
    for ig in range(0,8):
	#cvlist.append( TCanvas(graphnamelist[ig],graphtitlelist[ig], 1200, 600) )
	if option.graph:
	    graphlist.append( TGraphErrors( nLumis ) )
	    graphlist.append( TGraphErrors( nLumis ) )
        else:
	    graphlist.append( TH1F("name","title",nLumis,0,nLumis) )
	    graphlist.append( TH1F("name2","title2",nLumis,0,nLumis) )
        
	graphlist[2*ig]  .SetName (graphnamelist[ig]  + " " + tag1)
        graphlist[2*ig]  .SetTitle(graphtitlelist[ig] + " " + tag1)
	graphlist[2*ig+1].SetName (graphnamelist[ig]  + " " + tag2)
        graphlist[2*ig+1].SetTitle(graphtitlelist[ig] + " " + tag2)

    counter = 1;
    for b in range(2):
        if(b==0):
            beamSpots = beamSpots1;
        else:
            beamSpots = beamSpots2;
        for beamSpot in beamSpots:
            #print "Filling beamspot:", counter, "/", len(beamSpots1)+len(beamSpots2)
            counter += 1
            #print beamSpot.IOVfirst, ":" , beamSpot.IOVlast

            for lumi in range(int(beamSpot.IOVfirst),int(beamSpot.IOVlast)+1):
                run = int(beamSpot.Run);
                for ig in range(0,8):
                    if graphnamelist[ig] == 'X':
                        datay = beamSpot.X
                        datayerr = beamSpot.Xerr
                    if graphnamelist[ig] == 'Y':
                        datay = beamSpot.Y
                        datayerr = beamSpot.Yerr
                    if graphnamelist[ig] == 'Z':
                        datay = beamSpot.Z
                        datayerr = beamSpot.Zerr
                    if graphnamelist[ig] == 'SigmaZ':
                        datay = beamSpot.sigmaZ
                        datayerr = beamSpot.sigmaZerr
                    if graphnamelist[ig] == 'dxdz':
                        datay = beamSpot.dxdz
                        datayerr = beamSpot.dxdzerr
                    if graphnamelist[ig] == 'dydz':
                        datay = beamSpot.dydz
                        datayerr = beamSpot.dydzerr
                    if graphnamelist[ig] == 'beamWidthX':
                        datay = beamSpot.beamWidthX
                        datayerr = beamSpot.beamWidthXerr
                    if graphnamelist[ig] == 'beamWidthY':
                        datay = beamSpot.beamWidthY
                        datayerr = beamSpot.beamWidthYerr

	            if option.graph:
		    	#print binMap[run][lumi], ":" , float(datay)
                        graphlist[2*ig+b].SetPoint(binMap[run][lumi], binMap[run][lumi]-0.5, float(datay) )
                        graphlist[2*ig+b].SetPointError(binMap[run][lumi], 0.5, float(datayerr) )
                    else:
		    	#print binMap[run][lumi], ":" , float(datay)
                        graphlist[2*ig+b].SetBinContent(binMap[run][lumi], float(datay) )
                        graphlist[2*ig+b].SetBinError(binMap[run][lumi], float(datayerr) )

    for ig in range(0,8):
        oldRun = -1
        oldFill = -1
        sortedRunLumis = binMap.keys()
        sortedRunLumis.sort()
        for run in sortedRunLumis:
            print "Labling run:", counter,"/",len(binMap)*8,"/",len(binMap[run])
            counter += 1
            if option.fill:
                print "Old run = ",str(oldRun),"   New run = ",str(run),"  OldFill = ",str(oldFill),"   NewFill = ",str(RunFillList[run])
                #
                if (oldFill == RunFillList[run]):
                    oldRun = run
                    continue
                print "===> PRINT NEW FILL"
                oldRun = run 
                oldFill = RunFillList[run]
                print "Run = ",str(run),"   Fill = ",str(RunFillList[run])
            sortedLumis = binMap[run].keys()
            sortedLumis.sort()
#            for lumi in binMap[run]:
            for lumi in sortedLumis:
#                datax = str(run) + ":" + str(lumi)
                datax = str(run) # only plotting 1 per run, so just use run num to improve readability
                if option.fill:
                    datax = str(RunFillList[run])
                #datax = str(run)
                print "data x = ", str(datax),"/",len(binMap[run])
                graphlist[2*ig]  .GetXaxis().SetBinLabel(binMap[run][lumi] , str(datax) )
                graphlist[2*ig+1].GetXaxis().SetBinLabel(binMap[run][lumi] , str(datax) )

                if(len(binMap[run]) > 0):
                   break ;
        #graphlist[2*ig].Draw('P E1 X0')
        #graphlist[2*ig+1].SetMarkerStyle(4)
        graphlist[2*ig+1].SetMarkerColor(2)
        graphlist[2*ig+1].SetLineColor(2)
        #graphlist[2*ig+1].Draw('P E1 X0 SAME')
        graphlist[2*ig].GetXaxis().SetTitle(graphXaxis)
        graphlist[2*ig].GetYaxis().SetTitle(graphYaxis[ig])
        #graphlist[2*ig].Fit('pol1')
	#cvlist[ig].Update()
        #cvlist[ig].SetGrid()
        if option.Print:
            suffix = ''
            if option.suffix:
                suffix = option.suffix
	    cvlist[ig].Print(graphnamelist[ig]+"_"+suffix+".png")
        if option.wait:
            raw_input( 'Press ENTER to continue\n ' )
        #graphlist[0].Print('all')

    if option.output:
        outroot = TFile(option.output,"RECREATE")
        for ig in graphlist:
            ig.Write()

        outroot.Close()
        print " plots have been written to "+option.output
