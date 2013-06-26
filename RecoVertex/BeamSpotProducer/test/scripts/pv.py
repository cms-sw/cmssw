#! /usr/bin/env python

#import ROOT
from ROOT import *
#gROOT, TFile, TCanvas, TH1F, TH1I, TLegend, TH2F, gPad

#from ROOT import TCanvas, TH1F, TH1I

import sys,os, math
sys.path.append( os.environ['HOME'])


try:
    from RooAlias import *
    SetStyle()
except:
    print "I cannot find RooAlias.py, ignore it and use plain style"
    gROOT.SetStyle('Plain')

gROOT.Reset()

gSystem.AddIncludePath(" -I$CMSSW_BASE/src/RecoVertex/BeamSpotProducer/interface");
gSystem.AddIncludePath(" -I$CMSSW_BASE/src");
#gSystem.AddLinkedLibs(" -L$CMSSW_BASE/lib/$SCRAM_ARCH -lRecoVertexBeamSpotProducer");

gROOT.SetMacroPath("$CMSSW_BASE/src/:.")

gROOT.ProcessLine(".L RecoVertex/BeamSpotProducer/interface/BeamSpotFitPVData.h+")
gROOT.ProcessLine(".L RecoVertex/BeamSpotProducer/src/BeamSpotTreeData.cc+")
gROOT.ProcessLine(".L BSVectorDict.h+")
gROOT.ProcessLine(".L RecoVertex/BeamSpotProducer/src/FcnBeamSpotFitPV.cc+")

class BeamSpot:
    def __init__(self):
        self.Type = -1
        self.X = 0.
        self.Xerr = 0.
        self.Y = 0.
        self.Yerr = 0.
        self.Z = 0.
        self.Zerr = 0.
        self.sigmaZ = 0.
        self.sigmaZerr = 0.
        self.dxdz = 0.
        self.dxdzerr = 0.
        self.dydz = 0.
        self.dydzerr = 0.
        self.beamWidthX = 0.
        self.beamWidthXerr = 0.
        self.beamWidthY = 0.
        self.beamWidthYerr = 0.
        self.EmittanceX = 0.
        self.EmittanceY = 0.
        self.betastar = 0.
        self.IOVfirst = 0
        self.IOVlast = 0
        self.IOVBeginTime = 0
        self.IOVEndTime = 0
        self.Run = 0
    def Reset(self):
        self.Type = -1
        self.X = self.Y = self.Z = 0.
        self.Xerr = self.Yerr = self.Zerr = 0.
        self.sigmaZ = self.sigmaZerr = 0.
        self.dxdz = self.dydz = 0.
        self.dxdzerr = self.dydzerr = 0.
        self.beamWidthX = self.beamWidthY = 0.
        self.beamWidthXerr = self.beamWidthYerr = 0.
        self.EmittanceX = self.EmittanceY = self.betastar = 0.
        self.IOVfirst = self.IOVlast = 0
        self.Run = 0

# 3D fit
def Fit3D( pvStore ):

    errorScale_ = 0.9
    sigmaCut_ = 5.0
    fbeamspot = BeamSpot()
    
    fcn = FcnBeamSpotFitPV(pvStore)
    minuitx = TFitterMinuit()
    minuitx.SetMinuitFCN(fcn)
 
    # fit parameters: positions, widths, x-y correlations, tilts in xz and yz
    minuitx.SetParameter(0,"x",0.,0.02,-10.,10.)
    minuitx.SetParameter(1,"y",0.,0.02,-10.,10.)
    minuitx.SetParameter(2,"z",0.,0.20,-30.,30.)
    minuitx.SetParameter(3,"ex",0.015,0.01,0.,10.)
    minuitx.SetParameter(4,"corrxy",0.,0.02,-1.,1.)
    minuitx.SetParameter(5,"ey",0.015,0.01,0.,10.)
    minuitx.SetParameter(6,"dxdz",0.,0.0002,-0.1,0.1)
    minuitx.SetParameter(7,"dydz",0.,0.0002,-0.1,0.1)
    minuitx.SetParameter(8,"ez",1.,0.1,0.,30.)
    minuitx.SetParameter(9,"scale",errorScale_,errorScale_/10.,errorScale_/2.,errorScale_*2.)

    # first iteration without correlations
    ierr = 0
    minuitx.FixParameter(4)
    minuitx.FixParameter(6)
    minuitx.FixParameter(7)
    minuitx.FixParameter(9)
    minuitx.SetMaxIterations(100)
    #       minuitx.SetPrintLevel(3)
    minuitx.SetPrintLevel(0)
    minuitx.CreateMinimizer()
    ierr = minuitx.Minimize()
    if ierr == 1:
	print "3D beam spot fit failed in 1st iteration"
	return (False, fbeamspot)
    
    # refit with harder selection on vertices
    
    fcn.setLimits(minuitx.GetParameter(0)-sigmaCut_*minuitx.GetParameter(3),
		   minuitx.GetParameter(0)+sigmaCut_*minuitx.GetParameter(3),
		   minuitx.GetParameter(1)-sigmaCut_*minuitx.GetParameter(5),
		   minuitx.GetParameter(1)+sigmaCut_*minuitx.GetParameter(5),
		   minuitx.GetParameter(2)-sigmaCut_*minuitx.GetParameter(8),
		   minuitx.GetParameter(2)+sigmaCut_*minuitx.GetParameter(8));
    ierr = minuitx.Minimize();
    if ierr == 1:
	print "3D beam spot fit failed in 2nd iteration"
	return (False, fbeamspot)
    
    # refit with correlations
    
    minuitx.ReleaseParameter(4);
    minuitx.ReleaseParameter(6);
    minuitx.ReleaseParameter(7);
    ierr = minuitx.Minimize();
    if ierr == 1:
	print "3D beam spot fit failed in 3rd iteration"
	return (False, fbeamspot)
    # store results

    fbeamspot.beamWidthX = minuitx.GetParameter(3);
    fbeamspot.beamWidthY = minuitx.GetParameter(5);
    fbeamspot.sigmaZ = minuitx.GetParameter(8);
    fbeamspot.beamWidthXerr = minuitx.GetParError(3);
    fbeamspot.beamWidthYerr = minuitx.GetParError(5);
    fbeamspot.sigmaZerr = minuitx.GetParError(8);
    fbeamspot.X = minuitx.GetParameter(0)
    fbeamspot.Y = minuitx.GetParameter(1)
    fbeamspot.Z = minuitx.GetParameter(2)
    fbeamspot.dxdz = minuitx.GetParameter(6)
    fbeamspot.dydz = minuitx.GetParameter(7)
    fbeamspot.Xerr = minuitx.GetParError(0)
    fbeamspot.Yerr = minuitx.GetParError(1)
    fbeamspot.Zerr = minuitx.GetParError(2)
    fbeamspot.dxdzerr = minuitx.GetParError(6)
    fbeamspot.dydzerr = minuitx.GetParError(7)

    return (True, fbeamspot)


def main():

    # input files
    tfile = TFile("BeamFit_LumiBased_NewAlignWorkflow_1042_139407_1.root")
    tfile.cd()
    
    fchain = ROOT.gDirectory.Get( 'PrimaryVertices' )
    entries = fchain.GetEntriesFast()
    
    aData = BeamSpotTreeData()
    
    aData.setBranchAddress(fchain);
    
    histox = {}
    histoy = {}
    histoz = {}
    histobx = TH1I("bx","bx",5000,0,5000)
    histolumi = TH1I("lumi","lumi",3000,0,3000)
    
#pvStore = []

    pvStore = ROOT.vector( BeamSpotFitPVData )(0)

    for jentry in xrange( entries ):
	# get the next tree in the chain
	ientry = fchain.LoadTree(jentry)
	if ientry < 0:
	    break

	# verify file/tree/chain integrity
	nb = fchain.GetEntry( jentry )
	if nb <= 0 or not hasattr( fchain, 'lumi' ):
	    continue

    
    #run = int( fchain.bunchCrossing )
    #lumi = int( fchain.lumi )
    #bx = int( fchain.bunchCrossing )
	run = aData.getRun()
	if ientry == 0 :
	    therun = run
	if run != therun:
	    print "FILES WITH DIFFERENT RUNS?? "+str(runt) + " and "+str(therun)
	    break

	pvdata = aData.getPvData()
    
	bx = int( pvdata.bunchCrossing )
	histobx.Fill(bx)
	lumi = aData.getLumi()
	histolumi.Fill(lumi)

	pvx = pvdata.position[0]
	pvy = pvdata.position[1]
	pvz = pvdata.position[2]

	if histox.has_key(bx) == False:
	    print "bx: "+str(bx)
	    histox[bx] = TH2F("x_"+str(bx),"x_"+str(bx),100,0,0.2,300,0,1500)#TH1F("x_"+str(bx),"x_"+str(bx),100,0,0.2)
	    histoy[bx] = TH1F("y_"+str(bx),"y_"+str(bx),100,0,0.2)
	    histoz[bx] = TH1F("z_"+str(bx),"z_"+str(bx),100,0,0.2)

	histox[bx].Fill(pvx,lumi)
	histoy[bx].Fill(pvy)
	histoz[bx].Fill(pvz)
    
    	pvStore.push_back( pvdata )
        #if ientry > 10:
	#break

    # fit
    results = Fit3D( pvStore )

    if results[0]:
        print "Results:"
        print " X = " +str(results[1].X)
        print "width X = " +str(results[1].beamWidthX)

    # plots
    cvbx = TCanvas("bx","bx",700,700)

    histobx.Draw()
    histobx.SetXTitle("bunch ID")

    cvlumi = TCanvas("lumi","lumi",700,700)
    
    histolumi.Draw()
    histolumi.SetXTitle("lumi section")
    
    cvx = TCanvas("cvx","cvx",700,700)

    i = 0
    for ibx in histox.keys():
	if i==0:
	    t1 = histox[ibx].ProjectionX("xall_"+str(ibx))
	    t1.Draw()
	else:
	    t1 = histox[ibx].ProjectionX("xall_"+str(ibx))
	    t1.Draw("same")

	i =+ 1
    
    raw_input ("Enter to quit:")


if __name__ == "__main__":
    main()




