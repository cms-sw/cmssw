#!/usr/bin/env python
#____________________________________________________________
#
#
# A very simple way to make plots with ROOT via an XML file
#
# Francisco Yumiceva
# yumiceva@fnal.gov
#
# Fermilab, 2010
#
#____________________________________________________________

"""
   plotBeamSpotDB

   A very simple script to plot the beam spot data stored in condDB

   usage: %prog -t <tag name>
   -b, --batch : Run ROOT in batch mode.
   -c, --create  = CREATE: name for beam spot data file.
   -d, --data    = DATA: input beam spot data file.
   -i, --initial = INITIAL: First run.
   -f, --final   = FINAL: Last run.
   -n, --noplot : Only extract beam spot data, plots are not created..
   -o, --output  = OUTPUT: filename of ROOT file with plots.
   -t, --tag     = TAG: tag name.
   -w, --wait : Pause script after plotting a new histograms.
   
   Francisco Yumiceva (yumiceva@fnal.gov)
   Fermilab 2010
   
"""


import os, string, re, sys, math
import commands

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

from ROOT import TFile, TGraphErrors, TGaxis
from ROOT import TCanvas

#_______________OPTIONS________________
import optparse

USAGE = re.compile(r'(?s)\s*usage: (.*?)(\n[ \t]*\n|$)')

def nonzero(self): # will become the nonzero method of optparse.Values
    "True if options were given"
    for v in self.__dict__.itervalues():
        if v is not None: return True
    return False

optparse.Values.__nonzero__ = nonzero # dynamically fix optparse.Values

class ParsingError(Exception): pass

optionstring=""

def exit(msg=""):
    raise SystemExit(msg or optionstring.replace("%prog",sys.argv[0]))

def parse(docstring, arglist=None):
    global optionstring
    optionstring = docstring
    match = USAGE.search(optionstring)
    if not match: raise ParsingError("Cannot find the option string")
    optlines = match.group(1).splitlines()
    try:
        p = optparse.OptionParser(optlines[0])
        for line in optlines[1:]:
            opt, help=line.split(':')[:2]
            short,long=opt.split(',')[:2]
            if '=' in opt:
                action='store'
                long=long.split('=')[0]
            else:
                action='store_true'
            p.add_option(short.strip(),long.strip(),
                         action = action, help = help.strip())
    except (IndexError,ValueError):
        raise ParsingError("Cannot parse the option string correctly")
    return p.parse_args(arglist)

#__________END_OPTIONS_______________________________________________

class BeamSpot:
    def __init__(self):
        self.type = ""
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
	self.beamWdithXerr = 0.
        self.beamWidthY = 0.
	self.beamWidthYerr = 0.
	self.EmittanceX = 0.
	self.EmittanceY = 0.
	self.betastar = 0.
	self.IOVfirst = 0
	self.IOVlast = 0
    def Reset(self):
	self.X = self.Y = self.Z = 0.
	self.Xerr = self.Yerr = self.Zerr = 0.
	self.sigmaZ = self.sigmaZerr = 0.
	self.dxdz = self.dydz = 0.
	self.dxdzerr = self.dydzerr = 0.
	self.beamWidthX = self.beamWidthY = 0.
	self.beamWidthXerr = self.beamWidthYerr = 0.
	self.EmittanceX = self.EmittanceY = self.betastar = 0.
	self.IOVfirst = self.IOVlast = 0

class IOV:
    def __init__(self):
	self.since = 1
	self.till = 1

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
    ROOT.gStyle.SetPadLeftMargin(0.16)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)

    ROOT.gStyle.SetTextFont(42) #132
    ROOT.gStyle.SetTextSize(0.09)
    ROOT.gStyle.SetLabelFont(42,"xyz")
    ROOT.gStyle.SetTitleFont(42,"xyz")
    ROOT.gStyle.SetLabelSize(0.035,"xyz")
    ROOT.gStyle.SetTitleSize(0.045,"xyz")
    ROOT.gStyle.SetTitleOffset(1.5,"y")

    # use bold lines and markers
    ROOT.gStyle.SetMarkerStyle(8)
    ROOT.gStyle.SetHistLineWidth(2)
    ROOT.gStyle.SetLineWidth(1)
    #ROOT.gStyle.SetLineStyleString(2,"[12 12]") // postscript dashes

    # do not display any of the standard histogram decorations
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetOptStat(0) #("m")
    ROOT.gStyle.SetOptFit(0)
    
    #ROOT.gStyle.SetPalette(1,0)
    ROOT.gStyle.cd()
    ROOT.gROOT.ForceStyle()
#########################################

if __name__ == '__main__':

    # style
    SetStyle()
    
    printCanvas = False
    printFormat = "png"
    printBanner = False
    Banner = "CMS Preliminary"

    # COMMAND LINE OPTIONS
    #################################
    option,args = parse(__doc__)
    if not args and not option: exit()
    
    tag = ''
    if not option.tag and not option.data: 
	print " need to provide DB tag name or beam spot data file"
	exit()
    else:
	tag = option.tag

    if option.batch:
        ROOT.gROOT.SetBatch()
        
    datafilename = "tmp_beamspot.dat"
    if option.create:
        datafilename = option.create
    
    getDBdata = True
    if option.data:
        getDBdata = False
    
    firstRun = 1
    lastRun  = 4999999999

    if option.initial:
        firstRun = int(option.initial)
    if option.final:
        lastRun = int(option.final)
    
                                                
    # GET IOVs
    ################################

    if getDBdata:

        print " read DB to get list of IOVs for the given tag"
        acommand = 'cmscond_list_iov -c frontier://PromptProd/CMS_COND_31X_BEAMSPOT -P /afs/cern.ch/cms/DB/conddb -t '+ tag
        tmpstatus = commands.getstatusoutput( acommand )
        tmplistiov = tmpstatus[1].split('\n')
        #print tmplistiov

        iovlist = []
        passline = False
        iline = jline = 0
        totlines = len(tmplistiov)
        for line in tmplistiov:
	
            if line.find('since') != -1:
                passline = True
                jline = iline
            if passline and iline > jline and iline < totlines-1:
                linedata = line.split()
	        #print linedata
                aIOV = IOV()
                aIOV.since = int(linedata[0])
                aIOV.till = int(linedata[1])
                iovlist.append( aIOV )
            iline += 1
    
        print " total number of IOVs = " + str(len(iovlist))


        #  GET DATA
        ################################

        print " get beam spot data from DB for IOVs. This can take a few minutes ..."

        tmpfile = open(datafilename,'w')

        for iIOV in iovlist:
            passiov = False
	    #print "since = " + str(iIOV.since) + " till = "+ str(iIOV.till)
            if iIOV.since >= firstRun and lastRun < 0 and iIOV.since <= firstRun:
                print " IOV: " + str(iIOV.since)
                passiov = True
            if iIOV.since >= firstRun and lastRun > 0 and iIOV.till < lastRun:
                print " IOV: " + str(iIOV.since) + " to " + str(iIOV.till)
                passiov = True
            if passiov:
                acommand = 'getBeamSpotDB.py '+ tag + " " + str(iIOV.since)
                status = commands.getstatusoutput( acommand )
                tmpfile.write(status[1])
    
        print " beam spot data collected and stored in file " + datafilename
    
        tmpfile.close()


    if option.noplot:
        print " no plots requested, exit now."
        sys.exit()

    
    # PROCESS DATA
    ###################################

    # check if input data exists if given
    if option.data:
        if os.path.exists(option.data):
            datafilename = option.data
        else:
            print " input beam spot data DOES NOT exist, file " + option.data
            exit()
    
    tmpfile = open(datafilename)

    listbeam = []
    tmpbeam = BeamSpot()
    tmpbeamsize = 0

    for line in tmpfile:
	
	if line.find('X0') != -1:
	    tmpbeam.X = line.split()[2]
	    tmpbeam.Xerr = line.split()[4]
	    tmpbeamsize += 1
            #print " x = " + str(tmpbeam.X)
	if line.find('Y0') != -1:
	    tmpbeam.Y = line.split()[2]
	    tmpbeam.Yerr = line.split()[4]
	    tmpbeamsize += 1
            #print " y =" + str(tmpbeam.Y)
	if line.find('Z0') != -1 and line.find('Sigma Z0') == -1:
	    tmpbeam.Z = line.split()[2]
	    tmpbeam.Zerr = line.split()[4]
	    tmpbeamsize += 1
            #print " z =" + str(tmpbeam.Z)
	if line.find('Sigma Z0') !=-1:
	    tmpbeam.sigmaZ = line.split()[3]
	    tmpbeam.sigmaZerr = line.split()[5]
	    tmpbeamsize += 1
	if line.find('dxdz') != -1:
	    tmpbeam.dxdz = line.split()[2]
	    tmpbeam.dxdzerr = line.split()[4]
	    tmpbeamsize += 1
	if line.find('dydz') != -1:
	    tmpbeam.dydz = line.split()[2]
	    tmpbeam.dydzerr = line.split()[4]
	    tmpbeamsize += 1
	if line.find('Beam Width X') != -1:
	    tmpbeam.beamWidthX = line.split()[4]
	    tmpbeam.beamWidthXerr = line.split()[6]
	    tmpbeamsize += 1
	if line.find('Beam Width Y') != -1:
	    tmpbeam.beamWidthY = line.split()[4]
	    tmpbeam.beamWidthYerr = line.split()[6]
	    tmpbeamsize += 1
	#if line.find('Run ') != -1:
        if line.find('for runs')  != -1:
	    #tmpbeam.IOVfirst = line.split()[6].strip(',')
            tmpbeam.IOVfirst = line.split()[2]
            tmpbeam.IOVlast = line.split()[4]
	    tmpbeamsize += 1
	if tmpbeamsize == 9:
            #print " from object " + str(tmpbeam.X)
            if int(tmpbeam.IOVfirst) >= firstRun and int(tmpbeam.IOVlast) <= lastRun:
                listbeam.append(tmpbeam)
            tmpbeamsize = 0
	    tmpbeam = BeamSpot()
	    
    print " got total number of IOVs = " + str(len(listbeam)) + " from file "+datafilename
    #print " run " + str(listbeam[3].IOVfirst ) + " " + str( listbeam[3].X )

    # MAKE PLOTS
    ###################################
   
    TGaxis.SetMaxDigits(8)

    graphlist = []
    graphnamelist = ['X','Y','Z','SigmaZ','dxdz','dydz']
    graphtitlelist = ['beam spot X','beam spot Y','beam spot Z','beam spot #sigma_Z','beam spot dX/dZ','beam spot dY/dZ']
    graphXaxis = 'Run number'
    graphYaxis = ['beam spot X [cm]','beam spot Y [cm]','beam spot Z [cm]', 'beam spot #sigma_Z [cm]', 'beam spot dX/dZ', 'beam spot dY/dZ']

    cvlist = []

    for ig in range(0,6):
	cvlist.append( TCanvas(graphnamelist[ig],graphtitlelist[ig], 800, 600) )
	graphlist.append( TGraphErrors( len(listbeam) ) )
	graphlist[ig].SetName(graphnamelist[ig])
        graphlist[ig].SetTitle(graphnamelist[ig])
	ipoint = 0
	for ii in range(0,len(listbeam)):
	    
	    ibeam = listbeam[ii]
	    datax = dataxerr = 0.
	    datay = datayerr = 0.
	    if graphnamelist[ig] == 'X':
		datay = ibeam.X
		datayerr = ibeam.Xerr
	    if graphnamelist[ig] == 'Y':
		datay = ibeam.Y
		datayerr = ibeam.Yerr
	    if graphnamelist[ig] == 'Z':
		datay = ibeam.Z
		datayerr = ibeam.Zerr
	    if graphnamelist[ig] == 'SigmaZ':
		datay = ibeam.sigmaZ
		datayerr = ibeam.sigmaZerr
	    if graphnamelist[ig] == 'dxdz':
		datay = ibeam.dxdz
		datayerr = ibeam.dxdzerr
	    if graphnamelist[ig] == 'dydz':
		datay = ibeam.dydz
		datayerr = ibeam.dydzerr
	    
	    datax = ibeam.IOVfirst
	    if datax == '1':
		print " skip in plot IOV = "+ str(ibeam.IOVfirst) + " to " + str(ibeam.IOVlast)
		graphlist[ig].Set( len(listbeam) -1 )
		continue
	    #print str(ipoint) + " x = " + str(datax) + " y = " + str(datay)
	    graphlist[ig].SetPoint(ipoint, float(datax), float(datay) )
	    graphlist[ig].SetPointError(ipoint, float(dataxerr), float(datayerr) )
	    ipoint += 1

	
	graphlist[ig].Draw('AP')
	graphlist[ig].GetXaxis().SetTitle(graphXaxis)
	graphlist[ig].GetYaxis().SetTitle(graphYaxis[ig])
	cvlist[ig].Update()
        if option.wait:
            raw_input( 'Press ENTER to continue\n ' )
        #graphlist[0].Print('all')

    if option.output:
        outroot = TFile(option.output,"RECREATE")
        for ig in graphlist:
            ig.Write()

        outroot.Close()
        print " plots have been written to "+option.output
        
    

    # CLEAN temporal files
    ###################################
    #os.system('rm tmp_beamspotdata.log')
