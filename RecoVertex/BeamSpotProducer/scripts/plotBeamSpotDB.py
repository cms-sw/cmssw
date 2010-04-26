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
   -a, --auth    = AUTH: DB authorization path. online(/nfshome0/popcondev/conddb).
   -b, --batch : Run ROOT in batch mode.
   -c, --create  = CREATE: name for beam spot data file.
   -d, --data    = DATA: input beam spot data file.
   -D, --destDB  = DESTDB: destination DB string. online(oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT).
   -i, --initial = INITIAL: First IOV. Options: run number, or run:lumi, eg. \"133200:21\"
   -f, --final   = FINAL: Last IOV. Options: run number, or run:lumi
   -g, --graph : create a TGraphError instead of a TH1 object
   -n, --noplot : Only extract beam spot data, plots are not created..
   -o, --output  = OUTPUT: filename of ROOT file with plots.
   -p, --payload = PAYLOAD: filename of output text file. Combine and splits lumi IOVs.
   -P, --Print : create PNG plots from canvas.
   -t, --tag     = TAG: Database tag name.
   -I, --IOVbase = IOVBASE: options: runbase(default), lumibase, timebase
   -w, --wait : Pause script after plotting a new histograms.
   -W, --weighted : Create a weighted result for a range of lumi IOVs, skip lumi IOV combination and splitting.
   
   Francisco Yumiceva (yumiceva@fnal.gov)
   Fermilab 2010
   
"""


import os, string, re, sys, math
import commands, time

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
from ROOT import TCanvas, TH1F

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

def cmp_list_run(a,b):
    if int(a.IOVfirst) < int(b.IOVfirst): return -1
    if int(a.IOVfirst) == int(b.IOVfirst): return 0
    if int(a.IOVfirst) > int(b.IOVfirst): return 1

def cmp_list_lumi(a,b):
    if int(a.Run) < int(b.Run): return -1
    if int(a.Run) == int(b.Run):
	if int(a.IOVfirst) < int(b.IOVfirst): return -1
	if int(a.IOVfirst) == int(b.IOVfirst): return 0
	if int(a.IOVfirst) > int(b.IOVfirst): return 1
    if int(a.Run) > int(b.Run) : return 1

def weight(x1, x1err,x2,x2err):
    #print "x1 = "+str(x1)+" +/- "+str(x1err)+" x2 = "+str(x2)+" +/- "+str(x2err)
    x1 = float(x1)
    x1err = float(x1err)
    x2 = float(x2)
    x2err = float(x2err)
    tmperr = 0.
    if x1err < 1e-6:
	x1 = x2/(x2err * x2err)
	tmperr = 1/(x2err*x2err)
    else:
	x1 = x1/(x1err*x1err) + x2/(x2err * x2err)
	tmperr = 1/(x1err*x1err) + 1/(x2err*x2err)
    x1 = x1/tmperr
    x1err = 1/tmperr
    x1err = math.sqrt(x1err)
    return (str(x1), str(x1err))

#__________END_OPTIONS_______________________________________________

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
class IOV:
    def __init__(self):
	self.since = 1
	self.till = 1
        self.RunFirst = 1
        self.RunLast  = 1
        self.LumiFirst = 1
        self.LumiLast = 1

def dump( beam, file):
    end = "\n"
    file.write("Runnumber "+beam.Run+end)
    file.write("BeginTimeOfFit "+str(beam.IOVBeginTime)+end)
    file.write("EndTimeOfFit "+str(beam.IOVEndTime)+end)
    file.write("LumiRange "+str(beam.IOVfirst)+" - "+str(beam.IOVlast)+end)
    file.write("Type "+str(beam.Type)+end)
    file.write("X0 "+str(beam.X)+end)
    file.write("Y0 "+str(beam.Y)+end)
    file.write("Z0 "+str(beam.Z)+end)
    file.write("sigmaZ0 "+str(beam.sigmaZ)+end)
    file.write("dxdz "+str(beam.dxdz)+end)
    file.write("dydz "+str(beam.dydz)+end)
    file.write("BeamWidthX "+beam.beamWidthX+end)
    file.write("BeamWidthY "+beam.beamWidthY+end)
    file.write("Cov(0,j) "+str(math.pow(float(beam.Xerr),2))+" 0 0 0 0 0 0"  +end)
    file.write("Cov(1,j) 0 "+str(math.pow(float(beam.Yerr),2))+" 0 0 0 0 0"  +end)
    file.write("Cov(2,j) 0 0 "+str(math.pow(float(beam.Zerr),2))+" 0 0 0 0"  +end)
    file.write("Cov(3,j) 0 0 0 "+str(math.pow(float(beam.sigmaZerr),2))+" 0 0 0"  +end)
    file.write("Cov(4,j) 0 0 0 0 "+str(math.pow(float(beam.dxdzerr),2))+" 0 0"  +end)
    file.write("Cov(5,j) 0 0 0 0 0 "+str(math.pow(float(beam.dydzerr),2))+" 0"  +end)
    file.write("Cov(6,j) 0 0 0 0 0 0 "+str(math.pow(float(beam.beamWidthXerr),2))  +end)
    file.write("EmittanceX 0"+end)
    file.write("EmittanceY 0"+end)
    file.write("BetaStar 0"+end)

def delta(x,xerr,nextx,nextxerr):
    return math.fabs( float(x) - float(nextx) )/math.sqrt(math.pow(float(xerr),2) + math.pow(float(nextxerr),2))

# lumi tools CondCore/Utilities/python/timeUnitHelper.py
def pack(high,low):
    """pack high,low 32bit unsigned int to one unsigned 64bit long long
       Note:the print value of result number may appear signed, if the sign bit is used.
    """
    h=high<<32
    return (h|low)

def unpack(i):
    """unpack 64bit unsigned long long into 2 32bit unsigned int, return tuple (high,low)
    """
    high=i>>32
    low=i&0xFFFFFFFF
    return(high,low)

def unpackLumiid(i):
    """unpack 64bit lumiid to dictionary {'run','lumisection'}
    """
    j=unpack(i)
    return {'run':j[0],'lumisection':j[1]}


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
   
    IOVbase = 'runbase'
    if option.IOVbase:
        if option.IOVbase != "runbase" and option.IOVbase != "lumibase" and option.IOVbase != "timebase":
            print "\n\n unknown iov base option: "+ option.IOVbase +" \n\n\n"
            exit()
	IOVbase = option.IOVbase
    
    firstRun = "1"
    lastRun  = "4999999999"
    if IOVbase == "lumibase":
	firstRun = "1:1"
	lastRun = "4999999999:4999999999"
    
    if option.initial:
        firstRun = option.initial
    if option.final:
        lastRun = option.final
        
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

        otherArgs = ''
        if option.destDB:
            otherArgs = " -d " + option.destDB
            if option.auth:
                otherArgs = otherArgs + " -a "+ option.auth
        
        print " get beam spot data from DB for IOVs. This can take a few minutes ..."

        tmpfile = open(datafilename,'w')

        for iIOV in iovlist:
            passiov = False
            tmprunfirst = firstRun
            tmprunlast = lastRun
            tmplumifirst = 1
            tmplumilast = 9999999
            if IOVbase=="lumibase":
                #tmprunfirst = int(firstRun.split(":")[0])
                #tmprunlast  = int(lastRun.split(":")[0])
                #tmplumifirst = int(firstRun.split(":")[1])
                #tmplumilast  = int(lastRun.split(":")[1])
                tmprunfirst = pack( int(firstRun.split(":")[0]) , int(firstRun.split(":")[1]) )
                tmprunlast  = pack( int(lastRun.split(":")[0]) , int(lasstRun.split(":")[1]) )
	    #print "since = " + str(iIOV.since) + " till = "+ str(iIOV.till)
            if iIOV.since >= int(tmprunfirst) and int(tmprunlast) < 0 and iIOV.since <= int(tmprunfirst):
                print " IOV: " + str(iIOV.since)
                passiov = True
            if iIOV.since >= int(tmprunfirst) and int(tmprunlast) > 0 and iIOV.till < int(tmprunlast):
                print " IOV: " + str(iIOV.since) + " to " + str(iIOV.till)
                passiov = True
            if passiov:
                acommand = 'getBeamSpotDB.py -t '+ tag + " -r " + str(iIOV.since) +otherArgs
                if IOVbase=="lumibase":
                    tmprun = unpack(iIOV.since)[0]
                    tmplumi = unpack(iIOV.since)[1]
                    acommand = 'getBeamSpotDB.py -t '+ tag + " -r " + str(tmprun) +" -l "+tmplumi +otherArgs
                status = commands.getstatusoutput( acommand )
                tmpfile.write(status[1])
    
        print " beam spot data collected and stored in file " + datafilename
    
        tmpfile.close()

    
    # PROCESS DATA
    ###################################

    # check if input data exists if given
    if option.data:
        if os.path.isdir(option.data):
            tmp = commands.getstatusoutput("ls "+option.data)
            files = tmp[1].split()
            datafilename = "combined_all.txt"
            output = open(datafilename,"w")
            
            for f in files:
                input = open(option.data +"/"+f)
                output.writelines(input.readlines())
            output.close()
            print " data files have been collected in "+datafilename
            
        elif os.path.exists(option.data):
            datafilename = option.data
        else:
            print " input beam spot data DOES NOT exist, file " + option.data
            exit()
    
    tmpfile = open(datafilename)

    listbeam = []
    tmpbeam = BeamSpot()
    tmpbeamsize = 0

    inputfiletype = 0
    if tmpfile.readline().find('Runnumber') != -1:
	inputfiletype = 1
    tmpfile.seek(0)

    if inputfiletype ==1:
	
	for line in tmpfile:

	    if line.find('Type') != -1:
		tmpbeam.Type = int(line.split()[1])
		tmpbeamsize += 1
	    if line.find('X0') != -1:
		tmpbeam.X = line.split()[1]
		#tmpbeam.Xerr = line.split()[4]
		tmpbeamsize += 1
            #print " x = " + str(tmpbeam.X)
	    if line.find('Y0') != -1:
		tmpbeam.Y = line.split()[1]
		#tmpbeam.Yerr = line.split()[4]
		tmpbeamsize += 1
            #print " y =" + str(tmpbeam.Y)
	    if line.find('Z0') != -1 and line.find('sigmaZ0') == -1:
		tmpbeam.Z = line.split()[1]
		#tmpbeam.Zerr = line.split()[4]
		tmpbeamsize += 1
	    if line.find('sigmaZ0') !=-1:
		tmpbeam.sigmaZ = line.split()[1]
		#tmpbeam.sigmaZerr = line.split()[5]
		tmpbeamsize += 1
            if line.find('dxdz') != -1:
		tmpbeam.dxdz = line.split()[1]
		#tmpbeam.dxdzerr = line.split()[4]
		tmpbeamsize += 1
	    if line.find('dydz') != -1:
		tmpbeam.dydz = line.split()[1]
		#tmpbeam.dydzerr = line.split()[4]
		tmpbeamsize += 1
	    if line.find('BeamWidthX') != -1:
		tmpbeam.beamWidthX = line.split()[1]
		#tmpbeam.beamWidthXerr = line.split()[6]
		tmpbeamsize += 1
	    if line.find('BeamWidthY') != -1:
		tmpbeam.beamWidthY = line.split()[1]
		#tmpbeam.beamWidthYerr = line.split()[6]
		tmpbeamsize += 1
	    if line.find('Cov(0,j)') != -1:
		tmpbeam.Xerr = str(math.sqrt( float( line.split()[1] ) ) )
		tmpbeamsize += 1
	    if line.find('Cov(1,j)') != -1:
		tmpbeam.Yerr = str(math.sqrt( float( line.split()[2] ) ) )
		tmpbeamsize += 1
	    if line.find('Cov(2,j)') != -1:
		tmpbeam.Zerr = str(math.sqrt( float( line.split()[3] ) ) )
		tmpbeamsize += 1
	    if line.find('Cov(3,j)') != -1:
		tmpbeam.sigmaZerr = str(math.sqrt( float( line.split()[4] ) ) )
		tmpbeamsize += 1
            if line.find('Cov(4,j)') != -1:
		tmpbeam.dxdzerr = str(math.sqrt( float( line.split()[5] ) ) )
		tmpbeamsize += 1
	    if line.find('Cov(5,j)') != -1:
		tmpbeam.dydzerr = str(math.sqrt( float( line.split()[6] ) ) )
		tmpbeamsize += 1
	    if line.find('Cov(6,j)') != -1:
		tmpbeam.beamWidthXerr = str(math.sqrt( float( line.split()[7] ) ) )
                tmpbeam.beamWidthYerr = tmpbeam.beamWidthXerr
		tmpbeamsize += 1
	    if line.find('LumiRange')  != -1 and IOVbase=="lumibase":
	    #tmpbeam.IOVfirst = line.split()[6].strip(',')
		tmpbeam.IOVfirst = line.split()[1]
		tmpbeam.IOVlast = line.split()[3]
		tmpbeamsize += 1
            if line.find('Runnumber') != -1:
		tmpbeam.Run = line.split()[1]
		if IOVbase == "runbase":
		    tmpbeam.IOVfirst = line.split()[1]
		    tmpbeam.IOVlast = line.split()[1]
		tmpbeamsize += 1
            if line.find('BeginTimeOfFit') != -1:
		tmpbeam.IOVBeginTime = line.split()[1] +" "+line.split()[2] +" "+line.split()[3]
		if IOVbase =="timebase":
		    tmpbeam.IOVfirst =  time.mktime( time.strptime(line.split()[1] +  " " + line.split()[2] + " " + line.split()[3],"%Y.%m.%d %H:%M:%S %Z") )
		tmpbeamsize += 1
            if line.find('EndTimeOfFit') != -1:
		tmpbeam.IOVEndTime = line.split()[1] +" "+line.split()[2] +" "+line.split()[3]
		if IOVbase =="timebase":
		    tmpbeam.IOVlast = time.mktime( time.strptime(line.split()[1] +  " " + line.split()[2] + " " + line.split()[3],"%Y.%m.%d %H:%M:%S %Z") )
		tmpbeamsize += 1
	    if tmpbeamsize == 20:
		if IOVbase=="lumibase":
		    tmprunfirst = int(firstRun.split(":")[0])
		    tmprunlast  = int(lastRun.split(":")[0])
		    tmplumifirst = int(firstRun.split(":")[1])
		    tmplumilast  = int(lastRun.split(":")[1])
		    acceptiov1 = acceptiov2 = False
		    # check lumis in the same run
		    if tmprunfirst == tmprunlast and int(tmpbeam.Run)==tmprunfirst:
			if int(tmpbeam.IOVfirst) >= tmplumifirst and int(tmpbeam.IOVlast)<=tmplumilast:
			    acceptiov1 = acceptiov2 = True
		    # if different runs make sure you select the correct range of lumis
		    elif int(tmpbeam.Run) == tmprunfirst:
			if int(tmpbeam.IOVfirst) >= tmplumifirst: acceptiov1 = True
		    elif int(tmpbeam.Run) == tmprunlast:
			if int(tmpbeam.IOVlast) <= tmplumilast: acceptiov2 = True
		    elif tmprunfirst <= int(tmpbeam.Run) and tmprunlast >= int(tmpbeam.Run): 
			acceptiov1 = acceptiov2 = True
			
		    if acceptiov1 and acceptiov2:
			if tmpbeam.Type != 2:
			    print "invalid fit, skip Run "+str(tmpbeam.Run)+" IOV: "+str(tmpbeam.IOVfirst) + " to "+ str(tmpbeam.IOVlast)
			else:
			    listbeam.append(tmpbeam)

		elif int(tmpbeam.IOVfirst) >= int(firstRun) and int(tmpbeam.IOVlast) <= int(lastRun):
		    if tmpbeam.Type != 2:
			print "invalid fit, skip Run "+str(tmpbeam.Run)+" IOV: "+str(tmpbeam.IOVfirst) + " to "+ str(tmpbeam.IOVlast)
		    else:
			listbeam.append(tmpbeam)
	
		tmpbeamsize = 0
		tmpbeam = BeamSpot()
    else:

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
		if int(tmpbeam.IOVfirst) >= int(firstRun) and int(tmpbeam.IOVlast) <= int(lastRun):
		    listbeam.append(tmpbeam)
		tmpbeamsize = 0
		tmpbeam = BeamSpot()
	    

    print " got total number of IOVs = " + str(len(listbeam)) + " from file "+datafilename
    #print " run " + str(listbeam[3].IOVfirst ) + " " + str( listbeam[3].X )

    ###################################
    if IOVbase == "lumibase":
	listbeam.sort( cmp = cmp_list_lumi )
    else:
	listbeam.sort( cmp = cmp_list_run )
    
    # first clean list of data for consecutive duplicates and bad fits
    tmpremovelist = []
    for ii in range(0,len(listbeam)):
        
        ibeam = listbeam[ii]
        datax = ibeam.IOVfirst
        #print str(ii) + "  " +datax
        if datax == '1' and IOVbase =="runbase":
            print " iov = 1? skip this IOV = "+ str(ibeam.IOVfirst) + " to " + str(ibeam.IOVlast)
            tmpremovelist.append(ibeam)
        
        if ii < len(listbeam) -1:
            #print listbeam[ii+1].IOVfirst
	    if IOVbase =="lumibase":
		if ibeam.Run == listbeam[ii+1].Run and ibeam.IOVfirst == listbeam[ii+1].IOVfirst:
		    print " duplicate IOV = "+datax+", keep only last duplicate entry"
		    tmpremovelist.append(ibeam)
	    elif datax == listbeam[ii+1].IOVfirst:
                print " duplicate IOV = "+datax+", keep only last duplicate entry"
                tmpremovelist.append(ibeam)

    for itmp in tmpremovelist:
        listbeam.remove(itmp)

    # CREATE FILE FOR PAYLOADS
    ################################
    if IOVbase == "lumibase" and option.payload:
	newlistbeam = []
	tmpbeam = BeamSpot()
	docreate = True
	countlumi = 0
	tmprun = ""
        maxNlumis = 100
        if option.weighted:
            maxNlumis = 999999999
	for ii in range(0,len(listbeam)):
	
	    ibeam = listbeam[ii]
	    inextbeam = BeamSpot()
	    iNNbeam = BeamSpot()
	    if docreate:
		tmpbeam.IOVfirst = ibeam.IOVfirst
		tmpbeam.IOVBeginTime = ibeam.IOVBeginTime
		tmpbeam.Run = ibeam.Run
		tmpbeam.Type = 2
	    docheck = False
	    docreate = False
	    
	    # check last iov
	    if ii < len(listbeam) - 1: 
		inextbeam = listbeam[ii+1]
		docheck = True
		if ii < len(listbeam) -2:
		    iNNbeam = listbeam[ii+2]
	    else:
		print "close payload because end of data has been reached. Run "+ibeam.Run
		docreate = True
            # check we run over the same run
	    if ibeam.Run != inextbeam.Run:
		print "close payload because end of run "+ibeam.Run
		docreate = True
	    # check maximum lumi counts
	    if countlumi == maxNlumis:
		print "close payload because maximum lumi sections accumulated within run "+ibeam.Run
		docreate = True
		countlumi = 0
	    # weighted average position
	    (tmpbeam.X, tmpbeam.Xerr) = weight(tmpbeam.X, tmpbeam.Xerr, ibeam.X, ibeam.Xerr)
	    (tmpbeam.Y, tmpbeam.Yerr) = weight(tmpbeam.Y, tmpbeam.Yerr, ibeam.Y, ibeam.Yerr)
	    (tmpbeam.Z, tmpbeam.Zerr) = weight(tmpbeam.Z, tmpbeam.Zerr, ibeam.Z, ibeam.Zerr)
	    (tmpbeam.sigmaZ, tmpbeam.sigmaZerr) = weight(tmpbeam.sigmaZ, tmpbeam.sigmaZerr, ibeam.sigmaZ, ibeam.sigmaZerr)
	    (tmpbeam.dxdz, tmpbeam.dxdzerr) = weight(tmpbeam.dxdz, tmpbeam.dxdzerr, ibeam.dxdz, ibeam.dxdzerr)
	    (tmpbeam.dydz, tmpbeam.dydzerr) = weight(tmpbeam.dydz, tmpbeam.dydzerr, ibeam.dydz, ibeam.dydzerr)
	    #print "wx = " + ibeam.beamWidthX + " err= "+ ibeam.beamWidthXerr
	    (tmpbeam.beamWidthX, tmpbeam.beamWidthXerr) = weight(tmpbeam.beamWidthX, tmpbeam.beamWidthXerr, ibeam.beamWidthX, ibeam.beamWidthXerr)
	    (tmpbeam.beamWidthY, tmpbeam.beamWidthYerr) = weight(tmpbeam.beamWidthY, tmpbeam.beamWidthYerr, ibeam.beamWidthY, ibeam.beamWidthYerr)

            if option.weighted:
                docheck = False
	    # check offsets
	    if docheck:
		deltaX = delta(ibeam.X, ibeam.Xerr, inextbeam.X, inextbeam.Xerr) > 1.5
		deltaY = delta(ibeam.Y, ibeam.Yerr, inextbeam.Y, inextbeam.Yerr) > 1.5
		deltaZ = delta(ibeam.Z, ibeam.Zerr, inextbeam.Z, inextbeam.Zerr) > 2.5
				
		deltasigmaZ = delta(ibeam.sigmaZ, ibeam.sigmaZerr, inextbeam.sigmaZ, inextbeam.sigmaZerr) > 2.5
		deltadxdz   = delta(ibeam.dxdz, ibeam.dxdzerr, inextbeam.dxdz, inextbeam.dxdzerr) > 2.5
		deltadydz   = delta(ibeam.dydz, ibeam.dydzerr, inextbeam.dydz, inextbeam.dydzerr) > 2.5
		
		deltawidthX = delta(ibeam.beamWidthX, ibeam.beamWidthXerr, inextbeam.beamWidthX, inextbeam.beamWidthXerr) > 3
		deltawidthY = delta(ibeam.beamWidthY, ibeam.beamWidthYerr, inextbeam.beamWidthY, inextbeam.beamWidthYerr) > 3

		#if iNNbeam.Type != -1:
		#    deltaX = deltaX and delta(ibeam.X, ibeam.Xerr, iNNbeam.X, iNNbeam.Xerr) > 1.5
		#    deltaY = deltaY and delta(ibeam.Y, ibeam.Yerr, iNNbeam.Y, iNNbeam.Yerr) > 1.5
		#    deltaZ = deltaZ and delta(ibeam.Z, ibeam.Zerr, iNNbeam.Z, iNNbeam.Zerr) > 1.5
		#		
		#    deltasigmaZ = deltasigmaZ and delta(ibeam.sigmaZ, ibeam.sigmaZerr, iNNbeam.sigmaZ, iNNbeam.sigmaZerr) > 2.5
		#    deltadxdz   = deltadxdz and delta(ibeam.dxdz, ibeam.dxdzerr, iNNbeam.dxdz, iNNbeam.dxdzerr) > 2.5
		#    deltadydz   = deltadydz and delta(ibeam.dydz, ibeam.dydzerr, iNNbeam.dydz, iNNbeam.dydzerr) > 2.5
		#
		#    deltawidthX = deltawidthX and delta(ibeam.beamWidthX, ibeam.beamWidthXerr, iNNbeam.beamWidthX, iNNbeam.beamWidthXerr) > 3
		#    deltawidthY = deltawidthY and delta(ibeam.beamWidthY, ibeam.beamWidthYerr, iNNbeam.beamWidthY, iNNbeam.beamWidthYerr) > 3

		if deltaX or deltaY or deltaZ or deltasigmaZ or deltadxdz or deltadydz or deltawidthX or deltawidthY:
		    docreate = True
		    #print "shift here: x="+str(deltaX)+" y="+str(deltaY)
		    #print "x1 = "+ibeam.X + " x1err = "+ibeam.Xerr
		    #print "x2 = "+inextbeam.X + " x2err = "+inextbeam.Xerr
		    #print "Lumi1: "+str(ibeam.IOVfirst) + " Lumi2: "+str(inextbeam.IOVfirst)
		    #print " x= "+ibeam.X+" +/- "+ibeam.Xerr
		    #print "weighted average x = "+tmpbeam.X +" +//- "+tmpbeam.Xerr
		    print "close payload because of movement in X= "+str(deltaX)+", Y= "+str(deltaY) + ", Z= "+str(deltaZ)+", sigmaZ= "+str(deltasigmaZ)+", dxdz= "+str(deltadxdz)+", dydz= "+str(deltadydz)+", widthX= "+str(deltawidthX)+", widthY= "+str(deltawidthY)
	    if docreate:
		tmpbeam.IOVlast = ibeam.IOVlast
		tmpbeam.IOVEndTime = ibeam.IOVEndTime
		print "  Run: "+tmpbeam.Run +" Lumi1: "+str(tmpbeam.IOVfirst) + " Lumi2: "+str(tmpbeam.IOVlast)
		newlistbeam.append(tmpbeam)
		tmpbeam = BeamSpot()
	    tmprun = ibeam.Run
	    countlumi += 1
	
        
	npayload = 1
        payloadfile = open(option.payload,"w")
	for iload in newlistbeam:
            # print new list
            name = option.payload
            #name = name.replace(".txt","")
            #name = name + "_" +str(npayload) +".txt"
            #payloadfile = open(name,"w")
	    dump( iload, payloadfile )
            #payloadfile.close()
            npayload += 1
        payloadfile.close()
    if option.noplot:
        print " no plots requested, exit now."
        sys.exit()
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
        graphXaxis = 'Lumi section'
    if IOVbase == 'timebase':
        graphXaxis = "Time"
    
    graphYaxis = ['beam spot X [cm]','beam spot Y [cm]','beam spot Z [cm]', 'beam spot #sigma_{Z} [cm]', 'beam spot dX/dZ', 'beam spot dY/dZ','beam width X [cm]', 'beam width Y [cm]']

    cvlist = []

    for ig in range(0,8):
	cvlist.append( TCanvas(graphnamelist[ig],graphtitlelist[ig], 1800, 600) )
	if option.graph:
	    graphlist.append( TGraphErrors( len(listbeam) ) )
        else:
	    graphlist.append( TH1F("name","title",len(listbeam),0,len(listbeam)) )
        
	graphlist[ig].SetName(graphnamelist[ig])
        graphlist[ig].SetTitle(graphtitlelist[ig])
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
	    if graphnamelist[ig] == 'beamWidthX':
		datay = ibeam.beamWidthX
		datayerr = ibeam.beamWidthXerr
	    if graphnamelist[ig] == 'beamWidthY':
		datay = ibeam.beamWidthY
		datayerr = ibeam.beamWidthYerr

            datax = ibeam.IOVfirst
	    if IOVbase=="lumibase":
		datax = str(ibeam.Run) + ":" + str(ibeam.IOVfirst)
		if ibeam.IOVfirst != ibeam.IOVlast:
		    datax = str(ibeam.Run) + ":" + str(ibeam.IOVfirst)+"-"+str(ibeam.IOVlast)
            #print datax
	    if option.graph:
		#datax = pack( int(ibeam.Run) , int(ibeam.IOVfirst) )
		if IOVbase=="lumibase":
		    datax = (float(ibeam.IOVlast) - float(ibeam.IOVfirst))/2 + float(ibeam.IOVfirst)
		    dataxerr =  (float(ibeam.IOVlast) - float(ibeam.IOVfirst))/2
		graphlist[ig].SetPoint(ipoint, float(datax), float(datay) )
		graphlist[ig].SetPointError(ipoint, float(dataxerr), float(datayerr) )
	    else:
		graphlist[ig].GetXaxis().SetBinLabel(ipoint +1 , str(datax) )
		graphlist[ig].SetBinContent(ipoint +1, float(datay) )
		graphlist[ig].SetBinError(ipoint +1, float(datayerr) )

            ipoint += 1


	if IOVbase=="timebase":
            graphlist[ig].GetXaxis().SetTimeDisplay(1);
            graphlist[ig].GetXaxis().SetTimeFormat("#splitline{%Y/%m/%d}{%H:%M}")
	if option.graph:
	    graphlist[ig].Draw('AP')
        else:
	    graphlist[ig].Draw('P E1 X0')
	    graphlist[ig].GetXaxis().SetTitle(graphXaxis)
	    graphlist[ig].GetYaxis().SetTitle(graphYaxis[ig])
            #graphlist[ig].Fit('pol1')
	cvlist[ig].Update()
        cvlist[ig].SetGrid()
        if option.Print:
	    cvlist[ig].Print(graphnamelist[ig]+".png")
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
