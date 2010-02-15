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
   -c, --create  = CREATE: name for beam spot data file.
   -d, --data    = DATA: input beam spot data file.
   -i, --initial = INITIAL: First run.
   -f, --final   = FINAL: Last run.
   -n, --noplot  = NOPLOT: no make plots only extract beam spot data.
   -o, --output  = OUTPUT: output ROOT filename.
   -t, --tag     = TAG: tag name.
   -w, --wait : Pause script after plotting a new superposition of histograms.
   
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

from ROOT import TFile
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
        self.Y = 0.
        self.Z = 0.
        self.sigmaZ = 0.
        self.dXdZ = 0.
        self.dYdZ = 0.
        self.beamWidthX = 0.
        self.beamWidthY = 0.
    def Reset(self):
	self.X = self.Y = self.Z = 0.
	self.sigmaZ = 0.
	self.dXdZ = self.dYdZ
	self.beamWidthX = self.beamWidthY = 0.

class IOV:
    def __init__(self):
	self.since = 1
	self.till = 1



if __name__ == '__main__':

    # style
    #thestyle = Style.Style()
    #thestyle.SetStyle()

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

    datafilename = "tmp_beamspot.dat"
    if option.create:
        datafilename = option.create
    
    getDBdata = True
    if option.data:
        getDBdata = False
    
    
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

        firstRun = 1
        lastRun  = 4999999999

        if option.initial:
            firstRun = int(option.initial)
        if option.final:
            lastRun = int(option.final)
    
    
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
	    tmpbeamsize += 1
            #print " x = " + str(tmpbeam.X)
	if line.find('Y0') != -1:
	    tmpbeam.Y = line.split()[2]
	    tmpbeamsize += 1
            #print " y =" + str(tmpbeam.Y)
	if line.find('Z0') != -1 and line.find('Sigma Z0') == -1:
	    tmpbeam.Z = line.split()[2]
	    tmpbeamsize += 1
            #print " z =" + str(tmpbeam.Z)
	if tmpbeamsize == 3:
            #print " from object " + str(tmpbeam.X)
	    listbeam.append(tmpbeam)
	    tmpbeamsize = 0
	    tmpbeam.Reset()
    
    print " total number of IOVs " + str(len(listbeam))
	

    # CLEAN temporal files
    ###################################
    #os.system('rm tmp_beamspotdata.log')
