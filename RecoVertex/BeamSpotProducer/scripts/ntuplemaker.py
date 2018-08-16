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
   ntuplemaker

   A very simple script to plot the beam spot data stored in condDB

   usage: %prog -t <tag name>
   -a, --auth    = AUTH: DB authorization path. online(/nfshome0/popcondev/conddb).
   -b, --batch : Run ROOT in batch mode.
   -c, --create  = CREATE: name for beam spot data file.
   -d, --data    = DATA: input beam spot data file.
   -D, --destDB  = DESTDB: destination DB string. online(oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT).
   -i, --initial = INITIAL: First IOV. Options: run number, or run:lumi, eg. \"133200:21\"
   -f, --final   = FINAL: Last IOV. Options: run number, or run:lumi
   -o, --output  = OUTPUT: filename of ROOT file with plots.
   -x, --xcrossing = XCROSSING : Bunch crossing number.
   
   Francisco Yumiceva (yumiceva@fnal.gov)
   Fermilab 2010
   
"""
from __future__ import print_function


import os, string, re, sys, math
import commands, time
from BeamSpotObj import BeamSpot
from IOVObj import IOV
from CommonMethods import *

try:
    import ROOT
except:
    print("\nCannot load PYROOT, make sure you have setup ROOT in the path")
    print("and pyroot library is also defined in the variable PYTHONPATH, try:\n")
    if (os.getenv("PYTHONPATH")):
        print(" setenv PYTHONPATH ${PYTHONPATH}:$ROOTSYS/lib\n")
    else:
        print(" setenv PYTHONPATH $ROOTSYS/lib\n")
        sys.exit()

from ROOT import *
from array import array

def getFill( json, run ):

    thefill = 0
    run = int(run)
    keys = json.keys()

    for i in keys:

        run0 = int(json[i][0])
        run1 = int(json[i][1])
        if run>= run0 and run<=run1:
            thefill = i

    return int(thefill)

if __name__ == '__main__':



    # fill and runs
    FillList = {}
    runsfile = open("FillandRuns.txt")
    for line in runsfile:
        if line.find('fill:') != -1:
            aline = line.split()
            afill = aline[1]
            run0 = aline[3]
            run1 = aline[5]
            FillList[int(afill)] = [int(run0),int(run1)]

    #print FillList

    # create ntuple
    gROOT.ProcessLine(
        "struct spot {\
        Float_t   position[3];\
        Float_t   posError[3];\
        Float_t   width[3];\
        Float_t   widthError[3];\
        Float_t   slope[2];\
        Float_t   slopeError[2];\
        Float_t   time[2];\
        Int_t     run;\
        Int_t     lumi[2];\
        Int_t     fill;\
        };" );

    bntuple = spot()
    fntuple = TFile( 'bntuple.root', 'RECREATE' )
    tbylumi = TTree( 'bylumi', 'beam spot data lumi by lumi' )
    tbylumi.Branch('fill', AddressOf( bntuple, 'fill'), 'fill/I' )
    tbylumi.Branch('run', AddressOf( bntuple, 'run'), 'run/I' )
    tbylumi.Branch('lumi', AddressOf( bntuple, 'lumi'), 'lumi[2]/I' )
    tbylumi.Branch('position', AddressOf( bntuple, 'position'),'position[3]/F')
    tbylumi.Branch('posErr', AddressOf( bntuple, 'posError'),'posError[3]/F')
    tbylumi.Branch('width', AddressOf( bntuple, 'width'),'width[3]/F')
    tbylumi.Branch('widthErr', AddressOf( bntuple, 'widthError'),'widthError[3]/F')
    tbylumi.Branch('slope', AddressOf( bntuple, 'slope'),'slope[2]/F')
    tbylumi.Branch('slopeErr', AddressOf( bntuple, 'slopeError'),'slopeError[2]/F')
    tbylumi.Branch('time', AddressOf( bntuple, 'time'),'time[2]/F')

    tbyIOV = TTree( 'byIOV', 'beam spot data by IOV' )
    tbyIOV.Branch('fill', AddressOf( bntuple, 'fill'), 'fill/I' )
    tbyIOV.Branch('run', AddressOf( bntuple, 'run'), 'run/I' )
    tbyIOV.Branch('lumi', AddressOf( bntuple, 'lumi'), 'lumi[2]/I' )
    tbyIOV.Branch('position', AddressOf( bntuple, 'position'),'position[3]/F')
    tbyIOV.Branch('posErr', AddressOf( bntuple, 'posError'),'posError[3]/F')
    tbyIOV.Branch('width', AddressOf( bntuple, 'width'),'width[3]/F')
    tbyIOV.Branch('widthErr', AddressOf( bntuple, 'widthError'),'widthError[3]/F')
    tbyIOV.Branch('slope', AddressOf( bntuple, 'slope'),'slope[2]/F')
    tbyIOV.Branch('slopeErr', AddressOf( bntuple, 'slopeError'),'slopeError[2]/F')
    tbyIOV.Branch('time', AddressOf( bntuple, 'time'),'time[2]/F')

    tbyrun = TTree( 'byrun', 'beam spot data by run' )
    tbyrun.Branch('fill', AddressOf( bntuple, 'fill'), 'fill/I' )
    tbyrun.Branch('run', AddressOf( bntuple, 'run'), 'run/I' )
    tbyrun.Branch('lumi', AddressOf( bntuple, 'lumi'), 'lumi[2]/I' )
    tbyrun.Branch('position', AddressOf( bntuple, 'position'),'position[3]/F')
    tbyrun.Branch('posErr', AddressOf( bntuple, 'posError'),'posError[3]/F')
    tbyrun.Branch('width', AddressOf( bntuple, 'width'),'width[3]/F')
    tbyrun.Branch('widthErr', AddressOf( bntuple, 'widthError'),'widthError[3]/F')
    tbyrun.Branch('slope', AddressOf( bntuple, 'slope'),'slope[2]/F')
    tbyrun.Branch('slopeErr', AddressOf( bntuple, 'slopeError'),'slopeError[2]/F')
    tbyrun.Branch('time', AddressOf( bntuple, 'time'),'time[2]/F')


    # COMMAND LINE OPTIONS
    #################################
    option,args = parse(__doc__)
    if not args and not option: exit()

    if not option.data: 
        print(" need to provide beam spot data file")
        exit()

    if option.batch:
        ROOT.gROOT.SetBatch()

    datafilename = "tmp_beamspot.dat"
    if option.create:
        datafilename = option.create

    getDBdata = True
    if option.data:
        getDBdata = False

    IOVbase = 'lumibase'
    firstRun = "0:0"
    lastRun = "4999999999:4999999999"

    if option.initial:
        firstRun = option.initial
    if option.final:
        lastRun = option.final

    # GET IOVs
    ################################

    if getDBdata:

        print(" read DB to get list of IOVs for the given tag")
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

        print(" total number of IOVs = " + str(len(iovlist)))


        #  GET DATA
        ################################

        otherArgs = ''
        if option.destDB:
            otherArgs = " -d " + option.destDB
            if option.auth:
                otherArgs = otherArgs + " -a "+ option.auth

        print(" get beam spot data from DB for IOVs. This can take a few minutes ...")

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
                print(" IOV: " + str(iIOV.since))
                passiov = True
            if iIOV.since >= int(tmprunfirst) and int(tmprunlast) > 0 and iIOV.till <= int(tmprunlast):
                print(" IOV: " + str(iIOV.since) + " to " + str(iIOV.till))
                passiov = True
            if iIOV.since >= int(tmprunlast) and iIOV.till >= 4294967295:
                print(" IOV: " + str(iIOV.since) + " to " + str(iIOV.till))
                passiov = True                
            if passiov:
                acommand = 'getBeamSpotDB.py -t '+ tag + " -r " + str(iIOV.since) +otherArgs
                if IOVbase=="lumibase":
                    tmprun = unpack(iIOV.since)[0]
                    tmplumi = unpack(iIOV.since)[1]
                    acommand = 'getBeamSpotDB.py -t '+ tag + " -r " + str(tmprun) +" -l "+tmplumi +otherArgs
                status = commands.getstatusoutput( acommand )
                tmpfile.write(status[1])

        print(" beam spot data collected and stored in file " + datafilename)

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
                if os.path.isdir(option.data+"/"+f) is False:
                    input = open(option.data +"/"+f)
                    output.writelines(input.readlines())
            output.close()
            print(" data files have been collected in "+datafilename)

        elif os.path.exists(option.data):
            datafilename = option.data
        else:
            print(" input beam spot data DOES NOT exist, file " + option.data)
            exit()

    listbeam = []

    if option.xcrossing:
        listmap = readBeamSpotFile(datafilename,listbeam,IOVbase,firstRun,lastRun)
        # bx
        print("List of bunch crossings in the file:")
        print(listmap.keys())
        listbeam = listmap[option.Xrossing]
    else:
        readBeamSpotFile(datafilename,listbeam,IOVbase,firstRun,lastRun)

    sortAndCleanBeamList(listbeam,IOVbase)


    ###################################    

    for ii in range(0,len(listbeam)):

        ibeam = listbeam[ii]

        bntuple.position = array('f', [float(ibeam.X), float(ibeam.Y), float(ibeam.Z)])
        bntuple.posError = array('f', [float(ibeam.Xerr),float(ibeam.Yerr),float(ibeam.Zerr)])
        bntuple.width = array('f', [float(ibeam.beamWidthX), float(ibeam.beamWidthY), float(ibeam.sigmaZ)])
        bntuple.widthError = array('f',[float(ibeam.beamWidthXerr),float(ibeam.beamWidthYerr),float(ibeam.sigmaZerr)])
        bntuple.run = int(ibeam.Run)
        bntuple.fill = int( getFill( FillList, int(ibeam.Run) ) )
        bntuple.lumi = array('i', [int(ibeam.IOVfirst),int(ibeam.IOVlast)])
        line = ibeam.IOVBeginTime
        begintime = time.mktime( time.strptime(line.split()[0] +  " " + line.split()[1] + " " + line.split()[2],"%Y.%m.%d %H:%M:%S %Z") )
        line = ibeam.IOVEndTime
        endtime = time.mktime( time.strptime(line.split()[0] +  " " + line.split()[1] + " " + line.split()[2],"%Y.%m.%d %H:%M:%S %Z") )
        bntuple.time = array('f', [begintime, endtime])
        tbylumi.Fill()


    iovlist = listbeam
    iovlist = createWeightedPayloads("tmp.txt",iovlist,False)

    for ii in range(0,len(iovlist)):

        ibeam = iovlist[ii]

        bntuple.position = array('f', [float(ibeam.X), float(ibeam.Y), float(ibeam.Z)])
        bntuple.posError = array('f', [float(ibeam.Xerr),float(ibeam.Yerr),float(ibeam.Zerr)])
        bntuple.width = array('f', [float(ibeam.beamWidthX), float(ibeam.beamWidthY), float(ibeam.sigmaZ)])
        bntuple.widthError = array('f',[float(ibeam.beamWidthXerr),float(ibeam.beamWidthYerr),float(ibeam.sigmaZerr)])
        bntuple.run = int(ibeam.Run)
        bntuple.fill = int( getFill( FillList, int(ibeam.Run) ) )
        bntuple.lumi = array('i', [int(ibeam.IOVfirst),int(ibeam.IOVlast)])
        line = ibeam.IOVBeginTime
        begintime = time.mktime( time.strptime(line.split()[0] +  " " + line.split()[1] + " " + line.split()[2],"%Y.%m.%d %H:%M:%S %Z") )
        line = ibeam.IOVEndTime
        endtime = time.mktime( time.strptime(line.split()[0] +  " " + line.split()[1] + " " + line.split()[2],"%Y.%m.%d %H:%M:%S %Z") )
        bntuple.time = array('f', [begintime, endtime])

        tbyIOV.Fill()

    weightedlist = listbeam
    weightedlist = createWeightedPayloads("tmp.txt",weightedlist,True)

    for ii in range(0,len(weightedlist)):

        ibeam = weightedlist[ii]

        bntuple.position = array('f', [float(ibeam.X), float(ibeam.Y), float(ibeam.Z)])
        bntuple.posError = array('f', [float(ibeam.Xerr),float(ibeam.Yerr),float(ibeam.Zerr)])
        bntuple.width = array('f', [float(ibeam.beamWidthX), float(ibeam.beamWidthY), float(ibeam.sigmaZ)])
        bntuple.widthError = array('f',[float(ibeam.beamWidthXerr),float(ibeam.beamWidthYerr),float(ibeam.sigmaZerr)])
        bntuple.run = int(ibeam.Run)
        bntuple.fill = int( getFill( FillList, int(ibeam.Run) ) )
        bntuple.lumi = array('i', [int(ibeam.IOVfirst),int(ibeam.IOVlast)])
        line = ibeam.IOVBeginTime
        begintime = time.mktime( time.strptime(line.split()[0] +  " " + line.split()[1] + " " + line.split()[2],"%Y.%m.%d %H:%M:%S %Z") )
        line = ibeam.IOVEndTime
        endtime = time.mktime( time.strptime(line.split()[0] +  " " + line.split()[1] + " " + line.split()[2],"%Y.%m.%d %H:%M:%S %Z") )
        bntuple.time = array('f', [begintime, endtime])

        tbyrun.Fill()


    os.system('rm tmp.txt')
    fntuple.cd()
    tbylumi.Write()
    tbyIOV.Write()
    tbyrun.Write()
    fntuple.Close()

    # CLEAN temporal files
    ###################################
    #os.system('rm tmp_beamspotdata.log')
