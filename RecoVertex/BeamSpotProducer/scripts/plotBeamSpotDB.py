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
   -s, --suffix = SUFFIX: Suffix will be added to plots filename.
   -t, --tag     = TAG: Database tag name.
   -T, --Time : create plots with time axis.
   -I, --IOVbase = IOVBASE: options: runbase(default), lumibase, timebase
   -w, --wait : Pause script after plotting a new histograms.
   -W, --weighted : Create a weighted result for a range of lumi IOVs, skip lumi IOV combination and splitting.
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
        print(" need to provide DB tag name or beam spot data file")
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
            print("\n\n unknown iov base option: "+ option.IOVbase +" \n\n\n")
            exit()
        IOVbase = option.IOVbase

    firstRun = "0"
    lastRun  = "4999999999"
    if IOVbase == "lumibase":
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
        mydestdb = 'frontier://PromptProd/CMS_COND_31X_BEAMSPOT'
        if option.destDB:
            mydestdb = option.destDB
        acommand = 'cmscond_list_iov -c '+mydestdb+' -P /afs/cern.ch/cms/DB/conddb -t '+ tag
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
                tmprunlast  = pack( int(lastRun.split(":")[0]) , int(lastRun.split(":")[1]) )
            #print "since = " + str(iIOV.since) + " till = "+ str(iIOV.till)
            if iIOV.since >= int(tmprunfirst) and int(tmprunlast) < 0 and iIOV.since <= int(tmprunfirst):
                print(" IOV: " + str(iIOV.since))
                passiov = True
            if iIOV.since >= int(tmprunfirst) and int(tmprunlast) > 0 and iIOV.till <= int(tmprunlast):
                print(" a IOV: " + str(iIOV.since) + " to " + str(iIOV.till))
                passiov = True
            #if iIOV.since >= int(tmprunlast) and iIOV.till >= 4294967295:
            #    print " b IOV: " + str(iIOV.since) + " to " + str(iIOV.till)
            #    passiov = True                
            if passiov:
                acommand = 'getBeamSpotDB.py -t '+ tag + " -r " + str(iIOV.since) +otherArgs
                if IOVbase=="lumibase":
                    tmprun = unpack(iIOV.since)[0]
                    tmplumi = unpack(iIOV.since)[1]
                    acommand = 'getBeamSpotDB.py -t '+ tag + " -r " + str(tmprun) +" -l "+str(tmplumi) +otherArgs
                    print(acommand)
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

    if IOVbase == "lumibase" and option.payload:
        weighted = True;
        if not option.weighted:
            weighted = False
        createWeightedPayloads(option.payload,listbeam,weighted)
    if option.noplot:
        print(" no plots requested, exit now.")
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
    if IOVbase == 'timebase' or option.Time:
        graphXaxis = "Time"
        #dh = ROOT.TDatime(2010,06,01,00,00,00)
        ROOT.gStyle.SetTimeOffset(0) #dh.Convert())

    graphYaxis = ['beam spot X [cm]','beam spot Y [cm]','beam spot Z [cm]', 'beam spot #sigma_{Z} [cm]', 'beam spot dX/dZ', 'beam spot dY/dZ','beam width X [cm]', 'beam width Y [cm]']

    cvlist = []

    for ig in range(0,8):
        cvlist.append( TCanvas(graphnamelist[ig],graphtitlelist[ig], 1200, 600) )
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
                if IOVbase=="lumibase":
                    #first = int( pack( int(ibeam.Run) , int(ibeam.IOVfirst) ) )
                    #last = int( pack( int(ibeam.Run) , int(ibeam.IOVlast) ) )
                    first = ibeam.IOVfirst
                    last = ibeam.IOVlast
                    if option.Time:
                        atime = ibeam.IOVBeginTime
                        first = time.mktime( time.strptime(atime.split()[0] +  " " + atime.split()[1] + " " + atime.split()[2],"%Y.%m.%d %H:%M:%S %Z") )
                        atime = ibeam.IOVEndTime
                        last = time.mktime( time.strptime(atime.split()[0] +  " " + atime.split()[1] + " " + atime.split()[2],"%Y.%m.%d %H:%M:%S %Z") )
                        da_first = TDatime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(first - time.timezone)))
                        da_last = TDatime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(last - time.timezone)))
                        if ipoint == 0:
                            ## print local time
                            da_first.Print()
                            ## print gmt time
                            print("GMT = " + str(time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(first - time.timezone))))
                            reftime = first
                            ptm = time.localtime(reftime)
                            da = TDatime(time.strftime('%Y-%m-%d %H:%M:%S',ptm))
                            if time.daylight and ptm.tm_isdst:
                                offset_daylight = time.timezone - time.altzone
                            ROOT.gStyle.SetTimeOffset(da.Convert(1) - 3600)

                    datax = (float(last) - float(first))/2 + float(first) - da.Convert() + 3600
                    ## Comment out this block if running on Mac ##
                    if time.daylight and ptm.tm_isdst:
                        datax += offset_daylight
                    ##################################

                    dataxerr =  (float(last) - float(first))/2
                graphlist[ig].SetPoint(ipoint, float(datax), float(datay) )
                graphlist[ig].SetPointError(ipoint, float(dataxerr), float(datayerr) )
            else:
                graphlist[ig].GetXaxis().SetBinLabel(ipoint +1 , str(datax) )
                graphlist[ig].SetBinContent(ipoint +1, float(datay) )
                graphlist[ig].SetBinError(ipoint +1, float(datayerr) )

            ipoint += 1
        if option.Time:
            ## print local time
            da_last.Print()
            print("GMT = " + str(time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(last - time.timezone))))
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
        print(" plots have been written to "+option.output)



    # CLEAN temporal files
    ###################################
    #os.system('rm tmp_beamspotdata.log')
