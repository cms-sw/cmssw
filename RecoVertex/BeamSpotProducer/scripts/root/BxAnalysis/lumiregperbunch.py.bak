#!/usr/bin/env python
import sys,commands,os,calendar
from ROOT import gDirectory,TFile

from online_beamspot_reader import BeamspotMeasurement
from online_beamspot_reader import paragraphs, \
     start_of_new_beamspot_measurement

FILL=''
OUTDIR="./perbunchfiles/"

runlstime={}


class bsmeas(object):
    def __init__(self, x=None,y=None,z=None,ex=None,ey=None,ez=None,
                 wx=None,wy=None,wz=None,ewx=None,ewy=None,ewz=None,
                 dxdz=None,dydz=None,edxdz=None,edydz=None):
        self.x = x
        self.y = y
        self.z = z
        self.ex = ex
        self.ey = ey
        self.ez = ez
        self.wx = ex
        self.wy = ey
        self.wz = ez
        self.ewx = ex
        self.ewy = ey
        self.ewz = ez
        self.dxdz = dxdz
        self.dydz = dydz
        self.edxdz = edxdz
        self.edydz = edydz
        



def timeof(run,lumisection):
    # first check if this run is already in the list, otherwise read it
    if run not in runlstime.keys():
        print "Reading lumi time from lumireg localcopy files"
        filename="localcopy/BeamFitResults_Run"+run+".txt"
        if not os.path.exists(filename):
            print "WARNING: file ",filename," does not exist. Returning null."
            return -1

        # reading file
        lstime={}
        in_file = open(filename)
        pieces = paragraphs(in_file,start_of_new_beamspot_measurement,True)
        for piece in pieces:
            if len(piece) < 1:
                continue
            try:
                tmp = BeamspotMeasurement(piece)
            except Exception as err:
                print >> sys.stderr, \
                      "    ERROR Found corrupt " \
                      "beamspot measurement entry!"
                print >> sys.stderr, \
                      "    !!! %s !!!" % str(err)
                continue
            # Argh!
            runfromfile=tmp.run_number
            (lumimin,lumimax)=tmp.lumi_range
            time_begin=tmp.time_begin
            time_end=tmp.time_end
            time_begin=calendar.timegm(time_begin.timetuple())
            time_end=calendar.timegm(time_end.timetuple())-23 # assume end of lumisection
            lstime[lumimin]=time_begin
            lstime[lumimax]=time_end
            
        # order lumisections and make a list
        lslist=sorted(lstime.keys())
        lstimesorted=[]
        for ls in lslist:
            lstimesorted.append((ls,lstime[ls]))
        runlstime[run]=lstimesorted

        #            print runfromfile
        #            print lumirange
        #            print time_begin, calendar.timegm(time_begin.timetuple())
        #            print time_end, calendar.timegm(time_end.timetuple())
        
        in_file.close()
    # now give a time
    dcloselumi=999999
    closelumi=-1
    closetime=-1
    lstimesorted=runlstime[run]
    
    for pair in lstimesorted:
        (lumi,time)=pair
        if abs(lumisection-lumi)<dcloselumi:
            dcloselumi=abs(lumisection-lumi)
            closelumi=lumi
            closetime=time
    if closelumi!=-1:
        finaltime=closetime+(lumisection-closelumi)*23
    else:
        finaltime=-1
        
    return finaltime


def readroot():
    rls=[]
    bxlist=[]
    allmeas={}
    
    DIRES=['X0','Y0','Z0','width_X0','Width_Y0','Sigma_Z0','dxdz','dydz']
    # DIRES=['X0']
    rootfile="BxAnalysis_Fill_"+FILL+".root"
    filein=TFile(rootfile)
    for dire in DIRES:
        filein.cd(dire)
        # get list of histograms
        histolist=gDirectory.GetListOfKeys()
        iter = histolist.MakeIterator()
        key = iter.Next()
        while key:
            if key.GetClassName() == 'TH1F':
                td = key.ReadObj()
                histoname = td.GetName()
                if "bx" in histoname:
#                    print histoname
                    bx=histoname.split('_')[-1]
                    if bx not in bxlist:
# this is to be removed                        
#                        if len(bxlist)>=2:
#                            key = iter.Next()
#                            continue
# end to be removed                        
                        bxlist.append(bx)
                        allmeas[bx]={}
#                    print bx,histoname
                    histo=gDirectory.Get(histoname)
                    nbin=histo.GetNbinsX()

                    thisbx=allmeas[bx]
                    
                    for bin in range(1,nbin+1):
                        label=histo.GetXaxis().GetBinLabel(bin)
                        label=label.strip()
                        if ":" not in label:
                            # not a valid label of type run:lumi-lumi, skip it
                            continue
                        
                        cont=histo.GetBinContent(bin)
                        if cont!=cont:
                            # it's a nan
                            cont=-999.0
                        err=histo.GetBinError(bin)
                        if err!=err:
                            err=-999.0
                        #                        if len(bxlist)==1:
                        #                            rls.append(label)
                        #                            print label
                        #                        else:
                        if label not in rls:
                            print "New range:",label," found in ",histoname
                            rls.append(label)
                                
                        if label in thisbx.keys():
                            thismeas=thisbx[label]
                        else:
                            thisbx[label]=bsmeas()
                            thismeas=thisbx[label]
                        #  now filling up
                        if dire=='X0':
                            thismeas.x=cont
                            thismeas.ex=err
                        if dire=='Y0':
                            thismeas.y=cont
                            thismeas.ey=cont
                        if dire=='Z0':
                            thismeas.z=cont
                            thismeas.ez=err
                        if dire=='width_X0':
                            thismeas.wx=cont
                            thismeas.ewx=err
                        if dire=='Width_Y0':
                            thismeas.wy=cont
                            thismeas.ewy=err
                        if dire=='Sigma_Z0':
                            thismeas.wz=cont
                            thismeas.ewz=err
                        if dire=='dxdz':
                            thismeas.dxdz=cont
                            thismeas.edxdz=err
                        if dire=='dydz':
                            thismeas.dydz=cont
                            thismeas.edydz=err

    
            key = iter.Next()
        
        
    #    for name in pippo:
    #        print name
    
    filein.Close()

    # let's try to show it
#    for bx in allmeas.keys():
#        print "bx=",bx
#        bxmeas=allmeas[bx]
#        for meas in bxmeas.keys():
#            print "meas time=",meas
#            thismeas=bxmeas[meas]
#            print thismeas.x,thismeas.ex,thismeas.y,thismeas.ey,thismeas.z,thismeas.ez
#            print thismeas.wx,thismeas.ewx,thismeas.wy,thismeas.ewy,thismeas.wz,thismeas.ewz
#            print thismeas.dxdz,thismeas.edxdz,thismeas.dydz,thismeas.edydz
    return allmeas

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print "Usage: :",sys.argv[0]," <fillnr>"
        sys.exit(1)
    FILL=sys.argv[1]

    allmeas=readroot()
    # now write it
    
    for bx in allmeas.keys():
        print "writing bx=",bx
        bxmeas=allmeas[bx]
        lines={}
        for meas in bxmeas.keys():
            # first derive time in unix time
            runno=meas.split(':')[0]
            runno=runno.strip()
            lumirange=meas.split(':')[1]
            lumimin=lumirange.split('-')[0]
            lumimin=lumimin.strip()
            lumimax=lumirange.split('-')[1]
            lumimax=lumimax.strip()
            lumimid=int((int(lumimin)+int(lumimax))/2.)
            meastime=timeof(runno,lumimid)
            print runno,str(lumimid),meastime
            
            thismeas=bxmeas[meas]
#            print thismeas.x,thismeas.ex,thismeas.y,thismeas.ey,thismeas.z,thismeas.ez
#            print thismeas.wx,thismeas.ewx,thismeas.wy,thismeas.ewy,thismeas.wz,thismeas.ewz
#            print thismeas.dxdz,thismeas.edxdz,thismeas.dydz,thismeas.edydz
            line=str(meastime)+" "
            line+="1 "
            line+="%11.7f %11.7f %11.7f %11.7f %11.7f %11.7f " % (-thismeas.x*10,thismeas.ex*10,
                                                            thismeas.y*10,thismeas.ey*10,
                                                            -thismeas.z*10,thismeas.ez*10)
            line+="%11.7f %11.7f %11.7f %11.7f %11.7f %11.7f " % (thismeas.wx*10,thismeas.ewx*10,
                                                            thismeas.wy*10,thismeas.ewy*10,
                                                            thismeas.wz*10,thismeas.ewz*10)
            line+="%11.7f %11.7f %11.7f %11.7f" % (thismeas.dxdz,thismeas.edxdz,-thismeas.dydz,thismeas.edydz)
            line+="\n"
            
            # validate it
            if (thismeas.x != 0.0 and thismeas.y != 0.0 and thismeas.z != 0.0 and
                thismeas.wx != 0.0 and thismeas.wy != 0.0 and thismeas.wz != 0.0 and
                thismeas.dxdz != 0.0 and thismeas.dydz != 0.0 ):
                lines[meastime]=line

            
        # now write it
        WORKDIR=OUTDIR+FILL
        os.system("mkdir -p "+WORKDIR)
        rfbucket=(int(bx)-1)*10+1
        filename=WORKDIR+"/"+FILL+"_lumireg_"+str(rfbucket)+"_CMS.txt"
        file=open(filename,'w')
        sortedtimes=sorted(lines.keys())
        for meastime in sortedtimes:
            file.write(lines[meastime])
        file.close()

    #    print timeof("168264",25)
