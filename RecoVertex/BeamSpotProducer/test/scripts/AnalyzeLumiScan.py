#!/usr/bin/env python
#____________________________________________________________
#
#  AnalyzeLumiScan
#
# A very simple way to run beam fit for lumi scan
# Running with RecoVertex/BeamSpotProducer package
# Create log and beam fit txt files in the dir named <Run number>
#
# Geng-yuan Jeng
# Geng-yuan.Jeng@cern.ch
#
# Fermilab, 2009
#
#____________________________________________________________

from __future__ import print_function
from builtins import range
import sys,os,re,string
import commands

##dataset = "ExpressPhysics/BeamCommissioning09-Express-v2/FEVT"
dataset = "/MinimumBias/BeamCommissioning09-Dec19thReReco_341_v1/RECO"
def get_list_files(run,ls1,ls2,mode):
    lfiles = []
    ## DBS query
    dbsquery = "dbs search --noheader --query=\"find file where dataset="+dataset
    dbsquery += " and run="+str(run)+" and lumi>="+str(ls1)+" and lumi<="+str(ls2)+"\" > tmplist.txt"
## For running on ExpressPhysics datasets ordered by lumi, uncomment the following line and comment out the previous one
##    dbsquery += " and run="+str(run)+" and lumi>="+str(ls1)+" and lumi<="+str(ls2)+" order by lumi asc\" > tmplist.txt"
    #print dbsquery
    os.system(dbsquery)
    n=0
    fileHandle = open("tmplist.txt")
    lineList = fileHandle.readlines()
    nfiles = len(lineList)-1

    lineList = map(string.strip, lineList)
    if mode == 0: ## caf
        prefix=""
    elif mode == 1: ## lxplus
        prefix="rfio:/castor/cern.ch/cms"
    elif mode == 2: ## cmslpc
	prefix="dcache:/pnfs/cms/WAX/11"
    else:
        print("Mode = "+str(mode)+" is not supported")

    for f in lineList:
        #print n
        if f.find("store") != -1:
            #print f
            if n < nfiles:
                lfiles.append("'" + prefix + f + "',\n")
            else:
                lfiles.append("'" + prefix + f + "'\n")
        n=n+1
    fileHandle.close()
    os.system("rm tmplist.txt")
    return lfiles

def main():
    
    if len(sys.argv) < 4:
        print("\n [Usage] python AnalyzeLumiScan.py <LumiScanLists.txt> <caf/lxplus/cmslpc> <local/batch>")
        sys.exit()

    cfidir = "../../python/"
    cfifilepath = "RecoVertex.BeamSpotProducer."
    
    lumilistfile = sys.argv[1]
    mode = sys.argv[2]
    jobmode = sys.argv[3]
    if mode == "caf":
        fmode = 0
    elif mode == "lxplus":
        fmode = 1
    elif mode == "cmslpc":
	fmode = 2
	jobmode = "local" ## temporary
    else:
        print("Mode not supported")
        sys.exit()

    if jobmode != "local" and jobmode != "batch":
        print("Jobe mode not supported")
        sys.exit()
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
                if i.lower() == "max":
                    dbsquery ="dbs search --noheader --query=\"find max(lumi) where dataset="+dataset
                    dbsquery += " and run ="+run[3:len(run)-1]+"\""
##                    print dbsquery
                    i = os.popen(dbsquery).read()
                runsinfo.setdefault(run,[]).append(int(i))
##    print runsinfo

    for i in range(len(runsinfo)):
        d=runsinfo.keys()[i]
        if os.path.exists(d):
            print("Directory \""+d[:len(d)-1]+"\" exists!!!")
            sys.exit() ## remove for test

    for i in range(len(runsinfo)):
        d=runsinfo.keys()[i]
        os.system("mkdir -p "+d[:len(d)-1])
        print("Output and log files will be saved to: ", end=' ')
        print(d[:len(d)-1])

        ## Create input cfi files according to run and lumi sections
        lumilist=runsinfo.get(d)
        for j in range((len(lumilist)+1)/2):
            files = get_list_files(int(d[3:len(d)-1]),lumilist[j*2],lumilist[j*2+1],fmode)
            if lumilist[j*2] < 10:
                minlumi = "00"+str(lumilist[j*2])
            elif lumilist[j*2] < 100:
                minlumi = "0"+str(lumilist[j*2])
            elif lumilist[j*2] < 1000:
                minlumi = str(lumilist[j*2])
            else:
                print("Lumi range greater than 1000!!!")
                sys.exit()

            if lumilist[j*2+1] < 10:
                maxlumi = "00"+str(lumilist[j*2+1])
            elif lumilist[j*2+1] < 100:
                maxlumi = "0"+str(lumilist[j*2+1])
            elif lumilist[j*2+1] < 1000:
                maxlumi = str(lumilist[j*2+1])
            else:
                print("Lumi range greater than 1000!!!")
                sys.exit()

            tagName = d[:len(d)-1]+"LS"+minlumi+"to"+maxlumi
##            print tagName
            
            fouttmp = open("Input_template_cfi.py")
            cfiname = cfidir+"LumiScan_"+tagName+"_cfi.py"
            fout = open(cfiname,"w")

            for line in fouttmp:
                if line.find("INPUTFILES")!=-1:
                    for f in files:
                        fout.write(f)

                if not "INPUTFILES" in line:
                    fout.write(line)

            fout.close()
            
            ## Copy cfi file to run directories
            os.system("cp "+cfiname+" "+d)            
            ## Create cfg files for cmsRun
            cfinametag = cfifilepath+"LumiScan_"+tagName+"_cfi"
            #print cfinametag
            lumirange = d[3:len(d)-1]+":"+str(lumilist[j*2])+"-"+d[3:len(d)-1]+":"+str(lumilist[j*2+1])
            #print lumirange
            asciioutfile = d+"LumiScan_"+tagName+".txt"
            #print asciioutfile
            treeoutfile = d+"LumiScan_"+tagName+".root"
            #print treeoutfile
            replacetag = [('INPUT_FILE',cfinametag),
                          ('LUMIRANGE',lumirange),
                          ('ASCIIFILE',asciioutfile),
                          ('OUTPUTFILE',treeoutfile)
                          ]

            tmpcfgfile = open("analyze_lumiscan_template_cfg.py")
            cfgtagname = "analyze_lumiscan_"+tagName+"_cfg.py"
            newcfgfile = open(cfgtagname,"w")
            for line in tmpcfgfile:
                for itag in replacetag:
                    line = line.replace(itag[0],itag[1])

                newcfgfile.write(line)

            newcfgfile.close()
            
            os.system("mv "+cfgtagname+" "+d)
            if jobmode == "local":
                runjobcmd = "cmsRun "+d+cfgtagname+" >& "+d+"LumiScan_"+tagName+".log &"
                print(runjobcmd)
                os.system(runjobcmd)

            ## Create job and submitting to batch
            if jobmode == "batch":
                workdir = os.environ['PWD']        
##                print workdir
                runlogtag = d+"LumiScan_"+tagName+".log"
                replacetag1 = [('PWD',workdir),
                               ('CFGFILE',d+cfgtagname),
                               ('CFGRUNLOG',runlogtag)
                               ]
                tmpbjobfile = open("AnalyzeLumiScanJob_template.sh","r")
                bjobName = "AnalyzeLumiScanJob_"+tagName+".sh"
                newbjobfile = open(bjobName,'w')

                for line in tmpbjobfile.readlines():
                    for itag in replacetag1:
                        line = line.replace(itag[0],itag[1])
                    newbjobfile.write(line)
                newbjobfile.write(line)
                os.system("chmod +x "+bjobName)
                if os.environ['SCRAM_ARCH'] == "slc5_ia32_gcc434":
                    submitjobcmd = "bsub -q 8nh -R \"type=SLC5_64\" "+bjobName
                    print(submitjobcmd)
                else:
                    submitjobcmd = "bsub -q 8nh "+bjobName
                    print(submitjobcmd)

                os.system(submitjobcmd)
                
    print("End of submitting jobs")

#_________________________________    
if __name__ =='__main__':
    sys.exit(main())
