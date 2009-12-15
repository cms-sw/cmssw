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

import sys,os,re,string
import commands

def get_list_files(run,ls1,ls2,mode):
    lfiles = []
    dbsquery = "dbs search --query=\"find file where dataset=/ExpressPhysics/*/FEVT and run="+str(run)+" and lumi>="+str(ls1)+" and lumi<="+str(ls2)+" order by lumi asc\" > tmplist.txt"
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
    else:
        print "Mode = "+str(mode)+" is not supported"

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
        print "\n [Usage] python AnalyzeLumiScan.py <LumiScanLists.txt> <caf/lxplus> <local/batch>"
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
    else:
        print "Mode not supported"
        sys.exit()

    if jobmode != "local" and jobmode != "batch":
        print "Jobe mode not supported"
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
                runsinfo.setdefault(run,[]).append(int(i))
##    print runsinfo

    for i in range(len(runsinfo)):
        d=runsinfo.keys()[i]
        if os.path.exists(d):
            print "Directory \""+d[:len(d)-1]+"\" exists!!!"
            sys.exit() ## remove for test

    for i in range(len(runsinfo)):
        d=runsinfo.keys()[i]
        os.system("mkdir -p "+d[:len(d)-1])
        print "Output and log files will be saved to: ",
        print d[:len(d)-1]

        ## Create input cfi files according to run and lumi sections
        lumilist=runsinfo.get(d)
        for j in range((len(lumilist)+1)/2):
            files = get_list_files(int(d[3:len(d)-1]),lumilist[j*2],lumilist[j*2+1],fmode)
            tagName = d[:len(d)-1]+"LS"+str(lumilist[j*2])+"-"+str(lumilist[j*2+1])
            print tagName
            
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
            if jobmode == "local":
                runjobcmd = "cmsRun "+cfgtagname+" >& "+d+"LumiScan_"+tagName+".log &"
                print runjobcmd
                os.system(runjobcmd)

            ## Create job and submitting to batch
            if jobmode == "batch":
                workdir = os.environ['PWD']        
##                print workdir
                runlogtag = d+"LumiScan_"+tagName+".log"
                replacetag1 = [('PWD',workdir),
                               ('CFGFILE',cfgtagname),
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

                submitjobcmd = "bsub -q 8nh "+bjobName
                print submitjobcmd
                os.system(submitjobcmd)

    print "End of submitting jobs"

#_________________________________    
if __name__ =='__main__':
    sys.exit(main())
