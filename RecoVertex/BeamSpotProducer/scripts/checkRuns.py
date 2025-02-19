#!/usr/bin/env python
import sys,os,commands,re
import xmlrpclib
from CommonMethods import *
try: # FUTURE: Python 2.6, prior to 2.6 requires simplejson
    import json
except:
    try:
        import simplejson as json
    except:
        print "Please set a crab environment in order to get the proper JSON lib"
        sys.exit(1)

#####################################################################################
def getListOfRunsAndLumiFromFile(firstRun=-1,fileName=""):
    file = open(fileName);
    jsonFile = file.read();
    file.close()
    jsonList=json.loads(jsonFile);
    
    selected_dcs = {};
    for element in jsonList:
        selected_dcs[long(element)]=jsonList[element]
    return selected_dcs
    
#####################################################################################
def getListOfRunsAndLumiFromRR(firstRun=-1,error=""):
    RunReg  ="http://pccmsdqm04.cern.ch/runregistry"
    #RunReg  = "http://localhost:40010/runregistry"
    #Dataset=%Online%
    Group   = "Collisions10"
    
    # get handler to RR XML-RPC server
    FULLADDRESS=RunReg + "/xmlrpc"
    #print "RunRegistry from: ",FULLADDRESS
    #firstRun = 153000
    server = xmlrpclib.ServerProxy(FULLADDRESS)
    #sel_runtable="{groupName} ='" + Group + "' and {runNumber} > " + str(firstRun) + " and {datasetName} LIKE '" + Dataset + "'"
    sel_runtable="{groupName} ='" + Group + "' and {runNumber} > " + str(firstRun)
        
    tries = 0;
    maxAttempts = 3
    while tries<maxAttempts:
        try:
            run_data = server.DataExporter.export('RUN', 'GLOBAL', 'csv_runs', sel_runtable)
            break
        except:
            tries += 1
            print "Trying to get run data. This fails only 2-3 times so don't panic yet...", tries, "/", maxAttempts
            time.sleep(1)
            print "Exception type: ", sys.exc_info()[0]
        if tries==maxAttempts:
            error = "Ok, now panic...run registry unaccessible...I'll get the runs from a json file!"
            print error;
            return {};
            
    listOfRuns=[]
    runErrors = {}
    for line in run_data.split("\n"):
        run=line.split(',')[0]
        if run.isdigit():
            listOfRuns.append(run)
                    
    tries = 0
    maxAttempts = 3
    firstRun = listOfRuns[len(listOfRuns)-1];
    lastRun  = listOfRuns[0];
    sel_dcstable="{groupName} ='" + Group + "' and {runNumber} >= " + str(firstRun) + " and {runNumber} <= " + str(lastRun) + " and {parDcsBpix} = 1 and {parDcsFpix} = 1 and {parDcsTibtid} = 1 and {parDcsTecM} = 1 and {parDcsTecP} = 1 and {parDcsTob} = 1 and {parDcsEbminus} = 1 and {parDcsEbplus} = 1 and {parDcsEeMinus} = 1 and {parDcsEePlus} = 1 and {parDcsEsMinus} = 1 and {parDcsEsPlus} = 1 and {parDcsHbheA} = 1 and {parDcsHbheB} = 1 and {parDcsHbheC} = 1 and {parDcsH0} = 1 and {parDcsHf} = 1"
    while tries<maxAttempts:
        try:
            dcs_data = server.DataExporter.export('RUNLUMISECTION', 'GLOBAL', 'json'    , sel_dcstable)
            break
        except:
            tries += 1
            print "I was able to get the list of runs and now I am trying to access the detector status", tries, "/", maxAttempts
            time.sleep(1)
            print "Exception type: ", sys.exc_info()[0]
            
    if tries==maxAttempts:
        error = "Ok, now panic...run registry unaccessible...I'll get the runs from a json file!"
        print error;
        return {};
            
    selected_dcs={}
    jsonList=json.loads(dcs_data)
    
    #for element in jsonList:
    for element in listOfRuns:
        #if element in listOfRuns:
        if element in jsonList:
            selected_dcs[long(element)]=jsonList[element]
        else:
            #print "WARNING: Run " + element + " is a collision10 run with 0 lumis in Run Registry!"
            selected_dcs[long(element)]= [[]]
    return selected_dcs
        
#####################################################################################
def main():
    filesDir = "LatestRuns/Results/";
    fileList = ls(filesDir)
    listOfRunsAndLumi = {};
    #listOfRunsAndLumi = getListOfRunsAndLumiFromRR(-1);
    if(not listOfRunsAndLumi):
        listOfRunsAndLumi = getListOfRunsAndLumiFromFile(-1,"/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions10/7TeV/StreamExpress/Cert_132440-149442_7TeV_StreamExpress_Collisions10_JSON_v3.txt");

    runKeys = listOfRunsAndLumi.keys();            
    runKeys.sort();
    runFiles = [];
    for fileName in fileList:
        regExp = re.search('(\D+)(\d+)_(\d+)_(\d+).txt',fileName);
        if(not regExp):
            error = "Can't find reg exp";
            exit(error);
        runFiles.append(long(regExp.group(3)));    

    #for run in runKeys:
    #    if(run not in runFiles):   
    #        print "Can't find run", run, "in the files!"        

    runsAndLumisInRR = {};        
    for run in runKeys:
        RRList = [];
        for lumiRange in listOfRunsAndLumi[run]:
            if lumiRange != []:
                for l in range(lumiRange[0],lumiRange[1]+1):
                    RRList.append(long(l));
        #print run, "->", RRList;            
        runsAndLumisInRR[run] = RRList;

    runsAndLumisProcessed = {}
    for fileName in fileList:
        file = open(filesDir+fileName)
        for line in file:
            if line.find("Runnumber") != -1:
                run = long(line.replace('\n','').split(' ')[1])
            elif line.find("LumiRange") != -1:
                lumiLine = line.replace('\n','').split(' ')
                begLumi = long(lumiLine[1])
                endLumi = long(lumiLine[3])
                if begLumi != endLumi:
                    error = "The lumi range is greater than 1 for run " + str(run) + " " + line + " in file: " + runListDir + fileName
                    exit(error)
                else:
                    if not run in runsAndLumisProcessed:
                        runsAndLumisProcessed[run] = []
                    if begLumi in runsAndLumisProcessed[run]:
                        print "Lumi " + str(begLumi) + " in event " + str(run) + " already exist. This MUST not happen but right now I will ignore this lumi!"
                    else:
                        runsAndLumisProcessed[run].append(begLumi)
        file.close()
        #print run, "->", runsAndLumisProcessed[run];            

    for run in runKeys:               
        missingLumis = [];    
        for lumi in runsAndLumisInRR[run]:
            #print str(counter) + "->" + str(lumi)
            #counter += 1
            if(run not in runFiles):   
                print "Can't find run", run, "in the files!"        
                break ;
            elif( not lumi in runsAndLumisProcessed[run]):
                missingLumis.append(lumi)
        if(len(missingLumis) != 0):        
            print "In run", run, "these lumis are missing ->", missingLumis          
                                                     
                            
if __name__ == "__main__":
        main()
    
