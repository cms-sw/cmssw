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
def getUploadedIOVs(tagName,destDB="oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT"):
    listIOVCommand = "cmscond_list_iov -c " + destDB + " -P /afs/cern.ch/cms/DB/conddb -t " + tagName
    dbError = commands.getstatusoutput( listIOVCommand )
    if dbError[0] != 0 :
        if dbError[1].find("metadata entry \"" + tagName + "\" does not exist") != -1:
            exit(dbError[1])
        else:
            exit("ERROR: Can\'t connect to db because:\n" + dbError[1])
            
            
    aCommand = listIOVCommand + " | grep DB= | awk \'{print $1}\'"
    #print aCommand
    output = commands.getstatusoutput( aCommand )
            
    #WARNING when we pass to lumi IOV this should be long long
    if output[1] == '':
        exit("ERROR: The tag " + tagName + " exists but I can't get the value of the last IOV")
                
    runs = []    
    for run in output[1].split('\n'):
        runs.append(long(run))
                
    return runs

#####################################################################################
def getListOfRunsAndLumiFromRR(lastRun=-1,runErrors={}):
    RunReg  ="http://pccmsdqm04.cern.ch/runregistry"
    #RunReg  = "http://localhost:40010/runregistry"
    #Dataset=%Online%
    Group   = "Collisions10"

    # get handler to RR XML-RPC server
    FULLADDRESS=RunReg + "/xmlrpc"
    #print "RunRegistry from: ",FULLADDRESS
    server = xmlrpclib.ServerProxy(FULLADDRESS)
    #sel_runtable="{groupName} ='" + Group + "' and {runNumber} > " + str(lastRun) + " and {datasetName} LIKE '" + Dataset + "'"
    sel_runtable="{groupName} ='" + Group + "' and {runNumber} > " + str(lastRun)
    sel_dcstable="{groupName} ='" + Group + "' and {runNumber} > " + str(lastRun) + " and {parDcsBpix} = 1 and {parDcsFpix} = 1 and {parDcsTibtid} = 1 and {parDcsTecM} = 1 and {parDcsTecP} = 1 and {parDcsTob} = 1 and {parDcsEbminus} = 1 and {parDcsEbplus} = 1 and {parDcsEeMinus} = 1 and {parDcsEePlus} = 1 and {parDcsEsMinus} = 1 and {parDcsEsPlus} = 1 and {parDcsHbheA} = 1 and {parDcsHbheB} = 1 and {parDcsHbheC} = 1 and {parDcsH0} = 1 and {parDcsHf} = 1"

    tries = 0;
    while tries<10:
        try:
            run_data = server.DataExporter.export('RUN'           , 'GLOBAL', 'csv_runs', sel_runtable)
            dcs_data = server.DataExporter.export('RUNLUMISECTION', 'GLOBAL', 'json'    , sel_dcstable)
            #print run_data
            #print dcs_data
            break
        except:
            print "Something wrong in accessing runregistry, retrying in 5s...."
            tries += 1
            time.sleep(5)
        if tries==10:
            error = "Run registry unaccessible.....exiting now"
            sys.exit(error)
                
                
    listOfRuns=[]
    for line in run_data.split("\n"):
        run=line.split(',')[0]
        if run.isdigit():
            listOfRuns.append(run)
            runErrors[long(run)] = line.split(',')[19:27]
                        
                        
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
    printExtra = False
    tagNumber = "14"
    if len(sys.argv) >= 2:
        if not sys.argv[1].isdigit():
            exit("USAGE: ./checkPayloads.py (optional tagNumber)")
        else:
            tagNumber = sys.argv[1]
    destDB = ""
    if(len(sys.argv) >= 3):
        destDB = sys.argv[2]
    #132573 Beam lost immediately
    #132958 Bad strips
    #133081 Bad pixels bad strips
    #133242 Bad strips
    #133472 Bad strips
    #133473 Only 20 lumisection, run duration 00:00:03:00 
    #133509 Should be good!!!!!!!!!!
    #136290 Bad Pixels bad strips
    #138560 Bad pixels bad strips
    #138562 Bad HLT bad L1T, need to rescale the Jet Triggers
    #139363 NOT in the bad list but only 15 lumis and stopped for DAQ problems
    #139455 Bad Pixels and Strips and stopped because of HCAL trigger rate too high
    #140133 Beams dumped
    #140182 No pixel and Strips with few entries
    #141865 Pixel are bad but Strips work. Run is acceptable but need relaxed cuts since there are no pixels. BeamWidth measurement is bad 80um compared to 40um
    #142461 Run crashed immediately due to PIX, stable beams since LS1
    #142465 PostCollsions10, beams lost, HCAl DQM partly working
    #142503 Bad pixels bad strips
    #142653 Strips not in data taking
    #143977 No Beam Strips and Pixels bad
    
    knownMissingRunList = [132573,132958,133081,133242,133472,133473,136290,138560,138562,139455,140133,140182,142461,142465,142503,142653,143977]
    tagName = "BeamSpotObjects_2009_v" + tagNumber + "_offline"
    print "Checking payloads for tag " + tagName
    runErrors = {}
    listOfRunsAndLumiFromRR = getListOfRunsAndLumiFromRR(-1,runErrors)
    if(destDB != ""):
        listOfIOVs = getUploadedIOVs(tagName,destDB) 
    else:
        listOfIOVs = getUploadedIOVs(tagName)
    RRRuns = listOfRunsAndLumiFromRR.keys()
    RRRuns.sort()
    for run in RRRuns:
        #print listOfRunsAndLumiFromRR[run]
        if run not in listOfIOVs:
            extraMsg = ""
            if listOfRunsAndLumiFromRR[run] == [[]]:
                extraMsg = " but it is empty in the RR"
                if not printExtra: continue
            if run in knownMissingRunList :
                extraMsg = " but this run is know to be bad " #+ runErrors[run]
                if not printExtra: continue
            print "Run: " + str(run) + " is missing for DB tag " + tagName + extraMsg 
    

if __name__ == "__main__":
    main()
            
