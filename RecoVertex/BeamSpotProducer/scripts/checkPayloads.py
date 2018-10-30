#!/usr/bin/env python
from __future__ import print_function
import sys,os,commands,re
import xmlrpclib
from CommonMethods import *

try: # FUTURE: Python 2.6, prior to 2.6 requires simplejson
    import json
except:
    try:
        import simplejson as json
    except:
        print("Please set a crab environment in order to get the proper JSON lib")
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
            print("Trying to get run data. This fails only 2-3 times so don't panic yet...", tries, "/", maxAttempts)
            time.sleep(1)
            print("Exception type: ", sys.exc_info()[0])
        if tries==maxAttempts:
            error = "Ok, now panic...run registry unaccessible...I'll get the runs from a json file!"
            print(error);
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
            print("I was able to get the list of runs and now I am trying to access the detector status", tries, "/", maxAttempts)
            time.sleep(1)
            print("Exception type: ", sys.exc_info()[0])

        if tries==maxAttempts:
            error = "Ok, now panic...run registry unaccessible...I'll get the runs from a json file!"
            print(error);
            return {};

    #This is the original and shold work in the furture as soon as the server will be moved to a more powerfull PC
    #while tries<maxAttempts:
    #    try:
    #        run_data = server.DataExporter.export('RUN'           , 'GLOBAL', 'csv_runs', sel_runtable)
    #        dcs_data = server.DataExporter.export('RUNLUMISECTION', 'GLOBAL', 'json'    , sel_dcstable)
    #        #print run_data
    #        #print dcs_data
    #        break
    #    except:
    #        print "Something wrong in accessing runregistry, retrying in 5s...."
    #        tries += 1
    #        time.sleep(2)
    #        print "Exception type: ", sys.exc_info()[0]
    #
    #    if tries==maxAttempts:
    #        error = "Run registry unaccessible.....exiting now"
    #        sys.exit(error)



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
    usage = "USAGE: ./checkPayloads.py (optional tagNumber) (optional \"lumi\") (optional \"z\" (optional destDB)"
    printExtra = False
    tagNumber = "14"
    dbBase = ""
    sigmaZ = ""

    if len(sys.argv) >= 2:
        if not sys.argv[1].isdigit():
            exit(usage)
        else:
            tagNumber = sys.argv[1]
    if len(sys.argv) >= 3:
        if not sys.argv[2] == "lumi":
            exit(usage)
        else:
            dbBase = "_LumiBased"
    if len(sys.argv) >= 4:
        if not sys.argv[3] == "z":
            exit(usage)
        else:
            sigmaZ = "_SigmaZ"
    destDB = ""
    if(len(sys.argv) > 4):
        destDB = sys.argv[4]
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
    #148859 Strips and Pixels HV off waiting for beam 

    knownMissingRunList = [132573,132958,133081,133242,133472,133473,136290,138560,138562,139455,140133,140182,142461,142465,142503,142653,143977,148859]
    tagName = "BeamSpotObjects_2009" + dbBase + sigmaZ + "_v" + tagNumber + "_offline"
    print("Checking payloads for tag " + tagName)
    listOfRunsAndLumi = {};
    #listOfRunsAndLumi = getListOfRunsAndLumiFromRR(-1);
    if(not listOfRunsAndLumi):
        listOfRunsAndLumi = getListOfRunsAndLumiFromFile(-1,"/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions10/7TeV/StreamExpress/Cert_132440-149442_7TeV_StreamExpress_Collisions10_JSON_v3.txt");
    tmpListOfIOVs = []
    if(destDB != ""):
        tmpListOfIOVs = getUploadedIOVs(tagName,destDB) 
    else:
        tmpListOfIOVs = getUploadedIOVs(tagName)


    listOfIOVs = []
    if(dbBase == ''):
        listOfIOVs = tmpListOfIOVs
    else:
        for iov in tmpListOfIOVs:
            if((iov >> 32) not in listOfIOVs):
                listOfIOVs.append(iov >>32)
    RRRuns = sorted(listOfRunsAndLumi.keys())
    for run in RRRuns:
        #print listOfRunsAndLumiFromRR[run]
        if run not in listOfIOVs:
            extraMsg = ""
            if listOfRunsAndLumi[run] == [[]]:
                extraMsg = " but it is empty in the RR"
                if not printExtra: continue
            if run in knownMissingRunList :
                extraMsg = " but this run is know to be bad " #+ runErrors[run]
                if not printExtra: continue
            print("Run: " + str(run) + " is missing for DB tag " + tagName + extraMsg) 


if __name__ == "__main__":
    main()

