#! /usr/bin/env python

import os
import time
import sys
import re
import random
from threading import Thread

scriptPath = os.path.dirname( os.path.abspath(sys.argv[0]) )
if scriptPath not in sys.path:
    sys.path.append(scriptPath)

        
class testit(Thread):
    def __init__(self,dirName, commandList):
        Thread.__init__(self)
        self.dirName = dirName
        self.commandList = commandList
        self.status=-1
        self.report=''
        self.nfail=[]
        self.npass=[]

        return
    
    def run(self):

        startime='date %s' %time.asctime()
        exitCodes = []

        for command in self.commandList:

            if not os.path.exists(self.dirName):
                os.makedirs(self.dirName)

            commandbase = command.replace(' ','_').replace('/','_')
            logfile='%s.log' % commandbase[:150].replace("'",'').replace('"','').replace('../','')
            
            executable = 'cd '+self.dirName+'; '+command+' > '+logfile+' 2>&1'

            ret = os.system(executable)
            exitCodes.append( ret )

        endtime='date %s' %time.asctime()
        tottime='%s-%s'%(endtime,startime)
    
        for i in range(len(self.commandList)):
            command = self.commandList[i]
            exitcode = exitCodes[i]
            if exitcode != 0:
                log='%s : FAILED - time: %s s - exit: %s\n' %(command,tottime,exitcode)
                self.report+='%s\n'%log
                self.nfail.append(1)
                self.npass.append(0)
            else:
                log='%s : PASSED - time: %s s - exit: %s\n' %(command,tottime,exitcode)
                self.report+='%s\n'%log
                self.nfail.append(0)
                self.npass.append(1)

        return

class StandardTester(object):

    def __init__(self, nThrMax=4):

        self.threadList = []
        self.maxThreads = nThrMax
        self.prepare()

        return

    def activeThreads(self):

        nActive = 0
        for t in self.threadList:
            if t.isAlive() : nActive += 1

        return nActive

    def prepare(self):
    
        self.devPath = os.environ['LOCALRT'] + '/src/'
        self.relPath = self.devPath
        if os.environ.has_key('CMSSW_RELEASE_BASE') and (os.environ['CMSSW_RELEASE_BASE'] != ""): self.relPath = os.environ['CMSSW_RELEASE_BASE'] + '/src/'

        lines = { 'read312RV' : ['cmsRun '+self.file2Path('Utilities/ReleaseScripts/scripts/read312RV_cfg.py')], 
                  'fastsim1'  : ['cmsRun '+self.file2Path('FastSimulation/Configuration/test/IntegrationTestFake_cfg.py')],
                  'fastsim2'  : ['cmsRun '+self.file2Path('FastSimulation/Configuration/test/IntegrationTest_cfg.py')],
                  #'fastsim3'  : ['cmsRun '+self.file2Path('FastSimulation/Configuration/test/ExampleWithHLT_1E31_cfg.py')],
                  'fastsim4'  : ['cmsRun '+self.file2Path('FastSimulation/Configuration/test/IntegrationTestWithHLT_cfg.py')],
                  'pat1'      : ['cmsRun '+self.file2Path('PhysicsTools/PatAlgos/test/IntegrationTest_cfg.py')],
                }

        hltTests = { 'hlt1' : ['cmsDriver.py TTbar_Tauola.cfi -s GEN,SIM,DIGI,L1,DIGI2RAW --mc --scenario=pp -n 10 --conditions auto:startup_GRun --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAW --fileout file:RelVal_Raw_GRun_STARTUP.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_GRun.py'), 
                               'cmsDriver.py RelVal -s HLT:GRun,RAW2DIGI,L1Reco,RECO --mc --scenario=pp -n 10 --conditions auto:startup_GRun --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --processName=HLTRECO --filein file:RelVal_Raw_GRun_STARTUP.root --fileout file:RelVal_Raw_GRun_STARTUP_HLT_RECO.root'], 
                     'hlt2' : ['cmsDriver.py TTbar_Tauola.cfi -s GEN,SIM,DIGI,L1,DIGI2RAW --mc --scenario=HeavyIons -n 10 --conditions auto:starthi_HIon --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAW --fileout file:RelVal_Raw_HIon_STARTUP.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_HIon.py'),
                               'cmsDriver.py RelVal -s HLT:HIon,RAW2DIGI,L1Reco,RECO --mc --scenario=HeavyIons -n 10 --conditions auto:starthi_HIon --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --processName=HLTRECO --filein file:RelVal_Raw_HIon_STARTUP.root --fileout file:RelVal_Raw_HIon_STARTUP_HLT_RECO.root'],
                     'hlt3' : ['cmsDriver.py TTbar_Tauola.cfi -s GEN,SIM,DIGI,L1,DIGI2RAW --mc --scenario=pp -n 10 --conditions auto:startup_PIon --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAW --fileout file:RelVal_Raw_PIon_STARTUP.root', 
                               'cmsRun ' + self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_PIon.py'),
                               'cmsDriver.py RelVal -s HLT:PIon,RAW2DIGI,L1Reco,RECO --mc --scenario=pp -n 10 --conditions auto:startup_PIon --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --processName=HLTRECO --filein file:RelVal_Raw_PIon_STARTUP.root --fileout file:RelVal_Raw_PIon_STARTUP_HLT_RECO.root'],
                     'hlt4' : ['cmsDriver.py RelVal -s L1REPACK --data --scenario=pp -n 10 --conditions auto:hltonline_GRun --relval 9000,50 --datatier "RAW" --eventcontent RAW --fileout file:RelVal_Raw_GRun_DATA.root --filein /store/data/Run2012A/MuEG/RAW/v1/000/191/718/14932935-E289-E111-830C-5404A6388697.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnData_HLT_GRun.py'),
                               'cmsDriver.py RelVal -s HLT:GRun,RAW2DIGI,L1Reco,RECO --data --scenario=pp -n 10 --conditions auto:com10_GRun --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --processName=HLTRECO --filein file:RelVal_Raw_GRun_DATA.root --fileout file:RelVal_Raw_GRun_DATA_HLT_RECO.root'],
                     'hlt5' : ['cmsDriver.py RelVal -s L1REPACK --data --scenario=HeavyIons -n 10 --conditions auto:hltonline_HIon --relval 9000,50 --datatier "RAW" --eventcontent RAW --fileout file:RelVal_Raw_HIon_DATA.root --filein /store/hidata/HIRun2011/HIHighPt/RAW/v1/000/182/838/F20AAF66-F71C-E111-9704-BCAEC532971D.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnData_HLT_HIon.py'),
                               'cmsDriver.py RelVal -s HLT:HIon,RAW2DIGI,L1Reco,RECO --data --scenario=HeavyIons -n 10 --conditions auto:com10_HIon --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --processName=HLTRECO --filein file:RelVal_Raw_HIon_DATA.root --fileout file:RelVal_Raw_HIon_DATA_HLT_RECO.root'],
                     'hlt6' : ['cmsDriver.py RelVal -s L1REPACK --data --scenario=pp -n 10 --conditions auto:hltonline_PIon --relval 9000,50 --datatier "RAW" --eventcontent RAW --fileout file:RelVal_Raw_PIon_DATA.root --filein /store/data/Run2012A/MuEG/RAW/v1/000/191/718/14932935-E289-E111-830C-5404A6388697.root',
                               'cmsRun ' + self.file2Path('HLTrigger/Configuration/test/OnData_HLT_PIon.py'),
                               'cmsDriver.py RelVal -s HLT:PIon,RAW2DIGI,L1Reco,RECO --data --scenario=pp -n 10 --conditions auto:com10_PIon --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --processName=HLTRECO --filein file:RelVal_Raw_PIon_DATA.root --fileout file:RelVal_Raw_PIon_DATA_HLT_RECO.root'],
                     }

        self.commands={}
        for dirName, command in lines.items():
            self.commands[dirName] = command

        for dirName, commandList in hltTests.items():
            self.commands[dirName] = commandList
        return
	
    def dumpTest(self):
        print ",".join(self.commands.keys())
        return

    def file2Path(self,rFile):

        fullPath = self.relPath + rFile
        if os.path.exists(self.devPath + rFile): fullPath = self.devPath + rFile
        return fullPath

    def runTests(self, testList = None):

        actDir = os.getcwd()

        if not os.path.exists('addOnTests'):
            os.makedirs('addOnTests')
        os.chdir('addOnTests')

        nfail=0
    	npass=0
    	report=''
    	
    	print 'Running in %s thread(s)' % self.maxThreads
    	
        for dirName, command in self.commands.items():

    	    if testList and not dirName in testList:
                del self.commands[dirName]
                continue

            # make sure we don't run more than the allowed number of threads:
    	    while self.activeThreads() >= self.maxThreads:
    	        time.sleep(10)
                continue
    	    
    	    print 'Preparing to run %s' % str(command)
    	    current = testit(dirName, command)
    	    self.threadList.append(current)
    	    current.start()
            time.sleep(random.randint(1,5)) # try to avoid race cond by sleeping random amount of time [1,5] sec 
            
        # wait until all threads are finished
        while self.activeThreads() > 0:
    	    time.sleep(5)
    	    
    	# all threads are done now, check status ...
    	for pingle in self.threadList:
    	    pingle.join()
            for f in pingle.nfail: nfail  += f
            for p in pingle.npass: npass  += p
    	    report += pingle.report
    	    print pingle.report
            sys.stdout.flush()
            
    	reportSumm = '\n %s tests passed, %s failed \n' %(npass,nfail)
    	print reportSumm
    	
    	runall_report_name='runall-report.log'
    	runall_report=open(runall_report_name,'w')
    	runall_report.write(report+reportSumm)
    	runall_report.close()

        # get the logs to the logs dir:
        print '==> in :', os.getcwd()
        print '    going to copy log files to logs dir ...'
        if not os.path.exists('logs'):
            os.makedirs('logs')
        for dirName in self.commands:
            cmd = "for L in `ls "+dirName+"/*.log`; do cp $L logs/cmsDriver-`dirname $L`_`basename $L` ; done"
            print "going to ",cmd
            os.system(cmd)

        import pickle
        pickle.dump(self.commands, open('logs/addOnTests.pkl', 'w') )

        os.chdir(actDir)
        
    	return

    def upload(self, tgtDir):

        print "in ", os.getcwd()

        if not os.path.exists(tgtDir):
            os.makedirs(tgtDir)
        
        cmd = 'tar cf - addOnTests.log addOnTests/logs | (cd '+tgtDir+' ; tar xf - ) '
        try:
            print 'executing: ',cmd
            ret = os.system(cmd)
            if ret != 0:
                print "ERROR uploading logs:", ret, cmd
        except Exception, e:
            print "EXCEPTION while uploading addOnTest-logs : ", str(e)
            
    	return

                
def main(argv) :

    import getopt
    
    try:
        opts, args = getopt.getopt(argv, "dj:t:", ["nproc=", 'uploadDir=', 'tests=','noRun','dump'])
    except getopt.GetoptError, e:
        print "unknown option", str(e)
        sys.exit(2)
        
    np        = 4
    uploadDir = None
    runTests  = True
    testList  = None
    dump      = False
    for opt, arg in opts :
        if opt in ('-j', "--nproc" ):
            np=int(arg)
        if opt in ("--uploadDir", ):
            uploadDir = arg
        if opt in ('--noRun', ):
            runTests = False
        if opt in ('-d','--dump', ):
            dump = True
        if opt in ('-t','--tests', ):
            testList = arg.split(",")

    tester = StandardTester(np)
    if dump:
        tester.dumpTest()
    else:
        if runTests:
            tester.runTests(testList)
        if uploadDir:
            tester.upload(uploadDir)
    return
    
if __name__ == '__main__' :
    main(sys.argv[1:])
