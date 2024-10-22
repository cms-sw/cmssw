#! /usr/bin/env python3

from __future__ import print_function
from builtins import range
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

    def run(self):
        try:
            os.makedirs(self.dirName)
        except:
            pass

        with open(self.dirName+'/cmdLog', 'w') as clf:
            clf.write(f'# {self.dirName}\n')

            for cmdIdx, command in enumerate(self.commandList):
                clf.write(f'\n{command}\n')

                time_start = time.time()
                exitcode = os.system(f'cd {self.dirName} && {command} > step{cmdIdx+1}.log 2>&1')
                time_elapsed_sec = round(time.time() - time_start)

                timelog = f'elapsed time: {time_elapsed_sec} sec (ended on {time.asctime()})'
                logline = f'[{self.dirName}:{cmdIdx+1}] {command} : '
                if exitcode != 0:
                    logline += 'FAILED'
                    self.nfail.append(1)
                    self.npass.append(0)
                else:
                    logline += 'PASSED'
                    self.nfail.append(0)
                    self.npass.append(1)
                logline += f' - {timelog} - exit: {exitcode}'
                self.report += logline+'\n\n'

class StandardTester(object):

    def __init__(self, nThrMax=4):

        self.threadList = []
        self.maxThreads = nThrMax
        self.prepare()

        return

    def activeThreads(self):

        nActive = 0
        for t in self.threadList:
            if t.is_alive() : nActive += 1

        return nActive

    def prepare(self):

        self.devPath = os.environ['LOCALRT'] + '/src/'
        self.relPath = self.devPath
        if 'CMSSW_RELEASE_BASE' in os.environ and (os.environ['CMSSW_RELEASE_BASE'] != ""): self.relPath = os.environ['CMSSW_RELEASE_BASE'] + '/src/'

        lines = { 'read312RV' : ['cmsRun '+self.file2Path('Utilities/ReleaseScripts/scripts/read312RV_cfg.py')], 
                  'fastsim'   : ["cmsDriver.py TTbar_8TeV_TuneCUETP8M1_cfi  --conditions auto:run1_mc --fast  -n 100 --eventcontent AODSIM,DQM --relval 100000,1000 -s GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,L1Reco,RECO,VALIDATION  --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --datatier GEN-SIM-DIGI-RECO,DQMIO --beamspot Realistic8TeVCollision"],
                  'fastsim1'  : ["cmsDriver.py TTbar_13TeV_TuneCUETP8M1_cfi --conditions auto:run2_mc_l1stage1 --fast  -n 100 --eventcontent AODSIM,DQM --relval 100000,1000 -s GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,L1Reco,RECO,VALIDATION  --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --datatier GEN-SIM-DIGI-RECO,DQMIO --beamspot NominalCollision2015 --era Run2_25ns"],
                  'fastsim2'  : ["cmsDriver.py TTbar_13TeV_TuneCUETP8M1_cfi --conditions auto:run2_mc --fast  -n 100 --eventcontent AODSIM,DQM --relval 100000,1000 -s GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,L1Reco,RECO,VALIDATION  --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --datatier GEN-SIM-DIGI-RECO,DQMIO --beamspot NominalCollision2015 --era Run2_2016"],
                  'pat1'      : ['cmsRun '+self.file2Path('PhysicsTools/PatAlgos/test/IntegrationTest_cfg.py')],
                }

        hltTests = {}
        hltFlag_data = 'realData=True globalTag=@ inputFiles=@'
        hltFlag_mc = 'realData=False globalTag=@ inputFiles=@'
        from Configuration.HLT.addOnTestsHLT import addOnTestsHLT
        hltTestsToAdd = addOnTestsHLT()
        for key in hltTestsToAdd:
            if '_data_' in key:
                hltTests[key] = [hltTestsToAdd[key][0],
                                 'cmsRun '+self.file2Path(hltTestsToAdd[key][1])+' '+hltFlag_data,
                                 hltTestsToAdd[key][2]]
            elif '_mc_' in key:
                hltTests[key] = [hltTestsToAdd[key][0],
                                 'cmsRun '+self.file2Path(hltTestsToAdd[key][1])+' '+hltFlag_mc,
                                 hltTestsToAdd[key][2]]
            else:
                hltTests[key] = [hltTestsToAdd[key][0],
                                 'cmsRun '+self.file2Path(hltTestsToAdd[key][1]),
                                 hltTestsToAdd[key][2]]

        self.commands = {}
        for dirName, command in lines.items():
            self.commands[dirName] = command

        for dirName, commandList in hltTests.items():
            self.commands[dirName] = commandList
        return

    def dumpTest(self):
        print(",".join(self.commands.keys()))
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

        print('Running in %s thread(s)' % self.maxThreads)

        if testList:
            self.commands = {d:c for d,c in self.commands.items() if d in testList}
        for dirName, command in self.commands.items():

            # make sure we don't run more than the allowed number of threads:
            while self.activeThreads() >= self.maxThreads:
                time.sleep(10)
                continue

            print('Preparing to run %s' % str(command))
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
            print(pingle.report)
            sys.stdout.flush()

        reportSumm = '\n %s tests passed, %s failed \n' %(npass,nfail)
        print(reportSumm)

        runall_report_name='runall-report.log'
        runall_report=open(runall_report_name,'w')
        runall_report.write(report+reportSumm)
        runall_report.close()

        # get the logs to the logs dir:
        print('==> in :', os.getcwd())
        print('    going to copy log files to logs dir ...')
        if not os.path.exists('logs'):
            os.makedirs('logs')
        for dirName in self.commands:
            cmd = "for L in `ls "+dirName+"/*.log`; do cp $L logs/cmsDriver-`dirname $L`_`basename $L` ; done"
            print("going to ",cmd)
            os.system(cmd)

        import pickle
        pickle.dump(self.commands, open('logs/addOnTests.pkl', 'wb'), protocol=2)

        os.chdir(actDir)

        return

    def upload(self, tgtDir):

        print("in ", os.getcwd())

        if not os.path.exists(tgtDir):
            os.makedirs(tgtDir)

        cmd = 'tar cf - addOnTests.log addOnTests/logs | (cd '+tgtDir+' ; tar xf - ) '
        try:
            print('executing: ',cmd)
            ret = os.system(cmd)
            if ret != 0:
                print("ERROR uploading logs:", ret, cmd)
        except Exception as e:
            print("EXCEPTION while uploading addOnTest-logs : ", str(e))

        return


def main(argv) :

    import getopt

    try:
        opts, args = getopt.getopt(argv, "dj:t:", ["nproc=", 'uploadDir=', 'tests=','noRun','dump'])
    except getopt.GetoptError as e:
        print("unknown option", str(e))
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
