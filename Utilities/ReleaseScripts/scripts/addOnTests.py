#! /usr/bin/env python

import os
import time
import sys
import re
import random
from threading import Thread

scriptPath = os.path.dirname( os.path.abspath(sys.argv[0]) )
print "scriptPath:", scriptPath
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

        startDir = os.getcwd()
        
        for command in self.commandList:

            if not os.path.exists(self.dirName):
                os.makedirs(self.dirName)

            commandbase = command.replace(' ','_').replace('/','_')
            logfile='%s.log' % commandbase[:150].replace("'",'').replace('../','')
            
            if os.path.exists( os.path.join(os.environ['CMS_PATH'],'cmsset_default.sh') ) :
                executable = 'source $CMS_PATH/cmsset_default.sh; eval `scram run -sh`;'
            else:
                executable = 'source $CMS_PATH/sw/cmsset_default.sh; eval `scram run -sh`;'
            # only if needed! executable += 'export FRONTIER_FORCERELOAD=long;' # force reload of db
            executable += 'cd '+self.dirName+';'
            executable += '%s > %s 2>&1' %(command, logfile)

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

        os.chdir(startDir)
        
        return

class StandardTester(object):

    def __init__(self, nThrMax=4):

        self.threadList = []
        self.maxThreads = nThrMax

        return

    def activeThreads(self):

        nActive = 0
        for t in self.threadList:
            if t.isAlive() : nActive += 1

        return nActive

    def prepare(self):

        cmd = 'ln -s /afs/cern.ch/user/a/andreasp/public/IBTests/read*.py .'
        try:
            os.system(cmd)
        except:
            pass


        tstPkgs = { 'FastSimulation' : [ 'Configuration' ],
	            'HLTrigger'      : [ 'Configuration' ],
		    'PhysicsTools'   : [ 'PatAlgos'      ],
		  }

	#-ap: make sure the actual package is there, not just the subsystem ...
        # and set symlinks accordingly ...
        pkgPath = os.environ['CMSSW_BASE'] + '/src/'
        relPath = '$CMSSW_RELEASE_BASE/src/'
        cmd = ''
        for tstSys in tstPkgs:
          if not os.path.exists(pkgPath + tstSys):
             cmd  = 'ln -s ' + relPath + tstSys + ' .;'
             try:
                print 'setting up symlink for ' + tstSys + ' using ' + cmd
                os.system(cmd)
             except:
                pass
          else:
	    for tstPkg in tstPkgs[tstSys]:
              if not os.path.exists(pkgPath + tstSys + "/" + tstPkg):
                 cmd  = 'mkdir -p ' + tstSys + '; ln -s ' + relPath + tstSys + '/' + tstPkg + ' ' + tstSys +';'
              else:
                 cmd  = 'mkdir -p ' + tstSys + '; ln -s ' + pkgPath + tstSys + '/' + tstPkg + ' ' + tstSys +';'
              try:
                print 'setting up symlink for ' + tstSys + '/' + tstPkg + ' using ' + cmd
                os.system(cmd)
              except:
                pass

        return


    def runTests(self):

    	# make sure we have a way to set the environment in the threads ...
    	if not os.environ.has_key('CMS_PATH'):
    	    cmsPath = '/afs/cern.ch/cms'
    	    print "setting default for CMS_PATH to", cmsPath
    	    os.environ['CMS_PATH'] = cmsPath

        lines = { 'read312RV' : ['cmsRun ../read312RV_cfg.py'], 
                  'fastsim1' : ['cmsRun ../FastSimulation/Configuration/test/IntegrationTestFake_cfg.py'],
                  'fastsim2' : ['cmsRun ../FastSimulation/Configuration/test/IntegrationTest_cfg.py'],
                  #'fastsim3' : ['cmsRun ../FastSimulation/Configuration/test/ExampleWithHLT_1E31_cfg.py'],
                  'fastsim4' : ['cmsRun ../FastSimulation/Configuration/test/IntegrationTestWithHLT_cfg.py'],
                  'pat1'     : ['cmsRun ../PhysicsTools/PatAlgos/test/IntegrationTest_cfg.py'],
                }

        hltTests = { 'hlt1' : ['cmsDriver.py TTbar_Tauola.cfi -s GEN,SIM,DIGI,L1,DIGI2RAW -n 10 --conditions auto:startup --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAW --fileout file:RelVal_DigiL1Raw_GRun.root',
                      'cmsRun ../HLTrigger/Configuration/test/OnLine_HLT_GRun.py' ], 
                     'hlt2' : ['cmsDriver.py TTbar_Tauola.cfi -s GEN,SIM,DIGI,L1,DIGI2RAW -n 10 --conditions auto:starthi --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAW --fileout file:RelVal_DigiL1Raw_HIon.root',
                      'cmsRun ../HLTrigger/Configuration/test/OnLine_HLT_HIon.py'],
                     'hlt3' : ['cmsRun ../HLTrigger/Configuration/test/OnData_HLT_GRun.py'],
                     'hlt4' : ['cmsRun ../HLTrigger/Configuration/test/OnData_HLT_HIon.py'],
                     }

    	commands={}

        actDir = os.getcwd()

        if not os.path.exists('addOnTests'):
            os.makedirs('addOnTests')
        os.chdir('addOnTests')

        self.prepare()

    	for dirName, command in lines.items():
    	        commands[dirName] = command
    	        # print 'Will do: '+command

        for dirName, commandList in hltTests.items():
            cmds = commandList
            commands[dirName] = cmds

        nfail=0
    	npass=0
    	report=''
    	
    	print 'Running in %s thread(s)' % self.maxThreads
    	
        for dirName, command in commands.items():

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
        for dirName in commands.keys():
            cmd = "for L in `ls "+dirName+"/*.log`; do cp $L logs/cmsDriver-`dirname $L`_`basename $L` ; done"
            print "going to ",cmd
            os.system(cmd)

        import pickle
        pickle.dump(commands, open('logs/addOnTests.pkl', 'w') )

        os.chdir(actDir)
        
    	return

    def upload(self, tgtDir):

        print "in ", os.getcwd()

    	# wait until all threads are finished
        while self.activeThreads() > 0:
    	    time.sleep(5)

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
        opts, args = getopt.getopt(argv, "j", ["nproc=", 'uploadDir=', 'noRun'])
    except getopt.GetoptError, e:
        print "unknown option", str(e)
        sys.exit(2)
        
# check command line parameter

    np=4 # default: four threads

    uploadDir = None
    runTests  = True
    for opt, arg in opts :
        if opt in ('-j', "--nproc" ):
            np=arg
        if opt in ("--uploadDir", ):
            uploadDir = arg
        if opt in ('--noRun', ):
            runTests = False

    tester = StandardTester(np)
    if runTests:
        tester.runTests()
    if uploadDir:
        tester.upload(uploadDir)
    

if __name__ == '__main__' :
    main(sys.argv[1:])
