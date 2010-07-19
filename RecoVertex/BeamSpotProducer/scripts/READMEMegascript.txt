1) Manually running the BeamSpotWorkflow.py script

Just typing BeamSpotWorkflow.py -h will show the possible options of the script

The 3 most common options are 
-z -> changes the sigmaZ form the calculated value to 10cm
-u -> upload the valuse into the DB

-c -> allow to specify your custom cfg file otherwise it uses the default BeamSpotWorkflow.cfg

Example:
./BeamSpotWorkflow.py -c BeamSpotWorkflow_run.cfg -z -u

2)Cfg file structure: (extra lines can be commented with a # at the beginning)

a) SOURCE_DIR  = /castor/cern.ch/cms/store/caf/user/uplegger/Workflows/361_patch4/express_T0_v11/
   Any directory ( castor or hard disk) where you have the txt files produced by the CMSSW beamspot workflow 

b) ARCHIVE_DIR = /afs/cern.ch/cms/CAF/CMSCOMM/COMM_BSPOT/automated_workflow/good_archive/
   Any directory where you want to store the beamspot files. The files from SOURCE_DIR will be copied to the ARCHIVE_DIR

c) WORKING_DIR = /afs/cern.ch/cms/CAF/CMSCOMM/COMM_BSPOT/automated_workflow/good_tmp
   After the files are copied in the ARCHIVE_DIR, then they will be copied in the WORKING_DIR. Every time you run the script, the 
   WORKING_DIR will be WIPED OUT first. In case you are running MORE SCRIPTS AT THE SAME TIME you can keep the same ARCHIVE_DIR but 
   you MUST to use a different WORKING_DIR for each script to avoid conflicts 

d) DBTAG       = BeamSpotObjects_2009_v14_offline
   Database tag you want to update. Currently we have (BeamSpotObjects_2009_v14_offline, BeamSpotObjects_2009_SigmaZ_v14_offline, 
   BeamSpotObjects_2009_lumi_v14_offline, BeamSpotObjects_2009_lumi_SigmaZ_v14_offline. I use BeamSpotObjects_2009_v13_offline for testing)

e) DATASET     = /StreamExpress/Run2010A-TkAlMinBias-v4/ALCARECO
   Dataset which your txt files have been produced by. You can specify multiple DATASETS splitting them with a comma(,) like:
  /StreamExpress/Commissioning10-StreamTkAlMinBias-v7/ALCARECO,
  /StreamExpress/Commissioning10-StreamTkAlMinBias-v8/ALCARECO,
  /StreamExpress/Commissioning10-StreamTkAlMinBias-v9/ALCARECO,
  /StreamExpress/Run2010A-StreamTkAlMinBias-v1/ALCARECO,
  /StreamExpress/Run2010A-TkAlMinBias-v2/ALCARECO,
  /StreamExpress/Run2010A-TkAlMinBias-v3/ALCARECO,
  /StreamExpress/Run2010A-TkAlMinBias-v4/ALCARECO
  This is a very nice feature when you are reprocessing the all dataset

f) FILE_IOV_BASE = lumibase
   The iov base of the txt files. Recently we have been produceing files fitting every lumisection so it has been for a long time lumibase ( can be runbase)

g) DB_IOV_BASE   = runnumber
   Iov base in the database for the tag you want to upload. Right now the official tag has runnumber iovs. The other possibility is lumiid

h) DBS_TOLERANCE_PERCENT = 10
   Percentage of missing lumisection that can be tolerated between the lumi section processed and the ones that dbs says should have been processed.
   When querying dbs the script asks how many lumisections were present in the files that the workflow processed. The number of lumi processed and the one 
   in dbs should always match but unfortunately it is not the case. 10% should let you pass all the files that have been processed so far.

i) DBS_TOLERANCE = 20
   Number of missing lumisection that can be tolerated between the lumi section processed and the ones that dbs says should have been processed.
   Sometimes a run  has few lumisections so in case the workflow doesn't process a few, the percentage of not processed lumis doesn't pass the 
   previous tolerance.

l) RR_TOLERANCE = 10
   Percentage of missing lumisection that can be tolerated between the lumi section processed and the ones that are considered good in the run registry.
   If there are too many lumis unprocessed, when comapared to dbs, the script check if the ones that have been processed at least cover the
   one that are considered good in the run registry. 

m) MISSING_FILES_TOLERANCE = 2
   Number of missing files that can be tolerated before the script can continue. It is important to keep this number low 2 max 3 especially
   when running it in a cron job. In fact, ithe script can be triggered when few files are still being processed and you don't want to do that 
   if the number of missing files is still big.

n) MISSING_LUMIS_TIMEOUT = 14400
   There are few timeouts in the script (for example if there are still many files missing), and after a certain number of seconds = MISSING_LUMIS_TIMEOUT
   hte script keep running.

o) EMAIL       = uplegger@cern.ch,yumiceva@fnal.gov
   Comma separated list of people who will receive an e-mail in case of big troubles. There are some conditions that must be validated by
   a person so typically the script stop working and send an e-mail to the persons in this list who will have to take action.

3) Cron job shell script.
   In python/tools there is the beamspotWorkflow_cron.sh shell script which runs the workflow automatically.
   
//--------------------------------------------------------------------------------------------------
   export STAGE_HOST=castorcms.cern.ch
   source /afs/cern.ch/cms/sw/cmsset_default.sh
   cd /afs/cern.ch/user/u/uplegger/scratch0/CMSSW/CMSSW_3_6_1_patch4/src/
   logFileName="/afs/cern.ch/user/u/uplegger/www/Logs/MegaScriptLog.txt"
   echo >> $logFileName
   echo "Begin running the script on " `date` >> $logFileName
   if [ ! -e .lock ]
   then
     touch .lock
     eval `scramv1 runtime -sh`
     python $CMSSW_BASE/src/RecoVertex/BeamSpotProducer/scripts/BeamSpotWorkflow_T0.py -u -c BeamSpotWorkflow_T0.cfg >> $logFileName
     rm .lock
   else
     echo "There is already a megascript runnning...exiting" >> $logFileName
   fi
   echo "Done on " `date` >> $logFileName
//--------------------------------------------------------------------------------------------------

   REMEMBER: 
   a) cd /afs/cern.ch/user/u/uplegger/scratch0/CMSSW/CMSSW_3_6_1_patch4/src/
      is the CMSSW area where your script is!
   b) logFileName="/afs/cern.ch/user/u/uplegger/www/Logs/MegaScriptLog.txt"
      is my area which is web accessible, so I can check the output of the script once in a while
   c) python $CMSSW_BASE/src/RecoVertex/BeamSpotProducer/scripts/BeamSpotWorkflow_T0.py -u -c BeamSpotWorkflow_T0.cfg >> $logFileName
      Runs the script WITH the BeamSpotWorkflow_T0.cfg cfg file and saves the output in the logfilename that I can check online
   d) if [ ! -e .lock ] then touch .lock
      It creates a .lock file in /afs/cern.ch/user/u/uplegger/scratch0/CMSSW/CMSSW_3_6_1_patch4/src/ 
      This lock file prevent 2 megascripts to run at the same time. It is in the shell script so should be removed 99.9% of the times
      but it already happened to me that it was not removed once. 


4) Running the cron job:
   acrontab -e 
   let you edit your cron jobs while 
   acrontab -l 
   shows what your cron job file is. 
//--------------------------------------------------------------------------------------------------
   5 * * * * lxplus258 /afs/cern.ch/user/u/uplegger/scratch0/CMSSW/CMSSW_3_6_1_patch4/src/RecoVertex/BeamSpotProducer/python/tools/beamspotWorkflow_cron.sh >& /afs/cern.ch/user/u/uplegger/www/Logs/CronJob.log
   25 * * * * lxplus301 /afs/cern.ch/user/u/uplegger/scratch0/CMSSW/CMSSW_3_6_1_patch4/src/RecoVertex/BeamSpotProducer/python/tools/beamspotWorkflow_cron.sh >& /afs/cern.ch/user/u/uplegger/www/Logs/CronJob.log
   45 * * * * lxplus256 /afs/cern.ch/user/u/uplegger/scratch0/CMSSW/CMSSW_3_6_1_patch4/src/RecoVertex/BeamSpotProducer/python/tools/beamspotWorkflow_cron.sh >& /afs/cern.ch/user/u/uplegger/www/Logs/CronJob.log
   3 0,13 * * * lxplus301 /afs/cern.ch/user/u/uplegger/scratch0/CMSSW/CMSSW_3_6_1_patch4/src/RecoVertex/BeamSpotProducer/python/tools/mvLogFile_cron.sh
//--------------------------------------------------------------------------------------------------
   Right now I am running the megascript cron job from 3 different machines every 20 minutes.
   I am also running twice a day another script that moves the log files away to keep the one on the web small.
   
