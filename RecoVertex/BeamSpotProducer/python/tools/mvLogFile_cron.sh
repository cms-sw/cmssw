export STAGE_HOST=castorcms.cern.ch
logFileName="/afs/cern.ch/user/u/uplegger/www/Logs/MegaScriptLog.txt"
to="/afs/cern.ch/cms/CAF/CMSCOMM/COMM_BSPOT/automated_workflow/logs/MegaScriptLog_"`date '+%y-%m-%d_%H:%M'`".txt"
mv -f $logFileName $to
touch $logFileName
 
