#!/usr/bin/env tcsh
# write your command in place of $argv and encapsulate the command for this script
# in nohup . ie "nohup ./runbackground.csh &"
set SRCPATH=`printf $CMSSW_SEARCH_PATH | awk '{split($1,a,":") ; print a[1]}' | tr -d '\n'`
echo $SRCPATH
cd $SRCPATH 
eval `scramv1 runtime -csh`
scramv1 b; rehash
cd -
$argv >& background.log 
