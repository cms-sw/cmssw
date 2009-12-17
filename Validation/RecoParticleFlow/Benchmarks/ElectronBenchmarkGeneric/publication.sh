#!/bin/sh
eval `scram ru -sh`
echo $TITLE
../Tools/indexGen.py -f -c "$TITLE" -t "$TITLE"
#publish
if [ -z "$?WEB_PUBLICATION" ] ; then
    echo "exit"
    exit
    else
    if [ "$WEB_PUBLICATION" = "true" ] ; then
    ../Tools/submit.py -f -e $DBS_SAMPLE$E_SELECTION 
    fi
fi
# Restore benchmark file
../Tools/listBenchmarks.py "*$DBS_RELEASE*/*" -a -u 
