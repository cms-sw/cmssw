#!/bin/sh
eval `scram ru -sh`
echo $TITLE

#copy back into the content into the local directory; not needed when everything is run sequentially
#cp -r $DBS_RELEASE/$DBS_SAMPLE Plots_BarrelAndEndcap
#cp $DBS_RELEASE/$DBS_SAMPLE/benchmark.root benchmark.root

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
