#!/bin/sh
eval `scram ru -sh`
for file in `../Tools/listBenchmarks.py "*$DBS_RELEASE*/*" -a -u | grep http `
do
type=`echo $file | awk -v FS="_RelVal" '{print $2}'`

case $type in
    TTbarefromW)
    Comment="Electrons from W in ttbar events"
    ;;
    TTbarefromb)
    Comment="Electrons from b in ttbar events"
    ;;
    SingleElectronPt35)
    Comment="Single electrons Pt=35 GeV"
    ;;
    SingleElectronPt10)
    Comment="Single electrons Pt=10 GeV"
    ;;
    ZEEefromZ)
    Comment="Electrons from Z->ee events"
    ;;
    *)
    Comment="unknown"
    ;;
esac
#probably the worst trick with awk I know...
echo $file | awk '{print "   * [["$1"]["ENVIRON["DBS_RELEASE"]"'": $Comment"']]"}'
done
