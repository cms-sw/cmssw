#!/bin/bash

jobname=pythiagun_jpsi_Pt310_embd_myregit_modlowpt_timecent_20140806

# indir=/afs/cern.ch/user/e/echapon/workspace/private/jpsi_PbPb_5_3_17/CMSSW_5_3_17/test/$jobname/res/
indir=/tmp/echapon/$jobname
outdir=parsedOutputs/$jobname/
# files="$indir/*stdout"
files="$indir/*log"
tmp=/tmp/echapon/

mkdir -p $outdir $tmp

echo  "time reports per module"
rm -f $outdir/timereport.dat
for file in $files; do
   egrep '^TimeReport' $file | grep -A1000 "Module Summary" | egrep -v 'event|CPU|Module|complete|Summary|activated' >> $outdir/timereport.dat
done
awk '{print $2 " " $3 " " $4 " " $5 " " $6 " " $7 " " $8}' $outdir/timereport.dat>$tmp/tmp.txt
mv $tmp/tmp.txt $outdir/timereport.dat

echo "max vsize"
grep "Peak virtual size" $files | awk '{print $5}' > $outdir/peakvsize.dat

echo "event time statistics"
time=$tmp/time.tmp
cpu=$tmp/cpu.tmp

rm -f $outdir/timestats.dat

for file in $files; do
   grep -A4 "Time Summary" $file > $time
   grep -A4 "CPU Summary" $file > $cpu

   mint=`grep Min $time | awk -F':' '{print $2}'`
   maxt=`grep Max $time | awk -F':' '{print $2}'`
   avgt=`grep Avg $time | awk -F':' '{print $2}'`
   tott=`grep Total $time | awk -F':' '{print $2}'`
   minc=`grep Min $cpu | awk -F':' '{print $2}'`
   maxc=`grep Max $cpu | awk -F':' '{print $2}'`
   avgc=`grep Avg $cpu | awk -F':' '{print $2}'`
   totc=`grep Total $cpu | awk -F':' '{print $2}'`

   echo $mint $maxt $avgt $tott $minc $maxc $avgc $totc >> $outdir/timestats.dat
done

rm $time
rm $cpu


echo "time reports per event and per module"
rm -rf $outdir/timemodule.dat
for file in $files; do
   grep TimeModule $file | awk '{print $4 " " $5 " " $6}' >> $outdir/timemodule.dat
done


echo "Centrality"
rm -rf $outdir/centrality.dat
for file in $files; do
   grep -A1 CENTRALITY $file | grep -v '\-\-' | grep -v "CENTRALITY" >> $outdir/centrality.dat
done
