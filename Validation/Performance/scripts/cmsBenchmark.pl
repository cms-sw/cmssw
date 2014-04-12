#!/usr/bin/perl
#G.Benelli
#Script derived from cmsCreateSimPerfTestPyRelVal.pl
#to create a simple CMSSW benchmark to profile hardware.

#Get some environment variables to use
$CMSSW_BASE=$ENV{'CMSSW_BASE'};
$CMSSW_RELEASE_BASE=$ENV{'CMSSW_RELEASE_BASE'};
$CMSSW_VERSION=$ENV{'CMSSW_VERSION'};
$HOST=$ENV{'HOST'};

#Default to 0 the number of times running the cmsScimark2 benchmarks
$cmsScimark2NumOfTimes=0;
$cmsScimark2LargeNumOfTimes=0;

if (@ARGV)
{
    $Core=$ARGV[0];
    if ($ARGV[1])
    {
	$cmsScimark2NumOfTimes=$ARGV[1];
    }
    if ($ARGV[2])
    {
        $cmsScimark2LargeNumOfTimes=$ARGV[2];
    }
}
else
{
    print "Please input the cpu core number on which you want to run cmsBenchmark.pl!\n";
    print "Usage [to run on cpu1 with no leading and trailing cmsScimarks and cmsScimarkLarge]: cmsBenchmarkControl.pl 1\n";
    print "Usage [to run on cpu1 with 5 leading and trailing cmsScimarks and 3 leading and trainlng cmsScimarkLarge]: cmsBenchmarkControl.pl 1 5 3\n";
    exit;
}

#Default number of events for each set of tests:
$TimeSizeNumOfEvts=100;

#Adding a check for a local version of the packages
$PerformancePkg="$CMSSW_BASE/src/Validation/Performance";
if (-e $PerformancePkg)
{
    $BASE_PERFORMANCE=$PerformancePkg;
    print "**Using LOCAL version of Validation/Performance instead of the RELEASE version**\n";
}
else
{
    $BASE_PERFORMANCE="$CMSSW_RELEASE_BASE/src/Validation/Performance";
}
$PyRelValPkg="$CMSSW_BASE/src/Configuration/PyReleaseValidation";
if (-e $PyRelValPkg)
{
    $BASE_PYRELVAL=$PyRelValPkg;
    print "**Using LOCAL version of Configuration/PyReleaseValidation instead of the RELEASE version**\n";
}
else
{
    $BASE_PYRELVAL="$CMSSW_RELEASE_BASE/src/Configuration/PyReleaseValidation";
}

#This is the "prefix" to submit commands to run individual cpus:                                                                         
$Taskset="taskset -c $Core";
#Setting the path for the various commands when submitting on multiple cores: 
$cmsDriver="$Taskset $BASE_PYRELVAL/scripts/cmsDriver.py";
$cmsSimPyRelVal="$Taskset $BASE_PERFORMANCE/scripts/cmsSimPyRelVal.pl";
$cmsRelvalreport="$Taskset $BASE_PYRELVAL/scripts/cmsRelvalreport.py";
$cmsScimark="$Taskset cmsScimark2";

$date=`date`;
$path=`pwd`;
$tags=`showtags -r`;
#Information for the logfile
print "$date";
print "$HOST\n";
print "Local path: $path";
print "\$CMSSW_BASE is $CMSSW_BASE\n";
print "\$CMSSW_VERSION is $CMSSW_VERSION\n";
print $tags;
#Adding an independent benchmark of the machine before running
$cmsScimark2log="cmsScimark2_cpu".$Core.".log";
$cmsScimark2largelog="cmsScimark2_large_cpu".$Core.".log";
open(SCIMARK,">$cmsScimark2log")||die "Could not open file $cmsScimark2log:$!\n";
open(SCIMARKLARGE,">$cmsScimark2largelog")||die "Could not open file $cmsScimark2largelog:$!\n";
$date=`date`;
print SCIMARK "Initial Benchmark\n";
print SCIMARK "$date$HOST\n";
for ($i=0;$i<$cmsScimark2NumOfTimes;$i++)
{
    $j=$i+1;
    print "$cmsScimark \[$j/$cmsScimark2NumOfTimes\]\n";
    print SCIMARK "$cmsScimark \[$j/$cmsScimark2NumOfTimes\]\n";
    system("$cmsScimark >> $cmsScimark2log");
}
$date=`date`;
print SCIMARK $date;
$date=`date`;
print SCIMARKLARGE "Initial Benchmark\n";
print SCIMARKLARGE "$date$HOST\n";
for ($i=0;$i<$cmsScimark2LargeNumOfTimes;$i++)
{
    $j=$i+1;
    print "$cmsScimark -large \[$j/$cmsScimark2NumOfTimes\]\n";
    print SCIMARKLARGE "$cmsScimark -large \[$j/$cmsScimark2NumOfTimes\]\n";
    system("$cmsScimark -large >> $cmsScimark2largelog");
}
$date=`date`;
print SCIMARKLARGE $date;
@Candle=(
    HiggsZZ4LM190, 
    MinBias,
    SingleElectronE1000, 
    SingleMuMinusPt1000, 
    SinglePiMinusPt1000, 
    TTbar,
    QCD_80_120
    );

%CmsDriverCandle=(
    $Candle[0]=>"\"HZZLLLL\"",
    $Candle[1]=>"\"MINBIAS\"",
    $Candle[2]=>"\"E -e 1000\"",
    $Candle[3]=>"\"MU- -e pt10\"",
    $Candle[4]=>"\"PI- -e 1000\"",
    $Candle[5]=>"\"TTBAR\"",
    $Candle[6]=>"\"QCD -e 80_120\""
    );
%CmsDriverCandleNoBrackets=(
    $Candle[0]=>"HZZLLLL",
    $Candle[1]=>"MINBIAS",
    $Candle[2]=>"E -e 1000",
    $Candle[3]=>"MU- -e pt10",
    $Candle[4]=>"PI- -e 1000",
    $Candle[5]=>"TTBAR",
    $Candle[6]=>"QCD -e 80_120"
    );
#Running TimingReport, TimeReport, SimpleMemoryCheck, EdmSize on all 7 candles
#With $TimeSizeNumOfEvts events each
foreach (@Candle)
{
    print "mkdir $_"."_cpu".$Core." 
cd $_"."_cpu".$Core." 
$cmsSimPyRelVal $TimeSizeNumOfEvts $CmsDriverCandle{$_} 0123
$cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& "."$_".".log
cd ..\n";
    system(
	   "mkdir $_"."_cpu".$Core." ;
	   cd $_"."_cpu".$Core." ;
	   $cmsSimPyRelVal $TimeSizeNumOfEvts $CmsDriverCandle{$_} 0123;
	   $cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& "."$_".".log;
	   cd .."
	   );
}


#Adding an independent benchmark of the machine after running
$date=`date`;
print "$date\n";
print SCIMARK "Final Benchmark\n";
print SCIMARK "$date$HOST\n";
for ($i=0;$i<$cmsScimark2NumOfTimes;$i++)
{
    $j=$i+1;
    print "$cmsScimark \[$j/$cmsScimark2NumOfTimes\]\n";
    print SCIMARK "$cmsScimark \[$j/$cmsScimark2NumOfTimes\]\n";
    system("$cmsScimark >> $cmsScimark2log");
}
$date=`date`;
print SCIMARK $date;
$date=`date`;
print SCIMARKLARGE "Final Benchmark\n";
print SCIMARKLARGE "$date$HOST\n";
for ($i=0;$i<$cmsScimark2LargeNumOfTimes;$i++)
{
    $j=$i+1;
    print "$cmsScimark -large \[$j/$cmsScimark2NumOfTimes\]\n";
    print SCIMARKLARGE "$cmsScimark -large \[$j/$cmsScimark2NumOfTimes\]\n";
    system("$cmsScimark -large >> $cmsScimark2largelog");
}
$date=`date`;
print SCIMARKLARGE $date;
close SCIMARK;
close SCIMARKLARGE;
exit;

