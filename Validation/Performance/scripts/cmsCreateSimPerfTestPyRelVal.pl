#!/usr/bin/perl

#Get some environment variables to use
$CMSSW_BASE=$ENV{'CMSSW_BASE'};
$CMSSW_RELEASE_BASE=$ENV{'CMSSW_RELEASE_BASE'};
$CMSSW_VERSION=$ENV{'CMSSW_VERSION'};
$HOST=$ENV{'HOST'};

#Default number of events for each set of tests:
$TimeSizeNumOfEvts=100;
$IgProfNumOfEvts=5;
$ValgrindNumOfEvts=1;
#Number of times running the cmsScimark2 benchmarks
$cmsScimark2NumOfTimes=10;
$cmsScimark2LargeNumOfTimes=10;

#To fix pie-chart issues until PerfReport3
system("source /afs/cern.ch/user/d/dpiparo/w0/perfreport2.1installation/share/perfreport/init_matplotlib.sh");

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
#Setting the path for the cmsDriver.py command:
$cmsDriver="$BASE_PYRELVAL/scripts/cmsDriver.py";
$cmsSimPyRelVal="$BASE_PERFORMANCE/scripts/cmsSimPyRelVal.pl";
$cmsRelvalreport="$BASE_PYRELVAL/scripts/cmsRelvalreport.py";

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
open(SCIMARK,">cmsScimark2.log")||die "Could not open file cmsScimark2.log:$!\n";
open(SCIMARKLARGE,">cmsScimark2_Large.log")||die "Could not open file cmsScimark2_Large.log:$!\n";
$date=`date`;
print SCIMARK "Initial Benchmark\n";
print SCIMARK "$date$HOST\n";
for ($i=0;$i<$cmsScimark2NumOfTimes;$i++)
{
    $scimark=`cmsScimark2`;
    print SCIMARK "$scimark\n";
}
$date=`date`;
print SCIMARK $date;
$date=`date`;
print SCIMARKLARGE "Initial Benchmark\n";
print SCIMARKLARGE "$date$HOST\n";
for ($i=0;$i<$cmsScimark2LargeNumOfTimes;$i++)
{
    $scimarklarge=`cmsScimark2 -large`;
    print SCIMARKLARGE "$scimarklarge\n";
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
    ZPrimeJJM700
    );

%CmsDriverCandle=(
    $Candle[0]=>"\"HZZLLLL\"",
    $Candle[1]=>"\"MINBIAS\"",
    $Candle[2]=>"\"E -e 1000\"",
    $Candle[3]=>"\"MU- -e pt1000\"",
    $Candle[4]=>"\"PI- -e pt1000\"",
    $Candle[5]=>"\"TTBAR\"",
    $Candle[6]=>"\"ZPJJ\""
    );
%CmsDriverCandleNoBrackets=(
    $Candle[0]=>"HZZLLLL",
    $Candle[1]=>"MINBIAS",
    $Candle[2]=>"E -e 1000",
    $Candle[3]=>"MU- -e pt1000",
    $Candle[4]=>"PI- -e pt1000",
    $Candle[5]=>"TTBAR",
    $Candle[6]=>"ZPJJ"
    );
#Running TimingReport, TimeReport, SimpleMemoryCheck, EdmSize on all 7 candles
#With $TimeSizeNumOfEvts events each
foreach (@Candle)
{
   system(
	"mkdir "."$_"."_TimeSize;
	cd "."$_"."_TimeSize;
	$cmsDriver $CmsDriverCandleNoBrackets{$_} -n $TimeSizeNumOfEvts --step=GEN --customise=Simulation.py >& "."$_"."_GEN.log;
	$cmsSimPyRelVal $TimeSizeNumOfEvts $CmsDriverCandle{$_} 0123;
	$cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& "."$_".".log;
	cd .."
	);
}

#Running IgProfPerf, IgProfMem (TOTAL, LIVE, ANALYSE) on $IgProfNumOfEvts ZPrimeJJ events
system(
    "mkdir ZPrimeJJM700_IgProf;
    cd ZPrimeJJM700_IgProf;
    $cmsDriver $CmsDriverCandleNoBrackets{$Candle[6]} -n $IgProfNumOfEvts --step=GEN --customise=Simulation.py >& ZPrimeJJM700_GEN.log;
    $cmsSimPyRelVal $IgProfNumOfEvts $CmsDriverCandle{$Candle[6]} 4567;
    $cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& ZPrimeJJM700.log;
    cd .."
    );

#Running ValgrindFCE callgrind and memcheck on $ValgrindNumOfEvts ZPrimeJJ event (DIGI only)
system(
    "mkdir ZPrimeJJM700_Valgrind;
    cd ZPrimeJJM700_Valgrind;
    $cmsDriver $CmsDriverCandleNoBrackets{$Candle[6]} -n $ValgrindNumOfEvts --step=GEN,SIM --fileout=ZPJJ__SIM.root --customise=Simulation.py >& ZPrimeJJM700_GEN_SIM.log;
    $cmsSimPyRelVal $ValgrindNumOfEvts "."$CmsDriverCandle{$Candle[6]}"." 89;grep -v SIM SimulationCandles_"."$CMSSW_VERSION".".txt \>tmp; 
    mv tmp SimulationCandles_"."$CMSSW_VERSION".".txt;
    $cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& ZPrimeJJM700.log;
    cd .."
    );

#Running ValgrindFCE callgrind and memcheck on $ValgrindNumOfEvts SingleMuMinus event (SIM only)
system(
    "mkdir SingleMuMinusPt1000_Valgrind;
    cd SingleMuMinusPt1000_Valgrind;
    $cmsDriver $CmsDriverCandleNoBrackets{$Candle[3]} -n $ValgrindNumOfEvts --step=GEN --customise=Simulation.py >& SingleMuMinusPt1000_GEN.log
    $cmsSimPyRelVal $ValgrindNumOfEvts "."$CmsDriverCandle{$Candle[3]}"." 89;grep -v DIGI SimulationCandles_"."$CMSSW_VERSION".".txt \>tmp; 
    mv tmp SimulationCandles_"."$CMSSW_VERSION".".txt;
    $cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& SingleMuMinusPt1000.log;
    cd .."
    );
#Adding an independent benchmark of the machine after running
$date=`date`;
print SCIMARK "Final Benchmark\n";
print SCIMARK "$date$HOST\n";
for ($i=0;$i<$cmsScimark2NumOfTimes;$i++)
{
    $scimark=`cmsScimark2`;
    print SCIMARK "$scimark\n";
}
$date=`date`;
print SCIMARK $date;
$date=`date`;
print SCIMARKLARGE "Final Benchmark\n";
print SCIMARKLARGE "$date$HOST\n";
for ($i=0;$i<$cmsScimark2LargeNumOfTimes;$i++)
{
    $scimarklarge=`cmsScimark2 -large`;
    print SCIMARKLARGE "$scimarklarge\n";
}
$date=`date`;
print SCIMARKLARGE $date;
close SCIMARK;
close SCIMARKLARGE;
exit;

