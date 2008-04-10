#!/usr/bin/perl

#Get some environment variables to use
$CMSSW_BASE=$ENV{'CMSSW_BASE'};
$CMSSW_RELEASE_BASE=$ENV{'CMSSW_RELEASE_BASE'};
$CMSSW_VERSION=$ENV{'CMSSW_VERSION'};
$HOST=$ENV{'HOST'};

#Default CASTOR directory:
if ($#ARGV==0)
{
    $CASTOR_DIR=$ARGV[0];
    print "User provided the following custom CASTOR directory where to archive the results of the performance suite:\n$CASTOR_DIR\n";
}
elsif ($#ARGV<0)
{
    $CASTOR_DIR="/castor/cern.ch/user/r/relval/performance/";
    print "Default CASTOR directory where the tarball with the results will be archived is:\n$CASTOR_DIR\n";
}
else
{
    print "Usage: cmsCreateSimPerfTestPyRelVal.pl [optional CASTOR directory argument]:
E.G.:
cmsCreateSimPerfTestPyRelVal.pl
(this will archive the results in a tarball on /castor/cern.ch/user/r/relval/performance/)
OR
cmsCreateSimPerfTestPyRelVal.pl \"/castor/cern.ch/user/y/yourusername/yourdirectory/\"
(this will archive the results in a tarball on /castor/cern.ch/user/y/yourusername/yourdirectory/)\n";
    exit;
}

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
    print "**[cmsCreateSimPerfTestPyRelVal.pl]Using LOCAL version of Validation/Performance instead of the RELEASE version**\n";
}
else
{
    $BASE_PERFORMANCE="$CMSSW_RELEASE_BASE/src/Validation/Performance";
}
$PyRelValPkg="$CMSSW_BASE/src/Configuration/PyReleaseValidation";
if (-e $PyRelValPkg)
{
    $BASE_PYRELVAL=$PyRelValPkg;
    print "**[cmsCreateSimPerfTestPyRelVal.pl]Using LOCAL version of Configuration/PyReleaseValidation instead of the RELEASE version**\n";
}
else
{
    $BASE_PYRELVAL="$CMSSW_RELEASE_BASE/src/Configuration/PyReleaseValidation";
}

#This is the "prefix" to submit all commands to run on cpu1:
$Taskset="taskset -c 1";

#Setting the path for the commands (and cpu core affinity for some):
$cmsDriver="$Taskset $BASE_PYRELVAL/scripts/cmsDriver.py";
$cmsSimPyRelVal="$Taskset $BASE_PERFORMANCE/scripts/cmsSimPyRelVal.pl";
$cmsRelvalreport="$Taskset $BASE_PYRELVAL/scripts/cmsRelvalreport.py";
$cmsScimarkLaunch="$BASE_PERFORMANCE/scripts/cmsScimarkLaunch.csh";
$cmsScimarkParser="$BASE_PERFORMANCE/scripts/cmsScimarkParser.py";
$cmsScimarkStop="$BASE_PERFORMANCE/scripts/cmsScimarkStop.pl";

#To help performance reproducibility when running on 1 core:
#Submit executables only on core cpu1,
#while running cmsScimark on the other cores
print "Submitting cmsScimarkLaunch to run on core cpu0\n";
system("taskset -c 0 $cmsScimarkLaunch 0&");
print "Submitting cmsScimarkLaunch to run on core cpu2\n";
system("taskset -c 2 $cmsScimarkLaunch 2&");
print "Submitting cmsScimarkLaunch to run on core cpu3\n";
system("taskset -c 3 $cmsScimarkLaunch 3&");

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
    $j=$i+1;
    print "$Taskset cmsScimark2 \[$j/$cmsScimark2NumOfTimes\]\n";
    $scimark=`$Taskset cmsScimark2`;
    print SCIMARK "$Taskset cmsScimark2 \[$j/$cmsScimark2NumOfTimes\]\n";
    print SCIMARK "$scimark\n";
}
$date=`date`;
print SCIMARK $date;
$date=`date`;
print SCIMARKLARGE "Initial Benchmark\n";
print SCIMARKLARGE "$date$HOST\n";
for ($i=0;$i<$cmsScimark2LargeNumOfTimes;$i++)
{
    $j=$i+1;
    print "$Taskset cmsScimark2 -large \[$j/$cmsScimark2NumOfTimes\]\n";
    $scimarklarge=`$Taskset cmsScimark2 -large`;
    print SCIMARKLARGE "$Taskset cmsScimark2 -large \[$j/$cmsScimark2NumOfTimes\]\n";
    print SCIMARKLARGE "$scimarklarge\n";
}
$date=`date`;
print SCIMARKLARGE $date;
@Candle=(
    HiggsZZ4LM190, 
    MinBias,
    SingleElectronE1000, 
    SingleMuMinusPt10, 
    SinglePiMinusE1000, 
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
if ($TimeSizeNumOfEvts>0)
{
    print "Launching the TimeSize tests (TimingReport, TimeReport, SimpleMemoryCheck, EdmSize with $TimeSizeNumOfEvts events each\n"; 
    foreach (@Candle)
    {
	print "mkdir "."$_"."_TimeSize
cd "."$_"."_TimeSize
$cmsSimPyRelVal $TimeSizeNumOfEvts $CmsDriverCandle{$_} 0123
$cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& "."$_".".log
cd ..\n";
	
	system(
	       "mkdir "."$_"."_TimeSize;
	       cd "."$_"."_TimeSize;
	       $cmsSimPyRelVal $TimeSizeNumOfEvts $CmsDriverCandle{$_} 0123;
	       $cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& "."$_".".log;
	       cd .."
	      );
    }
}
#Running IgProfPerf, IgProfMem (TOTAL, LIVE, ANALYSE) on $IgProfNumOfEvts QCD_80_120 events
if ($IgProfNumOfEvts>0)
{
    print "Launching the IgProf tests with $IgProfNumOfEvts each\n";
    print "mkdir QCD_80_120_IgProf
cd QCD_80_120_IgProf
$cmsSimPyRelVal $IgProfNumOfEvts $CmsDriverCandle{$Candle[6]} 4567
$cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& QCD_80_120.log
cd ..\n";
    system(
	   "mkdir QCD_80_120_IgProf;
           cd QCD_80_120_IgProf;
           $cmsSimPyRelVal $IgProfNumOfEvts $CmsDriverCandle{$Candle[6]} 4567;
           $cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& QCD_80_120.log;
           cd .."
          );
}
if ($ValgrindNumOfEvts>0)
{
    print "Launching the Valgrind tests with $ValgrindNumOfEvts events each\n";
#Running ValgrindFCE callgrind and memcheck on $ValgrindNumOfEvts QCD_80_120 event (DIGI, RECO, DIGI PILEUP, RECO PILEUP only)
    print "mkdir QCD_80_120_Valgrind
cd QCD_80_120_Valgrind
cp -pR ../QCD_80_120_IgProf/QCD_80_120_SIM.root .
$cmsSimPyRelVal $ValgrindNumOfEvts "."$CmsDriverCandle{$Candle[6]}"." 89;grep -v SIM SimulationCandles_"."$CMSSW_VERSION".".txt \>tmp
mv tmp SimulationCandles_"."$CMSSW_VERSION".".txt
$cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& QCD_80_120.log
cd ..\n";

    system(
	   "mkdir QCD_80_120_Valgrind;
           cd QCD_80_120_Valgrind;
           #Copying over the SIM.root file from the IgProf profiling directory to avoid re-running it
           cp -pR ../QCD_80_120_IgProf/QCD_80_120_SIM.root .;
           $cmsSimPyRelVal $ValgrindNumOfEvts "."$CmsDriverCandle{$Candle[6]}"." 89;grep -v SIM SimulationCandles_"."$CMSSW_VERSION".".txt \>tmp; 
           mv tmp SimulationCandles_"."$CMSSW_VERSION".".txt;
           $cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& QCD_80_120.log;
           cd .."
	  );

#Running ValgrindFCE callgrind and memcheck on $ValgrindNumOfEvts SingleMuMinus event (SIM only)
    print "mkdir SingleMuMinusPt10_Valgrind
cd SingleMuMinusPt10_Valgrind
$cmsSimPyRelVal $ValgrindNumOfEvts "."$CmsDriverCandle{$Candle[3]}"." 89;
grep -v DIGI SimulationCandles_"."$CMSSW_VERSION".".txt \>tmp
grep -v RECO tmp \>tmp1
mv tmp1 SimulationCandles_"."$CMSSW_VERSION".".txt
$cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& SingleMuMinusPt10.log\n
cd ..\n";

    system(
	   "mkdir SingleMuMinusPt10_Valgrind;
           cd SingleMuMinusPt10_Valgrind;
           $cmsSimPyRelVal $ValgrindNumOfEvts "."$CmsDriverCandle{$Candle[3]}"." 89;
           grep -v DIGI SimulationCandles_"."$CMSSW_VERSION".".txt \>tmp;
           grep -v RECO tmp \>tmp1; 
           mv tmp1 SimulationCandles_"."$CMSSW_VERSION".".txt;
           $cmsRelvalreport -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& SingleMuMinusPt10.log;
           cd .."
	  );
}#if $ValgrindNumOfEvts>0
#Adding an independent benchmark of the machine after running
$date=`date`;
print SCIMARK "Final Benchmark\n";
print SCIMARK "$date$HOST\n";
for ($i=0;$i<$cmsScimark2NumOfTimes;$i++)
{
        $j=$i+1;
    print "$Taskset cmsScimark2 \[$j/$cmsScimark2NumOfTimes\]\n";
    #$scimark=`$Taskset cmsScimark2`;
    print SCIMARK "$Taskset cmsScimark2 \[$j/$cmsScimark2NumOfTimes\]\n";
    print SCIMARK "$scimark\n";
}
$date=`date`;
print SCIMARK $date;
$date=`date`;
print SCIMARKLARGE "Final Benchmark\n";
print SCIMARKLARGE "$date$HOST\n";
for ($i=0;$i<$cmsScimark2LargeNumOfTimes;$i++)
{
        $j=$i+1;
    print "$Taskset cmsScimark2 -large \[$j/$cmsScimark2NumOfTimes\]\n";
    #$scimarklarge=`$Taskset cmsScimark2 -large`;
    print SCIMARKLARGE "$Taskset cmsScimark2 -large \[$j/$cmsScimark2NumOfTimes\]\n";
    print SCIMARKLARGE "$scimarklarge\n";
}
$date=`date`;
print SCIMARKLARGE $date;
close SCIMARK;
close SCIMARKLARGE;
print "Stop all cmsScimarkLaunch jobs\n";
print "$cmsScimarkStop\n";
system("$cmsScimarkStop");
#Create a tarball of the work directory 
$TarFile=$CMSSW_VERSION.'_'.$HOST.'_work.tar';
print "tar -cvf $TarFile *; gzip $TarFile";
system("tar -cvf $TarFile *; gzip $TarFile");
$TarFileGzip=$TarFile.".gz";
$CastorFile=$CASTOR_DIR.$TarFile;
#Archive the tarball in CASTOR
print "rfcp $TarFileGzip $CastorFile";
system("rfcp $TarFileGzip $CastorFile");
exit;

