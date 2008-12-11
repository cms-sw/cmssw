#!/usr/bin/perl
#Script to stop and log the output of cmsScimark

#Get some environment variables to use
#$CMSSW_BASE=$ENV{'CMSSW_BASE'};
#$CMSSW_RELEASE_BASE=$ENV{'CMSSW_RELEASE_BASE'};

#Adding a check for a local version of the packages
#$PerformancePkg="$CMSSW_BASE/src/Validation/Performance";
#if (-e $PerformancePkg)
#{
#    $BASE_PERFORMANCE=$PerformancePkg;
#    print "**[cmsScimarkStop.pl]Using LOCAL version of Validation/Performance instead of the RELEASE version**\n";
#}
#else
#{
#    $BASE_PERFORMANCE="$CMSSW_RELEASE_BASE/src/Validation/Performance";
#}
#Get the PID of the running cmsScimarkLaunch.csh
$PsOutput=`ps -ef|grep cmsScimarkLaunch`;
print $PsOutput;
@Ps=split('\n',$PsOutput);
#Kill all the cmsScimarkLaunch.csh processes
foreach (@Ps)
{
    @PsTokens=split(' ',$_);
    if (($PsTokens[7] eq "/bin/csh")&&($PsTokens[8] =~ /cmsScimarkLaunch.csh/))
    {
	$kill=`kill -9 $PsTokens[1]`;
	print "kill -9 $PsTokens[1] (was $_)\n";
	$cpucore=$PsTokens[9];
	print "Killed cmsScimarkLaunch.csh on cpu$cpucore\n";
	$ResultsDir="cmsScimarkResults_cpu".$cpucore;
#	print "$ResultsDir\n";
	$ResultsLog="cmsScimark_".$cpucore.".log";
#	print "$ResultsLog\n";
	if (!(-e $ResultsDir))
	{
	    $mkdir=`mkdir $ResultsDir`;
	    print "mkdir $ResultsDir\n";
	}
	#Script assumes that cmsScimarkParser.py is already in the release
        #Otherwise, one needs to add ./ in front of it
	$cmsScimarkParse=`cmsScimarkParser.py -i $ResultsLog -o $ResultsDir`;
	print "cmsScimarkParser.py -i $ResultsLog -o $ResultsDir\n";
    }
    else
    {
	print "There was no matching in the ps -ef results";
    }
}
exit;
