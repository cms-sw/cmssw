#!/usr/bin/perl
#GBenelli Dec21 
#This script is designed to run on a local directory 
#after the user has created a local CMSSW release,
#initialized its environment variables by executing
#in the release /src directory:
#eval `scramv1 runtime -csh`
#project CMSSW
#The script will create a SimulationCandles.txt ASCII
#file, input to cmsRelvalreport.py, to launch the 
#standard simulation performance suite.

#Input arguments are three:
#1-Number of events to put in the cfg files
#2-Name of the candle(s) to process (either AllCandles, or NameOfTheCandle)
#3-Profiles to run (with code below)
#E.g.: ./cmsSimPyRelVal.pl 50 AllCandles 012

#Get some environment variables to use
$CMSSW_BASE=$ENV{'CMSSW_BASE'};
$CMSSW_RELEASE_BASE=$ENV{'CMSSW_RELEASE_BASE'};
#Adding a check for a local version of the packages
$PyRelValPkg="$CMSSW_BASE/src/Configuration/PyReleaseValidation";
if (-e $PyRelValPkg)
{
    $BASE_PYRELVAL=$PyRelValPkg;
    print "**[cmsSimPyRelVal.pl]Using LOCAL version of Configuration/PyReleaseValidation instead of the RELEASE version**\n";
}
else
{
    $BASE_PYRELVAL="$CMSSW_RELEASE_BASE/src/Configuration/PyReleaseValidation";
}
#Setting the path for the cmsDriver.py command:
$cmsDriver="$BASE_PYRELVAL/scripts/cmsDriver.py";

if (($#ARGV != 2)&&($#ARGV != 3)&&($#ARGV != 4)) {
	print "Usage: cmsSimPyRelVal.pl NumberOfEventsPerCfgFile Candles Profile [cmsDriverOptions] [processingStepsOption]
Candles codes:
 AllCandles
 \"HZZLLLL\"
 \"MINBIAS\"
 \"E -e 1000\"
 \"MU- -e pt10\"
 \"PI- -e 1000\"
 \"TTBAR\"
 \"QCD -e 80_120\"
Profile codes (multiple codes can be used):
 0-TimingReport
 1-TimeReport
 2-SimpleMemoryCheck
 3-EdmSize
 4-IgProfPerf
 5-IgProfMemTotal
 6-IgProfMemLive
 7-IgProfAnalyse
 8-ValgrindFCE
 9-ValgrindMemCheck

Option for cmsDriver.py can be specified as a string to be added to all cmsDriver.py commands:
\"--conditions FakeConditions\"

Examples: 
./cmsSimulationCandles.pl 10 AllCandles 1 
OR 
./cmsSimulationCandles.pl 50 \"HZZLLLL\" 012\n
OR
./cmsSimulationCandles.pl 100 \"TTBAR\" 45 \"--conditions FakeConditions\"\n
OR
./cmsSimulationCandles.pl 100 \"MINBIAS\" 89 \"--conditions FakeConditions\" \"--usersteps=GEN-SIM,DIGI\"\n";
	exit;
}
$NumberOfEvents=$ARGV[0];
$WhichCandles=$ARGV[1];
if ($ARGV[2] =~/--usersteps/)
{
    $userSteps=$ARGV[2];
}
else
{
    $ProfileCode=$ARGV[2];
}
if ($#ARGV==3)
{
    if ($ARGV[3] =~/--usersteps/)
    {
	#First split the option using the "=" to get actual user steps
	@userStepsTokens=split /=/,$ARGV[3];
	$userSteps= $userStepsTokens[1];
	#print "$userSteps\n";
	#Then split the user steps into "steps"
	@StepsTokens=split /,/,$userSteps;
	foreach (@StepsTokens)
	{
	    #Then transform the combined steps (GEN-SIM, RAW2DIGI-RECO) 
	    #from using the "-" to using the "," to match cmsDriver.py convention
	    if ($_ =~ /-/)
	    {
		s/-/,/;
	    }
	    #print "$_\n";
	    #Finally collect all the steps into the @Steps array:
	    push @Steps, $_;
	}
    }
    else
    {
	$cmsDriverOptions=$ARGV[3];
	print "Using user-specified cmsDriver.py options: $cmsDriverOptions\n";
    }
}
#Ugly cut and pastes for now, since we need to rewrite in python anyway...

if ($#ARGV==4)
{
    if ($ARGV[4] =~/--usersteps/)
    {
	#First split the option using the "=" to get actual user steps
	@userStepsTokens=split /=/,$ARGV[4];
	$userSteps= $userStepsTokens[1];
	#print "$userSteps\n";
	#Then split the user steps into "steps"
	@StepsTokens=split /,/,$userSteps;
	foreach (@StepsTokens)
	{
	    #Then transform the combined steps (GEN-SIM, RAW2DIGI-RECO) 
	    #from using the "-" to using the "," to match cmsDriver.py convention
	    if ($_ =~ /-/)
	    {
		s/-/,/;
	    }
	    #print "$_\n";
	    #Finally collect all the steps into the @Steps array:
	    push @Steps, $_;
	}
	$cmsDriverOptions=$ARGV[3];
	print "Using user-specified cmsDriver.py options: $cmsDriverOptions\n";
    }
    elsif ($ARGV[3] =~/--usersteps/)
    {
	#First split the option using the "=" to get actual user steps
	@userStepsTokens=split /=/,$ARGV[3];
	$userSteps= $userStepsTokens[1];
	#print "$userSteps\n";
	#Then split the user steps into "steps"
	@StepsTokens=split /,/,$userSteps;
	foreach (@StepsTokens)
	{
	    #Then transform the combined steps (GEN-SIM, RAW2DIGI-RECO) 
	    #from using the "-" to using the "," to match cmsDriver.py convention
	    if ($_ =~ /-/)
	    {
		s/-/,/;
	    }
	    #print "$_\n";
	    #Finally collect all the steps into the @Steps array:
	    push @Steps, $_;
	}
	$cmsDriverOptions=$ARGV[4];
	print "Using user-specified cmsDriver.py options: $cmsDriverOptions\n";
    }
}
#Getting some important environment variables:
$CMSSW_RELEASE_BASE=$ENV{'CMSSW_RELEASE_BASE'};
$CMSSW_VERSION=$ENV{'CMSSW_VERSION'};

if ($WhichCandles eq "AllCandles")
{
    @Candle=("HZZLLLL",
	     "MINBIAS",
	     "E -e 1000",
	     "MU- -e pt10",
	     "PI- -e 1000",
	     "TTBAR",
	     "QCD -e 80_120"
	     );
    print "ALL standard simulation candles will be PROCESSED:\n";
}
else
{
   
    @Candle=($WhichCandles);
    print "ONLY @Candle will be PROCESSED\n";
}
#Need a little hash to match the candle with the ROOT name used by cmsDriver.py.
%FileName=(
	   "HZZLLLL"=>"HZZLLLL_190",
	    "MINBIAS"=>"MINBIAS_",
	    "E -e 1000"=>"E_1000",
	    "MU- -e pt10"=>"MU-_pt10",
	    "PI- -e 1000"=>"PI-_1000",
	    "TTBAR"=>"TTBAR_",
	    "QCD -e 80_120"=>"QCD_80_120"
	   );
#Creating and opening the ASCII input file for the relvalreport script:
$SimCandlesFile= "SimulationCandles"."_".$CMSSW_VERSION.".txt";
open(SIMCANDLES,">$SimCandlesFile")||die "Couldn't open $SinCandlesFile to save - $!\n";
print SIMCANDLES "#Candles file automatically generated by cmsSimPyRelVal.pl for $CMSSW_VERSION\n\n";
#For now the two steps are built in, this can be added as an argument later
#Added the argument option so now this will only be defined if it was not defined already:
if (!@Steps)
{
    print "The default steps will be run:\n";
    @Steps=(
	   "GEN,SIM",
	   #"SIM",#To run SIM only need GEN done already!
	   "DIGI",
	   #Adding L1 step
	   #"L1",
	   #Adding DIGI2RAW step
	   #"DIGI2RAW",
	   #Adding HLT step
	   #"HLT",
	   #Adding RAW2DIGI step together with RECO
	   #"RAW2DIGI,RECO"
	   );
    foreach (@Steps)
    {
	print "$_\n";
    }
}
else
{
    print "You defined your own steps to run:\n";
    foreach (@Steps)
	{
	    print "$_\n";
	}
}
#Convenient hash to map the correct Simulation Python fragment:
%CustomiseFragment=(
		    #Added the Configuration/PyReleaseValidation/ in front of the fragments names 
		    #Since the customise option was broken in 2_0_0_pre8, when the fragments were
		    #moved into Configuration/PyReleaseValidation/python from /data.
		    #unfortunately the full path Configuration/PyReleaseValidation/python/MyFragment.py
		    #does not seem to work.
		    "GEN,SIM"=>"Configuration/PyReleaseValidation/SimulationG4.py",
		    #"SIM"=>"SimulationG4.py", #To run SIM only need GEN done already!
		    "DIGI"=>"Configuration/PyReleaseValidation/Simulation.py"
		    #,
		    #The following part should be edited to implement the wanted step and 
		    #define the appropriate customise fragment path
		    #Adding RAW2DIGI,RECO step
		    #"RAW2DIGI,RECO"=>"Configuration/PyReleaseValidation/Simulation.py",
		    #Adding L1 step
		    #"L1"=>"Configuration/PyReleaseValidation/Simulation.py",
		    #Adding DIGI2RAW step
		    #"DIGI2RAW"=>"Configuration/PyReleaseValidation/Simulation.py",
		    #Adding HLT step
		    #"HLT"=>"Configuration/PyReleaseValidation/Simulation.py",
		    );
#This option will not be used for now since the event content names keep changing, it can be edited by hand pre-release by pre-release if wanted (Still moved all to FEVTDEBUGHLT for now, except RECO, left alone).
%EventContent=(
	       #Use FEVTSIMDIGI for all steps but HLT and RECO
	       "GEN,SIM"=>"FEVTDEBUGHLT",
	       "DIGI"=>"FEVTDEBUGHLT",
	       #The following part should be edited to implement the wanted step and define the appropriate
	       #event content
	       #"L1"=>"FEVTDEBUGHLT",
	       #"DIGI2RAW"=>"FEVTDEBUGHLT",
	       #Use FEVTSIMDIGIHLTDEBUG for now
	       #"HLT"=>"FEVTDEBUGHLT",
	       #Use RECOSIM for RECO step
	       #"RAW2DIGI,RECO"=>"RECOSIM"
	       );
#The allowed profiles are:
@AllowedProfile=(
	  "TimingReport",
	  "TimeReport",
	  "SimpleMemReport",
	  "EdmSize",
	  "IgProfperf",
	  "IgProfMemTotal",
	  "IgProfMemLive",
	  "IgProfMemAnalyse",
	  "valgrind",
	  "memcheck_valgrind",
	  "None"
	   );
#Based on the profile code create the array of profiles to run:
for ($i=0;$i<10;$i++)
{
    if ($ProfileCode=~/$i/)
    {
	if (((($i==0)&&(($ProfileCode=~/1/)||($ProfileCode=~/2/)))||(($i==1)&&($ProfileCode=~/2/)))||((($i==5)&&(($ProfileCode=~/6/)||($ProfileCode=~/7/)))||(($i==6)&&($ProfileCode=~/7/))))
	{
	    $Profile[++$#Profile]="$AllowedProfile[$i]"." @@@ reuse";
	}
	else
	{
	    $Profile[++$#Profile]=$AllowedProfile[$i];
	}
    }
}
#Hash for the profiler to run
%Profiler=(
	   "TimingReport"=>"Timing_Parser",
	   "TimingReport @@@ reuse"=>"Timing_Parser",#Ugly fix to be able to handle the reuse case
	   "TimeReport"=>"Timereport_Parser",
	   "TimeReport @@@ reuse"=>"Timereport_Parser",#Ugly fix to be able to handle the reuse case
	   "SimpleMemReport"=>"SimpleMem_Parser",
           "EdmSize"=>"Edm_Size",
	   "IgProfperf"=>"IgProf_perf.PERF_TICKS",
	   "IgProfMemTotal"=>"IgProf_mem.MEM_TOTAL",
	   "IgProfMemTotal @@@ reuse"=>"IgProf_mem.MEM_TOTAL",#Ugly fix to be able to handle the reuse case
	   "IgProfMemLive"=>"IgProf_mem.MEM_LIVE",
	   "IgProfMemLive @@@ reuse"=>"IgProf_mem.MEM_LIVE",#Ugly fix to be able to handle the reuse case
	   "IgProfMemAnalyse"=>"IgProf_mem.ANALYSE",
	   "valgrind"=>"ValgrindFCE",
	   "memcheck_valgrind"=>"Memcheck_Valgrind",
	   "None"=>"None"
	   );
#Hash to switch from keyword to .cfi use of cmsDriver.py:
%KeywordToCfi=(
	       #For now use existing:
	       "HZZLLLL"=>"H200ZZ4L.cfi",
	       #But for consistency we should add H190ZZ4LL.cfi into the Configuration/Generator/data:
	       #"HZZLLLL"=>"H190ZZ4L.cfi",
	       "MINBIAS"=>"MinBias.cfi",
	       #For now test using existing (wrong but not to change rest of the code):
	       #"E -e 1000"=>"SingleElectronPt1000.cfi",
	       #But we'd like:
	       "E -e 1000"=>"SingleElectronE1000.cfi",
	       "MU- -e pt10"=>"SingleMuPt10.cfi",
	       #For now use existing (wrong but not to change rest of the code):
	       #"PI- -e 1000"=>"SinglePiPt1000.cfi",
	       #But we'd like:
	       "PI- -e 1000"=>"SinglePiE1000.cfi",
	       "TTBAR"=>"TTbar.cfi",
	       "QCD -e 80_120"=>"QCD_Pt_80_120.cfi"
	       );
foreach (@Candle)
{
    $candle=$_;
    print "*Candle $candle\n";
    $stepIndex=0;
    foreach (@Steps)
    {
	print SIMCANDLES "#$FileName{$candle}\n";
	$step=$_;
	print SIMCANDLES "#Step $step\n";
	print "$step\n";
	if ($step eq "DIGI2RAW")
	{
	    #print "DIGI2RAW\n";
	    #print "$step\n";
	    @SavedProfile=@Profile;
	    @Profile=("None");
	}
	if ($step eq "HLT")
	{
	    @Profile=@SavedProfile;
	}
	foreach (@Profile)
	{
	    if ($_ eq "EdmSize")
	    {
		if ($step eq "GEN,SIM") #Hack since we use SIM and not GEN,SIM extension (to facilitate DIGI)
		{
		    $step="GEN\,SIM";
		}
		$Command="$FileName{$candle}"."_"."$step".".root ";
	    }
	    else
	    {
		if (!$CustomiseFragment{$step})
		{
		    #Temporary hack to have potentially added steps use the default Simulation.py fragment
		    #This should change once each group customises its customise python fragments.
		    $CustomisePythonFragment=$CustomiseFragment{"DIGI"}
		}
		else
		{
		    $CustomisePythonFragment=$CustomiseFragment{$step}
		}
		#Adding a fileout option too to avoid dependence on future convention changes in cmsDriver.py:
		$OutputFileOption="--fileout=$FileName{$candle}"."_"."$step".".root";
		$OutputStep=$step;

		#Use --filein (have to for L1, DIGI2RAW, HLT) to add robustness
		if ($step eq "GEN,SIM") #there is no input file for GEN,SIM!
		{
		    $InputFileOption="";
		}
		#Special hand skipping of HLT since it is not stable enough, so it will not prevent 
		#RAW2DIGI,RECO from running
		elsif ($Steps[$stepIndex-1] eq "HLT")
		{
		    $InputFileOption="--filein file:$FileName{$candle}"."_"."$Steps[$stepIndex-2]".".root ";
		}
		else
		{
		    $InputFileOption="--filein file:$FileName{$candle}"."_"."$Steps[$stepIndex-1]".".root ";
		}
		#Adding .cfi to use new method of using cmsDriver.py
		#$Command="$cmsDriver $candle -n $NumberOfEvents --step=$step $FileIn{$step}$InputFile --customise=$CustomiseFragment{$step} ";
		#$Command="$cmsDriver $KeywordToCfi{$candle} -n $NumberOfEvents --step=$step $InputFileOption $OutputFileOption --eventcontent=$EventContent{$step} --customise=$CustomiseFragment{$step} $cmsDriverOptions";
		$Command="$cmsDriver $KeywordToCfi{$candle} -n $NumberOfEvents --step=$step $InputFileOption $OutputFileOption --customise=$CustomisePythonFragment $cmsDriverOptions";
	    }
	    print SIMCANDLES "$Command @@@ $Profiler{$_} @@@ $FileName{$candle}_"."$OutputStep"."_"."$_"."\n";
	}
	$stepIndex++;
    }
    #Add the extra "step" DIGI with PILE UP only for QCD_80_120:
#After digi pileup steps:
#Freeze this for now since we will only run by default the GEN-SIM,DIGI and DIGI pileup steps
@AfterPileUpSteps=(
		   #"L1",
		   #"DIGI2RAW",
		   #"HLT",
		   #"RAW2DIGI,RECO"
		   );
    if ($candle eq "QCD -e 80_120")
    {
	#First run the DIGI with PILEUP (using the MixingModule.py)
	#Hardcode stuff for this special step
	print SIMCANDLES "#$FileName{$candle}\n";
	print SIMCANDLES "#DIGI PILE-UP STEP\n";
	print "DIGI PILEUP\n";
	foreach (@Profile)
	{
	    if ($_ eq "EdmSize")
	    {
		$Command="$FileName{$candle}"."_DIGI_PILEUP.root ";
	    }
	    else
	    {
		$InputFileOption="--filein file:$FileName{$candle}"."_GEN,SIM.root ";
		$OutputFileOption="--fileout=$FileName{$candle}"."_DIGI_PILEUP.root";
		#Adding .cfi to use new method of using cmsDriver.py
		#$Command="$cmsDriver $candle -n $NumberOfEvents --step=$step $FileIn{$step}$InputFile --customise=$CustomiseFragment{$step} ";
		#$Command="$cmsDriver $KeywordToCfi{$candle} -n $NumberOfEvents --step=DIGI $InputFileOption $OutputFileOption --PU --eventcontent=FEVTSIMDIGI --customise=Configuration/PyReleaseValidation/MixingModule.py $cmsDriverOptions";
		$Command="$cmsDriver $KeywordToCfi{$candle} -n $NumberOfEvents --step=DIGI $InputFileOption $OutputFileOption --PU --customise=Configuration/PyReleaseValidation/MixingModule.py $cmsDriverOptions";
	    }
	    print SIMCANDLES "$Command @@@ $Profiler{$_} @@@ $FileName{$candle}_DIGI_PILEUP_"."$_"."\n";
	}
	#Very messy solution for now:
	#Setting the stepIndex variable to 2, i.e. RECO step
	$stepIndex=2;
	$FileIn{"RECO"}="--filein file:";
	foreach (@AfterPileUpSteps)
	{
	    print SIMCANDLES "#$FileName{$candle}\n";
	    $step=$_;
	    print SIMCANDLES "#Step $step PILEUP\n";
	    print "$step PILEUP\n";
	    if ($step eq "DIGI2RAW")
	    {
		@SavedProfile=@Profile;
		@Profile=("None");
	    }
	    if ($step eq "HLT")
	    {
		@Profile=@SavedProfile;
	    }
	    
	    foreach (@Profile)
	    {
		if ($_ eq "EdmSize")
		{
		    $Command="$FileName{$candle}"."_"."$step"."_PILEUP".".root ";
		}
		else
		{
		    if (!$CustomiseFragment{$step})
		    {
			#Temporary hack to have potentially added steps use the default Simulation.py fragment
			#This should change once each group customises its customise python fragments.
			$CustomisePythonFragment=$CustomiseFragment{"DIGI"}
		    }
		    else
		    {
			$CustomisePythonFragment=$CustomiseFragment{$step}
		    }
		    $OutputStep=$step."_PILEUP";
		    $InputFileOption="$FileName{$candle}"."_"."$Steps[$stepIndex-1]"."_PILEUP".".root ";
		    $OutputFileOption="--fileout=$FileName{$candle}"."_"."$step"."_PILEUP.root";
		    #Adding .cfi to use new method of using cmsDriver.py
		    #$Command="$cmsDriver $candle -n $NumberOfEvents --step=$step $FileIn{$step}$InputFile --customise=$CustomiseFragment{$step} ";
		    #$Command="$cmsDriver $KeywordToCfi{$candle} -n $NumberOfEvents --step=$step $InputFileOption $OutputFileOption --eventcontent=$EventContent{$step} --customise=$CustomiseFragment{$step} $cmsDriverOptions";
		    $Command="$cmsDriver $KeywordToCfi{$candle} -n $NumberOfEvents --step=$step $InputFileOption $OutputFileOption  --customise=$CustomisePythonFragment $cmsDriverOptions"; 
		}
		print SIMCANDLES "$Command @@@ $Profiler{$_} @@@ $FileName{$candle}_"."$OutputStep"."_"."$_"."\n";
	    }
	    $stepIndex++;
	}
    }
}
exit;
