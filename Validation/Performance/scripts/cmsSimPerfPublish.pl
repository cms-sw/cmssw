#!/usr/bin/perl
#G.Benelli Jan 22 2007
#A little script to move Simulation Performance Suite
#relevant html and log files into our public area
#/afs/cern.ch/cms/sdt/web/performance/simulation/
#Set here the standard number of events (could become an option... or could be read from the log...)
$TimeSizeNumOfEvents=100;
$IgProfNumOfEvents=5;
$ValgrindNumOfEvents=1;
$LocalWebArea=$ARGV[0]; 

#Some nomenclature
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
    $Candle[0]=>"HZZLLLL_190",
    $Candle[1]=>"MINBIAS",
    $Candle[2]=>"E_1000",
    $Candle[3]=>"MU_1000",
    $Candle[4]=>"PI-_1000",#For now leave it like this to match cmsSimPyRelVal
    $Candle[5]=>"TTBAR",
    $Candle[6]=>"ZPJJ"
    );
@Profile=(
          "TimingReport",
          "TimeReport",
          "SimpleMemReport",
          "EdmSize",
          "IgProfperf",
          "IgProfMemTotal",
          "IgProfMemLive",
          "IgProfMemAnalyse",
          "valgrind",
          "memcheck_valgrind"
	   );
@DirName=(
	  "TimeSize",
	  "IgProf",
	  "Valgrind"
	  );
@Step=(
       "SIM",
       "DIGI"
       );
%StepLowCaps=(
	      "SIM"=>"sim",
	      "DIGI"=>"digi"
	      );
%NumOfEvents=(
	    $DirName[0]=>"$TimeSizeNumOfEvents",
	    $DirName[1]=>"$IgProfNumOfEvents",
	    $DirName[2]=>"$ValgrindNumOfEvents"
	    );
%OutputHtml=(
	     $Profile[0]=>"*TimingReport.html",
	     $Profile[1]=>"TimeReport.html",
	     $Profile[2]=>"*.html",
	     $Profile[3]=>"objects_pp.html",
	     $Profile[4]=>"overall.html",
	     $Profile[5]=>"overall.html",
	     $Profile[6]=>"overall.html",
	     $Profile[7]=>"doBeginJob_output.html",
	     $Profile[8]=>"overall.html",
	     $Profile[9]=>"beginjob.html"
	     );
@IgProfMemAnalyseOut=(
		      "doBeginJob_output.html",
		      "doProduce_output.html",
		      "mem.html"
		      );
@memcheck_valgrindOut=(
		       "beginjob.html",
		       "edproduce.html",
		       "esproduce.html"
		       );
#Get the CMSSW_VERSION from the environment
$CMSSW_VERSION=$ENV{'CMSSW_VERSION'};
$CMSSW_RELEASE_BASE=$ENV{'CMSSW_RELEASE_BASE'};
$CMSSW_BASE=$ENV{'CMSSW_BASE'};
$HOST=$ENV{'HOST'};
$LocalPath=`pwd`;
$ShowTagsResult=`showtags -r`;

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

#Define the web publishing area
if ($LocalWebArea eq "local")
{ 
    $WebArea="/tmp/gbenelli/"."$CMSSW_VERSION";
    print "Creating local directory $WebArea\n";
    system("mkdir $WebArea");
}
else
{
    $WebArea="/afs/cern.ch/cms/sdt/web/performance/simulation/"."$CMSSW_VERSION";
}
#Dump some info in a file   
opendir(WEBDIR,$WebArea)||die "The area $WebArea does not exist!\nRun the appropriate script to request AFS space first, or wait for it to create it!\n";
@Contents=readdir(WEBDIR);
#    $CheckDir=`ls $WebArea`;
#    if ($CheckDir eq "")
if ($#Contents==1)#@Contents will have only 2 entries . and .. if the dir is empty.
{
    print "The area $WebArea is ready to be populated!\n";
}
else
{
    print "The area $WebArea already exists!\n";
    exit;
}
$date=`date`;
@LogFiles=`ls cms*.log`;
print "Found the following log files:\n";
print @LogFiles;
$ExecutionDateSec=0;
foreach (@LogFiles)
{
    chomp($_);
    if ($_=~/^cmsCreateSimPerfTest/)
    {
	$ExecutionDateLastSec=`stat --format=\%Z $_`;
	$ExecutionDateLast=`stat --format=\%y $_`;
	print "Execution date for $_ was: $ExecutionDateLast";
	if ($ExecutionDateLastSec>$ExecutionDateSec)
	{
	    $ExecutionDateSec=$ExecutionDateLastSec;
	    $ExecutionDate=$ExecutionDateLast;
	}	
    }
}
print "Copying the logfiles to $WebArea/.\n";
system("cp -pR cms*.log $WebArea/.");
#Copy the perf_style.css file from Validation/Performance/doc
print "Copying perf_style.css style file to $WebArea/.\n";
system("cp -pR $BASE_PERFORMANCE/doc/perf_style.css $WebArea/.");
@Dir=`ls`;
chomp(@Dir);
#Produce a small logfile with basic info on the Production area
$LogFile="$WebArea/"."ProductionLog.txt";
open(LOG,">$LogFile")||die "CAnnot open file $_!\n$!\n";
print "Writing Production Host, Location, Release and Tags information in $LogFile\n"; 
print LOG "These performance tests were executed on host $HOST and published on $date\n";
print LOG "They were run in $LocalPath\n";
print LOG "Results of showtags -r in the local release:\n$ShowTagsResult\n";
close(LOG);
#Produce a small index.html file to navigate the html reports/logs etc
$IndexFile="$WebArea"."/index.html";
open(INDEX,">$IndexFile")||die "Cannot open file $IndexFile!\n$!\n";
print "Writing an index.html file with links to the profiles report information for easier navigation\n"; 
$TemplateHtml="$BASE_PERFORMANCE"."/doc/index.html";
open(TEMPLATE,"<$TemplateHtml")||die "Couldn't open file $TemplateHtml - $!\n";
#Loop line by line to build our index.html based on the template one
while (<TEMPLATE>)
{
    $NewFileLine=$_;
    if ($_=~/\CMSSW_VERSION/)
    {
	print INDEX $CMSSW_VERSION;
	next;
	}
    if ($_=~/\HOST/)
    {
	print INDEX $HOST;
	next;
    }
    if ($_=~/LocalPath/)
	{
	    print INDEX $LocalPath;
	    next;
	}
    if ($_=~/ProductionDate/)
    {
	print INDEX $ExecutionDate;
	next;
    }
    if ($_=~/LogfileLinks/)
    {
	print INDEX "<br><br>";
	foreach (@LogFiles)
	{
	    chomp($_);
	    #$LogFileLink="$WebArea/"."$_";
	    print INDEX "<a href="."$_"."> $_ <\/a>";
	    print INDEX "<br><br>";
	}
	next;
    }
    if ($_=~/DirectoryBrowsing/)
    {
	#Create a subdirectory DirectoryBrowsing to circumvent the fact the dir is not browsable if there is an index.html in it.
	system("mkdir $WebArea/DirectoryBrowsing");
	print INDEX "Click <a href=\.\/DirectoryBrowsing\/\.>here<\/a> to browse the directory containing all results (except the root files)\n";
	next;
    }
    if ($_=~/PublicationDate/)
    {
	print INDEX $date;
	next;
    }
    if ($_=~/\CandlesHere/)
    {
	foreach (@Candle)
	{
	    print INDEX "<table cellpadding=20px border=1><td width=25% bgcolor=#FFFFFE >";
	    print INDEX "<h2>";
	    print INDEX $_;
	    $CurrentCandle=$_;
	    print INDEX "<\/h2>";
	    print INDEX "<dir style=\"font-size: 13\"> \n <p>";
	    foreach (@DirName)
	    {
		chomp($_);
		$CurDir=$_;
		$LocalPath="$CurrentCandle"."_$CurDir";
		@CandleLogFiles=`find $LocalPath \-name \"\*.log\"`;
		if (@CandleLogFiles)
		{
		    print INDEX "<br><b>Logfiles for $CurDir<\/b><br>";
		    foreach (@CandleLogFiles)
		    {
			chomp($_);
			print "Found $_ in $LocalPath\n";
			system("cp -pR $_ $WebArea/.");
			print INDEX "<a href="."$_".">$_ <\/a>";
		    }
		}
		foreach (@Profile)
		{
		    $CurrentProfile=$_;
		    foreach (@Step)
		    {
			$ProfileTemplate="$CurrentCandle"."_"."$CurDir"."/"."*_"."$_"."_"."$CurrentProfile"."*/"."$OutputHtml{$CurrentProfile}";
			$ProfileTemplateLowCaps="$CurrentCandle"."_"."$CurDir"."/"."*_"."$StepLowCaps{$_}"."_"."$CurrentProfile"."*/"."$OutputHtml{$CurrentProfile}";
			    $ProfileReportLink=`ls $ProfileTemplate 2>/dev/null`;
			if ( $ProfileReportLink !~ /^$CurrentCandle/)#no match with caps try low caps
			{
			    $ProfileReportLink=`ls $ProfileTemplateLowCaps 2>/dev/null`;
			}
			if ($ProfileReportLink=~/$CurrentProfile/)#It could also not be there
			{
			    if ($PrintedOnce==0)
			    {
				print INDEX "<br>";
				print INDEX "<b>$CurDir</b>";
				$PrintedOnce=1;
			    }
			    print INDEX "<li><a href="."$ProfileReportLink".">$CurrentProfile $_ ("."$NumOfEvents{$CurDir}"." events)<\/a>";
			    if ($CurrentProfile eq $Profile[7])
			    {
				for ($i=1;$i<3;$i++)
				{
				    $ProfileTemplate="$CurrentCandle"."_"."$CurDir"."/"."*_"."$_"."_"."$CurrentProfile"."*/"."$IgProfMemAnalyseOut[$i]";
				    $ProfileTemplateLowCaps="$CurrentCandle"."_"."$CurDir"."/"."*_"."$StepLowCaps{$_}"."_"."$CurrentProfile"."*/"."$IgProfMemAnalyseOut[$i]";
				    $ProfileReportLink=`ls $ProfileTemplate 2>/dev/null`;
				    if ( $ProfileReportLink !~ /^$CurrentCandle/)#no match with caps try low caps
				    {
					$ProfileReportLink=`ls $ProfileTemplateLowCaps 2>/dev/null`;
				    }
				    if ($ProfileReportLink=~/$CurrentProfile/)#It could also not be there
				    {
					print INDEX "<li><a href="."$ProfileReportLink".">$CurrentProfile $IgProfMemAnalyseOut[$i] $_ ("."$NumOfEvents{$CurDir}"." events)<\/a>";
				    }
				}
			    }
			    if ($CurrentProfile eq $Profile[9])
			    {
				for ($i=1;$i<3;$i++)
				{
				    $ProfileTemplate="$CurrentCandle"."_"."$CurDir"."/"."*_"."$_"."_"."$CurrentProfile"."*/"."$memcheck_valgrindOut[$i]";
				    $ProfileTemplateLowCaps="$CurrentCandle"."_"."$CurDir"."/"."*_"."$StepLowCaps{$_}"."_"."$CurrentProfile"."*/"."$memcheck_valgrindOut[$i]";
				    $ProfileReportLink=`ls $ProfileTemplate 2>/dev/null`;
				    if ( $ProfileReportLink !~ /^$CurrentCandle/)#no match with caps try low caps
				    {
					$ProfileReportLink=`ls $ProfileTemplateLowCaps 2>/dev/null`;
				    }
				    if ($ProfileReportLink=~/$CurrentProfile/)#It could also not be there
				    {
					print INDEX "<li><a href="."$ProfileReportLink".">$CurrentProfile $memcheck_valgrindOut[$i] $_ ("."$NumOfEvents{$CurDir}"." events)<\/a>";
				    }
				}
			    }
			}
		    }
		}
		$PrintedOnce=0;
	    }
	    print INDEX "<\/p>";
	    print INDEX "</dir>";
	    print INDEX "<hr>";
	    print INDEX "<br>";
	    print INDEX "<\/td><\/table>";
	}	    
	next;
    }
    print INDEX $NewFileLine;
    
} #End of while loop on template html file
foreach (@Dir)
{
    $CurrentDir=$_;
    foreach (@DirName)
    {
	if ($CurrentDir=~/$_/)#To get rid of possible spurious dirs
	{
	    print "Copying $CurrentDir to $WebArea\n";
	    $CopyDir=`cp -pR $CurrentDir $WebArea/.`;
	    $RemoteDirRootFiles="$WebArea/"."$CurrentDir/*.root";
	    $RemoveRootFiles=`rm -Rf $RemoteDirRootFiles`; 
	}
    }
}
#Creating symbolic links to the web area in subdirectory to allow directory browsing:
@DirectoryContent=`ls $WebArea`;
foreach (@DirectoryContent)
{
    chomp($_);
    if (($_ ne "index.html")&&($_ ne "DirectoryBrowsing"))
    {
	system("ln -s $WebArea/$_ $WebArea/DirectoryBrowsing/$_");
    }
}
print INDEX "\<\/body\>\n";
print INDEX "\<\/html\>\n";
close(INDEX);

exit;

