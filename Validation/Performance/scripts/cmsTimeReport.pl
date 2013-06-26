#!/usr/bin/perl
#GBenelli Nov 9 2007
#This script is designed to process a cmsRun output logfile and 
#spit out the TimeReport lines in an html file in a user-specified directory
#So it takes 2 arguments:
#1-input logfile (output of cmsRun, usually will have the CandleName.log)
#2-output directory (usually CandleName_TimeReport, in which the script 
#will create the TimeReport.html file)

if ($#ARGV != 1) {
	print "Usage: ./TimeReport.pl InputLogfile HtmlOutputDir\nE.g: ./TimeReport.pl HiggsZZ4LM190_sim_G4.log HiggsZZ4LM190_sim_G4_TimeReport OR ./TimeReport.pl ZPrimeJJM700_sim_G4.log ZPrimeJJM700_sim_G4_TimeReport \n";
	exit;
}

$InputFile=$ARGV[0];
$OutputDir=$ARGV[1];

#Open user-defined input file
open(INFILE, "<$InputFile")||die "Couldn't open file $InputFile - $!\n";
#Create user-defined directory (if non-existent):
if (!(-e $OutputDir))
{
    system("mkdir $OutputDir");
}
#Open standard TimeReport.html file to dump the TimeReport:
$TimeReport=$OutputDir."/TimeReport.html";
open(OUTFILE,">$TimeReport")||die "Couldn't open file $TimeReport to save - $!\n";
$date=`date`;
$path=`pwd`;
chomp($path);
$LogFile=$path."/".$InputFile;
$CMSSW_VERSION=$ENV{'CMSSW_VERSION'};
$CMSSW_BASE=$ENV{'CMSSW_BASE'};
print OUTFILE "\<html\>\n";
print OUTFILE "\<body\>\n";
print OUTFILE "\<h2 allign=\"center\"\>Time Report for $CMSSW_VERSION\<\/h2\>\n";
print OUTFILE "\<h3\>Extracted from $LogFile \<br\>by TimeReport.pl on $date \<\/h3\>\n";
print OUTFILE "\<h4\>Local Test Release:$CMSSW_BASE\<\/h4\>\n";
print OUTFILE "\<table align=\"center\", border=2\>\n";
while (<INFILE>) #Loop line by line
{
#    if (/^TimeReport/)
    @word=split(/\s+/);
    if ($word[0] eq "TimeReport")
    {
	#Here we could match the lines to get the times as individual variables
	#Could parse out the leading TimeReport word, space out the numbers...
	#print OUTFILE;
	print OUTFILE "\<tr\>";
	foreach (@word)
	{
	    if ($_ ne "TimeReport")
	    {
		print OUTFILE "\<td align=\"center\"\>$_\<\/td\>";
	    }
	}
	print OUTFILE "\<tr\>\n";
    }
}
print OUTFILE "\<\/table\>\n";
print OUTFILE "\<\/body\>\n";
print OUTFILE "\<\/html\>\n";
close(INFILE);
close(OUTFILE);
exit;
