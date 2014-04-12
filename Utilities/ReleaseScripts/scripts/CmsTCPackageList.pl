#!/usr/bin/env perl
# 
# 
#
# David Lange, LLNL. October 18, 2005: Port to CMS Tag Collector
#
# Script to return a list of tags from CmsTC for a given release
# Required option --rel

use strict;
use Getopt::Long;

Getopt::Long::config('bundling_override');
 
my %options;
my @packs;
my @subsys;
my @ignoredPacks;

GetOptions(\%options,'h','help','rel=s','pack=s'=>\@packs,
	   'subsys=s'=>\@subsys,
	   'file=s','listOnlyVersion','ignorepack=s'=>\@ignoredPacks);

if ( !$options{'rel'} || $options{'h'} || $options{'help'} ) {
    print "CmsTCPackageList.pl usage:\n";
    print "    CmsTCPackageList.pl --rel <release>\n \n";
    print "Options:\n";
    print "  --pack <packages> Print only for specified package(s)\n";
    print "  --subsys <subsystems> Print only for specified subsystem(s)\n";
    print "         (Either space separated in quotes or use multiple --pack/--subsys options for multiple packages or subsystems)\n";
    print "  --ignorepack <packages> Ignore specified package(s)\n";
    print "         (Either space separated in quotes or use multiple --ignorepack options for multiple packages)\n";
    print "  --listOnlyVersion   Output only the version, not the corresponding package name\n";
    print "  --file <file>   Send output to file\n";
    print "\n";
    exit;
} 

my $rel= $options{'rel'} ? $options{'rel'} : die "Need a release (--rel)";

# check that wget exists
my @wgets=split(' ',`whereis -b wget`,9);
die "wget does not seem to be in your current path. Exiting.\n" if ($#wgets == 1);

my %packages;
my $onlySpecifiedPackages=0;
if ( @packs ) {
    $onlySpecifiedPackages=1;
    foreach (@packs )  {
	my @spTmp=split(' ',$_,999);
	foreach (@spTmp) {
	    $packages{$_}=1;
	}
    }
}

my %subsystems;
my $onlySpecifiedSubsystems=0;
if ( @subsys ) {
    $onlySpecifiedSubsystems=1;
    foreach (@subsys )  {
	my @spTmp=split(' ',$_,999);
	foreach (@spTmp) {
	    $subsystems{$_}=1;
	}
    }
}

my %ignorePackages;
if ( @ignoredPacks ) {
    foreach (@ignoredPacks )  {
	my @spTmp=split(' ',$_,999);
	foreach (@spTmp) {
	    $ignorePackages{$_}=1;
	}
    }
}

my $onlySpecified= $onlySpecifiedSubsystems + $onlySpecifiedPackages;


#figure out what version of wget we have
# --no-check-certificate needed for 1.10 and above
my $wgetVers=`wget --version`;
my @splitOutput=split(' ',$wgetVers,9);
my @splitVers=split('\.',$splitOutput[2],9);
my $wgetVersion=1000*$splitVers[0]+$splitVers[1];

my $options="";
$options="--no-check-certificate" if ( $wgetVersion>1009);

open(CMSTCQUERY,"wget ${options}  -nv -o /dev/null -O- 'https://cmstags.cern.ch/tc/public/CreateTagList?release=${rel}' |");

my %tags;
while ( <CMSTCQUERY> ) {
    if ( $_ =~ /td/) {
	my @sp1=split(' ',$_,99);
	my $pack=$sp1[2];
	my $tag=$sp1[5];
	$tags{$pack}=$tag;
    }
}

close CMSTCQUERY;

my $filename= $options{'file'} ? $options{'file'} : "&STDOUT";
open (OUTFILE,">$filename") or die "can not open output file $filename";

my $hasAPackage=0;
my $key;
foreach $key (sort keys %tags) {
    if ( $key eq "-1" ) {
# error condition.. missing release	
# should be a better way to catch this
	close OUTFILE;
	print "Release $rel does not exist in CmsTC\n";
	exit;
    }
    $hasAPackage++;
    my @spTmp1=split('/',$key,2);
    my $subsystem=$spTmp1[0];

    my $includeKey=0;
    $includeKey=1 if (($onlySpecified==0) || ($packages{$key}==1) ||
		      ($subsystems{$subsystem}==1));

    if ( $includeKey == 1 ) {
	if ( !$ignorePackages{$key} ) {
	    if ( $options{'listOnlyVersion'} ) {
		print OUTFILE "$tags{$key} \n";
	    }
	    else{
		print OUTFILE "$key $tags{$key} \n";
	    }
	}
    }
}

close OUTFILE;

print "No packages found in release ${rel}\n" if ( $hasAPackage==0);





