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
GetOptions(\%options,'h','help','rel=s','pack=s'=>\@packs,'file=s');

if ( !$options{'rel'} || $options{'h'} || $options{'help'} ) {
    print "CmsTCPackageList.pl usage:\n";
    print "    CmsTCPackageList.pl --rel <release>\n \n";
    print "Options:\n";
    print "  --pack <packages> Print only for specified package(s)\n";
    print "         (Either space separated in quotes or use multiple --pack options for multiple packages)\n";
    print "  --file <file>   Send output to file\n";
    print "\n";
    exit;
} 

my $rel= $options{'rel'} ? $options{'rel'} : die "Need a release (--rel)";

my %packages;
my $onlySpecifiedPackages=0;
#if ( $options{'pack'} ) {
if ( @packs ) {
    $onlySpecifiedPackages=1;
    foreach (@packs )  {
	my @spTmp=split(' ',$_,999);
	foreach (@spTmp) {
	    $packages{$_}=1;
	}
    }
}

my $user="cmstcreader";
my $pass="CmsTC";

open(CMSTCQUERY,"/usr/bin/wget -nv -o /dev/null -O- 'http://${user}:${pass}\@cmsdoc.cern.ch/swdev/CmsTC/cgi-bin/CreateTagList?release=${rel}' |");

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
foreach $key (keys %tags) {
    if ( $key eq "-1" ) {
# error condition.. missing release	
# should be a better way to catch this
	close OUTFILE;
	print "Release $rel does not exist in CmsTC\n";
	exit;
    }
    $hasAPackage++;
    if ( ($onlySpecifiedPackages==0) || ($packages{$key}==1) ) {
	print OUTFILE "$key $tags{$key} \n";
    }
}

close OUTFILE;

print "No packages found in release ${rel}\n" if ( $hasAPackage==0);





