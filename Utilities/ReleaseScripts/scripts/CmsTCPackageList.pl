#!/usr/local/bin/perl
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
GetOptions(\%options,'rel=s');

if ( !$options{'rel'} ) {
    print "CmsTCPackageList.pl usage:\n";
    print "    CmsTCPackageList.pl --rel <release>\n";
    print "\n";
    exit;
} 

my $rel= $options{'rel'} ? $options{'rel'} : die "Need a release (--rel)";

my $user="cmstcreader";
my $pass="CmsTC";

open(CMSTCQUERY,"/usr/bin/wget -nv -O- 'http://${user}:${pass}\@cmsdoc.cern.ch/swdev/CmsTC/cgi-bin/CreateTagList?release=${rel}' |");

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

my $key;
foreach $key (keys %tags) {
    print "$key $tags{$key} \n";
}




