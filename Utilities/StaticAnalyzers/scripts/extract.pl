#!/usr/bin/env perl
use Text::Balanced qw/:ALL/;
use strict;
use warnings;
use Data::Dumper;
use Scalar::Util 'blessed';
my @queue =<>;

while  (@queue){
my $string=shift @queue;
$string =~ s/\&gt\;/\>/g;
$string =~ s/\&lt\;/\</g;
my @fields = extract_multiple($string,
	[
	{ B => sub { extract_bracketed($_[0],'<>','[^<>]++') } },
	]);

#print "\nshift: $string\n";
#print Dumper(\@fields);
foreach (@fields) {	
	if ( defined blessed($_) )
		{
		my $temp = $$_;
		$temp =~ s/<//;
		$temp =~ s/>//;
#		print "unshift: $temp\n";
		unshift (@queue, $temp); 
		}
	else
		{
		$,="\n";print join (' ',split (' ',$_)), "\n" }; 
	}
}
