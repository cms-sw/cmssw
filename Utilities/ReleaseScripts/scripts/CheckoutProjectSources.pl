#!/usr/bin/env perl 

#____________________________________________________________________ 
# File: CheckoutProjectSources.pl
#____________________________________________________________________ 
#  
# Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
# Update: 2005-10-28 12:12:08+0200
# Revision: $Id: CheckoutProjectSources.pl,v 1.4 2009/02/06 08:05:48 andreasp Exp $ 
#
# Copyright: 2005 (C) Shaun ASHBY
#
#--------------------------------------------------------------------
use Cwd;
use Getopt::Long ();

# Fixed parameters:
my $cvs = '/usr/bin/cvs';
my $projectroot='CMSSW';
my $cvsroot = ':gserver:cmscvs.cern.ch:/cvs_server/repositories/'.$projectroot;
my $tagfile;
my $outdir;
my $rv;
# Somewhere to store checked-out tags:
my $versionfile="CPSVersions";

# Getopt option variables:
my %opts;
my %options =
   ("outdir=s"        => sub { $outdir=$_[1] },
    "run"             => sub { $opt{RUN} = 1 },
    "query"           => sub { $opt{QUERY} = 1 },
    "help"            => sub { &usage(); exit(0)}
    );

# Get the options using Getopt:
Getopt::Long::config qw(default no_ignore_case require_order);

if (! Getopt::Long::GetOptions(\%opts, %options))
   {
   print "$0: Error with arguments.","\n";
   &usage();
   exit(1);
   }
else
   {
   # Check to make sure we have a file to read the tags from:
   ($tagfile)=(@ARGV);
   # If the file is found in cwd, create full path. Otherwise
   # assume that the full path was given:
   if (-f cwd()."/".$tagfile)
      {
      $tagfile = cwd()."/".$tagfile;
      }
      
   # Set where to put the processed files:
   $outdir ||= cwd()."/src";
   # Check to see if we're running a query. If so we can do it
   # and return without checking anything out:
   if ($opt{QUERY})
      {
      &query();
      exit(0);
      }
   # Only run if the run option is given:
   elsif ($opt{RUN})
      {
      # Create the output directory if it doesn't already exist:
      if (! -d $outdir)
	 {
	 system("mkdir",$outdir);
	 }
      
      # Move to the output directory:
      chdir $outdir;
      &checkout();
      }
   else
      {
      &usage();
      exit(1);
      }
   }

sub query()
   {
   open (TAGFILE,"$tagfile") || die "Unable to read file $tagfile: $!";   
   while (<TAGFILE>)
      {
      chomp;
      my ($pkg, $tag)=split;
      # Print package info:
      printf ("%-20s %-10s\n",$pkg,$tag);
      }
   
   close TAGFILE;   
   }

sub checkout()
   {
   # Get the list of packages and tags from the file specified.
   print "Running checkout: reading tags from $tagfile","\n";
   print "\n";
   
   open (TAGFILE,"$tagfile") || die "Unable to read file $tagfile: $!";

   # Somewhere to write the tags for future reference:
   open(VERSIONS, "> $versionfile") || die "$versionfile: $!","\n";
   # Keep a record of which tag was taken:
   print VERSIONS "TAGFILE file: ",$tagfile,"\n";
   
   while (<TAGFILE>)
      {
      chomp;
      my ($pkg, $tag)=split;
      # Check out the package:
      $rv = system($cvs,"-Q","-d",$cvsroot,"co","-P","-r",$tag,$pkg);
      # Check the status of the checkout and only write to VERSIONS if
      # the tag really exists:
      if ($rv == 0)
	 {
	 printf ("Package %-45s version %-10s checkout SUCCESSFUL\n",$pkg,$tag);
	 printf VERSIONS ("%-20s %-10s\n",$pkg,$tag);
	 }
      else
	 {
	 printf STDERR ("Package %-45s version %-10s checkout FAILED\n",$pkg,$tag);
	 printf STDERR "Checkout ERROR: tag $tag for package $pkg is not correct!","\n";
	 print "\n";
	 exit(1);
	 }
      }
   
   close VERSIONS;
   close TAGFILE;   
   }

sub usage()
   {
   my $string="\nUsage: CheckoutProjectSources.pl [--help|-h] [--run|-r OR --query|-q] <TAGFILE>\n";
   $string.="\n";
   $string.="<TAGFILE>             The file containing the list of package tags\n";
   $string.="--run|-r              Do the code checkout....\n";
   $string.="--query|-q            ..or just query tags on packages.\n";
   $string.="\n";
   print $string,"\n";
   }
