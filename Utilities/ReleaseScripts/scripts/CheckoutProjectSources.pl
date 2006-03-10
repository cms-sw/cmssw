#!/usr/bin/perl 
#____________________________________________________________________ 
# File: CheckoutProjectSources.pl
#____________________________________________________________________ 
#  
# Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
# Update: 2005-10-28 12:12:08+0200
# Revision: $Id: CheckoutProjectSources.pl,v 1.1 2005/12/13 16:52:16 argiro Exp nobody $ 
#
# Copyright: 2005 (C) Shaun ASHBY
#
#--------------------------------------------------------------------
use Cwd;
use Getopt::Long ();

# Fixed parameters:
my $cvs = '/usr/bin/cvs';
my $projectroot='CMSSW';
my $cvsroot = ':kserver:cmscvs.cern.ch:/cvs_server/repositories/'.$projectroot;
my $releaseconf = $projectroot.'/Release.conf';

my $outdir;
my $rconftag;

# Somewhere to keep track of packages and tag versions:
my $versionfile='Versions';

# Opt support:
# - Use a specific tag for the Release.conf;
# - Choose the output directory;

# Getopt option variables:
my %opts;
my %options =
   ("outdir=s"        => sub { $outdir=$_[1] },
    "tag=s"           => sub { $rconftag=$_[1] },
    "run"             => sub { $opt{RUN} = 1 },
    "query"           => sub { $opt{QUERY} = 1 }
    );

# Get the options using Getopt:
Getopt::Long::config qw(default no_ignore_case require_order);

if (! Getopt::Long::GetOptions(\%opts, %options))
   {
   print "$0: Error with arguments.","\n";
   exit(1);
   }
else
   {
   # Set where to put the processed files:
   $outdir ||= cwd()."/src";
   # Tag of Release.conf:
   $rconftag ||= 'HEAD';

   # Check to see if we're running a query. If so we can do it
   # and return without checking anything out:
   if ($opt{QUERY})
      {
      &query();
      exit(0);
      }

   # Only run if the run option is given:
   if ($opt{RUN})
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
      # Print debug info only:
      &dumpinfo();     
      }
   }

sub query()
   {
   print "Running query of $releaseconf version $rconftag","\n";
   &dumpinfo(); 
   print "\n";
   
   open (CVSCOHTML, $cvs.' -Qn -d '.$cvsroot.' co -p -r '.$rconftag.' '.$releaseconf.' |')
      || die "Unable to access cvs server $!";
   
   while (<CVSCOHTML>)
      {
      chomp;
      my ($pkg, $tag)=split;
      # Print package info:
      printf ("%-20s %-10s\n",$pkg,$tag);
      }
   
   close CVSCOHTML;   
   }

sub checkout()
   {
   # Get the list of packages and tags. Do this by checking out the Release.conf, using -p
   # option to CVS so that the file output is piped from CVS instead of being checked out to a
   # local directory and then being read back:
   open (CVSCOHTML, $cvs.' -Qn -d '.$cvsroot.' co -p -r '.$rconftag.' '.$releaseconf.' |')
      || die "Unable to access cvs server $!";
   
   # Somewhere to write the tags for future reference:
   open(VERSIONS, "> $versionfile") || die "$versionfile: $!","\n";
   # Keep a record of which tag was taken:
   print VERSIONS "CVSTAG: ",$rconftag,"\n";
   
   while (<CVSCOHTML>)
      {
      chomp;
      my ($pkg, $tag)=split;
      # Check out the package:
      printf VERSIONS ("%-20s %-10s\n",$pkg,$tag);
      print "Checking out ",$pkg," with tag ",$tag,"\n";
      system($cvs,"-Q","co","-P","-r",$tag,$pkg);
      }
   
   close VERSIONS;
   close CVSCOHTML;   
   }

sub dumpinfo()
   {   
   print "\n";
   print "Current parameters are:","\n";
   print "\n";
   print "-> CVSROOT = ",$cvsroot,"\n";
   print "-> Release.conf = ",$releaseconf,"\n";
   print "-> CVS tag = ",$rconftag,"\n";
   print "-> Output dir = ",$outdir,"\n";
   print "\n";
   print "Use the \"-run\" option to actually do the checkout.","\n";   
   }
