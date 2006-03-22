#!/usr/bin/perl 

#____________________________________________________________________ 
# File: CheckoutProjectSources.pl
#____________________________________________________________________ 
#  
# Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
# Update: 2005-10-28 12:12:08+0200
# Revision: $Id$ 
#
# Copyright: 2005 (C) Shaun ASHBY
#
#--------------------------------------------------------------------
use Cwd;
use Getopt::Long ();
use File::Spec;

# Fixed parameters:
my $cvs = '/usr/bin/cvs';
my $projectroot='CMSSW';
my $cvsroot = ':kserver:cmscvs.cern.ch:/cvs_server/repositories/'.$projectroot;
my $releaseconf = $projectroot.'/Release.conf';
my $tagfile;
my $outdir;
my $rconftag;
my $rv;

# Somewhere to keep track of packages and tag versions:
my $versionfile='Versions';

# Opt support:
# - Use a specific tag for the Release.conf;
# - Choose the output directory;

# Getopt option variables:
my %opts;
my %options =
   ("file=s"          => sub { $opt{TAGFROMFILE}=1; $tagfile=File::Spec->rel2abs($_[1]) },
    "outdir=s"        => sub { $outdir=$_[1] },
    "tag=s"           => sub { $rconftag=$_[1] },
    "run"             => sub { $opt{RUN} = 1 },
    "query"           => sub { $opt{QUERY} = 1 },
    "help"            => sub { &usage(); exit(0)},
    "debug"           => sub { $opt{DUMP} = 1 }
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
   # Set where to put the processed files:
   $outdir ||= cwd()."/src";
   # Tag of Release.conf:
   $rconftag ||= 'HEAD';

   # Check to see if we're running a query. If so we can do it
   # and return without checking anything out:
   if ($opt{QUERY})
      {
      if ($opt{TAGFROMFILE})
	 {
	 &queryfromtagfile();
	 exit(0);
	 }
      else
	 {
	 &query();
	 exit(0);
	 }
      }
   # Only run if the run option is given:
   elsif ($opt{RUN})
      {
      print "Checking out sources for version ",$rconftag," of Release.conf","\n";
      # Create the output directory if it doesn't already exist:
      if (! -d $outdir)
	 {
	 system("mkdir",$outdir);
	 }
      
      # Move to the output directory:
      chdir $outdir;
      if ($opt{TAGFROMFILE})
	 {
	 &checkoutfromtagfile();
	 }
      else
	 {
	 &checkout();
	 }
      }
   elsif ($opt{DUMP})
      {
      # Print debug info only:
      &dumpinfo();     
      }
   else
      {
      &usage();
      exit(1);
      }
   }

sub queryfromtagfile()
   {
   print "Running query on tags listed in file ",$tagfile,"\n";
   print "\n";
   
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

sub query()
   {
   print "Running query on version ",$rconftag," of Release.conf","\n";
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

sub checkoutfromtagfile()
   {
   # Get the list of packages and tags from the file specified.
   print "Running checkout using tags from file $tagfile","\n";
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
   close CVSCOHTML;   
   }

sub dumpinfo()
   {   
   print "\n";
   print "Current parameters are:","\n";
   print "\n";
   print "-> CVSROOT = ",$cvsroot,"\n";
   if ($tagfile)
      {
      print "-> Tags read from file. TAGFILE = ",$tagfile,"\n";	    
      }
   else
      {
      print "-> Release.conf = ",$releaseconf,"\n";
      print "-> CVS tag = ",$rconftag,"\n";
      }
   print "-> Output dir = ",$outdir,"\n";
   print "\n";
   print "Use the \"-run\" option to actually do the checkout.","\n";   
   }

sub usage()
   {
   my $string="\nUsage: CheckoutProjectSources.pl [--help|-h] [--tag|-t <CVSTAG>] [--run|-r OR --query|-q]\n";
   $string.="\n";
   $string.="--file|-f <TAGFILE>   Use the file TAGFILE as the source of tags rather than CVS\n";
   $string.="--tag|-t <CVSTAG>     Check out version CVSTAG of Release.conf\n";
   $string.="--run|-r              Do the code checkout....\n";
   $string.="--query|-q            ..or just query tags on packages.\n";
   $string.="\n";
   $string.="If CVSTAG isn't given, HEAD is assumed.\n";
   print $string,"\n";
   }
