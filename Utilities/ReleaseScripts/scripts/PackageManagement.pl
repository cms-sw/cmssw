#!/usr/bin/perl 
#____________________________________________________________________ 
# File: PackageManagement.pl
#____________________________________________________________________ 
#  
# Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
# (Tagcollector interface taken from CmsTCPackageList.pl (author D.Lange))
# Update: 2006-04-10 16:15:32+0200
# Revision: $Id$ 
#
# Copyright: 2006 (C) Shaun ASHBY
#
#--------------------------------------------------------------------
use Cwd;
use Getopt::Long ();

# Fixed parameters:
my $cvs = '/usr/bin/cvs';
my $projectroot='CMSSW';
my $cvsroot = ':kserver:cmscvs.cern.ch:/cvs_server/repositories/'.$projectroot;
my $outdir;
my $rv;
my $releaseid;
my $mypackagefile;
my $ignoredpackages;
my $wantedpackages;
my $versionfile;
my $packagelist;

# Getopt option variables:
my %opts;
my %options =
   ("release=s"       => sub { $releaseid=$_[1] },
    "mypackagefile=s" => sub { $opts{MYPACKAGES} = 1; $mypackagefile=$_[1] },
    "outdir=s"        => sub { $outdir=$_[1] },
    "query"           => sub { $opts{QUERY} = 1},
    "ignorepack=s"    => sub { $opts{IGNOREPACK} = 1; $ignoredpackages = [ split(" ",$_[1]) ] },
    "pack=s"          => sub { $opts{PACKAGES} = 1; $wantedpackages = [ split(" ",$_[1]) ]; $opts{MYPACKAGES} = 0; },
    "justtag"         => sub { $opts{SHOWTAGONLY} = 1 },
    "dumptags"        => sub { $opts{DUMPTAGLIST} = 1 },
    "verbose"         => sub { $opts{VERBOSE} = 1 },
    "help"            => sub { &usage(); exit(0)}
    );

# Get the options using Getopt:
Getopt::Long::config qw(default no_ignore_case require_order);

if (! Getopt::Long::GetOptions(\%opts, %options))
   {
   print STDERR "$0: Error with arguments.","\n";
   &usage();
   exit(1);
   }
else
   {
   # Check that we got a release name:
   die "PackageManagement: No release given! (--rel <RELEASE>)","\n",unless ($releaseid);
   # Check for conflicting options. If --justtag given, check for -q also
   # otherwise it makes no sense:
   if ($opts{SHOWTAGONLY} && !$opts{QUERY})
      {
      die "PackageManagement: \"--justtag\" only makes sense with \"--query\" and \"--pack X\" (i.e. one package on the cmd line).","\n";
      }

   # Somewhere to store checked-out tags:
   $versionfile="PackageVersions.".$releaseid;
   # Checkout to current dir unless overridden:
   $outdir||=cwd()."/src";   

   # Get the package list:
   $packagelist = &getpklistfromtc();

   # Dump the versions to a file if required:
   &dumptaglisttofile($versionfile), if ($opts{DUMPTAGLIST});

   # Look for packages to ignore and filter them out of the main package list:
   if ($opts{IGNOREPACK})
      {
      foreach my $ipack (@$ignoredpackages)
	 {
	 # Delete the ignored entries if they exist:
	 if (exists($packagelist->{$ipack}))
	    {
	    print "PackageManagement: Ignoring package \"",$ipack,"\"\n", if ($opts{VERBOSE});
	    delete $packagelist->{$ipack};
	    }
	 }
      }
   
   # Now see if we have a file containing the developers packages or whether the user
   # specified packages on the command line. In either case, make copies of the wanted tags:
   if ($opts{MYPACKAGES} && -f cwd()."/".$mypackagefile)
      {
      $mypackagefile = cwd()."/".$mypackagefile;
      my $mypacklist=&getmypackages($mypackagefile);
      if ($opts{QUERY})
	 {
	 &do_query($mypacklist);
	 }
      else
	 {
	 print "PackageManagement: Checking out packages listed in $mypackagefile.","\n", if ($opts{VERBOSE});
	 &do_checkout($mypacklist);
	 }
      }
   elsif ($opts{PACKAGES})
      {
      # Make a copy of just the wanted package info:
      my $wantedpks={};
      
      foreach my $wpk (@$wantedpackages)
	 {
	 if (exists($packagelist->{$wpk}))
	    {
	    $wantedpks->{$wpk} = $packagelist->{$wpk};
	    }
	 }
      
      if ($opts{QUERY})
	 {
	 &do_query($wantedpks);
	 }
      else
	 {
	 # Do the real checkout:
	 &do_checkout($wantedpks);
	 }
      }
   else
      {
      if ($opts{QUERY})
	 {
	 # A query: dump package/tag list:
	 print "PackageManagement: Querying full list of packages in the tag collector for release $releaseid.","\n", if ($opts{VERBOSE});
	 print "\n";
	 &do_query($packagelist);
	 }
      else
	 {
	 # Run the checkout:
	 print "PackageManagement: Checking out full list of packages in the tag collector for release $releaseid.","\n", if ($opts{VERBOSE});
	 print "\n";
	 &do_checkout($packagelist);	 
	 }
      }
   }

#### subroutines ####
sub getmypackages()
   {
   my ($mypackagefile)=@_;
   my $packlist={};
   # Open the file and copy tag info for selected packages:
   open(MYPACKAGELIST,"$mypackagefile") || die "PackageManagement: $!","\n";
   while (<MYPACKAGELIST>)
      {
      chomp;
      # In case the input file was a tag file dumped using --dump, only take the first part (i.e. the package name)
      # and discard the tag:
      if (my ($p) = ($_ =~ /(.*?)\s+?V[0-9][0-9]-[0-9][0-9]-[0-9][0-9].*?$/))
	 {
	 if (exists($packagelist->{$p}))
	    {
	    $packlist->{$p} = $packagelist->{$p};
	    }	 
	 }
      else
	 {
	 if (exists($packagelist->{$_}))
	    {
	    $packlist->{$_} = $packagelist->{$_};
	    }
	 }
      }
   close(MYPACKAGELIST);
   return $packlist;
   }

sub do_checkout()
   {
   my ($packagelist)=@_;
   die "PackageManagement: No packages to check out!","\n", unless (scalar (my $nkeys = keys %$packagelist) > 0);

   # Create the output directory if it doesn't already exist:
   if (! -d $outdir)
      {
      system("mkdir",$outdir);
      }
   
   # Move to the output directory:
   chdir $outdir;

   foreach my $pkg (sort keys %$packagelist)
      {
      chomp($pkg);
      # Check out the package:
      $rv = system($cvs,"-Q","-d",$cvsroot,"co","-P","-r",$packagelist->{$pkg}, $pkg);
      # Check the status of the checkout and report if a package tag doesn't exist:
      if ($rv == 0)
	 {
	 printf ("Package %-45s version %-10s checkout SUCCESSFUL\n",$pkg, $packagelist->{$pkg}), if ($opts{VERBOSE});
	 }
      else
	 {
	 printf STDERR ("Package %-45s version %-10s checkout FAILED\n",$pkg, $packagelist->{$pkg});
	 printf STDERR "Checkout ERROR: tag ".$packagelist->{$pkg}." for package $pkg is not correct!","\n";
	 print "\n";
	 exit(1);
	 }
      }
   }

sub do_query()
   {
   my ($packagelist)=@_;
   my ($npk)=scalar(keys %$packagelist);
   
   if ($opts{SHOWTAGONLY} && $npk > 1)
      {
      die "PackageManagement: \"--justtag\" only makes sense with \"--query\" and \"--pack X\" (i.e. one package on the cmd line).","\n";
      }
   else
      {
      map
	 {
	 if ($opts{SHOWTAGONLY})
	    {
	    printf ("%-10s\n",$packagelist->{$_});
	    }
	 else
	    {
	    printf ("%-45s %-10s\n",$_,$packagelist->{$_});
	    }
	 } sort keys %$packagelist;      
      }
   }

sub getpklistfromtc()
   {
   # Based on script by D.Lange.
   #
   # Subroutine to get a list of packages/tags for a given release:
   # Check the version of wget.
   # --no-check-certificate needed for 1.10 and above:
   my $wgetver = (`/usr/bin/wget --version` =~ /^GNU Wget 1\.1.*?/);
   my $options = ""; $options = "--no-check-certificate", if ($wgetver == 1);
   my $user="cmstcreader";
   my $pass="CmsTC";
   my $gotpacks=0;
   
   open(CMSTCQUERY,"/usr/bin/wget $options  -nv -o /dev/null -O- 'http://$user:$pass\@cmsdoc.cern.ch/swdev/CmsTC/cgi-bin/CreateTagList?release=$releaseid' |");
   
   my %tags;
   while ( <CMSTCQUERY> )
      {
      if ( $_ =~ /td/)
	 {
	 my @sp1=split(' ',$_,99);
	 my $pack=$sp1[2];
	 my $tag=$sp1[5];
	 $tags{$pack}=$tag;
	 $gotpacks++;
	 }
      }
   
   close CMSTCQUERY;
   # Die if no tags found (i.e. release doesn't exist):
   die "PackageManagement: No packages found in release $releaseid. Perhaps $releaseid doesn't exist?\n" if ($gotpacks == 0);
   return \%tags;
   }

sub dumptaglisttofile()
   {
   my ($versionfile)=@_;
   # Default dump of tags to a file:
   open (OUTFILE,">$versionfile") or die "PackageManagement: Cannot dump tag output file $filename.";   
   foreach my $pk (sort keys %$packagelist)
      {
      printf OUTFILE ("%-45s %-10s\n",$pk,$packagelist->{$pk});
      }
   close OUTFILE;
   }

sub usage()
   {
   my $string="\nUsage: PackageManagement.pl --release <REL> [--out <DIR>] [--dumptags] [OPTIONS]\n";
   $string.="\n";
   $string.="--release=<REL>             The release: either \"nightly\" or a release tag like \"CMSSW_x_y_z\".\n";
   $string.="\n";
   $string.="OPTIONS:\n";
   $string.="\n";
   $string.="--outdir=<DIR>              Check out packages to directory <DIR>. Create it if it doesn't exist.\n";
   $string.="--dumptags                  Dump all tags for the release REL to a file called \"PackageVersion.<REL>\".\n";
   $string.="--mypackagefile=<FILENAME>  Read the list of packages to check out from FILENAME.\n";
   $string.="--ignorepack=<PACKAGES>     Ignore packages listed in space-separated string PACKAGES.\n";
   $string.="--pack=<PACKAGES>           Only consider the packages listed in space-separated string PACKAGES.\n";
   $string.="--justtag | -j              Print just the CVS tag for the package given in \"--pack X\" option.\n";
   $string.="--query | -q                Query package lists to see tags. Don't perform any checkouts.\n";
   $string.="--verbose | -v              Be slightly verbose.\n";
   $string.="--help | -h                 Show this help and exit.\n";
   $string.="\n";
   print $string,"\n";
   }
