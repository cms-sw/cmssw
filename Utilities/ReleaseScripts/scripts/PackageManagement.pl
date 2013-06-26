#!/usr/bin/env perl
#____________________________________________________________________ 
# File: PackageManagement.pl
#____________________________________________________________________ 
#  
# Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
# (Tagcollector interface taken from CmsTCPackageList.pl (author D.Lange))
# Update: 2006-04-10 16:15:32+0200
# Revision: $Id: PackageManagement.pl,v 1.17 2013/01/24 14:16:44 muzaffar Exp $ 
#
# Copyright: 2006 (C) Shaun ASHBY
#
#--------------------------------------------------------------------
use Cwd;
use File::Spec;
use File::Temp qw/ tempfile /;
use Getopt::Long ();

# Fixed parameters:
my $cvs = '/usr/bin/cvs';
my $projectroot='CMSSW';
my $cvsroot = ':gserver:cmscvs.cern.ch:/cvs_server/repositories/'.$projectroot;
my $cvsrootAnon = ':pserver:anonymous@cmscvs.cern.ch:/cvs_server/repositories/'.$projectroot;
my $outdir;
my $rv;
my $releaseid;
my $baserel="none";
my $mypackagefile;
my $ignoredpackages;
my $wantedpackages;
my $versionfile;
my $packagelist;
my $startdir=cwd();
my $package_search_regexp = '*';
my $checkout="co";

# Support for colours in messages:
if ( -t STDIN && -t STDOUT && $^O !~ /MSWin32|cygwin/ )
   {
   $bold = "\033[1m";
   $normal = "\033[0m";
   $status = "\033[0;35;1m"; # Magenta
   $fail = "\033[0;31;1m";   # Red
   $pass = "\033[0;32;1m";   # Green
   $ignored = "\033[0;33;1m"; # Yellow
   $good = $bold.$pass;      # Status messages ([OK])
   $error = $bold.$fail;     #                 ([ERROR])
   }

# Getopt option variables:
my %opts; $opts{VERBOSE} = 1; $opts{USE_REGEXP}=0; # Verbose by default; Use package list as supplied by user (or all by default);
my %options =
   ("release=s"       => sub { $releaseid=$_[1] },
    "baserel=s"       => sub { $baserel=$_[1] },
    "mypackagefile=s" => sub { $opts{MYPACKAGES} = 1; $mypackagefile=$_[1] },
    "outdir=s"        => sub { $outdir=$_[1] },
    "query"           => sub { $opts{QUERY} = 1},
    "ignorepack=s"    => sub { $opts{IGNOREPACK} = 1; $ignoredpackages = [ split(" ",$_[1]) ] },
    "pack=s"          => sub { $opts{PACKAGES} = 1; $wantedpackages = [ split(" ",$_[1]) ]; $opts{MYPACKAGES} = 0; },
    "search=s"        => sub { $opts{USE_REGEXP} = 1; $opts{PACKAGES} = 0; $package_search_regexp = $_[1]; },
    "justtag"         => sub { $opts{SHOWTAGONLY} = 1 },
    "dumptags"        => sub { $opts{DUMPTAGLIST} = 1 },
    "anoncvs"         => sub { $opts{ANONCVS} = 1; $cvsroot=$cvsrootAnon; }, 
    "silent"          => sub { $opts{VERBOSE} = 0 },
    "export"          => sub { $checkout = "export" },
    "ignore-checkout-errors" => sub { $opts {IGNORECHECKOUTERRORS} = 1 },
    "-k" => sub { $opts {IGNORECHECKOUTERRORS} = 1 },
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


   # Check that MYPACKAGES option set if --rel HEAD option given:
   if (($releaseid eq 'HEAD' || $releaseid eq 'head') && !$opts{MYPACKAGES})
      {
      die "PackageManagement: \"--rel HEAD\" must only be used with \"--mypackagefile=FILE\" for checking out listed packages without versioning.","\n";
      }
   
   # Somewhere to store checked-out tags:
   $versionfile="PackageVersions.".$releaseid;
   # Checkout to current dir unless overridden:
   $outdir||="src";   

   # Get the package list:
   $packagelist = &getpklistfromtc();

   if ( $baserel ne "none" ) {
       my $tid=$releaseid;
       $releaseid=$baserel;
       my $packagelistbase=&getpklistfromtc();
       $releaseid=$tid;

       foreach my $ipack (sort keys %$packagelistbase) {
	   if (exists($packagelist->{$ipack})) {
	       if ( $packagelist->{$ipack} eq $packagelistbase->{$ipack} ) {
		   delete $packagelist->{$ipack};
	       }
	   }
       }
   }


   
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
   if ($opts{MYPACKAGES})
      {
      $mypfile = File::Spec->rel2abs($mypackagefile);
      if ($opts{QUERY})
	 {
	 my $mypacklist=&getmypackages($mypfile);
	 &do_query($mypacklist);
	 }
      else
	 {
	 print "PackageManagement: Checking out packages listed in $mypackagefile.","\n\n", if ($opts{VERBOSE});
	 if ($releaseid eq 'HEAD' || $releaseid eq 'head')
	    {
	    my $mypacklist=&getmypackages($mypfile,1);
	    # If the release is HEAD, check out the required packages from the HEAD
	    # rather than using the version info from the TC:
	    &do_checkout($mypacklist,1);
	    }
	 else
	    {
	    my $mypacklist=&getmypackages($mypfile);
	    # Do normal checkout of packages with specific versions:
	    &do_checkout($mypacklist);
	    }
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
	 else
	    {
	    die "PackageManagement: Unknown package \"".$wpk."\"","\n";
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
   elsif ($opts{USE_REGEXP})
      {
      my $wantedpks={};
      my $matched=0;
      print "PackageManagement: Checking out packages matching regexp \"$package_search_regexp\".","\n\n", if ($opts{VERBOSE});
      # Use a regular expression to build the list of wanted packages:      
      foreach my $list_package (sort keys %$packagelist)
	 {
	 if ($list_package =~ m|$package_search_regexp|)
	    {
	    $matched++;
	    $wantedpks->{$list_package} = $packagelist->{$list_package};
	    }
	 }

      # Check to see if at least one match was found:
      if ($matched)
	 {
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
	 print "PackageManagement: No packages found matching regexp \"$package_search_regexp\".","\n\n";
	 }
      }
   else
      {
      if ($opts{QUERY})
	 {
	 # A query: dump package/tag list:

	     if ($baserel ne "none") {
		 print "PackageManagement: Querying list of packages in the tag collector for release $releaseid different from $baserel.","\n", if ($opts{VERBOSE});
	     }
	     else{
		 print "PackageManagement: Querying full list of packages in the tag collector for release $releaseid.","\n", if ($opts{VERBOSE});
	     }
	     print "\n";
	 &do_query($packagelist);
	 }
      else
	 {
	 # Skip config and SCRAMToolbox packages:
	 delete $packagelist->{'config'}; delete $packagelist->{'SCRAMToolbox'};
	 # Run the checkout:
	     if ($baserel ne "none") {
		 print "PackageManagement: Checking out list of packages in the tag collector for release $releaseid different from $baserel.","\n", if ($opts{VERBOSE});
	     }
	     else{
		 print "PackageManagement: Checking out full list of packages in the tag collector for release $releaseid.","\n", if ($opts{VERBOSE});
	     }
	 print "\n";
	 &do_checkout($packagelist);	 
	 }
      }
   }

#### subroutines ####
sub getmypackages()
   {
   my ($mypackagefile,$noversions)=@_;
   my $packlist={};
   # Open the file and copy tag info for selected packages:
   open(MYPACKAGELIST,"$mypackagefile") || die "PackageManagement: Unable to read packages from $mypackagefile: $!","\n";
   while (<MYPACKAGELIST>)
      {
      chomp;
      # In case the input file was a tag file dumped using --dump, only take the first part (i.e. the package name)
      # and discard the tag:
      if (my ($p) = ($_ =~ /(.*?)\s+?V[0-9][0-9]-[0-9][0-9]-[0-9][0-9].*?$/))
	 {
	 if ($noversions)
	    {
	    # Don't check whether package exists or not:
	    $packlist->{$p}='HEAD';
	    }
	 else
	    {
	    if (exists($packagelist->{$p}))
	       {
	       $packlist->{$p} = $packagelist->{$p};
	       }
	    }
	 }
      else
	 {
	 if ($noversions)
	    {
	    # Don't check whether package exists or not:
	    $packlist->{$_}='HEAD';	    
	    }
	 else
	    {
	    if (exists($packagelist->{$_}))
	       {
	       $packlist->{$_} = $packagelist->{$_};
	       }
	    }
	 }
      }
   close(MYPACKAGELIST);
   return $packlist;
   }

sub do_checkout()
   {
   my ($packagelist,$noversions)=@_;
   die "PackageManagement: No packages to check out!","\n", unless (scalar (my $nkeys = keys %$packagelist) > 0);
   print "PackageManagement: Checking out packages from HEAD of CVS repo.","\n\n", if ($noversions && $opts{VERBOSE});
   my $statmsg="";

   # Create the output directory if it doesn't already exist:
   if (! -d $outdir)
      {
      system("mkdir",$outdir);
      }
   else
      {
      # The source dir already exists: each package directory will be checked to see
      # if it exists:
      $statmsg=" (up-to-date)";
      }
   
   # Move to the output directory:
   chdir $outdir;

   my %toCoList;
   foreach my $pkg (sort keys %$packagelist)
      {
      chomp($pkg);
      if (-d $pkg && -f $pkg."/CVS/Tag")
	 {
	 my $tagfile=$pkg."/CVS/Tag";
	 open(TAGFILE,"$tagfile") || die "PackageManagement: Can't read CVS/Tag for package $pkg","\n";
	 chomp(my ($CVSTAG)=(<TAGFILE>));
	 close(TAGFILE);
	 # Strip any characters from the start of the tag (before "V"):
	 $CVSTAG =~ s/^[A-Z](V.*?)/$1/g;
	 # Check to see if the tags match:
	 if ($packagelist->{$pkg} ne $CVSTAG)
	    {
	    $statmsg=" (updated)";
	    print "-> ".$status."Removing $pkg for a clean re-checkout:".$normal."\n",if ($opts{VERBOSE});
	    my $rv = system("rm","-rf",$pkg);
#	    $rv = system($cvs,"-Q","-d",$cvsroot,$checkout,"-P","-r",$packagelist->{$pkg}, $pkg);      
	    $toCoList{$packagelist->{$pkg}}.="$pkg ";
	    # Check the status of the checkout and report if a package tag doesn't exist:
	    }
	 }
      else
	 {
	 $statmsg="";
	 # A fresh area so do a complete checkout:
	 $toCoList{$packagelist->{$pkg}}.="$pkg ";
         }
  }
   my $cvspass = undef;
   my $cvspass_env="";
   if ($opts{ANONCVS})
       {
       $cvspass = File::Temp->new();
       print $cvspass "/1 :pserver:anonymous\@cmscvs.cern.ch:2401/cvs_server/repositories/CMSSW AA_:yZZ3e\n";
       $cvspass_env="CVS_PASSFILE=$cvspass"
       }
   print "$starting to co\n";
   foreach my $tag (keys %toCoList) {
       my $rv = system("$cvspass_env $cvs -Q -d $cvsroot $checkout -r $tag $toCoList{$tag}");
       if ($rv == 0)
       {
	   my @sp=split(' ',$toCoList{$tag});
	   foreach my $pkg (@sp) {
	       printf ("Package %-45s version %-13s checkout ".$good."SUCCESSFUL".$normal."\n",$pkg, $tag) if ($opts{VERBOSE});
	   }
       }
       elsif ($opts{IGNORECHECKOUTERRORS})
       {
    	   my @sp=split(' ',$toCoList{$tag});
    	   foreach my $pkg (@sp) {
    	       printf ("Package %-45s version %-13s checkout ".$ignored."ERRORS IGNORED".$normal."\n",$pkg, $tag) if ($opts{VERBOSE});
    	   }
       }
       else
       {
	   my @sp=split(' ',$toCoList{$tag});
	   foreach my $pkg (@sp) {
	       printf STDERR ("Package %-45s version %-13s checkout ".$error."FAILED".$normal."\n",$pkg, $tag);
	       printf STDERR "Checkout ERROR: tag ".$tag." for package $pkg is not correct!","\n";
	   }
	   print "\n";
	   exit(1);
       }	       	       	       	       
   }

   print "\nDone".$statmsg,"\n";
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
   print "\n";
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
   my $gotpacks=0;
   if ($wgetver)
      {
      open(CMSTCQUERY,"/usr/bin/wget $options  -nv -o /dev/null -O- 'https://cmstags.cern.ch/tc/public/CreateTagList?release=$releaseid' |");
      }
   else
      {
      open(CMSTCQUERY,"/usr/bin/curl -L -k --stderr /dev/null 'https://cmstags.cern.ch/tc/public/CreateTagList?release=$releaseid' |");
      }
   
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
   $string.="                            If the \"--mypackagefile=FILE\" option is used, the release version HEAD\n";
   $string.="                            is also available to check out listed packages without versioning.\n";
   $string.="\n";
   $string.="OPTIONS:\n";
   $string.="\n";
   $string.="--baserel=<RELEASE>         Check out only changes relative to release <RELEASE>\n";
   $string.="--outdir=<DIR>              Check out packages to directory <DIR>. Create it if it doesn't exist.\n";
   $string.="--dumptags                  Dump all tags for the release REL to a file called \"PackageVersion.<REL>\".\n";
   $string.="--mypackagefile=<FILENAME>  Read the list of packages to check out from FILENAME.\n";
   $string.="--ignorepack=<PACKAGES>     Ignore packages listed in space-separated string PACKAGES.\n";
   $string.="--pack=<PACKAGES>           Only consider the packages listed in space-separated string PACKAGES.\n";
   $string.="--search=<REGEXP>           Query/checkout packages matching the Perl regular expression REGEXP.\n";
   $string.="--justtag | -j              Print just the CVS tag for the package given in \"--pack X\" option.\n";
   $string.="--anoncvs                   Use the anonymous CVSROOT to check out packages\n";  
   $string.="--query | -q                Query package lists to see tags. Don't perform any checkouts.\n";
   $string.="--silent | -s               Be silent: don't print anything.\n";
   $string.="--export | -e               Do cvs expot instead of checkout.\n";
   $string.="--help | -h                 Show this help and exit.\n";
   $string.="\n";
   print $string,"\n";
   }
