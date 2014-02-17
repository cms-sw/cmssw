#!/usr/bin/env perl 
#____________________________________________________________________ 
# File: InstallCMSSWFromSource
#____________________________________________________________________ 
#  
# Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
# Update: 2006-04-26 17:51:03+0200
# Revision: $Id: InstallCMSSWFromSource.pl,v 1.6 2009/02/06 08:05:48 andreasp Exp $ 
#
# Copyright: 2006 (C) Shaun ASHBY
#
#--------------------------------------------------------------------
use Cwd;
use Getopt::Long ();
use Env;

$|=1;

# Fixed parameters:
my $cvs = '/usr/bin/cvs';
my $project = 'CMSSW';
# If CVSROOT is set, use it, otherwise use this default:
my $cvsroot=':gserver:cmscvs.cern.ch:/cvs_server/repositories/$project';
my $rv;

if ($ENV{CVSROOT})
   {
   $cvsroot = $ENV{CVSROOT};
   }

# Getopt option variables:
my %opts = ();
my ($releasever,$architecture,$releasearea,$is_release);
my $builddir;
my $is_release=0; # Build a prerelease by default 
my $scram="/afs/cern.ch/cms/utils/scramv1";
my $packman="/afs/cern.ch/cms/utils/PackageManagement.pl";
my $linkname="latest_prerelease";
my $linkarea;

# Boot with bootsrc (i.e. get the srcs) by default:
my $do_boot_src = 1;
my $symcheck = 0; # No sym checks by default
my $do_clean = 0; # Don't automatically remove the release ver if an area exists already
# The default architecture:
my $defarch="slc3_ia32_gcc323";


my %options =
   ("version=s"  => sub { $releasever = $_[1] },
    "builddir=s" => sub { $builddir = $_[1] },
    "release"    => sub { $is_release = 1; },
    "nosrc"      => sub { $do_boot_src = 0; },
    "clean"      => sub { $do_clean = 1; },
    "symcheck"   => sub { $symcheck = 1 },
    "arch=s"     => sub { $architecture = $_[1]; $ENV{SCRAM_ARCH} = $architecture; },
    "debug"      => sub { $opts{DEBUG} = 1; },
    "help"       => sub { &usage(); exit(0) }
    );

# Support for colours in messages:
if ( -t STDIN && -t STDOUT && $^O !~ /MSWin32|cygwin/ )
   {
   $bold = "\033[1m";
   $normal = "\033[0m";
   $mag  = "\033[0;35;1m";  # Magenta
   $blu  = "\033[0;32;1m";  # Blue
   $yell = "\033[0;34;1m";  # Yellow
   $fail = "\033[0;31;1m";  # Red
   $pass = "\033[0;32;1m";  # Green
   $good = $bold.$pass;     # Status messages ([OK])
   $error = $bold.$fail;    #                 ([ERROR])
   }

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
   $architecture ||= $defarch;
   die "$0: No release version. A version MUST be supplied.","\n", unless ($releasever);
   print "\n";
   print ">>> Going to build a ",($is_release) ? "release" : "pre-release",",version $releasever <<<<","\n"; 
   print "\n";
   print "Build architecture is $architecture.","\n",if ($opts{DEBUG});
   # Change to the release area:
   $releasearea = $builddir; # Use the user-supplied build dir
   $releasearea ||= "/afs/cern.ch/cms/Releases"; # or default to AFS area
   $linkarea = $releasearea."/CMSSW";
   $releasearea.= ($is_release) ? "/CMSSW" : "/CMSSW/prerelease";
   # Set the link name if we're building a release:
   $linkname = 'latest_release', if ($is_release);
   
   print "Release area (build dir) is $releasearea","\n";
   chdir $releasearea;
   print "Current working dir is ",cwd(),"\n",if ($opts{DEBUG});
   
   # Configure the area:
   my $rv = &configure();
   
   if ($rv == 0)
      {
      # Configured OK, so we can build:
      chomp(my $starttime = `date +%T-%F`);
      print "Starting build at $starttime.","\n";      
      my $bstat=0;
      $bstat=&build($symcheck);
      print "Making links in $releasearea","\n", if ($opts{DEBUG});
      &mklinkinRA();      
      chomp(my $endtime = `date +%T-%F`);
      printf (">>>>>> Build of %-15s finished at $endtime [ status %-1d ] <<<<<<<\n",$releasever,$bstat);
      print "\n";
      }
   else
      {
      die "$0: There was a problem configuring the area so cannot proceed with build.","\n";
      }
   }

sub build()
   {
   my ($symcheck)=@_;
   my $rv=0;
   
   if (!$symcheck)
      {
      print "Build: will not check symbols.\n",if ($opts{DEBUG});
      $ENV{SCRAM_NOSYMCHECK} = 'true';
      }
   
   chdir $releasearea."/".$releasever;
   my $btgt="release-build";
   $rv+=system($scram,"-arch",$architecture,"build","-v","-k",$btgt,">logs/$architecture/release-build.log","2>logs/$architecture/release-build-errors.log");
   die "$0: Error while building SCRAM project $releasever.","\n", if ($rv);
   $rv+=system("eval `$scram runt -sh`; SealPluginRefresh > logs/$architecture/SealPluginRefresh.log 2>&1");
   die "$0: Error when refreshing Plugin cache.","\n", if ($rv);
   # Only build the documentation if not donw already:
   if (! -f $releasearea."/".$releasever."/doc/html/index.html")
      {
      print ">>>> Building documentation for $releasever <<<<<<","\n";
      $rv+=system($scram,"-arch",$architecture,"build","-v","-k","doc",">logs/$architecture/docgen.log","2>logs/$architecture/docgen.log");
      die "$0: Error when generating documentation for $releasever.","\n", if ($rv);
      }
   $rv+=system($scram,"-arch",$architecture,"install");
   die "$0: Error when installing $releasever in the SCRAM database.","\n", if ($rv);
   # Clean up in the tmp dir:
   print "Cleaning up in /tmp: removing $releasearea/$releasever/tmp/$architecture/src","\n",if ($opts{DEBUG});
   $rv+=system("rm","-r","-f",$releasearea."/".$releasever."/tmp/".$architecture."/src");
   die "$0: Error when removing $releasearea/$releasever/tmp/$architecture/src.","\n", if ($rv);
   return $rv;
   }

sub configure()
   {
   my $rv=0;
   my $bootfile = ($do_boot_src) ? 'bootsrc' : 'boot';
   # Check to see if there's already a project area. If so, we must be configuring
   # for a second arch. Check that the user-supplied arch is different to the
   # default arch:
   if (-d $releasearea."/".$releasever && -d $releasearea."/".$releasever."/.SCRAM/$defarch" && $architecture ne $defarch)
      {      
      my $rv = &setup_other_arch();
      return $rv;
      }

   # Remove existing config dir:
   if (-d "config")
      {
      print "  ==> Config directory already exists....cleaning it.","\n",if ($opts{DEBUG});
      my $rv = system("rm","-rf",$releasearea."/config");
      die "$0: Unable to remove existing dir $releasearea/config","\n", if ($rv);
      }

   # Remove existing release version if asked to do so:
   if (-d $releasearea."/".$releasever && $do_clean)
      {
      print "  ==> $releasearea/$releasever already exists....ordered to clean it. I will.","\n",if ($opts{DEBUG});
      my $rv = system("rm","-rf",$releasearea."/".$releasever);
      die "$0: Unable to remove existing dir $releasearea/$releasever","\n", if ($rv);
      }
   
   # First, get the config contents:
   $rv += system($packman,"--rel","$releasever","--pack","config","-o",".");
   die "$0: Unable to check out config from CVS.","\n", if ($rv);
   print "Bootstrapping SCRAM area\n",if ($opts{DEBUG});
   print "Going to download sources too.\n",if ($do_boot_src && $opts{DEBUG});
   $rv += system($scram,"-arch",$architecture,"p","-b","config/$bootfile");
   die "$0: Unable to bootstrap SCRAM project.","\n", if ($rv);
   return $rv;
   }

sub setup_other_arch()
   {
   my $rv=0;
   print "Setting up for architecture $architecture","\n",if ($opts{DEBUG});
   chdir $releasearea."/".$releasever;
   # Run scram setup:
   $rv+=system($scram,"-arch",$architecture,"setup");
   die "$0: Unable to set up project for second arch $architecture.","\n", if ($rv);
   return $rv;
   }

sub mklinkinRA()
   {
   my ($from,$to);
   $from = $releasearea."/".$releasever;
   $to = $linkarea."/".$linkname; # Either "latest_prerelease" or "latest_release"
   my $rv = system("ln","-s","-f",$from,$to);
   die "$0: Unable to set up $linkname link in $releasearea.","\n", if ($rv);
   }

sub usage()
   {
   my $string.="\nUsage: InstallCMSSWFromSource.pl --version=<VERSION> [--arch=<ARCH>] [--release] [--builddir=<DIR>] [OPTIONS]\n";
   $string.="\n";
   $string.="--version=<VERSION>      The version of the project to build.\n";
   $string.="\nOPTIONS:\n";
   $string.="\n";
   $string.="--arch=<ARCH>            The architecture to build for. The default is $defarch. \n";
   $string.="--release                Signal that this release is a real release and not a pre-release (default).\n";
   $string.="--builddir=<DIR>         Where to build. Default is /afs/cern.ch/cms/Releases/CMSSW [/pre-release].\n";
   $string.="                         The subdirectory \"prerelease\" is added automatically if current release is a pre-release.\n";
   $string.="--nosrc                  Do not use the boot file which includes source code download.\n";
   $string.="--clean                  Remove the release area directory (and contents) if it already exists.\n";
   $string.="--symcheck               Turn on symbol checking in the build.\n";
   $string.="--debug| -d              Be more verbose.\n";
   $string.="--help | -h              Show help and return.\n";
   print $string,"\n";
   }

