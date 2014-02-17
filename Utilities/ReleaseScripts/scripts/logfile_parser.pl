#!/usr/bin/env perl
#____________________________________________________________________ 
# File: LogFileParser.pl
#____________________________________________________________________ 
#  
# Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
# Update: 2005-11-16 11:45:09+0100
# Revision: $Id: logfile_parser.pl,v 1.3 2013/01/24 14:16:44 muzaffar Exp $ 
#
# Copyright: 2005 (C) Shaun ASHBY
#
#--------------------------------------------------------------------

######## Needed to find Template Toolkit. Can be removed if PERL5LIB defined instead #######
BEGIN {
    if (! exists($ENV{PERL5LIB})) {
	# Elementary check of architecture:
	use Config; $PERLVERSION=$Config{version};
	# Default lib path for slc4:
	$LIBPATH = "/afs/cern.ch/cms/sw/slc4_ia32_gcc345/external/p5-template-toolkit/2.14/lib/site_perl/5.8.5/i386-linux-thread-multi";
	if ($PERLVERSION eq '5.8.0') { # Dirty: could be different version of Perl
	    $LIBPATH = "/afs/cern.ch/cms/sw/slc3_ia32_gcc323/external/p5-template-toolkit/2.14/lib/site_perl/5.8.0/i386-linux-thread-multi";
	}
	# Check for 64-bit arch:
	if ($Config{archname} =~ /64/) {
	    $LIBPATH="/afs/cern.ch/cms/sw/slc4_amd64_gcc345/external/p5-template-toolkit/2.14/lib/site_perl/5.8.5/x86_64-linux-thread-multi";
	}
    }
}

use Cwd;
use File::Find;

# To find Template Toolkit:
use lib $LIBPATH;

use Getopt::Long ();
use Storable;
use Time::localtime; # To get the day index;

my $buildsummary={};
my ($mainlogfile,$logfile,$workdir,$cmsrelease,$projectversion,$templatedir,$outputdir);
my %opts;
my %options =
    ("mainlog=s"     => sub { $mainlogfile = $_[1] }, # The logfile where SCRAM errors (missing packages etc.) can be found
     "workdir=s"     => sub { $workdir = $_[1] },     # The working dir from where log files can be found
     "release=s"     => sub { $projectversion = $_[1] },
     "templatedir=s" => sub { $templatedir = $_[1] }, # Where to find the templates
     "outputdir=s"   => sub { $outputdir = $_[1] },   # Where to write the HTML logs/summary
     "debug"         => sub { $debug = 1 },           # Show debugging info
     "help"          => sub { &usage(); exit(0) } );  # Show help and exit

my $data={};
my $dataobjs={};

my $pkglist=[];
my $currentpackage;
my $lastpkglog;
my $packagedata={};

my $tmstruct = localtime;
my $daynumber = $tmstruct->wday; # Get day index;

my $mailalertdir;
my $missingpackage="";
my $problem_bfs={};
my $scram_warnings=0;

my $package_version;

# Handle argument opts:
Getopt::Long::config qw(default no_ignore_case require_order);

if (! Getopt::Long::GetOptions(\%opts, %options)) {
    &usage(); exit(1);
} else {
    # We must have a project release version:
    die "ERROR: You must give a project release version (e.g. --release=CMSSW_xyz).\n", unless ($projectversion);
    # We must have a working dir, an output dir and a template location:
    die "ERROR: You must give a working directory which points to the project log files (--workdir=/a/dir/path/tmp/<ARCH>)\n", unless ($workdir && -d $workdir);
    die "ERROR: You must give an output directory where package information can be written (--outputdir=/a/dir/path/XXXX).\n", unless ($outputdir);
    system("mkdir","-p",$outputdir), unless (-d $outputdir);
    $templatedir||=$ENV{LOGFILEPARSER_TEMPLATE_DIR};
    die "ERROR: Unable to find the templates for the log results (use --templatedir=/path/2/templates).\n", unless ($templatedir && -d $templatedir);
    
    # Debugging off by default:
    $debug||=0;
    $mainlogfile||="";
    
    # For mail alerts:
    $mailalertdir=$ENV{MAILALERT_DIR}||$outputdir."/nightly-alerts";
    system("mkdir","-p",$mailalertdir), unless (-d $mailalertdir);
    
    # Isolate the numerical day:
    $buildsummary->{CMSSW_RELEASE}="CMSSW_".$daynumber;
    $buildsummary->{CMSSW_VERSION}=$projectversion;
    $buildsummary->{CMSSW_BUILD_ARCH}=$ENV{SCRAM_ARCH} || 'slc3_ia32_gcc323';
    $buildsummary->{CMSSW_BUILD_HOST}=`hostname`;
    $buildsummary->{DAY_INDEX}=$daynumber;

    # Some counters to keep track of the number of each kind of errors. These statistics are
    # for the whole build:
    $buildsummary->{N_COMPILATION_ERRORS} = 0;
    $buildsummary->{N_LINK_ERRORS} = 0;
    $buildsummary->{N_OTHER_ERRORS} = 0;
    $buildsummary->{TOTAL_FAILED_PACKAGES} = 0;

    # Create a date string:
    my $datestring;
    chomp($datestring=`date +'%F'`);
    $buildsummary->{CMSSW_DATESTRING}=$datestring;
    # Get the list of packages from the TC:
    my $packagelistfromTC = &getpklistfromtc();
    
    # Traverse the source code tree looking for developer files. Use these files to 
    # generate a list of responsibles for each package. First, check for cached info.
    # This saves a long wait.
    # Global store for package Administrators, Developers and everyone ('all'):
    $responsibles={};
    my $cachefilename = "./peoplecache.db";
    
    if (-f $cachefilename) {
	# Retrieve the info from cache:
	print "logfile_parser: Retrieving info on developers/admins cached in $cachefilename.\n";
	$responsibles = retrieve($cachefilename);
    } else {
	# Populate a new cache file:
	use vars qw/*name *dir *prune/;
	
	*name   = *File::Find::name;
	*dir    = *File::Find::dir;
	*prune  = *File::Find::prune;
	
	$|=1;
	print "logfile_parser: Traversing source tree under ".$workdir."/../../src\n";    
	# Read the list of files found under directory:
	File::Find::find({wanted => \&collect}, $workdir."/../../src");
	# Cache the contents of $responsibles:
	store($responsibles,$cachefilename);
	print "logfile_parser: Info on developers/admins cached in $cachefilename.\n";
	print "logfile_parser: Collected package admin/developer info.","\n";
    }
    
    # Loop over packages:
    foreach my $p (sort keys %$packagelistfromTC) {
	chomp($p);	
	$package_version = $packagelistfromTC->{$p};
	# Check to see if there's a logfile and if
	# so, process it. Also make sure file is not
	# zero size:
	$logfile=$workdir."/cache/log/src/$p/build.log";
	if (-f $logfile && -s $logfile) {
	    open(PLOG,"< $logfile") || die "$0: Unable to read package logfile $logfile!: $!","\n";
	    while(<PLOG>) {
		chomp;
		push(@$pkglist,$p);
		$currentpackage = $p;
		my $concatpackname = $p; $concatpackname =~ s/\///g;
		# A note on the status numbers:
		# 1 [Compilation Error], 2 [Link Error], 3 [Dictionary Error]
		# The last error wins except if status!=0, then don't set to 2
		# => status=1,3 wins over 2, which is what we want
		#
		# Delegate handling of per-package info to a dedicated object:
		if (! exists($dataobjs->{$currentpackage})) {
		    $dataobjs->{$currentpackage} = new PackageResults($currentpackage,$package_version,$responsibles->{$currentpackage});
		}
		
		# Look for error messages in the log file:
		# Linker errors. Report the missing library:
		if ($_ =~ m|/usr/bin/ld: cannot find -l(.*?)$|) {
		    $buildsummary->{N_LINK_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(2);
		    $dataobjs->{$currentpackage}->error(2,"LINK ERRORS (missing library \"".$1."\")");		    		
		    # Look for matches containing errors from gmake. Determine whether it was from
		    # a failed link command, or from a compilation error:
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/src/$concatpackname/.*?\.so|) {
		    $buildsummary->{N_LINK_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(2);
		    $dataobjs->{$currentpackage}->error(2,"LINK ERRORS for package library");
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/src/$concatpackname/.*?\.o|) {
		    $buildsummary->{N_COMPILATION_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(1);
		    $dataobjs->{$currentpackage}->error(1,"COMPILATION ERRORS for package");		    
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/bin/(.*?)/.*?\.o|) {
		    $buildsummary->{N_COMPILATION_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(1);
		    $dataobjs->{$currentpackage}->error(1,"COMPILATION ERRORS for executable $1 in bin");
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/bin/(.*?)/\1|) {
		    $buildsummary->{N_LINK_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(2);
		    $dataobjs->{$currentpackage}->error(2,"LINK ERRORS for executable $1 in bin");
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/bin/(.*?)/lib\1\.so|) {
		    $buildsummary->{N_LINK_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(2);
		    $dataobjs->{$currentpackage}->error(2,"LINK ERRORS for shared library $1 in bin");
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/test/stubs/lib(.*?)\.so|) {
		    $buildsummary->{N_LINK_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(2);
		    $dataobjs->{$currentpackage}->error(2,"LINK ERRORS for shared library $1 in test/stubs");
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/test/(.*?)/.*?\.so|) {
		    $buildsummary->{N_LINK_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(2);
		    $dataobjs->{$currentpackage}->error(2,"LINK ERRORS for shared library $1 in test");
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/test/stubs/.*?\.o|) {
		    $buildsummary->{N_COMPILATION_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(1);
		    $dataobjs->{$currentpackage}->error(1,"COMPILATION ERRORS for library in test/stubs");	    
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/test/(.*?)/.*?\.o|) {
		    $buildsummary->{N_COMPILATION_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(1);
		    $dataobjs->{$currentpackage}->error(1,"COMPILATION ERRORS for executable $1 in test");
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/test/(.*?)\.so|) {
		    $buildsummary->{N_LINK_ERRORS}++; 
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(2);
		    $dataobjs->{$currentpackage}->error(2,"LINK ERRORS for shared library $1 in test");
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/test/(.*?)\.o|) {
		    $buildsummary->{N_COMPILATION_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(1);
		    $dataobjs->{$currentpackage}->error(1,"COMPILATION ERRORS for executable $1 in test");		    
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/test/(.*?)/\1|) {
		    $buildsummary->{N_LINK_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(2);
		    $dataobjs->{$currentpackage}->error(2,"LINK ERRORS for executable $1 in test");
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/src/$concatpackname/classes_rflx\.cpp|) {
		    $buildsummary->{N_OTHER_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(3);
		    $dataobjs->{$currentpackage}->error(3,"DICTIONARY GEN ERRORS in package");
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/plugins/(.*?)/.*?\.o|) {
		    $buildsummary->{N_COMPILATION_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(1);
		    $dataobjs->{$currentpackage}->error(1,"COMPILATION ERRORS for SEAL PLUGIN $1 in plugins");
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/plugins/(.*?)/lib\1\.so|) {
		    $buildsummary->{N_LINK_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(2);
		    $dataobjs->{$currentpackage}->error(2,"LINK ERRORS for SEAL PLUGIN library $1 in plugins");
		} elsif ($_ =~ m|^gmake: \*\*\* .*?/src/$currentpackage/test/data/download\.url|) {
		    $buildsummary->{N_OTHER_ERRORS}++;
		    $dataobjs->{$currentpackage}->log($_);
		    $dataobjs->{$currentpackage}->status(3);
		    $dataobjs->{$currentpackage}->error(3,"DATA FILE COPY ERROR for file in data/download.url in test");
		} elsif ($_ =~ m|^gmake: \*\*\* .*$|) {
		    # For a misc error line which failed to match any of the other rules, dump
		    # some output for the line that a new regexp is needed for:
		    print STDERR "logfile_parser.pl: No regexps matched the line \"".$_."\"\n";
		} else {
		    # Just keep the logged info:
		    $dataobjs->{$currentpackage}->log($_);
		}		
	    }
	    # Keep track of the number of packages with errors:
	    $buildsummary->{TOTAL_FAILED_PACKAGES}++, if ($dataobjs->{$currentpackage}->status() != 0);
	    close(PLOG);
	} else {
	    print STDERR "WARNING: Missing log file (or file is zero-length) for $p\n";
	}
    }

    # See if user supplied location of main build log:
    if (-f $mainlogfile) {
	# Open the main logfile to look for SCRAM warnings and cyclic deps:
	open(BUILDLOG,"< $mainlogfile") || die "$0: Unable to read main logfile $mainlogfile!: $!","\n";
	# Create temp store for main package info (cycles..):
	my $mainobj=new PackageResults('MAIN',$projectversion);
	my $cycles=0;

	while(<BUILDLOG>) {
	    chomp;
	    # Collect cyclic deps messages. These messages will result in a complete
	    # build failure:
	    if ($_ =~ m|^SCRAM buildsystem ERROR:   Cyclic dependency (.*?) <--------> (.*?)$|) {
		$cycles=1;
		$mainobj->status(4);
		$mainobj->error(4,"$1 <--> $2");
	    } else {
		# Also scan for WARNING: messages from SCRAM concerning packages without BuildFiles
		# (or non-existent packages):
		if ($_ =~ m|^WARNING: Unable to find package/tool called (.*?)$|) {
		    $scram_warnings=1;
		    $missingpackage=$1;
		}
		
		if ($_ =~ m|.*?in current project area \(declared at (.*?)\)$| && $missingpackage ne "") {
		    if (exists($problem_bfs->{$1})) {
			push(@{$problem_bfs->{$1}},$missingpackage);
		    }
		    else {
			$problem_bfs->{$1} = [ $missingpackage ];
		    }	      
		    $missingpackage="";
		}
	    }	    
	}
	
	close(BUILDLOG);
	# If there were cycles, keep the data object for MAIN:
	if ($cycles) {
	    $dataobjs->{'MAIN'} = $mainobj;
	}
	
	# Dump missing packages/problem BuildFile report:
	my $scramlog=$outputdir."/scram_warnings.log";
	open(SCRAM_WARNINGS,"> $scramlog") || die "Unable to open $scramlog for writing:$!\n";;
	
	if ($scram_warnings) {
	    # Write to a file somewhere:
	    my ($pk, $dep);	
	    print SCRAM_WARNINGS "\n\n";
	    while (($pk, $dep)= each %{$problem_bfs}) {
		print SCRAM_WARNINGS "\n-> Location $pk has incorrect dependencies (to be removed from $pk/BuildFile): \n\t",join(" ",@$dep),"\n";
		print SCRAM_WARNINGS "\n";
	    }
	}
	else {
	    print SCRAM_WARNINGS "\n No SCRAM BuildFile warnings for this build: congratulations!","\n";
	}
	
	close(SCRAM_WARNINGS);
    } else {
	print "logfile_parser: No main log given. Skipping scanning for SCRAM messages.\n";
    }
     
    # Write the logs for each package:
    foreach my $p (sort keys %$dataobjs) {
	# Get the package data object:
	my $pdata = $dataobjs->{$p};
	# Check to see if there was a log for MAIN. If so
	# it means a big failure:
	if ($p eq 'MAIN') {
	    # Check status, just to make sure:
	    if ($pdata->status() == 4) {
		# Jump out of the loop:
		last;
	    }
	} else {
	    my $packagehtmllogfile = $pdata->subsystem()."_".$pdata->packname().".log.html";
	    # Prepare the data for the summary page:
	    if ($pdata->status()) {
		print "Package $p had errors.","\n", if ($debug);
		# Call the subroutine to prepare an email to be sent to the admins:
		&write_alert_mail($pdata, $templatedir);
	    }
	    
	    # Create an HTML log page for this package:
	    &log2html($pdata, $packagehtmllogfile, $templatedir, $outputdir);
	}
    }
    
    # Dump the main summary page:
    $buildsummary->{summarydata} = $dataobjs;
    
    &dumpmainpage($buildsummary, $templatedir, $outputdir);    
    # Dump out a final goodbye and stats on N packages processed:
    my $npackages=scalar(keys %$responsibles);
    print "logfile_parser: Looked at ",$npackages," packages.","\n";
    print "\n";
}

#### Subroutines ####
sub dumpmainpage() {
    my ($builddata,$templatedir,$outputdir)=@_;
    my $summaryfiletmpl="buildsummary.html.tmpl";
    
    use Template;
    
    # Template toolkit parameters:
    my $template_config = {
	INCLUDE_PATH => $templatedir,
	EVAL_PERL    => 1 
	};
    
    # Prepare the data for the bootstrap file and requirements:
    my $template_engine = Template->new($template_config) || die $Template::ERROR, "\n";   
    $template_engine->process($summaryfiletmpl, $builddata, $outputdir."/index.html")
	|| die "Template error: ".$template_engine->error;   
}

sub log2html() {
    my ($pdata,$packagehtmllogfile,$templatedir,$outputdir)=@_;
    my $packagehtmllogtmpl="package.log.html.tmpl";
    my $tdata = { package_data => $pdata };
    
    use Template;
    
    # Template toolkit parameters:
    my $template_config = {
	INCLUDE_PATH => $templatedir,
	EVAL_PERL    => 1 
	};

    # Prepare the data for the bootstrap file and requirements:
    my $template_engine = Template->new($template_config) || die $Template::ERROR, "\n";   
    $template_engine->process($packagehtmllogtmpl, $tdata, $outputdir."/".$packagehtmllogfile)
	|| die "Template error: ".$template_engine->error;
}

sub write_alert_mail() {
    my ($pdata, $templatedir)=@_;
    # Prepare the data for the template. Use the project version (e.g. CMSSW_xxxx-xx-xx)
    # as the release, rather than CMSSW_x: 
    my $tdata = {
	CMSSW_RELEASE => $projectversion,
	PACKAGE_OBJ => $pdata
	};
    
    use Template;
    
    # Template toolkit parameters:
    my $template_config = {
	INCLUDE_PATH => $templatedir,
	EVAL_PERL    => 1 
	};
    
    # Loop over all admins for the package:   
    foreach my $admin (split(" ",$pdata->responsibles("administrators"))) {
	# Set the name of the admin as SENDTO in the mail stub (ready to be interpreted by 
	# the mailing script):
	$tdata->{THIS_ADMIN} = $admin;
	# Generate a unique file name for the current package/administrator combination. For 
	# the mailing step, the first line of the mail stub contains the "to:" address.   
	srand();
	$fileid=int(rand 99999999)+1;
	print "logfile_parser: Preparing mail alert (fileid=",$fileid,") for package ".$pdata->fullname(),"\n";
	my $mailfile = $mailalertdir."/".$fileid.".mail";   
	my $template_engine = Template->new($template_config) || die $Template::ERROR, "\n";      
	$template_engine->process("alert_mail_stub.tmpl", $tdata, $mailfile)
	    || die "Template error: ".$template_engine->error;      
    }
}

sub collect() {
    my $persontype;
    my $persondata={ 'administrators' => [], 'developers' => []};
    
    if (my ($packagename) = ($name =~ m|.*?/src/(.*?)/.admin/developers|)) {
	open(DEVELOPERS, "$name") || die "$name: $!","\n";
	while(<DEVELOPERS>) {
	    chomp;
	    # Ignore comment lines:
	    next if ($_ =~ /^\#/);
	    # Look for the type of person tag (Administrators or Developers):
	    if ($_ =~ m|>(.*?)$|) {	    
		$persontype=lc($1);
	    }

	    if ($_ =~ m|.*:.(.*?)@(.*)|) {
		my $address="$1\@$2";	   
		# Check to avoid duplication of admins (where a developer file contains the
		# same email address more than once):
		if (! grep($address eq $_, @{$persondata->{$persontype}})) {
		    push(@{$persondata->{$persontype}},$address);
		}
	    }
	}
      
	# Close the file:
	close(DEVELOPERS);
	
	# Set the value of the administrators and developers entries for this package:
	$responsibles->{$packagename}->{'all'}="";
	foreach my $pertype ('administrators', 'developers') {
	    # Also store all persons as a single string:
	    $responsibles->{$packagename}->{'all'}.=" ".join(" ",@{$persondata->{$pertype}});
	    $responsibles->{$packagename}->{$pertype}=join(" ",@{$persondata->{$pertype}});
	}
    }   
}

sub getpklistfromtc() {
    # Based on script by D.Lange.
    #
    # Subroutine to get a list of packages/tags for a given release:
    # Check the version of wget.
    # --no-check-certificate needed for 1.10 and above:
    my $wgetver = (`wget --version` =~ /^GNU Wget 1\.1.*?/);
    my $options = ""; $options = "--no-check-certificate", if ($wgetver == 1);
    my $gotpacks=0;
    
    open(CMSTCQUERY,"wget $options  -nv -o /dev/null -O- 'https://cmstags.cern.ch/tc/public/CreateTagList?release=$projectversion' |");
    
    my %tags;
    while ( <CMSTCQUERY> ) {
	if ( $_ =~ /td/) {
	    my @sp1=split(' ',$_,99);
	    my $pack=$sp1[2];
	    my $tag=$sp1[5];
	    $tags{$pack}=$tag;
	    $gotpacks++;
	}
    }
    
    close CMSTCQUERY;
    # Die if no tags found (i.e. release doesn't exist):
    die "$0: No packages found in release $projectversion. Perhaps $projectversion doesn't exist?\n" if ($gotpacks == 0);
    return \%tags;
}

sub usage() {
    my $hstring="Usage:\n\n$0 [-h] [-d] --release=<version>  --workdir=<workingdir>  --outputdir=<dir>  [-t <DIR>]\n";
    $hstring.="\n";
    $hstring.="--mainlog | -m              Location of main logfile, if you want to see SCRAM messages. Optional.\n";
    $hstring.="--workdir | -w              The project working directory where the log files can be found.\n";
    $hstring.="--release | -r              The project release version (e.g. CMSSW_xxxx-xx-xx).\n";
    $hstring.="--outputdir | -o DIR        Where to write the HTML log files and summary page.\n";
    $hstring.="--templatedir | -t DIR      Set location where templates reside if LOGFILEPARSER_TEMPLATE_DIR not set.\n";
    $hstring.="\n";
    $hstring.="--debug | -d                Debug mode ON (off by default).\n";
    $hstring.="--help | -h                 Show usage information.\n";
    $hstring.="\n";
    print $hstring,"\n";
}

package PackageResults;

sub new() {
    my $proto=shift;
    my $class=ref($proto) || $proto;
    my $self={};
    bless($self,$class);
    my ($fullname, $version,$responsibles)=@_;
    $self->{FULLNAME} = $fullname;
    $self->{VERSION} = $version;
    $self->{RESPONSIBLES} = $responsibles || {};
    $self->{LOG} = [];
    $self->{STATUS} = 0;
    # Get the subsystem and package names from the full name:
    my ($subsystem,$packname)=split("/",$fullname);
    $self->{SUBSYSTEM} = $subsystem;
    $self->{PACKNAME} = $packname;
    
    return $self;
}

sub fullname() {
    my $self=shift;
    @_ ? $self->{FULLNAME} = shift
	: $self->{FULLNAME};
}

sub packname() {
    my $self=shift;
    @_ ? $self->{PACKNAME} = shift
	: $self->{PACKNAME};
}

sub subsystem() {
    my $self=shift;
    @_ ? $self->{SUBSYSTEM} = shift
	: $self->{SUBSYSTEM};
}

sub version() {
    my $self=shift;
    @_ ? $self->{VERSION} = shift
	: $self->{VERSION};
}

sub status() {
    my $self=shift;
    my ($status)=shift;
    # Only change status to 2 (link errors) if
    # current status is 0 (otherwise, impossible
    # to distinguish between link errors only and
    # compilation errors + link errors in same pkg.).
    if ($status) { # status is >= 1
	if ($self->{STATUS} == 0 && $status >= 2) {
	    $self->{STATUS} = $status;
	} elsif ($self->{STATUS} > 0) {
	}  else {
	    $self->{STATUS} = 1;
	}
    } else {
	return $self->{STATUS};
    }
}

sub log() {
    my $self=shift;
    if (@_) {
	push(@{$self->{LOG}},$_[0]);
    } else {
	return join("\n",@{$self->{LOG}});
    }
}

sub responsibles() {
    my $self=shift;
    my ($type)=@_;
    if (exists($self->{RESPONSIBLES}->{$type})) {
	return $self->{RESPONSIBLES}->{$type};
    } else {
	return "";
    }
}

sub error() {
    my $self=shift;
    my ($type,$msg)=@_;

    if (exists($self->{ERRORS}->{$type})) {
	if (exists($self->{ERRORS}->{$type}->{$msg})) {
	    $self->{ERRORS}->{$type}->{$msg}++;
	} else {
	    $self->{ERRORS}->{$type}->{$msg} = 1;
	}
    } else {
	$self->{ERRORS}->{$type}->{$msg} = 1;
    }

}

sub nErrorsByType() {
    my $self=shift;
    my ($type)=@_;
    my $n_errors=0;

    if (exists($self->{ERRORS}->{$type})) {
	foreach my $err (keys %{$self->{ERRORS}->{$type}}) {
	    $n_errors += $self->{ERRORS}->{$type}->{$err};
	}
    }
    return $n_errors;
}

sub package_errors() {
    my $self=shift;
    my ($stat)=@_;

    if ($stat) {
	if (exists($self->{ERRORS}->{$stat})) {
	    return $self->{ERRORS}->{$stat};
	}
    } else {
	return $self->{ERRORS};
    }
}
