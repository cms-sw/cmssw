#!/usr/bin/env perl
#______________  ______________________________________________________ 
# File: CreateCVSPackage.pl
#____________________________________________________________________ 
#  
# Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
# Update: 2006-04-28 09:50:38+0200
# Revision: $Id: CreateCVSPackage.pl,v 1.14 2010/02/24 15:51:33 andreasp Exp $ 
#
# Copyright: 2006 (C) Shaun ASHBY
#
#--------------------------------------------------------------------
use Cwd qw(cwd);
use Getopt::Long ();
use File::Basename;

# Fixed parameters:
my $cvs = '/usr/bin/cvs';
my $projectroot = 'CMSSW';
my $cvsroot = ':gserver:cmscvs.cern.ch:/cvs_server/repositories/'.$projectroot;

# Use CVSROOT to override:
if ($ENV{CVSROOT}) {
    $cvsroot = $ENV{CVSROOT};
}

# List of packages to create:
my $actionlist = new ActionList;
# Objects which will contain all the info for
# the developers/admins of the packages. By
# default, the principal admin is the first
# one to appear on the command line after the
# --admin option:
my $developerlist;
my $adminlist;
my ($packagelist,$developeridlist,$adminidlist);   
# Batch file mode:
my $batchfile;

my %opts; $opts{VERBOSE} = 0; # non-verbose by default;
$opts{DEBUG} = 0;             # Debugging off by default;
$opts{CLEAN} = 1;             # Remove the checked out directories by default;
$opts{BATCH} = $opts{USE_WGET} = $opts{TCQUERY} = 0;
# Normal operation is commandline rather than file or
# wget from DB as source of new packages/admin/developer info;
$opts{UPDATE_TC} = 0; # We require a second command to update the
                      # tag collector package list;

# Process options:
my %options = (
	       "packagename=s" => sub { $packagelist = [ split(" ",$_[1]) ]; },	       
	       "developers=s"  => sub { $developeridlist=&proc_ilist($_[1]); },
	       "admins=s"      => sub { $adminidlist=&proc_ilist($_[1]); },
	       "verbose"       => sub { $opts{VERBOSE} = 1; },
	       "debug"         => sub { $opts{DEBUG} = 1; },
	       "noclean"       => sub { $opts{CLEAN} = 0; },
	       "tcupdate"      => sub { $opts{UPDATE_TC} = 1; },
	       "usewget"       => sub { $opts{USE_WGET} = 1; },
	       "querytc"       => sub { $opts{TCQUERY} = 1; },
	       "batch=s"       => sub { $opts{BATCH} = 1; $batchfile=$_[1]; },
	       "help"          => sub { &usage(); exit(0) }
	       );

# Get the options using Getopt:
Getopt::Long::config qw(default no_ignore_case require_order);

if (! Getopt::Long::GetOptions(\%opts, %options)) {
    print STDERR "$0: Error with arguments.","\n";
} else {
    # Check for batch mode. If batch is active, no need to check for
    # the info from cmdline:
    if ($opts{BATCH} || $opts{USE_WGET}) {
	if ($opts{BATCH}) {
	    print "Running in batch mode: reading instructions from $batchfile.\n";
	    # Check that the file exists:
	    die basename($0).": Unable to read $batchfile, $!","\n", unless (-f $batchfile);
	    open(SOURCE,"< $batchfile") || die basename($0).": Unable to open $batchfile, $!","\n";
	} else {
	    # We use a wget request to the database (via a CGI script):
	    print "Running in batch mode: getting approved packages from the TagCollector.\n";
	    open(SOURCE,"wget --no-check-certificate -o /dev/null -nv -O- 'https://cmstags.cern.ch/cgi-bin/CmsTC/GetPacksForCVS?noupdate=1' |")
		|| die "Can't run wget request: $!","\n";
	}

	while (<SOURCE>) {
	    # Get rid of spaces at the end of the line:
	    $_ =~ s/(.*?)\s*?/$1/g;
	    my ($pack,$adminidlist,$developeridlist) = ($_ =~ /^(.*?)\s*?adm.*?:(.*?)\s*?devel.*?:(.*?)$/);
	    # Create new container for developer list:
	    my $adminlist = new AdminList(&proc_ilist($adminidlist));
	    my $developerlist = new DeveloperList(&proc_ilist($developeridlist));
	    # Add the infos to the action list:
	    $actionlist->additem(new ActionItem($pack, $developerlist, $adminlist));	    
	}
	close(SOURCE);
    } elsif ($opts{UPDATE_TC}) {
	# Reset the TC:
	print "Updating approved package list in the TagCollector.\n";
	print "Packages are assumed to have been created successfully!\n";
	my $rv = system("wget","--no-check-certificate","-o","/dev/null","-nv","-O-",'https://cmstags.cern.ch/cgi-bin/CmsTC/GetPacksForCVS');
	if ($rv != 0) {
	    print "There were ERRORS when updating the tag collector!","\n";
	}
	exit($rv);
    } elsif ($opts{TCQUERY}) {
	# Query th tag collector and dump the package list:
	open(WGET,"wget --no-check-certificate -o /dev/null -nv -O- 'https://cmstags.cern.ch/cgi-bin/CmsTC/GetPacksForCVS?noupdate=1' |") || die "Can't run wget request: $!","\n";
	my $n = 0;
	while(<WGET>) {
	    chomp;
	    print "TC_REQUEST QUERY: ",$_,"\n";
	    $n++;
	}
	print $n." package requests queued.\n";
	close(WGET);
	print "Done\n";
	exit(0);
    } else {
	# Build the list of developers and admins. This block will be active when
	# the --package, --admin and --developers opts are used together.
	# We must have a package name:
	die basename($0),": No package(s) given!\n", unless ($#$packagelist > -1);
	# We must have developer names as a list:
	die basename($0),": No developers given!\n", unless ($#$developeridlist > -1);
	# We must have a package admin:
	die basename($0),": No package admins given!\n", unless ($#$adminidlist > -1);
	# Create new container for developer list:
	my $adminlist = new AdminList($adminidlist);
	my $developerlist = new DeveloperList($developeridlist);
	foreach my $pack (@$packagelist) {
	    $actionlist->additem(new ActionItem($pack, $developerlist, $adminlist));    
	}
    }    
    # Run the actions:
    $actionlist->exec();
    print "\n";
}

#### Global functions (in package main) ####
sub proc_ilist() {
    my ($istring)=@_;
    my $list;
    # If the list is comma-separated:
    if ($istring =~ /,/) {      
	$list=[ split(",",$istring) ];
    } else {
	$list=[ split(" ",$istring) ];
    }
    return $list;
}

sub usage() {
    my $name=basename($0);
    my $string="\nUsage: $name --package=<PACKAGE> --admin=<ADMIN> --developers=<DEVLIST>[-h] [-v]\n";
    $string.="\nor     $name --batch=<FILENAME> [-h] [-v]\n";
    $string.="\nor     $name --usewget | --tcupdate [-h] [-v]\n";
    $string.="\n";
    $string.="--packagename=<PACKAGE>       Name of package to be created (in full,i.e. <sub>/<package>).\n";   
    $string.="\n";
    $string.="--admin=<ADMIN>               The administrators for this package. ADMIN should be given as a\n";
    $string.="                              quoted list of administrator login IDs.\n";
    $string.="                              This list can be comma or space separated.\n";
    $string.="\n";
    $string.="--developers=<DEVLIST>        The list of people to be registered as developers for this package.\n";
    $string.="                              DEVLIST should be given as a quoted list of login IDs.\n";
    $string.="                              This list can be comma or space separated.\n";
    $string.="--batch=<FILENAME>            Read the package and admin/developer info from FILENAME.\n";
    $string.="                              The file format should be:\n";
    $string.="                              <PACKAGE> admins:A,B,C,D developers:A,B,C,D\n";
    $string.="--usewget                     Use wget to run a CGI script on the server to check for and return\n";
    $string.="                              the current list of approved packages, their developers/admins.\n";
    $string.="--querytc                     Query the tag collector for new packages queued for creation.\n";
    $string.="--tcupdate                    Update the Tag Collector status for the created packages.\n";
    $string.="\n";
    $string.="--noclean                     Don't remove the checked out directories from the working area.\n";
    $string.="\n"; 
    $string.="OPTIONS:\n";
    $string.="--verbose | -v                Be verbose.\n";
    $string.="--debug   | -d                Debug mode. Show the info on admins and developers. Dump the generated\n";
    $string.="                              developers file to STDOUT and exit. Doesn't modify any files.\n";
    $string.="--help    | -h                Show help and exit.\n";
    print $string,"\n";
}

#### Packages ####
package ActionList;

sub new() {
    my $proto = shift;
    my $class = ref($proto) || $proto;
    my $self = {};
    bless($self, $class);
    # Package list:
    $self->{ITEMS} = [];
    return $self;
}

sub additem() {
    my $self=shift;
    my ($item)=@_;
    push(@{$self->{ITEMS}},$item);
}

sub exec() {
    my $self=shift;
    foreach my $it (@{$self->{ITEMS}}) {
	print ">>> Package ".$it->package()->fullname()."\n"; 
 	# First, check out the package. This will tell us if the package already exists.
	&CVS::co($it->package()->fullname()."/.admin");
	if (-d $it->package()->fullname()."/.admin") {
	    print "Package ",$it->package()->fullname()," already exists. Going to update the developer file.","\n";
	    # Check out the developer file and update it"
	    &CVS::co($it->package()->fullname()."/.admin/developers");
	    my $developers = Developers->new($it);
	    $developers->write();
	} else {
	    # The package does not exist. Create it:
	    print "Creating ",$it->package()->fullname(),"\n";
	    # First, try checking out the .admin directory of the subsystem. 
	    print "\tGetting SubSystem ",$it->package->modulename(),"/.admin\n";
	    &CVS::co($it->package->modulename()."/.admin");
	    if (-d $it->package->modulename()."/.admin") {
		# We have a .admin directory. Now check for NewLeaf file, creating
		# if necessary:
		print "\tUpdating/creating NewLeaf file.","\n";
		my $newleaf = NewLeaf->new($it);
		$newleaf->write();
		# At this point the package should've been created so check out
		# and update the developer file:
		&CVS::co($it->package()->fullname()."/.admin/developers");
		my $developers = Developers->new($it);
		$developers->write();		
	    } else {
		# SubSystem must be created. Check out project admin directory:
		print "Creating ",$it->package->modulename(),"\n";
		print "\tGetting project admin directory","\n";		
		&CVS::co("admin");
		if (-d "./admin") {
		    if (-f "./admin/AddModule") {
			# Add new module:
			print "\tFound AddModule file. Updating it.","\n";
			my $newmodule = AddModule->new($it);
			$newmodule->write();
		    } else {
			# Create the AddModule file:
			print "\tNo AddModule file found. Creating it.","\n";
			my $newmodule = AddModule->new($it);
			$newmodule->write();
		    }
		} else {
		    print "ERROR: No admin directory found for this project!\n";
		    exit(1);
		}		
		# Now proceed. First check out the newly-created subsystem:
		print "\tGetting SubSystem ",$it->package->modulename(),"\n";
		&CVS::co($it->package->modulename());
		if (-d $it->package->modulename()) {
		    # SubSystem exists. Get the .admin directory:
		    print "\tGetting ",$it->package->modulename(),"/.admin","\n";
		    &CVS::co($it->package->modulename()."/.admin");
		    if (-d $it->package->modulename()."/.admin") {
			# We have a .admin directory. Now check for NewLeaf file, creating
			# if necessary:
			print "\tUpdating/creating NewLeaf file.","\n";
			my $newleaf = NewLeaf->new($it);
			$newleaf->write();
			# At this point the package should've been created so check out
			# and update the developer file:
			&CVS::co($it->package()->fullname()."/.admin/developers");
			my $developers = Developers->new($it);
			$developers->write();
		    }
		}
	    }
	}
	print "\n";
    }
    print "\n== Finished. ==\n";
}

package ActionItem;

sub new($;$;$) {
    my $proto = shift;
    my $class = ref($proto) || $proto;
    my $self = {};
    bless($self, $class);
    my ($packagename) = shift;
    $self->{PACKAGE} = new Package($packagename);
    $self->{DEVELOPERLIST} = shift;    
    $self->{ADMINLIST} = shift;
    return $self;
}

sub package() {
    my $self=shift;
    return $self->{PACKAGE};
}

sub developers() {
    my $self=shift;
    return $self->{DEVELOPERLIST};
}

sub admins() {
    my $self=shift;
    return $self->{ADMINLIST};
}

package DeveloperList;

sub new($) {
    my $proto = shift;
    my $class = ref($proto) || $proto;
    my $self = {};
    bless($self, $class);
    $self->{IDS} = [];
    my ($idlist)=@_;
    # Read through the lists of developers to get full info:
    foreach my $loginid (@$idlist) {
	# Use phonebook command to get the full info for this person:
	chomp(my ($pbinfo)=`phonebook -login $loginid -t login -t surname -t firstname -t email `);
	# Check to see if user exists:
	if ($pbinfo =~ /not found/) {
	    print "WARNING: Userid \"$loginid\" not found in CCDB. Ignoring this user.","\n";
	    next;
	}
	my ($ccid,$lastname, $firstname,$email) = split(";",$pbinfo);   
	$lastname = ucfirst(lc($lastname));
	push(@{$self->{IDS}},new ID($loginid,"$firstname $lastname",$email));
    }

    # Simple check for existence of at least 1 developer:
    die "ERROR: No developers defined for this package.\n", unless ($#{$self->{IDS}} > -1);
    return $self;
}

sub ids() {
    my $self=shift;
    return $self->{IDS};
}

package AdminList;

sub new($) {
    my $proto = shift;
    my $class = ref($proto) || $proto;
    my $self = {};
    bless($self, $class);
    $self->{IDS} = [];
    my ($idlist)=@_;
    # Read through the lists of developers to get full info:
    foreach my $loginid (@$idlist) {
	# Use phone command to get the full info for this person:
	chomp(my ($pbinfo)=`phonebook -login $loginid -t login -t surname -t firstname -t email `);
	# Check to see if user exists:
	if ($pbinfo =~ /not found/) {
	    print "WARNING: Userid \"$loginid\" not found in CCDB. Ignoring this user.","\n";
	    next;
	}
	my ($ccid,$lastname, $firstname,$email) = split(";",$pbinfo);   
	$lastname = ucfirst(lc($lastname));
	push(@{$self->{IDS}},new ID($loginid,"$firstname $lastname",$email));
    }

    # Simple check for existence of at least 1 admin:
    die "ERROR: No admin defined for this package.\n", unless ($#{$self->{IDS}} > -1);
    return $self;
}

sub ids() {
    my $self=shift;
    return $self->{IDS};
}

sub principal() {
    my $self=shift;
    return $self->{IDS}->[0];
}

package ID;

sub new($;$;$) {
    my $proto = shift;
    my $class = ref($proto) || $proto;
    my $self = {};
    bless($self, $class);
    my ($loginid,$fullname,$email)=@_;
    $self->{LOGINID} = $loginid;
    $self->{FULLNAME} = $fullname;
    $self->{EMAIL} = $email;
    return $self;
}

sub loginid() {
    my $self=shift;
    return $self->{LOGINID};
}

sub fullname() {
    my $self=shift;
    return $self->{FULLNAME};
}

sub email() {
    my $self=shift;
    return $self->{EMAIL};
}

sub idstring() {
    my $self=shift;
    return $self->{LOGINID}." : ".$self->{FULLNAME}." : ".$self->{EMAIL};
}

package Package;

sub new() {
    my $proto = shift;
    my $class = ref($proto) || $proto;
    my $self = {};
    bless($self, $class);
    my ($pname) = @_;
    $self->{FULLNAME} = $pname;
    my ($subsys, $pk) = split("/",$self->{FULLNAME});
    $self->{MODULENAME} = $subsys;
    $self->{LEAFNAME} = $pk;    
    return $self;
}

sub fullname() {
    my $self=shift;
    return $self->{FULLNAME};
}

sub modulename() {
    my $self=shift;
    return $self->{MODULENAME};
}

sub leafname() {
    my $self=shift;
    return $self->{LEAFNAME};
}

package AddModule;
use File::Basename;
use Cwd;

sub new() {
    my $proto = shift;
    my $class = ref($proto) || $proto;
    my $self = {};
    bless($self, $class);
    my ($actionitem) = @_;
    $self->{CURRENT_ITEM} = $actionitem;
    return $self;
}

sub write() {
    my $self=shift;
    my $sdir=cwd();   
    my $module="";
    print "Creating new module ",$self->{CURRENT_ITEM}->package()->modulename(),"\n";
    $module.="# This is a administration file\n";
    $module.="# Fill in the Fields below and commit to add a new module\n";
    $module.="# Each Module must have someone defined as being responsible for it.\n";
    $module.="# email address and login name should refer to this person\n";
    $module.="# The Module/Project must be a pre-defined module.\n";
    $module.="\n";
    $module.="Login Name             : ".$self->{CURRENT_ITEM}->admins->principal()->loginid."\n";
    $module.="Valid Email            : ".$self->{CURRENT_ITEM}->admins->principal()->email."\n";
    $module.="Add to Module/Project  : CMSSW\n";
    $module.="New Module Name        : ".$self->{CURRENT_ITEM}->package()->modulename()."\n";
    # Change to the checked-out admin dir:
    chdir "./admin";
    # Write the file:
    open(FILE,"> AddModule") || die basename($0).": Unable to open AddModule for writing: $!","\n";
    print FILE $module,"\n";
    close(FILE);
    # Add and commit the file (if file already exists, the add is ignored):
    &CVS::add("AddModule");
    &CVS::ci("AddModule");
    chdir $sdir;
}

package NewLeaf;
use File::Basename;
use Cwd;

sub new() {
    my $proto = shift;
    my $class = ref($proto) || $proto;
    my $self = {};
    bless($self, $class);
    my ($actionitem) = @_;
    $self->{CURRENT_ITEM} = $actionitem;
    return $self;
}

sub write() {
    my $self=shift;
    my $sdir=cwd();
    my $leaf="";
    print "Creating new leaf ",$self->{CURRENT_ITEM}->package()->leafname(),"\n";
    $leaf.="# This is a administration file\n";
    $leaf.="# Fill in the Fields below and commit to add a new leaf\n";
    $leaf.="# Each Leaf must have someone defined as being responsible for it.\n";
    $leaf.="# email address and login name should refer to this person.\n";
    $leaf.="\n";
    $leaf.="Login Name             : ".$self->{CURRENT_ITEM}->admins->principal()->loginid."\n";
    $leaf.="Valid Email            : ".$self->{CURRENT_ITEM}->admins->principal()->email."\n";
    $leaf.="New Leaf Name          : ".$self->{CURRENT_ITEM}->package()->leafname()."\n";
    # Change to the checked-out admin dir:
    chdir $self->{CURRENT_ITEM}->package()->modulename()."/.admin";
    # Write the file:
    open(FILE,"> NewLeaf") || die basename($0).": Unable to open NewLeaf for writing: $!","\n";
    print FILE $leaf,"\n";
    close(FILE);
    # Add and commit the file (if file already exists, the add is ignored):
    &CVS::add("NewLeaf");
    &CVS::ci("NewLeaf");
    chdir $sdir;
}

package Developers;
use File::Basename;
use Cwd;

sub new() {
    my $proto = shift;
    my $class = ref($proto) || $proto;
    my $self = {};
    bless($self, $class);
    my ($actionitem) = @_;
    $self->{CURRENT_ITEM} = $actionitem;
    return $self;
}

sub write() {
    my $self=shift;
    my $sdir=cwd();
    print "Creating new developer file for ",$self->{CURRENT_ITEM}->package()->fullname(),"\n";
    my $developerfile="# Names of Developers with write access to this module\n";
    $developerfile.="#\n";
    $developerfile.="# There are two types of developers:\n";
    $developerfile.="# 1) Administrators - entitled to edit all files in the module ,\n";
    $developerfile.="#    in particular the .admin directory.  (Including this file) \n";
    $developerfile.="# 2) Regular Developers - entitled to edit all files in the module\n";
    $developerfile.="#    except those in the .admin directory.\n";
    $developerfile.="#\n";
    $developerfile.="# Entries must have the following format:\n";
    $developerfile.="#\n";
    $developerfile.="#      [logname] : [Firstname Familyname] : [emailaddress]\n";
    $developerfile.="#\n";
    $developerfile.="#      where [logname] is the login name of the user (in lower case)\n";
    $developerfile.="#            [Firstname Familyname] is the fullname of the user in free format\n";
    $developerfile.="#            [emailaddress] any email address of the user\n";
    $developerfile.="#\n";
    $developerfile.="#      IMPORTANT: The only entry that uniqely identifies the user\n";
    $developerfile.="#                 is the [loginname]. The rest of the entries are\n";
    $developerfile.="#                 used for information and clarity purposes.\n";
    $developerfile.="#\n";
    $developerfile.="# You can find the information required to add a user, using the \"phonebook\"\n";
    $developerfile.="# command from any CERN machine. \"phonebook user --all\" will give you a list of his\n";
    $developerfile.="# accounts and lognames too.\n";
    $developerfile.="# A safe assumption is to look for his ZH account on AFS/LXPLUS\n";
    $developerfile.="# Please remember to use lower case for the logname.\n";
    $developerfile.="# In case of doubts, please contact cvsadmin.cern.ch\n";
    $developerfile.="#\n";
    $developerfile.="# Important\n";
    $developerfile.="# ---------\n";
    $developerfile.="# --- Put names of regular developers after the >Developers Tag\n";
    $developerfile.="# --- Put names of administrators after the >Administrators Tag\n";
    $developerfile.="#\n";
    $developerfile.="# NB: This file was automatically generated by CreateCVSPackage.pl.\n";
    $developerfile.="#\n";
    $developerfile.=">Developers\n";
    map {
	$developerfile.=$_->idstring()."\n";
    } @{$self->{CURRENT_ITEM}->developers()->ids()};
    $developerfile.="\n";
    $developerfile.=">Administrators\n";
    map {
	$developerfile.=$_->idstring()."\n";
    } @{$self->{CURRENT_ITEM}->admins()->ids()};
    # Change to the checked-out admin dir:
    chdir $self->{CURRENT_ITEM}->package()->fullname()."/.admin";
    # Write the file:
    open(FILE,"> developers") || die basename($0).": Unable to open developers file for writing: $!","\n";
    print FILE $developerfile,"\n";
    close(FILE);
    # Add and commit the file (if file already exists, the add is ignored):
    &CVS::add("developers");
    &CVS::ci("developers");
    chdir $sdir;
}

package CVS;

sub co() {
    my $rv=0;
    my ($item)=@_;
    die "CVS: Nothing to check out!","\n", unless ($item);
    my $cmd = "$cvs -Q -d $cvsroot co -P $item >/dev/null 2>&1";
    $rv = system($cmd);
    return $rv;
}

sub add() {
    my $rv=0;
    my ($item)=@_;
    die "CVS: Nothing to add!","\n", unless ($item);
    my $cmd = "$cvs -Q -d $cvsroot add $item >/dev/null 2>&1";
    $rv = system($cmd);
    return $rv;
}

sub ci() {
    my $rv=0;
    my ($item)=@_;
    my ($message) = "CreateCVSPackage: Check in of $item.";
    die "CVS: Nothing to check in!","\n", unless ($item);
    my $cmd = "$cvs -Q -d $cvsroot ci -m \"$message\" $item >/dev/null 2>&1";
    $rv = system($cmd);
    return $rv;   
}
