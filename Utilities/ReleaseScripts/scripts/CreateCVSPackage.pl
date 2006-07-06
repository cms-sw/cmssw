#!/usr/bin/perl -w
#____________________________________________________________________ 
# File: CreateCVSPackage.pl
#____________________________________________________________________ 
#  
# Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
# Update: 2006-04-28 09:50:38+0200
# Revision: $Id: CreateCVSPackage.pl,v 1.8 2006/06/20 12:37:38 sashby Exp $ 
#
# Copyright: 2006 (C) Shaun ASHBY
#
#--------------------------------------------------------------------
use Cwd;
use Getopt::Long ();
use File::Basename;
use File::Path;

# Fixed parameters:
my $cvs = '/usr/bin/cvs';
my $projectroot='CMSSW';
my $cvsroot = ':kserver:cmscvs.cern.ch:/cvs_server/repositories/'.$projectroot;

# Use CVSROOT to override:
if ($ENV{CVSROOT})
   {
   $cvsroot = $ENV{CVSROOT};
   }

my ($subsystem,$packagename,$fullpackagename);
my ($principaladmin);
my $principaladmins={};
my $subsystemadmin;
my $packagelist;

# Arrays of developers and admins:
my ($developeridlist,$adminidlist);
# Hash which will contain all the info for
# the developers/admins of the package. By
# default, the principal admin is the first
# one to appear on the command line after the
# --admin option:
my $developers={};
my $batchfile;

my %opts; $opts{VERBOSE} = 0; # non-verbose by default;
$opts{DEBUG} = 0; # Debugging off by default;
$opts{CLEAN} = 1; # Remove the checked out directories by default;
$opts{BATCH} = $opts{USE_WGET} = $opts{TCQUERY} = 0; # Normal operation is commandline rather than file or
                                                     # wget from DB as source of new packages/admin/developer info;
$opts{UPDATE_TC} = 0; # We require a second command to update the tag collector package list;

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

if (! Getopt::Long::GetOptions(\%opts, %options))
   {
   print STDERR "$0: Error with arguments.","\n";
   }
else
   {
   # Check for batch mode. If batch is active, no need to check for
   # the info from cmdline:
   if ($opts{BATCH})
      {
      print "Running in batch mode: reading instructions from $batchfile.\n";
      &set_admins_from_file();      
      }
   elsif ($opts{USE_WGET})
      {
      # We use a wget request to the database (via a CGI script):
      print "Running in batch mode: getting approved packages from the TagCollector.\n";
      &set_admins_from_wget_req();      
      }
   elsif ($opts{UPDATE_TC})
      {
      print "Updating approved package list in the TagCollector.\n";
      # Reset the TC:
      my $rv = &update_tc();
      exit($rv);
      }
   elsif ($opts{TCQUERY})
      {
      # Query th tag collector and dump the package list:
      &query_tc_for_new_packages();
      print "Done\n";
      exit(0);
      }
   else
      {
      # Build the list of developers and admins:
      &set_admins();
      }
   
   # Loop over the packages stored in the developers info hash:
   foreach $pk (keys %$developers)
      {
      $fullpackagename = $pk;
      $principaladmin=$principaladmins->{$fullpackagename};
      &runit();      
      }
   
   print "Done\n";
   }

sub runit()
   {
   # We must have a package name:
   die basename($0),": No package name given!\n", unless ($fullpackagename);
   # We must have developer names as a list:
   die basename($0),": No developers given!\n", unless ($#$developeridlist > -1);
   # We must have a package admin:
   die basename($0),": No package admins given!\n", unless ($#$adminidlist > -1);
   
   # Get subsystem and package parts:
   ($subsystem,$packagename) = split("/",$fullpackagename);
   
   if ($opts{DEBUG})
      {
      print "Principal admin for package $fullpackagename will be $principaladmin.","\n";
      print "\n";
      print "Full list of developers and admins:\n";
      map
	 {
	 printf("%-20s %-10s %-1d\n",$developers->{$fullpackagename}->{$_}->[0],$developers->{$fullpackagename}->{$_}->[1],$developers->{$fullpackagename}->{$_}->[2]);
	 } keys %{$developers->{$fullpackagename}};
      print "\n";
      
      print &generate_NewLeaf($packagename);
      print "\n";
      print &generate_developers();
      print "\n";
      exit(0);
      }
      
   # Check out the .admin directory of the subsystem where the
   # the package is to be created:
   die basename($0),": Subsystem $subsystem already checked out in current directory.","\n", if (-d $subsystem);
   # Proceed with checkout:
   print "Checking out .admin directory of $subsystem.","\n",if ($opts{VERBOSE});
   $subsystemadmin=$subsystem."/.admin";
   &get($subsystemadmin);
   
   # Check to make sure that there's a NewLeaf file. Otherwise create it:
   my $nlmsg=".";
   if (! -f $subsystemadmin."/NewLeaf")
      {
      print "No NewLeaf file found in subsystem $subsystem. Going to create one.\n",if ($opts{VERBOSE});
      # Create, add and commit the NewLeaf file:
      &CreateNewLeaf($packagename);
      $nlmsg=" (and NewLeaf file created).";
      }
   else
      {
      # Just update the existing file:      
      print "The name of the package to create is $packagename. Here we go.","\n",if ($opts{VERBOSE});
      &UpdateNewLeaf($packagename);
      }
   
   # Now commit the added/changed NewLeaf file:
   &commit($subsystemadmin,"Adding new package $packagename to NewLeaf$nlmsg");
   # Now check out the new package:
   my $newpack=$subsystem."/".$packagename;
   &get($newpack);
   # Generate the developers file for the new package:
   &UpdateDevelopers($newpack);
   # Commit the changed developers file:
   &commit($newpack,"Updating developers file for $packagename.");
   # Clean up (skip if --noclean active):
   if ($opts{CLEAN})
      {
      my $ndel = rmtree($subsystem); # Returns number of deleted dirs. 0 means failure;
      die basename($0).": Unable to clean up - error removing $subsystem!","\n", if (!$ndel);
      }   
   }

sub set_admins()
   {
   foreach my $package (@$packagelist)
      {
      $developers->{$package} = {};
      # Set the principal admin:
      $principaladmins->{$package} = $adminidlist->[0];
      # Read through the lists of developers to get full info:
      foreach my $loginid (@$developeridlist)
	 {
	 # Use phone command to get the full info for this person:
	 chomp(my ($pbinfo)=`phone -loginid $loginid -FULL`);
	 my ($ccid,$pbdata,$email) = split(";",$pbinfo);   
	 my ($lastname,$firstname,@rest) = split(" ",$pbdata);
	 $lastname = ucfirst(lc($lastname));
	 # Store in the developers hash:
	 $developers->{$package}->{$loginid} = [ "$firstname $lastname", $email, 0 ];
	 }
      
      # Do the same for admins:
      foreach my $loginid (@$adminidlist)
	 {
	 # Use phone command to get the full info for this person:
	 chomp(my ($pbinfo)=`phone -loginid $loginid -FULL`);
	 my ($ccid,$pbdata,$email) = split(";",$pbinfo);   
	 my ($lastname,$firstname,@rest) = split(" ",$pbdata);
	 $lastname = ucfirst(lc($lastname));
	 # Store in the developers hash:
	 $developers->{$package} ->{$loginid} = [ "$firstname $lastname", $email, 1 ];
	 }
      }
   }

sub query_tc_for_new_packages()
   {
   open(WGET,"wget --no-check-certificate -o /dev/null -nv -O- 'https://cmsdoc.cern.ch/swdev/CmsTC/cgi-bin/GetPacksForCVS?noupdate=1' |")
      || die "Can't run wget request: $!","\n";
   my $n = 0;
   while(<WGET>)
      {
      chomp;
      print "TC_REQUEST QUERY: ",$_,"\n";
      $n++;
      }
   print $n." package requests queued.\n";
   close(WGET);
   }

sub set_admins_from_wget_req()
   {
   my ($adminstring,$developerstring);

   open(WGET,"wget --no-check-certificate -o /dev/null -nv -O- 'https://cmsdoc.cern.ch/swdev/CmsTC/cgi-bin/GetPacksForCVS?noupdate=1' |")
      || die "Can't run wget request: $!","\n";
   
   while (<WGET>)
      {
      # Get rid of spaces at the end of the line:
      $_ =~ s/(.*?)\s*?/$1/g;
      # Format is <PACKAGE> admins:A,B,C,D developers:A,B,C,D
      ($packagename,$adminstring,$developerstring) = ($_ =~ /^(.*?)\s*?adm.*?:(.*?)\s*?devel.*?:(.*?)$/);
      $adminidlist=&proc_ilist($adminstring);
      $developeridlist=&proc_ilist($developerstring);
      $developers->{$packagename} = {};
      # Set the principal admin:
      $principaladmins->{$packagename} = $adminidlist->[0];

      # Read through the lists of developers to get full info:
      foreach my $loginid (@$developeridlist)
	 {
	 $loginid =~ s/\s*//g;
	 # Store in the developers hash:
	 $developers->{$packagename}->{$loginid} = [ @{&getFullNameAndEmail($loginid)}, 0 ];
	 }
      
      # Do the same for admins:
      foreach my $loginid (@$adminidlist)
	 {
	 $loginid =~ s/\s*//g;	        
	 # Store in the developers hash:
	 $developers->{$packagename}->{$loginid} = [ @{&getFullNameAndEmail($loginid)}, 1 ];
	 }     
      }
   
   close(WGET);
   }

sub update_tc()
   {
   print "\n";
   print "Updating tag collector using wget (packages are assumed to have been created successfully)\n";
   my $rv = system("wget","--no-check-certificate","-o","/dev/null","-nv","-O-",'https://cmsdoc.cern.ch/swdev/CmsTC/cgi-bin/GetPacksForCVS');
   if ($rv != 0)
      {
      print "There were ERRORS when updating the tag collector!","\n";
      }
   print "\n";
   return $rv;
   }

sub set_admins_from_file()
   {
   my ($adminstring,$developerstring);
   # Check that the file exists:
   die basename($0).": Unable to read $batchfile, $!","\n", unless (-f $batchfile);
   open(BATCH,"< $batchfile") || die basename($0).": Unable to open $batchfile, $!","\n";
   while (<BATCH>)
      {
      # Get rid of spaces at the end of the line:
      $_ =~ s/(.*?)\s*?/$1/g;
      # Format is <PACKAGE> admins:A,B,C,D developers:A,B,C,D
      ($packagename,$adminstring,$developerstring) = ($_ =~ /^(.*?)\s*?adm.*?:(.*?)\s*?devel.*?:(.*?)$/);
      $adminidlist=&proc_ilist($adminstring);
      $developeridlist=&proc_ilist($developerstring);
      $developers->{$packagename} = {};
      # Set the principal admin:
      $principaladmins->{$packagename} = $adminidlist->[0];

      # Read through the lists of developers to get full info:
      foreach my $loginid (@$developeridlist)
	 {
	 $loginid =~ s/\s*//g;
	 # Store in the developers hash:
	 $developers->{$packagename}->{$loginid} = [ @{&getFullNameAndEmail($loginid)}, 0 ];
	 }
      
      # Do the same for admins:
      foreach my $loginid (@$adminidlist)
	 {
	 $loginid =~ s/\s*//g;	        
	 # Store in the developers hash:
	 $developers->{$packagename}->{$loginid} = [ @{&getFullNameAndEmail($loginid)}, 1 ];
	 }     
      }
   
   close(BATCH);
   }

sub getFullNameAndEmail()
   {
   my ($loginid)=@_;

   open(PHONE,"phone -loginid $loginid -FULL |");
   chomp(my $info = (<PHONE>));
   
   my ($ccid,$pbdata,$email) = split(";",$info);
   my $namestring;
   my (@name_elements);
   
   # If there's a phone number, extract the name list from the string:
   if (($namestring) = ($pbdata =~ /^(.*?)\s*?[0-9]* [0-9].*$/))
      {
      (@name_elements)=split(" ",$namestring);
      }
   else
      {
      # Otherwise, just use the full string returned from the phone command:
      (@name_elements)=split(" ",$pbdata);
      }
   
   my $firstname=pop(@name_elements);
   my $lastname=join(" ",map { ucfirst(lc($_)) } splice(@name_elements, 0, $#name_elements+1));
   close(PHONE);
   return [ "$firstname $lastname", $email ];
   }

sub CreateNewLeaf()
   {
   my ($packagename)=@_;
   my $sdir=cwd();
   chdir $subsystemadmin;
   print "Going to create entry for new package \"$packagename\" with admin\n",if ($opts{VERBOSE});
   print "(admin) \"$principaladmin\" in subsystem \"$subsystem\".\n",if ($opts{VERBOSE});
   open(NEWLEAF,"> NewLeaf") || die basename($0).": Unable to open Newleaf for writing (dir is $subsystemadmin): $!","\n";
   print NEWLEAF &generate_NewLeaf($packagename),"\n";
   close(NEWLEAF);
   # Now add it to the repository:
   my $rv = system($cvs,"-Q","-d",$cvsroot,"add", "NewLeaf");
   chdir $sdir;
   # Check the status of the add and report:
   if ($rv != 0)
      {
      die basename($0).": Unable to add $subsystemadmin/NewLeaf.","\n";
      }
   }

sub UpdateNewLeaf()
   {
   my ($packagename)=@_;
   print "Going to create entry for new package \"$packagename\" with admin\n",if ($opts{VERBOSE});
   print "(admin) \"$principaladmin\" in subsystem \"$subsystem\".\n",if ($opts{VERBOSE});
   open(NEWLEAF,"> $subsystemadmin/NewLeaf") || die basename($0).": Unable to open $subsystemadmin/Newleaf for writing: $!","\n";
   print NEWLEAF &generate_NewLeaf($packagename),"\n";
   close(NEWLEAF);
   }

sub UpdateDevelopers()
   {
   my ($newpack)=@_;
   print "Going to create developers file for new package \"$newpack\".\n",if ($opts{VERBOSE});
   open(DEVELOPERS,"> $newpack/.admin/developers") || die basename($0).": Unable to open $newpack/.admin/developers for writing: $!","\n";
   print DEVELOPERS &generate_developers();
   close(DEVELOPERS);
   }

sub get()
   {
   my $rv=0;
   my ($item)=@_;
   # If we don't have an arg, exit:
   die basename($0).": Nothing to check out!","\n", unless ($item);
   # Check out item from HEAD:
   $rv = system($cvs,"-Q","-d",$cvsroot,"co","-P", $item);      
   # Check the status of the checkout and report if a package tag doesn't exist:
   if ($rv != 0)
      {
      die basename($0).": Unable to check out $item!","\n";
      }
   
   return $rv;
   }

sub commit()
   {
   my ($location,$message)=@_;
   my $rv=0;
   my $idir=cwd();
   $message = "\"$message\"";
   chdir $location;
   $rv = system($cvs,"-Q","-d",$cvsroot,"ci","-m", $message);      
   # Check the status of the checkout and report if there was a problem:
   if ($rv != 0)
      {
      # Only warn because sometimes CVSpm has been successful even though
      # it exits with stat 1:
      warn(basename($0).": There may have been problems with checkin. Check Karma!\n");
      }

   chdir $idir;
   return $rv;
   }

sub proc_ilist()
   {
   my ($istring)=@_;
   my $list;
   # If the list is comma-separated:
   if ($istring =~ /,/)
      {      
      $list=[ split(",",$istring) ];
      }
   else
      {
      $list=[ split(" ",$istring) ];
      }
   return $list;
   }

sub generate_NewLeaf()
   {
   my $newleaf="";
   my ($packagename)=@_;
   $newleaf.="# This is a administration file\n";
   $newleaf.="# Fill in the Fields below and commit to add a new leaf\n";
   $newleaf.="# Each Leaf must have someone defined as being responsible for it.\n";
   $newleaf.="# email address and username should refer to this person\n";
   $newleaf.="#\n";
   $newleaf.="\n";
   $newleaf.="Valid Username : ".$principaladmin."\n";
   $newleaf.="Valid Email    : ".$developers->{$fullpackagename}->{$principaladmin}->[1]."\n";
   $newleaf.="New Leaf Name  : ".$packagename."\n";
   return $newleaf;
   }

sub generate_developers()
   {
   my $developerfile="# Names of Developers with write access to this module\n";
   $developerfile.="#\n";
   $developerfile.="# There are two types of developers:\n";
   $developerfile.="# 1) Administrators - entitled to edit all files in the module ,\n";
   $developerfile.="#    in particular the .admin directory.  (Including this file) \n";
   $developerfile.="# 2) Regular Developers - entitled to edit all files in the module\n";
   $developerfile.="#    except those in the .admin directory.\n";
   $developerfile.="#\n";
   $developerfile.="# You must use the full name of the developer as recorded on this system.\n";
   $developerfile.="# see :\n";
   $developerfile.="# http://cmsdoc.cern.ch/cmsoo/projects/swdevtools/developer_list.html\n";
   $developerfile.="# for a full list of names. If the developer you require is not on this\n";
   $developerfile.="# list then email the cvs administrator (cvsadmin\@cmscvs.cern.ch)\n";
   $developerfile.="#\n";
   $developerfile.="# Important\n";
   $developerfile.="# ---------\n";
   $developerfile.="# --- Put names of regular developers after the >Developers Tag\n";
   $developerfile.="# --- Put names of administrators after the >Administrators Tag\n";
   $developerfile.="#\n";
   $developerfile.="# Mailists\n";
   $developerfile.="# --------\n";
   $developerfile.="# The bug reporting system can automatically send mail to all the developers\n";
   $developerfile.="# and administrators. Add the email address after the name (seperated by a :)\n";
   $developerfile.="# of developers to include in the list.\n";
   $developerfile.="#\n";
   $developerfile.=">Developers\n";
   map
      {
      $developerfile.="$developers->{$fullpackagename}->{$_}->[0] : ".$developers->{$fullpackagename}->{$_}->[1]."\n";
      } keys %{$developers->{$fullpackagename}};
   $developerfile.="\n";
   $developerfile.=">Administrators\n";
   map
      {
      $developerfile.="$developers->{$fullpackagename}->{$_}->[0] : ".$developers->{$fullpackagename}->{$_}->[1]."\n", if ($developers->{$fullpackagename}->{$_}->[2]);
      } keys %{$developers->{$fullpackagename}};
   return $developerfile;
   }

sub usage()
   {
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
