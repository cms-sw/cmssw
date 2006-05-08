#!/usr/bin/perl
#____________________________________________________________________ 
# File: CreateCVSPackage.pl
#____________________________________________________________________ 
#  
# Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
# Update: 2006-04-28 09:50:38+0200
# Revision: $Id: CreateCVSPackage.pl,v 1.1 2006/04/28 14:52:20 sashby Exp $ 
#
# Copyright: 2006 (C) Shaun ASHBY
#
#--------------------------------------------------------------------
use Cwd;
use Getopt::Long ();
use File::Basename;

# Fixed parameters:
my $cvs = '/usr/bin/cvs';
my $projectroot='cmssw';
my $cvsroot = ':kserver:cmscvs.cern.ch:/cvs_server/repositories/'.$projectroot;

# Use CVSROOT to override:
if ($ENV{CVSROOT})
   {
   $cvsroot = $ENV{CVSROOT};
   }

my ($subsystem,$packagename);
my ($adminrealname,$adminemail);
my $subsystemadmin;
my $developers={};
my $administrators=[];
my $principaladmin;
my $padmin;

my %opts; $opts{VERBOSE} = 0; # non-verbose by default;
$opts{DEBUG} = 0; # Debugging off by default;
# Developers is a list of "Firstname Lastname:email,.. ", comma separated.
# The principal admin is "FirstName Lastname:loginid:email" but a list of
# admins can be provided:
my %options = (
	       "packagename=s" => sub { $packagename = $_[1] },
	       "developers=s"  => sub { my $dnamelist=[ split(",",$_[1]) ];
					map
					   {
					   # The developer info is Firstname Lastname:email@bb.cc:
					   my ($firstname, $lastname, $email) = ($_ =~ /\s*?([a-zA-Z\-]*)\s*?([a-zA-Z\-]*):(.*?)$/);
					   # Map the real name of the developer (or admin) to email address. 1/0 for
					   # administrator too, or not:
					   $developers->{$firstname." ".$lastname} = [ $email, 0 ];
					   } @$dnamelist;					
					},
	       "admin=s"       => sub { my $adminnamelist=[ split(",",$_[1]) ];
					map
					   {
					   my ($firstname, $lastname, $loginid, $email) = ( $_ =~ /\s*?([a-zA-Z\-]*)\s*?([a-zA-Z\-]*):(.*?):(.*?)$/);   
					   # Add the admin to the stack (in fact, we only use an array so we can grab the first admin
					   # in the list. All admins are added to the list of developers:
					   push(@$administrators, [ $firstname." ".$lastname, $loginid, $email ]);
					   } @$adminnamelist;
					},
	       "verbose"       => sub { $opts{VERBOSE} = 1; },
	       "debug"         => sub { $opts{DEBUG} = 1; },
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
   # We must have a package name:
   die basename($0),": No package name given!\n", unless ($packagename);
   # We must have developer names as a list:
   die basename($0),": No developers given!\n", unless ((scalar(keys %$developers)) > 0);
   # We must have a package admin:
   die basename($0),": No package admin given!\n", unless ($#$administrators > -1);
   # Form the admin name and email and real name from the input data. Take
   # the first admin on the first pass (to create the package) then add the
   # other admins when updating the list of developers:
   $principaladmin = $administrators->[0]; # Take the first admin
   $adminrealname = $principaladmin->[0];
   $padmin = $principaladmin->[1]; # Login id of the admin
   $adminemail = $principaladmin->[2];
   # Also add the package admins to the developers data:
   foreach my $adm (@$administrators)
      {
      $developers->{$adm->[0]} = [ $adm->[2], 1 ];
      }
   
   if ($opts{DEBUG})
      {
      print "Principal admin is \"$adminrealname ($padmin, $adminemail)\"","\n";
      print "Full list of developers and admins:\n";
      map
	 {
	 printf("%-20s %-10s %-1d\n",$_,$developers->{$_}->[0],$developers->{$_}->[1]);
	 } keys %$developers;
      print "\n";
      print &generate_developers($developers);
      print "\n";
      exit(0);
      }
   
   # Get subsystem and package parts:
   ($subsystem,$packagename) = split("/",$packagename);
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
      &CreateNewLeaf($padmin, $adminemail, $packagename);
      $nlmsg=" (and NewLeaf file created).";
      }
   else
      {
      # Just update the existing file:      
      print "The name of the package to create is $packagename. Here we go.","\n",if ($opts{VERBOSE});
      &UpdateNewLeaf($padmin, $adminemail, $packagename);
      }
   
   # Now commit the added/changed NewLeaf file:
   &commit($subsystemadmin,"Adding new package $packagename to NewLeaf$nlmsg");
   # Now check out the new package:
   my $newpack=$subsystem."/".$packagename;
   &get($newpack);   
   # Generate the developers file for the new package:
   &UpdateDevelopers($newpack,$developers);
   # Commit the changed developers file:
   &commit($newpack,"Updating developers file for $packagename.");
   print "Done\n";
   }

sub CreateNewLeaf()
   {
   my ($padmin,$adminemail,$packagename)=@_;
   print "Going to create entry for new package \"$packagename\" with admin\n",if ($opts{VERBOSE});
   print "(admin) \"$padmin\" in subsystem \"$subsystem\".\n",if ($opts{VERBOSE});
   open(NEWLEAF,"> $subsystemadmin/NewLeaf") || die basename($0).": Unable to open $subsystemadmin/Newleaf for writing: $!","\n";
   print NEWLEAF &generate_NewLeaf($padmin,$adminemail,$packagename),"\n";
   close(NEWLEAF);
   # Now add it to the repository:
   my $rv = system($cvs,"-Q","-d",$cvsroot,"add", "$subsystemadmin/NewLeaf");
   # Check the status of the add and report:
   if ($rv != 0)
      {
      die basename($0).": Unable to add $subsystemadmin/NewLeaf.","\n";
      }
   }

sub UpdateNewLeaf()
   {
   my ($padmin,$adminemail,$packagename)=@_;
   print "Going to create entry for new package \"$packagename\" with admin\n",if ($opts{VERBOSE});
   print "(admin) \"$padmin\" in subsystem \"$subsystem\".\n",if ($opts{VERBOSE});
   open(NEWLEAF,"> $subsystemadmin/NewLeaf") || die basename($0).": Unable to open $subsystemadmin/Newleaf for writing: $!","\n";
   print NEWLEAF &generate_NewLeaf($padmin,$adminemail,$packagename),"\n";
   close(NEWLEAF);
   }

sub UpdateDevelopers()
   {
   my ($newpack,$devhref)=@_;
   print "Going to create developers file for new package \"$newpack\".\n",if ($opts{VERBOSE});
   open(DEVELOPERS,"> $newpack/.admin/developers") || die basename($0).": Unable to open $newpack/.admin/developers for writing: $!","\n";
   print DEVELOPERS &generate_developers($devhref);
   close(DEVELOPERS);
   }

sub get()
   {
   my $rv=0;
   my ($item)=@_;

   # If we don;t have an arg, exit:
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

sub generate_NewLeaf()
   {
   my $newleaf="";
   my ($padmin,$email,$leafname)=@_;
   $newleaf.="# This is a administration file\n";
   $newleaf.="# Fill in the Fields below and commit to add a new leaf\n";
   $newleaf.="# Each Leaf must have someone defined as being responsible for it.\n";
   $newleaf.="# email address and username should refer to this person\n";
   $newleaf.="#\n";
   $newleaf.="\n";
   $newleaf.="Valid Username : $padmin\n";
   $newleaf.="Valid Email    : $email\n";
   $newleaf.="New Leaf Name  : $leafname\n";
   return $newleaf;
   }

sub generate_developers()
   {
   my ($devlist)=@_;
   my $developers="# Names of Developers with write access to this module\n";
   $developers.="# Names of Developers with write access to this module\n";
   $developers.="#\n";
   $developers.="# There are two types of developers:\n";
   $developers.="# 1) Administrators - entitled to edit all files in the module ,\n";
   $developers.="#    in particular the .admin directory.  (Including this file) \n";
   $developers.="# 2) Regular Developers - entitled to edit all files in the module\n";
   $developers.="#    except those in the .admin directory.\n";
   $developers.="#\n";
   $developers.="# You must use the full name of the developer as recorded on this system.\n";
   $developers.="# see :\n";
   $developers.="# http://cmsdoc.cern.ch/cmsoo/projects/swdevtools/developer_list.html\n";
   $developers.="# for a full list of names. If the developer you require is not on this\n";
   $developers.="# list then email the cvs administrator (cvsadmin@cmscvs.cern.ch)\n";
   $developers.="#\n";
   $developers.="# Important\n";
   $developers.="# ---------\n";
   $developers.="# --- Put names of regular developers after the >Developers Tag\n";
   $developers.="# --- Put names of administrators after the >Administrators Tag\n";
   $developers.="#\n";
   $developers.="# Mailists\n";
   $developers.="# --------\n";
   $developers.="# The bug reporting system can automatically send mail to all the developers\n";
   $developers.="# and administrators. Add the email address after the name (seperated by a :)\n";
   $developers.="# of developers to include in the list.\n";
   $developers.="#\n";
   $developers.=">Developers\n";
   map
      {
      $developers.="$_ : ".$devlist->{$_}->[0]."\n";
      } keys %$devlist;
   $developers.="\n";
   $developers.=">Administrators\n";
   map
      {
      $developers.="$_ : ".$devlist->{$_}->[0]."\n", if ($devlist->{$_}->[1]);
      } keys %$devlist;
   return $developers;
   }

sub usage()
   {
   my $name=basename($0);
   my $string="\nUsage: $name --package=<PACKAGE> --admin=<ADMIN> --developers=<DEVLIST> [-h] [-v]\n";
   $string.="\n";
   $string.="--packagename=<PACKAGE>       Name of package to be created (in full,i.e. <sub>/<package>).\n";   
   $string.="\n";
   $string.="--admin=<ADMIN>               The administrators for this package. ADMIN should be given as a\n";
   $string.="                              quoted comma-separated list of individual administrators\n";
   $string.="                              \"Firstname Lastname:LOGINID:Email,Firstname2 Lastname2:LOGINID2:Email2\".\n";
   $string.="\n";
   $string.="--developers=<DEVLIST>        The list of people to be registered as developers for this package.\n\n";
   $string.="                              DEVLIST is a quoted comma-separated list of individual developers:\n";
   $string.="                              \"Firstname1 Lastname1:Email1,Firstname2 Lastname2:email2\".\n";
   $string.="OPTIONS:\n";
   $string.="--verbose | -v                Be verbose.\n";
   $string.="--debug   | -d                Debug mode. Show the info on admins and developers. Dump the generated\n";
   $string.="                              developers file to STDOUT and exit. Doesn't modify any files.\n";
   $string.="--help    | -h                Show help and exit.\n";
   print $string,"\n";
   }
