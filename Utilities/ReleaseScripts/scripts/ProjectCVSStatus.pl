#!/usr/bin/env perl
#____________________________________________________________________ 
# File: ProjectCVSStatus.pl
#____________________________________________________________________ 
#  
# Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
# Update: 2006-04-11 14:46:41+0200
# Revision: $Id: ProjectCVSStatus.pl,v 1.6 2013/01/24 14:16:44 muzaffar Exp $ 
#
# Copyright: 2006 (C) Shaun ASHBY
#
#--------------------------------------------------------------------
use Cwd;
use Getopt::Long ();
use File::Find;
use vars qw/*name *dir *prune/;

*name   = *File::Find::name;
*dir    = *File::Find::dir;
*prune  = *File::Find::prune;

$|=1;

# Fixed parameters:
my $cvs = '/usr/bin/cvs';
###### testing ############
my $project = 'CMSSW';
# If CVSROOT is set, use it, otherwise use this default:
my $cvsroot=':gserver:cmscvs.cern.ch:/cvs_server/repositories/CMSSW';
my $rv;

if ($ENV{CVSROOT})
   {
   $cvsroot = $ENV{CVSROOT};
   }

# Base release for reference tags:
my $basereleaseid;
my $packagename;
my ($sourcedir,$current_extra_path,$srcpath,$packpath,$cvsstartdir,$findstartdir);
# Project data hash:
my $projectstatusobj=new ProjectStatus($project);
# Data obtained from the reference release:
my $referencedata;
# Getopt option variables:
my %opts = (COMPARE => 0,
	    MODONLY => 0,
	    SUMMARY => 1,
	    DEBUG => 0,
	    COMPPACKAGENAME => 0,
	    LOCALSCAN => 0
	    );

my %options =
   ("compare=s"       => sub { $opts{COMPARE} = 1; $basereleaseid=$_[1] },
    "sourcetree=s"    => sub { $srcpath=$_[1] },
    "path=s"          => sub { $packpath = $_[1] },
    "packname=s"      => sub { $opts{COMPPACKAGENAME} = 1; $packagename = $_[1] },
    "here"            => sub { $opts{LOCALSCAN} = 1 },
    "modifiedonly"    => sub { $opts{MODONLY} = 1 },
    "full"            => sub { $projectstatusobj->summary(0); },
    "debug"           => sub { $opts{DEBUG} = 1; },
    "help"            => sub { &usage(); exit(0) }
    );

# Support for colours in messages:
if ( -t STDIN && -t STDOUT && $^O !~ /MSWin32|cygwin/ )
   {
   $bold = "\033[1m";
   $normal = "\033[0m";
   $mod  = "\033[0;35;1m";  # Magenta
   $uptd  = "\033[0;32;1m"; # Blue
   $publ = "\033[0;34;1m";  # Yellow
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
   # Where are the sources:
   ($sourcedir)  = (cwd() =~ m|(.*?/src).*?|);
   # The rest of the line. If there's content it will
   # mean that the scanning can start from this extra
   # path, if required:
   $current_extra_path=$';#'####
      
   # Try to find the source tree:
   if ($sourcedir eq "" && $srcpath ne "" && -d cwd()."/".$srcpath)
      {
      $sourcedir=cwd()."/".$srcpath; # Try under supplied srcpath first
      }
   
   # Find src dir using default:
   if ($sourcedir eq "" && -d cwd()."/src")
      {
      $sourcedir=cwd()."/src";   
      }

   print "Source dir is $sourcedir","\n", if ($opts{DEBUG});
   
   # Figure out where in the source tree to start scanning from.
   # Default is the source dir:
   $findstartdir = $sourcedir;
   
   if ($packpath ne "")
      {
      # We have a package path so check that it exists
      # under the source tree:
      if (-d $sourcedir."/".$packpath)
	 {
	 print "Scanning from $sourcedir/$packpath","\n", if ($opts{DEBUG});
	 $findstartdir = $sourcedir."/".$packpath;
	 }
      else
	 {
	 print "Scanning from $sourcedir","\n", if ($opts{DEBUG});      
	 }   
      }
   elsif ($packpath eq "" && $current_extra_path ne "" && $opts{LOCALSCAN})
      {
      print "Scanning from $sourcedir".$current_extra_path."\n", if ($opts{DEBUG});
      $findstartdir = $sourcedir.$current_extra_path;  
      }
   else
      {
      print "Scanning from $sourcedir","\n", if ($opts{DEBUG});
      }
   
   # The start directory from which CVS will be run (all packages are passed to
   # the cvs update as relative to this path):
   $cvsstartdir=$sourcedir;
   
   print "CVS will start from $cvsstartdir","\n", if ($opts{DEBUG});
   print "Find will search from $findstartdir","\n\n", if ($opts{DEBUG});
   
   # Default source tree is src in current dir:
   die "ProjectCVSStatus: No source tree directory found!\n", unless (-d $sourcedir);

   if ($opts{COMPARE})
      {
      die "ProjectCVSStatus: No reference release given for comparison! (--compare <RELEASE>)","\n", unless ($basereleaseid);
      # Get the tags for the reference:
      $referencedata=&getReferenceTags();
      # Read the list of files found under directory:
      File::Find::find({wanted => \&collecttreeinfo}, $findstartdir);
      $projectstatusobj->compare_to_reference($packagename); # Maybe dump info for single package            	 
      }
   else
      {
      # Read the list of files found under directory:
      File::Find::find({wanted => \&collecttreeinfo}, $findstartdir);
      $projectstatusobj->show_file_info();
      }   
   }

#### subroutines ####
sub collecttreeinfo()
   {
   # Skip anything containing CVS in the name:
   next if ($name =~ m|CVS|);
   if (my ($packagename) = ($name =~ m|^/.*?/src/(.*/.*)/.admin/developers$|))
      {
      &checkcvsstatus($packagename);
      }
   }

sub checkcvsstatus()
   {
   my ($packagename)=@_;
   my $existingtgs=0;
   my $filedata=[]; # Array of file objects for this package
   my $sdir=cwd();
   my $file_object;
   my $was_modified=0;
   my $needs_tag_p=0;
   
   # Change to the main CVS start directory (the source directory):
   chdir $cvsstartdir;

   my $cvsstat="$cvs -q -d $cvsroot status -v $packagename";
   open(CVSSTATUS, "$cvsstat |") || die "ProjectCVSStatus: Can't run \"cvs status\" process.","\n";
   
   while(<CVSSTATUS>)
      {
      chomp;
      # The start of a file block:
      if (my ($filename, $status) = ($_ =~ /^File: (.*?)\s+Status: (.*?)$/))
	 {
	 $file_object=new FileObject($filename,$status);
	 # Quick check to keep track of packages which had files which were modified:
	 $was_modified++, if ($file_object->modified());
	 }
      elsif (my ($working_revision) = ($_ =~ /\s+Working revision:\s+([0-9\.]+).*?$/))
	 {
	 # Record the working revision:
	 $file_object->workingrev($working_revision);
	 }
      elsif (my ($repo_revision, $filepath) = ($_ =~ m|\s+Repository revision:\s+([0-9\.]+)\s+.*?$project/$packagename/(.*?),v$|))
	 {
	 $file_object->reporev($repo_revision);
	 $file_object->fullname($filepath);
	 } 
      elsif (my ($cvstag, $cvsrevision) = ($_ =~ /\s+Sticky Tag:\s+(V.*?) \(revision: (.*?)\)$/))
	 {
	 $file_object->stickytag($cvstag,$cvsrevision);
	 }
      elsif ($_ =~ /\s+Sticky Tag:\s+\(none\)$/)
	 {
	 $file_object->stickytag(0);
	 }
      elsif ($_ =~ /\s+Existing Tags:$/)
	 {
	 $existingtgs=1;
	 }
      elsif (my ($etag,$erev) = ($_ =~ /\s+([a-zA-Z\-0-9_]*?)\s+\(revision:\s([0-9\.]+)\)$/))
	 {
	 if ($existingtgs)
	    {
	    # Keep a record of the relationship between the symbolic tag and the CVS revision:
	    $file_object->add_tag_to_history($etag,$erev);
	    }
	 }
      elsif (my ($etag,$erev) = ($_ =~ /\s+([a-zA-Z\-0-9_]*?)\s+\(branch:\s([0-9\.]+)\)$/))
	 {
	 if ($existingtgs)
	    {
	    $file_object->add_tag_to_history('BRANCH',$erev);
	    }
	 }      
      elsif ($_ =~ /$/ && $existingtgs)
	 {
	 # Reset things:
	 $existingtgs = 0;
	 # Record whether the package requires a tag to be published:
	 $needs_tag_p = $file_object->get_previous_tag(), if ($file_object->needs_tag_publish());
	 # Store the FileObject object in an array:
	 push(@$filedata,$file_object);
	 }
      else
	 {
	 next;
	 }
      }

   close(CVSSTATUS);
   # Write the array of file objects to the main (project) data object:
   $projectstatusobj->add_package($packagename, $was_modified, $needs_tag_p, $filedata);
   # Change back to the start dir:
   chdir $sdir;
   }

sub getReferenceTags()
   {
   # Based on script by D.Lange.
   #
   # Subroutine to get a list of packages/tags for a given release:
   # Check the version of wget.
   # --no-check-certificate needed for 1.10 and above:
   my $wgetver = (`wget --version` =~ /^GNU Wget 1\.1.*?/);
   my $options = ""; $options = "--no-check-certificate", if ($wgetver == 1);
   my $gotpacks=0;
   
   open(CMSTCQUERY,"wget $options  -nv -o /dev/null -O- 'https://cmstags.cern.ch/tc/public/CreateTagList?release=$basereleaseid' |");
   
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
   die "ProjectCVSStatus: No packages found in release $basereleaseid. Perhaps $basereleaseid doesn't exist?\n" if ($gotpacks == 0);
   return \%tags;
   }

sub usage()
   {
   my $string="\nUsage: ProjectCVSStatus.pl [--compare <REL>] [OPTIONS]\n";
   $string.="\n";
   $string.="--sourcetree=<DIR>        The directory containing the source packages. Default is \"src\".\n";
   $string.="--path=<PATH>             The path from the source dir to the location to scan, e.g. SUBSYSTEM or SUBSYSTEM/PACKAGE.\n";
   $string.="                          Note that if this option is not used, the entire source tree will be\n";
   $string.="                          scanned: this takes ~11 minutes.\n";
   $string.="\n";
   $string.="--here                    Scan everything under current directory only.\n";
   $string.="--compare=<REL>           The reference release for comparisons of tags: either \"nightly\" or a release tag\n";
   $string.="                          like \"CMSSW_x_y_z\".\n";
   $string.="--packname=<PACKAGE>      Show comparison of tags for PACKAGE (use only with --compare option).\n";
   $string.=" \n";
   $string.="OPTIONS:\n";
   $string.="\n";
   $string.="--modifiedonly            Show just the files which have been modified.\n";   
   $string.="--full                    Show full details (revisions, status) for each file scanned. Default is just to\n";
   $string.="                          print a summary.\n";
   $string.="--debug                   Be slightly verbose.\n";
   $string.="--help                    Show this help and exit.\n";
   $string.="\n";
   print $string,"\n";
   }

#### Package to handle the file status info ####
package FileObject;

sub new()
   {
   my $proto=shift;
   my $class=ref($proto) || $proto;
   my $self={};   
   bless($self,$class);
   my ($filename,$status)=@_;
   $self->shortname($filename);
   $self->{TAG_INDEX_MARKER} = 1; # For first occurrence
   if ($status =~ /Locally Modified/)
      {
      # File not up-to-date:
      $self->modified(1);
      }
   return $self;
   }

sub shortname()
   {
   my $self=shift;
   @_ ? $self->{SHORTNAME} = shift
      : $self->{SHORTNAME};
   }

sub fullname()
   {
   my $self=shift;
   @_ ? $self->{FULLNAME} = shift
      : $self->{FULLNAME};
   }

sub modified()
   {
   my $self=shift;
   @_ ? $self->{MODIFIED} = shift
      : $self->{MODIFIED};
   }

sub workingrev()
   {
   my $self=shift;
   @_ ? $self->{WKREV} = shift
      : $self->{WKREV};
   }

sub reporev()
   {
   my $self=shift;
   @_ ? $self->{RREV} = shift
      : $self->{RREV};
   }

sub packagename()
   {
   my $self=shift;
   @_ ? $self->{RREV} = shift
      : $self->{RREV};
   }

sub projectname();
   {
   my $self=shift;
   @_ ? $self->{PROJECTNAME} = shift
      : $self->{PROJECTNAME};
   }

sub stickytag()
   {
   my $self=shift;
   my ($tag,$revision)=@_;

   if ($tag && $revision)
      {
      # Populate tag info:
      $self->{STICKYTAG} = $tag;
      $self->{STICKYREV} = $revision;
      }
   else
      {
      $self->{STICKYTAG} = 'none';
      }
   }

sub get_sticky_rev()
   {
   my $self=shift;
   if (exists($self->{STICKYTAG}) && $self->{STICKYTAG} ne 'none')
      {
      return $self->{STICKYREV};
      }
   return undef;
   }

sub get_sticky_tag()
   {
   my $self=shift;
   if (exists($self->{STICKYTAG}) && $self->{STICKYTAG} ne 'none')
      {
      return $self->{STICKYTAG};
      }
   return undef;
   }

sub add_tag_to_history()
   {
   my $self=shift;
   my ($tag,$rev)=@_;
   if (!exists($self->{EXISTINGTAGS}->{$tag}))
      {
      $self->{EXISTINGTAGS}->{$tag} = $rev;
      # Add the tag to the index. This is so that we can track
      # the ordering of tags/revisions in the tag list:
      $self->{TAG_INDEX}->{$self->{TAG_INDEX_MARKER}} = $tag;
      # Increment the tag marker:
      $self->{TAG_INDEX_MARKER}++;
      # Build a back-relation: rev -> tag list:
      if (exists($self->{REVISIONS}->{$rev}))
	 {
	 push(@{$self->{REVISIONS}->{$rev}},$tag);
	 }
      else
	 {
	 $self->{REVISIONS}->{$rev} = [ $tag ];
	 }      
      }
   }

sub get_revision_for_tag()
   {
   my $self=shift;
   my ($tag)=@_;
   if (exists($self->{EXISTINGTAGS}->{$tag}))
      {
      return $self->{EXISTINGTAGS}->{$tag};
      }
   return undef;
   }

sub get_taglist_for_revision()
   {
   my $self=shift;
   my ($rev)=@_;
   if (exists($self->{REVISIONS}->{$rev}))
      {
      # Return the array of tags for this revision:
      return $self->{REVISIONS}->{$rev};
      }
   return undef;
   }

sub get_previous_tag()
   {
   my $self=shift;
   foreach my $tg_idx (sort { $a <=> $b } keys %{$self->{TAG_INDEX}})
      {
      # Return as soon as we find a tag with a V at the front:
      if ($self->{TAG_INDEX}->{$tg_idx} =~ /^V.*/)
	 {
	 return $self->{TAG_INDEX}->{$tg_idx};
	 }
      }
   # Otherwise just return the latest tag:
   return $self->{TAG_INDEX}->{1};
   }

sub needs_tag_publish()
   {
   my $self=shift;
   # Get the list of tags corresponding to the working revision.
   # If this returns undef, the working rev is not tagged and
   # so we signal 'PUBLISH':
   my $wrev_taglist = $self->get_taglist_for_revision($self->workingrev());
   if (!$wrev_taglist)
      {
      return 1;
      }
   return 0;
   }

sub full_file_details()
   {
   my $self=shift;
   my $alert;
   
   if ($self->modified())
      {
      $alert=$main::bold.$main::mod."MODIFIED, NEEDS COMMIT".$main::normal;
      printf (" %-45s %-15s %-15s %-10s\n",$self->fullname(),$self->reporev(),$self->workingrev(),$alert);  
      }
   elsif ($self->needs_tag_publish())
      {
      my $lasttag = $self->get_previous_tag();
      $alert = $main::bold.$main::publ.'PUBLISH (last tag '.$lasttag.')'.$main::normal;
      printf (" %-45s %-15s %-15s %-10s\n",$self->fullname(),$self->reporev(),$self->workingrev(),$alert);
      }
   else
      {
      $alert=$main::bold.$main::uptd."UP-TO-DATE".$main::normal;
      printf (" %-45s %-15s %-15s %-10s\n",$self->fullname(),$self->reporev(),$self->workingrev(),$alert);
      }
   }

sub modified_file_details()
   {
   my $self=shift;
   my $alert;
   
   if ($self->modified())
      {
      $alert=$main::bold.$main::mod."MODIFIED, NEEDS COMMIT".$main::normal;
      printf (" %-45s %-15s %-15s %-10s\n",$self->fullname(),$self->reporev(),$self->workingrev(),$alert);  
      return 1;
      }

   if ($self->needs_tag_publish())
      {
      my $lasttag = $self->get_previous_tag();
      $alert = $main::bold.$main::publ.'PUBLISH (last tag '.$lasttag.')'.$main::normal;
      printf (" %-45s %-15s %-15s %-10s\n",$self->fullname(),$self->reporev(),$self->workingrev(),$alert);
      return 1;
      }

   return 0;
   }

#### Package to handle the project status info ####
package ProjectStatus;

sub new()
   {
   my $proto=shift;
   my $class=ref($proto) || $proto;
   my $self={};   
   bless($self,$class);
   $self->name(@_);
   $self->{MODIFIED}={};
   $self->{SUMMARY}=1;
   return $self;
   }

sub name()
   {
   my $self=shift;
   @_ ? $self->{PROJECTNAME} = shift
      : $self->{PROJECTNAME};
   }

sub add_package()
   {
   my $self=shift;
   my ($packagename,$hasmod,$lasttag,$fdata)=@_;
   $self->{PACKAGES}->{$packagename}=$fdata;
   $self->{MODIFIED}->{$packagename}=$hasmod;
   $self->{TOPUBLISH}->{$packagename}=$lasttag || 0;
   }

sub packages()
   {
   my $self=shift;
   return [ sort keys %{$self->{PACKAGES}} ];
   }

sub files_for_package()
   {
   my $self=shift;
   my ($packagename)=@_;
   return $self->{PACKAGES}->{$packagename};
   }
		      
sub modified()
   {
   my $self=shift;
   return [ sort keys %{$self->{MODIFIED}} ];
   }

sub nmodified_in_package()
   {
   my $self=shift;
   my ($packagename)=@_;
   return $self->{MODIFIED}->{$packagename};
   }

sub to_publish()
   {
   my $self=shift;
   my ($packagename)=@_;
   return $self->{TOPUBLISH}->{$packagename};
   }

sub summary()
   {
   my $self=shift;
   @_ ? $self->{SUMMARY} = shift
      : $self->{SUMMARY};   
   }

sub compare_to_reference()
   {
   my $self=shift;
   my ($pack)=@_;
   # Single package only:
   if ($pack eq '')
      {
      print "-"x100,"\n";
      printf ("%-44s %-25s   %-10s\n","Package Name","\"$basereleaseid\" Tag","Latest Tag");
      print "-"x100,"\n";
      # Handle all packages:
      foreach my $pk (@{$self->packages()})
	 {
	 # For the package being checked, check any file and get last tag (if the package
	 # is up-to-date and has been tagged, FileObject->get_previous_tag() will still
	 # return the latest tag).	 
	 # Take the first file object in the package:
	 my $ref_file = $self->files_for_package($pk)->[0];
	 die "No FileObject reference in compare_to_reference().\n", unless (ref($ref_file) eq 'FileObject');
	 my $this_package_tag = $ref_file->get_previous_tag(),"\n";
	 # Look for a sticky tag:
	 my $sticky_tag = $ref_file->get_sticky_tag();
	 
	 # Check that the package has a tag in the reference:
	 if (!exists($referencedata->{$pk}))
	    {
	    die "ProjectCVSStatus: compare_to_reference() - $pk doesn't have a tag in the reference release $basereleaseid!","\n";
	    }
	 
	 $this_package_tag = $main::bold.$main::mod.$this_package_tag.$main::normal, if ($this_package_tag ne $referencedata->{$pk});

	 if ($sticky_tag)
	    {
	    printf ("%-45s %-10s                 %-10s (Sticky: %-10s)\n",$pk,$referencedata->{$pk},$this_package_tag,$sticky_tag);
	    }
	 else
	    {
	    printf ("%-45s %-10s                 %-10s\n",$pk,$referencedata->{$pk},$this_package_tag);
	    }
	 }      
      }
   else
      {      
      print "Comparing tags for package $pack only.","\n";
      # Check that the required package is listed in the local package list:
      if (grep($pack eq $_,@{$self->packages()}))
	 {
	 # Take the first file object in the package:
	 my $ref_file = $self->files_for_package($pack)->[0];
	 die "No FileObject reference in compare_to_reference().\n", unless (ref($ref_file) eq 'FileObject');
	 my $this_package_tag = $ref_file->get_previous_tag(),"\n";
	 # Look for a sticky tag:
	 my $sticky_tag = $ref_file->get_sticky_tag();
	 
	 # Check that the package has a tag in the reference:
	 if (!exists($referencedata->{$pack}))
	    {
	    die "ProjectCVSStatus: compare_to_reference() - $pack doesn't have a tag in the reference release $basereleaseid!","\n";
	    }
	 
	 $this_package_tag = $main::bold.$main::mod.$this_package_tag.$main::normal, if ($this_package_tag ne $referencedata->{$pack});

	 print "-"x100,"\n";
	 printf ("%-44s %-25s   %-10s\n","Package Name","\"$basereleaseid\" Tag","Latest Tag");
	 print "-"x100,"\n";
	 
	 if ($sticky_tag)
	    {
	    printf ("%-45s %-10s                 %-10s (Sticky: %-10s)\n",$pack,$referencedata->{$pack},$this_package_tag,$sticky_tag);
	    }
	 else
	    {
	    printf ("%-45s %-10s                 %-10s\n",$pack,$referencedata->{$pack},$this_package_tag);
	    }	 
	 }
      else
	 {
	 print "No package called $pack in the current working area.","\n";
	 }
      }   
   }

sub show_file_info()
   {
   my $self=shift;

   if ($self->{SUMMARY})
      {
      foreach my $pk (@{$self->packages()})
	 {
	 my $alert;
	 my $nmodified = $self->nmodified_in_package($pk);
	 my $npublish = $self->to_publish($pk);
	 if ($nmodified)
	    {
	    $alert=$nmodified." ".$main::bold.$main::mod."MODIFIED FILES".$main::normal;
	    printf ("%-30s\n    %-20s\n",$main::bold.$pk.$main::normal,$alert);	    
	    }
	 elsif ($npublish)
	    {
	    $alert = $main::bold.$main::publ.'NEEDS TAG TO BE PUBLISHED (last tag '.$npublish.')'.$main::normal;
	    printf ("%-30s\n    %-20s\n",$main::bold.$pk.$main::normal,$alert);	    
	    }
	 else	   
	    {
	    $alert=$main::bold.$main::uptd."UP-TO-DATE".$main::normal;
	    printf ("%-30s\n    %-20s\n",$main::bold.$pk.$main::normal,$alert);
	    }
	 }
      }
   else
      {
      my $found_m=0;
      # Show full details of all modified files:      
      foreach my $pk (@{$self->packages()})
	 {
	 print $main::bold.$pk.$main::normal,"\n";
	 # Run over each file in the package:
	 foreach my $file_obj (@{$self->files_for_package($pk)})
	    {
	    if ($opts{MODONLY})
	       {
	       # If the function returns 0, we're up-to-date, otherwise increment the counter:
	       $found_m+=$file_obj->modified_file_details();
	       }
	    else
	       {
	       $file_obj->full_file_details();
	       }
	    }

	 # Check that we had a modified package or signal up-to-date (non-zero count):
	 if (!$found_m && $opts{MODONLY})
	    {
	    $alert=$main::bold.$main::uptd."UP-TO-DATE".$main::normal;
	    # If the function returns 0, we're up-to-date:
	    printf ("    %-20s\n",$alert);
	    }
	 }
      }
   }


1;
