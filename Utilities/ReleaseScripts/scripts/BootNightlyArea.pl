#!/usr/bin/env perl 
#____________________________________________________________________ 
# File: BootNightlyArea.pl
#____________________________________________________________________ 
#  
# Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
# Update: 2005-10-27 14:25:38+0200
# Revision: $Id: BootNightlyArea.pl,v 1.2 2013/05/24 10:33:16 muzaffar Exp $ 
#
# Copyright: 2005 (C) Shaun ASHBY
#
#--------------------------------------------------------------------
use strict;
use Cwd;
$|=1;

BEGIN
   {
   use Config;
   my $SCRAMINSTALLAREA = '/afs/cern.ch/cms/Releases/SCRAM/current/Installation/TT2/lib/perl5/site_perl';
   my $SCRAMSRC = $SCRAMINSTALLAREA."/".$Config{version}."/".$Config{archname};
   # Pick up the template toolkit from the current
   # version of SCRAM:
   unshift(@INC, $SCRAMSRC);
   }

my ($name,$version,$toolbox)=@ARGV;
my $templates=[ 'boot-nightly', 'requirements', 'BuildFile' ];

# We need the template toolkit:
use Template;

# Set the template dir:
my $templatedir='config';

# Set where to put the processed files:
my $outputdir='config';

# Template toolkit parameters:
my $template_config =
   {
   INCLUDE_PATH => $templatedir,
   EVAL_PERL    => 1 
      };

# Prepare the data for the bootstrap file and requirements:
my $template_engine = Template->new($template_config) || die $Template::ERROR, "\n";
my $projectdata = {};
my $packages = {};


# The main block:
if ($name ne '' && 
    $version ne '' && 
    $toolbox ne '')
   {
   # Check that the config directory is accessible in current dir:
   die "$0: Unable to find the config directory.","\n",unless (-d cwd()."/config");
   
   # The project data. This data is accessed by key in the template:
   $projectdata->{'PROJECT_NAME'} = $name;
   $projectdata->{'PROJECT_VERSION'} = $version;
   $projectdata->{'CONFIG_VERSION'} = $toolbox;

   # Process both files:
   foreach my $file (@$templates)
      {            
      $template_engine->process($file.".tmpl", $projectdata, $outputdir."/".$file)
	 || die "Template error: ".$template_engine->error;      
      }
   }
else
   {
   die "Usage: $0 <NAME> <VERSION> <TOOLBOX>","\n";
   }

