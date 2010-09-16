#!/usr/bin/env perl
use File::Basename;

my $cwd = `/bin/pwd`; chomp $cwd;
my $dir=shift || $cwd;

my $releasetop = &scramReleaseTop($dir);
if ($releasetop eq "")
{print "$dir is not a SCRAM-Base project area.\nUsage: $0 [project-path]\n"; exit(1);}

my %CACHE=();
&readEnvironment($releasetop);
my $src="src";
if(exists $CACHE{ENVIRONMENT}{SCRAM_SOURCEDIR})
{$src=$CACHE{ENVIRONMENT}{SCRAM_SOURCEDIR};}
my $srcdir="${releasetop}/${src}";

if (!-d "$srcdir")
{print "$srcdir does not exist."; exit(1);}

print "Searching files for duplicate data dictionary definitions.\n";
print "...";
&processDir ($srcdir);
print "\n";

my $msg=0;
foreach my $dict (keys %{$CACHE{DICT}})
{
  my @files=sort keys %{$CACHE{DICT}{$dict}};
  if (@files>1)
  {
    if($msg==0)
    {
      print "Multiple definitions of data dictionary found for following:\n";
      $msg=1;
    }
    print "  \"$dict\":\n";
    foreach my $file (@files)
    {
      print "      $file (".$CACHE{DICT}{$dict}{$file}.")\n";
    }
    print "\n";
  }
}
if ($msg==0)
{print "Congratulations, No multiple data dictionary definitions found.\n"}

sub processDir ()
{
  my $dir = shift || return;
  my $files;
  my $filefilter=".+\/classes_def\.xml";
  if(!opendir($files,"$dir")){ die "Can not open directory $dir.";}
  while (my $d = readdir $files)
  {
    my $file="${dir}/${d}";
    if($d=~/^(\..*|CVS|doc|html)$/){next;}
    if(-d "$file"){&processDir("$file");}
    if((-f "$file") && ($file=~/^$filefilter$/))
    {&processFile("$file");}
  }
  closedir($files);
}

sub processFile ()
{
  my $file=shift;
  print ".";
  my $linen=0;
  foreach my $line (`cat $file`)
  {
    chomp $line;
    $linen++;
    if($line=~/^.*<\s*class\s+name\s*=\s*\"([^"]+)\"\s*(\/\s*|\s*)>\s*$/)
    {
      my $dict = $1;
      $file=~/^$srcdir\/(.+)$/;
      my $relfile=$1;
      $CACHE{DICT}{$dict}{$relfile}=$linen;
    }
  }
}

sub scramReleaseTop(){
  return &checkWhileSubdirFound(shift,".SCRAM");
}

sub checkWhileSubdirFound(){
  my $dir=shift;
  my $subdir=shift;
  while((!-d "${dir}/${subdir}") && ($dir ne "/")){$dir=dirname($dir);}
  if(-d "${dir}/${subdir}"){return $dir;}
  return "";
}

sub readEnvironment (){
  my $dir=shift;
  my $env_file="${dir}/.SCRAM/Environment";
  foreach my $line (`cat $env_file`){
    chomp $line;
    if($line=~/^ *$/){next;}
    elsif($line=~/^ *#/){next;}
    elsif($line=~/^ *([^ ]+?) *= *(.+)$/){$CACHE{ENVIRONMENT}{$1}=$2;}
  }
}
