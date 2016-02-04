#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use SCRAMGenUtils;
use Getopt::Long;
$|=1;

if(&GetOptions(
	       "--release=s",\$rel,
	       "--out=s",\$output,
	       "--json",\$json,
	       "--help",\$help,
              ) eq ""){print STDERR "=====>ERROR: Wrong arguments.\n"; &usage_msg();}

if (defined $help){&usage_msg();}
if (!defined $rel){&usage_msg();}
if (defined $json){$json=1;}
else{$json=0;}

if (!-f "${rel}/.SCRAM/Environment"){die "Invalid release base path provided: $rel\n";}
my $ver=basename($rel);

my $pref=*STDOUT;
if (defined $output){open($pref,">$output") || die "Can not open file for writing: $output\n";}

my %cache=();
$cache{valid}{interface}{interface}=1;
$cache{valid}{src}{interface}=1;
$cache{valid}{plugins}{interface}=1;
$cache{valid}{test}{interface}=1;
$cache{valid}{bin}{interface}=1;
$cache{exception}{'GeneratorInterface\/.+\.F'}='SimDataFormats\/GeneratorProducts\/data\/.+\.inc';
$cache{exception}{'GeneratorInterface\/.+\.inc'}='SimDataFormats\/GeneratorProducts\/data\/.+\.inc';
$cache{exception}{'SimDataFormats\/GeneratorProducts\/src\/HepEvt.F'}='SimDataFormats\/GeneratorProducts\/data\/.+\.inc';
$cache{exception}{'Iguana\/GLBrowsers\/src\/.+\.cc'}='Iguana\/GLBrowsers\/pixmaps/.+\.xpm';

my $c=0;
print STDERR "Processing files in ${rel}/src ...";
foreach my $f (`find ${rel}/src -name "*" -type f`)
{
  chomp $f;
  my $d=dirname($f);
  my $rf=$f; $rf=~s/^$rel\/src\/*//;
  my ($based,$type,$pk)=&getFileInfo($rf);
  if (($type eq "") || (!exists $cache{valid}{$type})){next;}
  foreach my $l (`cat $f`)
  {
    chomp $l;
    if ($l=~/^\s*#\s*include\s*["<]([^">]+)[">]/)
    {
      my $inc=$1;
      if ((-f "${d}/${inc}") || (!-f "${rel}/src/${inc}")){next;} 
      my ($bd1,$t1,$pk1)=&getFileInfo($inc);
      if (($bd1 eq $based) || (exists $cache{valid}{$type}{$t1})){next;}
      my $skip=0;
      foreach my $exp (keys %{$cache{exception}})
      {
	if ($rf=~/^$exp$/)
	{
	  my $xi=$cache{exception}{$exp};
	  if ($inc=~/^$xi$/){$skip=1; last;}
	}
      }
      if ($skip){next;}
      $cache{ERRORS}{$ver}{$pk}{$rf}{$inc}=1;
    }
  }
  $c++;
  if ($c==2000){print STDERR ".";$c=0;last;}
}
print STDERR " Done\n";

if (!$json)
{
  foreach my $pk (sort keys %{$cache{ERRORS}{$ver}})
  {
    foreach my $rf (sort keys %{$cache{ERRORS}{$ver}{$pk}})
    {
      foreach my $inc (sort keys %{$cache{ERRORS}{$ver}{$pk}{$rf}}){print $pref "$rf  $inc\n";}
    }
  }
}
else
{
  my $str=&SCRAMGenUtils::writeJson($cache{ERRORS},0); chomp ($str); $str=~s/, *$//;
  print $pref  "$str\n";
}

if (defined $out){close($pref);}

sub getFileInfo()
{
  my $f=shift;
  if($f=~/^([A-Z][^\/]+)\/+([A-Za-z0-9][^\/]+)\/+([^\/]+)\/.+/){return ("$1/$2/$3",$3,"$1/$2");}
  return ("","","");
}

sub usage_msg()
{
  print STDERR "Usage: $0 --release <path> [--out <file>] [--json] [--help]\n",
               "  --release <path>   Release base path\n",
	       "  --out <file>       File path to write the output\n",
	       "  --json             JSON format output\n",
	       "  --help             Show this help message\n";
  exit 0;
}
