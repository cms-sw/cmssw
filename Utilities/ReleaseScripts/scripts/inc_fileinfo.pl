#!/usr/bin/env perl
use Cwd;
use File::Basename;
use lib "/build/muzaffar/includechecker/Utilities/ReleaseScripts/scripts";
use SCRAMGenUtils;

my $curdir=cwd();
my $file=shift || die "Missing file path to check";
if(!-f $file){die "No such file:$file";}

if($file!~/^\//){$file="${curdir}/${file}";}
$file=&SCRAMGenUtils::fixPath($file);
my $rel=dirname($file);
while(!-d "${rel}/.SCRAM"){$rel=dirname($rel);}
if(!-d "${rel}/.SCRAM"){die "not a scram base dir";}

$file=~s/^$rel\///;
$file=~s/^src\///;
my $arch=&SCRAMGenUtils::getScramArch();

my $cache={};
&getIncludes($file,$cache);
foreach my $inc (@{$cache->{order}})
{
  my $c=&readcachefile($inc);
  foreach my $i (keys %{$c->{ALL_INCLUDES_REMOVED}})
  {
    print "$inc >= $i\n";
  }
}
exit 0;

sub getIncludes ()
{
  my $f=shift;
  my $c=shift || {};
  if(exists $c->{all}{$f}){return;}
  my $cfile="${rel}/tmp/${arch}/includechecker/cache/files/${f}";
  if(!-f $cfile){return;}
  $c->{all}{$f}=1;
  push @{$c->{order}},$f;
  my $c1=&readcachefile($f);
  foreach my $f (keys %{$c1->{INCLUDES}})
  {&getIncludes($f,$c);}
}

sub readcachefile()
{
  my $file=shift;
  if(exists $cache->{data}{$file}){return $cache->{data}{$file};}
  my $cfile="${rel}/tmp/${arch}/includechecker/cache/files/${file}";
  my $c={};
  if(-f $cfile){$c=&SCRAMGenUtils::readHashCache($cfile);}
  $cache->{data}{$file}=$c;
  return $c;
}
