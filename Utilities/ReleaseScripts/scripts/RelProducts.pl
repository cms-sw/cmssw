#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use SCRAMGenUtils;

my $dir=dirname($0);
my $pwd=`/bin/pwd`; chomp $pwd; $pwd=&SCRAMGenUtils::fixPath($pwd);
if ($dir!~/^\//){$dir=&SCRAMGenUtils::fixPath("${pwd}/${dir}");}

my $release=&SCRAMGenUtils::scramReleaseTop($dir);
if($release eq ""){print STDERR "ERROR: Please run this script from a SCRAM-based area.\n"; exit 1;}
&SCRAMGenUtils::init ($release);

my $cachefile=&SCRAMGenUtils::fixCacheFileName(${release}."/.SCRAM/".$ENV{SCRAM_ARCH}."/ProjectCache.db");
my $projcache=&SCRAMGenUtils::readCache($cachefile);

my $data={};
foreach my $dir (reverse sort keys %{$projcache->{BUILDTREE}}){&updateProd($dir);}

foreach my $pack (sort keys %$data)
{
  my $str="";
  foreach my $p (sort keys %{$data->{$pack}{LIBRARY}}){$str.="lib$p.so,";}
  foreach my $p (sort keys %{$data->{$pack}{PLUGIN}}){$str.="plugin$p.so,";}
  $str=~s/,$//; $str.="|";
  foreach my $p (sort keys %{$data->{$pack}{BIN}}){$str.="$p,";}
  $str=~s/,$//; $str.="|";
  foreach my $p (sort keys %{$data->{$pack}{TEST}}){$str.="$p,";}
  $str=~s/,$//;  
  print "$pack:$str\n"; 
}

exit 0;

sub updateProd ()
{
  my $p=shift;
  if(exists $projcache->{BUILDTREE}{$p}{CLASS} && (exists $projcache->{BUILDTREE}{$p}{RAWDATA}{content}))
  {
    my $suffix=$projcache->{BUILDTREE}{$p}{SUFFIX};
    if($suffix ne ""){return 0;}
    my $class=$projcache->{BUILDTREE}{$p}{CLASS};
    my $c=$projcache->{BUILDTREE}{$p}{RAWDATA}{content};
    my $pack=dirname($p);
    if($class=~/^(LIBRARY|CLASSLIB|SEAL_PLATFORM)$/)
    {
      my $type="LIBRARY";
      if (&isPlugin($class,$c)){$type="PLUGIN";}
      my $name=$projcache->{BUILDTREE}{$p}{NAME};
      $data->{$pack}{$type}{$name}=1;
    }
    elsif ($class=~/^(TEST|BIN|PLUGINS|BINARY)$/){&updateProds($pack,$c,$class);}
  }
}

sub updateProds()
{
  my $pack=shift;
  my $c=shift;
  my $ptype=shift;
  foreach my $t (keys %{$c->{BUILDPRODUCTS}})
  {
    foreach my $prod (keys %{$c->{BUILDPRODUCTS}{$t}})
    {
      my $type=&getType($ptype,$t,$c->{BUILDPRODUCTS}{$t}{$prod}{content},$c);
      $data->{$pack}{$type}{$prod}=1;
    }
  }
}

sub getType()
{
  my $ptype=shift;
  my $stype=shift;
  my $c=shift;
  my $c1=shift;
  my $type=$stype;
  if ($stype eq "BIN")
  {
    if ($ptype eq "TEST"){$type=$ptype;}
  }
  elsif($stype eq "LIBRARY")
  {
    if (&isPlugin($ptype,$c,$c1)==1){$type="PLUGIN";}
  }
  else{die "ERROR: Unknown type: $stype\n";}
  return $type;
}

sub isPlugin()
{
  my $ptype=shift;
  my $c=shift;
  my $c1 = shift || undef;
  my $ok=0;
  if($ptype eq "PLUGINS"){$ok=1;}
  if ((exists $c->{FLAGS}) && (exists $c->{FLAGS}{EDM_PLUGIN})){$ok=$c->{FLAGS}{EDM_PLUGIN}[0];}
  elsif ((defined $c1) && (exists $c1->{FLAGS}) && (exists $c1->{FLAGS}{EDM_PLUGIN})){$ok=$c1->{FLAGS}{EDM_PLUGIN}[0];}
  return $ok;
}
