#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use SCRAMGenUtils;

my $pwd=`/bin/pwd`; chomp $pwd; $pwd=&SCRAMGenUtils::fixPath($pwd);
my $dir=shift || $pwd;
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
  foreach my $p (sort keys %{$data->{$pack}{IGLET}}){$str.="$p.iglet,";}
  $str=~s/,$//; $str.="|";
  foreach my $p (sort keys %{$data->{$pack}{BIN}}){$str.="$p,";}
  $str=~s/,$//; $str.="|";
  foreach my $p (sort keys %{$data->{$pack}{TEST}}){$str.="$p,";}
  $str=~s/,$//; $str.="|";
  foreach my $p (sort keys %{$data->{$pack}{SCRIPTS}}){$str.="$p,";}
  $str=~s/,$//;
  print "$pack:$str\n"; 
}
exit 0;

sub doSearch()
{
  foreach my $pack (sort keys %$data)
  {
    foreach my $p (sort keys %{$data->{$pack}{LIBRARY}})
    {
      if (!-e "${release}/lib/".$ENV{SCRAM_ARCH}."/lib${p}.so"){print "Missing LIBRARY: lib${p}.so\n";}
    }
    foreach my $p (sort keys %{$data->{$pack}{PLUGIN}})
    {
      if (!-e "${release}/lib/".$ENV{SCRAM_ARCH}."/plugin${p}.so"){print "Missing PLUGIN: plugin${p}.so\n";}
    }
    foreach my $p (sort keys %{$data->{$pack}{IGLET}})
    {
      if (!-e "${release}/lib/".$ENV{SCRAM_ARCH}."/${p}.iglet"){print "Missing IGLET: ${p}.iglet\n";}
    }
    foreach my $p (sort keys %{$data->{$pack}{BIN}})
    {
      if (!-e "${release}/bin/".$ENV{SCRAM_ARCH}."/${p}"){print "Missing BIN: $p\n";}
    }
    foreach my $p (sort keys %{$data->{$pack}{TEST}})
    {
      if (!-e "${release}/test/".$ENV{SCRAM_ARCH}."/${p}"){print "Missing TEST: $p\n";}
    }
    foreach my $p (sort keys %{$data->{$pack}{SCRIPTS}})
    {
      if (!-e "${release}/bin/".$ENV{SCRAM_ARCH}."/${p}"){print "Missing SCRIPTS: $p\n";}
    }
  }
}

sub doSearchRev()
{
  foreach my $file (`find ${release}/lib/$ENV{SCRAM_ARCH} -name "*" -type f | sed 's|${release}/lib/$ENV{SCRAM_ARCH}/||'`)
  {
    chomp $file;
    my $t="";
    my $n="";
    if ($file=~/^(plugin|lib)([^\.]+)\.so$/)
    {
      my $t="LIBRARY";
      my $n=$2;
      if ($1 eq "plugin"){$t="PLUGIN";}
    }
    elsif ($file=~/^([^\.]+)\.iglet$/)
    {
      $t="IGLET";
      $n=$1;
    }
    if ($n ne "")
    {
      my $found=0;
      foreach my $pack (sort keys %$data)
      {
        if (exists $data->{$pack}{$t}{$n}){$found=1; last;}
      }
      if (!$found){print "RMissing $t: $n ($file)\n";}
    }
  }
  foreach my $file (`find ${release}/test/$ENV{SCRAM_ARCH} -name "*" -type f | sed 's|${release}/test/$ENV{SCRAM_ARCH}/||'`)
  {
    chomp $file;
    my $found=0;
    foreach my $pack (sort keys %$data)
    {
      if (exists $data->{$pack}{TEST}{$file}){$found=1; last;}
    }
    if (!$found){print "Missing TEST: $file\n";}
  }
  foreach my $file (`find ${release}/bin/$ENV{SCRAM_ARCH} -maxdepth 1 -name "*" -type f | sed 's|${release}/bin/$ENV{SCRAM_ARCH}/||'`)
  {
    chomp $file;
    my $found=0;
    foreach my $pack (sort keys %$data)
    {
      if (exists $data->{$pack}{BIN}{$file}){$found=1; last;}
      if (exists $data->{$pack}{SCRIPTS}{$file}){$found=1; last;}
    }
    if (!$found){print "RMissing BIN: $file\n";}
  }
}

sub updateProd ()
{
  my $p=shift;
  if(exists $projcache->{BUILDTREE}{$p}{CLASS})
  {
    my $suffix=$projcache->{BUILDTREE}{$p}{SUFFIX};
    if($suffix ne ""){return 0;}
    my $class=$projcache->{BUILDTREE}{$p}{CLASS};
    my $pack=dirname($p);
    if (exists $projcache->{BUILDTREE}{$p}{RAWDATA}{content})
    {
      my $c=$projcache->{BUILDTREE}{$p}{RAWDATA}{content};
      if($class=~/^(LIBRARY|CLASSLIB|SEAL_PLATFORM)$/)
      {
        my $type="LIBRARY";
        if (&isPlugin($class,$c)){$type="PLUGIN";}
        my $name=$projcache->{BUILDTREE}{$p}{NAME};
        if (-f "${release}/src/${p}/iglet.cc"){$data->{$pack}{IGLET}{$name}=1;$type="LIBRARY";}
        elsif (($type eq "LIBRARY") && (-f "${release}/src/${p}/classes.h")  && (-f "${release}/src/${p}/classes_def.xml"))
        {$data->{$pack}{PLUGIN}{"${name}Capabilities"}=1;}
	$data->{$pack}{$type}{$name}=1;
        if ((exists $c->{FLAGS}) && (exists $c->{FLAGS}{INSTALL_SCRIPTS}))
        {
          foreach my $file (@{$c->{FLAGS}{INSTALL_SCRIPTS}}){$data->{$pack}{SCRIPTS}{$file}=1;}
        }
      }
      elsif ($class=~/^(TEST|BIN|PLUGINS|BINARY)$/){&updateProds($pack,$c,$class);}
    }
    elsif ($class=~/^SCRIPTS$/){&updateScripts($pack,$p);}
  }
}

sub updateScripts()
{
  my $pack=shift;
  my $dir=shift;
  foreach my $f (`ls ${release}/src/$dir`)
  {
    chomp $f;
    $data->{$pack}{SCRIPTS}{$f}=1;
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
      my $c1=$c->{BUILDPRODUCTS}{$t}{$prod}{content};
      if ((exists $c1->{FLAGS}) && (exists $c1->{FLAGS}{INSTALL_SCRIPTS}))
      {
        foreach my $file (@{$c1->{FLAGS}{INSTALL_SCRIPTS}}){$data->{$pack}{SCRIPTS}{$file}=1;}
      }
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
