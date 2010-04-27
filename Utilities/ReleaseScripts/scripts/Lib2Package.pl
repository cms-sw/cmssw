#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use SCRAMGenUtils;
use Getopt::Long;
$|=1;

my $pwd=`/bin/pwd`; chomp $pwd; $pwd=&SCRAMGenUtils::fixPath($pwd);

if(&GetOptions(
	       "--release=s",\$release,
	       "--product=s",\$product,
	       "--help",\$help,
              ) eq ""){print STDERR "#Wrong arguments.\n"; &usage_msg();}


if (defined $help){&usage_msg();}
if (!defined $product){&usage_msg();}

my $pname=$product;
if ($product=~/\.so$/)
{
  if($product=~/^lib(.+)\.so$/){$pname=$1;}
  elsif($product=~/^plugin(.+)(Capabilities|)\.so$/){$pname=$1;}
}
elsif($product=~/^(.+)\.iglet$/){$pname=$1;}

if (!defined $release){$release=$pwd;}
$release=&SCRAMGenUtils::scramReleaseTop($release);
if($release eq ""){print STDERR "ERROR: Please run this script from a SCRAM-based area or use --release <path> option.\n"; exit 1;}

&SCRAMGenUtils::init ($release);
my $arch=&SCRAMGenUtils::getScramArch();

my $cache={};
$cache=&initCache($release,$arch);

if (exists $cache->{PRODS}{$pname}){print "$product => ",$cache->{PRODS}{$pname},"\n";}
else{die "ERROR: No product found with name $pname ($product)\n";}

exit 0;

sub usage_msg()
{
  my $s=basename($0);
  print "Usage: $s --product <product> [--release <path>] [--help]\n\n",
        "e.g.\n",
        "  $s --release <path> --product libCore.so\n",
        "  $s --release <path> --product libFWCoreFramework.so\n",
        "  $s --release <path> --product lcg_EnvironmentAuthenticationService\n",
        "  $s --release <path> --product cmsRun\n";
        "  $s --release <path> -p FWCoreUtilities\n";
  exit 0;
}

##########################################################################
# Read Tools and Project cache of all externals and SCRAM-based projects #
##########################################################################
sub initCache()
{
  my ($release,$arch)=@_;
  my $cache={};
  $cache->{Caches}{TOOLS}=&SCRAMGenUtils::readCache("${release}/.SCRAM/${arch}/ToolCache.db.gz");
  foreach my $t (keys %{$cache->{Caches}{TOOLS}{SETUP}})
  {
    my $sbase="";
    if ($cache->{Caches}{TOOLS}{SETUP}{$t}{SCRAM_PROJECT} == 1)
    {
      my $bv=uc($t)."_BASE";
      $sbase=$cache->{Caches}{TOOLS}{SETUP}{$t}{$bv};
    }
    elsif ($t eq "self"){$sbase=$release;}
    if ($sbase ne "")
    {
      $cache->{Caches}{$t}=&SCRAMGenUtils::readCache("${sbase}/.SCRAM/${arch}/ProjectCache.db.gz");
      foreach my $d (keys %{$cache->{Caches}{$t}{BUILDTREE}}){&readPkgInfo($d,$t,$cache);}
      if ($t eq "self")
      {
        my $releaseTop=&SCRAMGenUtils::getFromEnvironmentFile("RELEASETOP",$release);
	if ($releaseTop ne "")
	{
	  $cache->{Caches}{$t}=&SCRAMGenUtils::readCache("${releaseTop}/.SCRAM/${arch}/ProjectCache.db.gz");
	  foreach my $d (keys %{$cache->{Caches}{$t}{BUILDTREE}}){&readPkgInfo($d,$t,$cache);}
	}
      }
      delete $cache->{Caches}{$t};
    }
    else{&readToolsInfo(lc($t),$cache);}
  }
  delete $cache->{Caches};
  return $cache;
}

sub readPkgInfo ()
{
  my ($d,$t,$cache)=@_;
  my $c=$cache->{Caches}{$t}{BUILDTREE}{$d};
  my $suffix=$c->{SUFFIX};
  if($suffix ne ""){return;}
  my $class=$c->{CLASS};
  my $name=$c->{NAME};
  my $c1=$c->{RAWDATA}{content};
  if($class=~/^(LIBRARY|CLASSLIB|SEAL_PLATFORM)$/o){$cache->{PRODS}{$name}=dirname($d);}
  elsif($class=~/^(TEST|BIN|PLUGINS|BINARY)$/o){&addProds($c1,$d,$cache);}
  elsif($class=~/^(PYTHON|SUBSYSTEM|DATA_INSTALL|SCRIPTS|PROJECT|IVS|PACKAGE)$/o){return;}
  else{print STDERR "WARNING: UNKNOW TYPE $class in $t/$d\n";}
}

sub addProds()
{
  my ($c,$pack,$cache)=@_;
  if (!defined $c){return;}
  if (exists $c->{BUILDPRODUCTS})
  {
    foreach my $t (keys %{$c->{BUILDPRODUCTS}})
    {
      foreach my $p (keys %{$c->{BUILDPRODUCTS}{$t}}){$cache->{PRODS}{$p}=$pack;}
    }
  }
}

sub readToolsInfo()
{
  my ($t,$cache)=@_;
  if (exists $cache->{Caches}{TOOLS}{SETUP}{$t}{LIB})
  {
    foreach my $l (@{$cache->{Caches}{TOOLS}{SETUP}{$t}{LIB}}){$cache->{PRODS}{$l}=$t;}
  }
}
