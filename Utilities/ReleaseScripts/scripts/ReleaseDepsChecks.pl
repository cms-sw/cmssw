#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use SCRAMGenUtils;
use Getopt::Long;
$|=1;

my $pwd=`/bin/pwd`; chomp $pwd; $pwd=&SCRAMGenUtils::fixPath($pwd);
my $defaultRuleFile="Utilities/ReleaseScripts/scripts/CMSSWReleaseDependency.rules";

if(&GetOptions(
	       "--release=s",\$release,
	       "--rules=s",\$ruleFile,
	       "--package=s",\$package,
	       "--product=s",\$product,
	       "--dependency=s",\$dependency,
	       "--detail",\$detail,
	       "--ignore-test",\$ignore_test,
	       "--source-only",\$source_only,
	       "--dump",\$dump,
	       "--help",\$help,
              ) eq ""){print STDERR "#Wrong arguments.\n"; &usage_msg();}


if (defined $help){&usage_msg();}

if (defined $detail){$detail=1;}
else{$detail=0;}

if (defined $dump){$dump=1;}
else{$dump=0;}

if (defined $ignore_test){$ignore_test='[^\/]+\/[^\/]+\/test\/';}
else{$ignore_test='';}

if (!defined $release){$release=$pwd;}
$release=&SCRAMGenUtils::scramReleaseTop($release);
if($release eq ""){print STDERR "ERROR: Please run this script from a SCRAM-based area or use --release <path> option.\n"; exit 1;}

&SCRAMGenUtils::init ($release);
my $arch=&SCRAMGenUtils::getScramArch();

my $cache={}; my $cache_config={};
$cache=&initCache($release,$arch);

if ($dependency eq "")
{
  if (!defined $ruleFile)
  {
    $ruleFile = "${release}/src/${defaultRuleFile}";
    if (!-f $ruleFile)
    {
      my $releaseTop = &SCRAMGenUtils::getFromEnvironmentFile("RELEASETOP",$release);
      $ruleFile = "${releaseTop}/src/${defaultRuleFile}";
    }
  }
  if (!-f $ruleFile){die "ERROR: Can not find Project dependency rules file: $ruleFile\n";}
  ($cache->{RULES},$cache_config)=&readRules($ruleFile);
  if ((!defined $package) && (exists $cache_config->{packages})){$package=join(",",%{$cache_config->{packages}});}
}
else
{
  my $rules=[];
  foreach my $dep (split(",",$dependency))
  {
    $dep=~s/^\s*//o;$dep=~s/\s*$//o;
    my $add=0;
    if ($package ne ""){push @$rules,["${package}/.*",$dep];$add=1;}
    if ($product ne ""){push @$rules,[".+/${product}",$dep];$add=1;}
    if(!$add){push @$rules,[".+",$dep];}
  }
  $cache->{RULES}{pos}{"not-allowed-uses"}=$rules;
}

my %packs=();
if (defined $product)
{
  foreach my $prod (split(/,/,$product))
  {
    $prod=~s/\s+//g;
    if ($prod eq ""){next;}
    if (exists $cache->{DEPS}{$prod}){$packs{$cache->{DEPS}{$prod}{PACK}}{$prod}=1;}
  }
}
if (defined $package)
{
  foreach my $pack (split(/,/,$package))
  {
    $pack=~s/\s+//g;
    if ($pack eq ""){next;}
    foreach my $p (keys %{$cache->{DEPS}})
    {
      my $pk = $cache->{DEPS}{$p}{PACK};
      if ($pk=~/^$pack(\/.+|)$/){$packs{$pk}{$p}=1;}
    }
  }
}
if ((!defined $product) && (!defined $package))
{
  foreach my $p (keys %{$cache->{DEPS}})
  {
    if ($cache->{DEPS}{$p}{TOOL} eq "self"){$packs{$cache->{DEPS}{$p}{PACK}}{$p}=1;}
  }
}

if (scalar(keys %packs)==0){die "ERROR: There are no products to check for dependency.\n";}

foreach my $pk (sort keys %packs)
{
  foreach my $prod (sort keys %{$packs{$pk}}){&checkDependency($prod,$pk,$cache);}
}

if ($dump)
{
  foreach my $type (keys %{$cache->{RULES}})
  {
    foreach my $sec (keys %{$cache->{RULES}{$type}})
    {
      print "$sec ($type)\n";
      foreach my $r (@{$cache->{RULES}{$type}{$sec}}){print "  ",$r->[0]," => ",$r->[1],"\n";}
    }
  }
}

sub usage_msg()
{
  my $s=basename($0);
  print "Usage: $s [--rules <rule-file>] [--package <package>] [--product <product>]\n",
        "          [--dependency <tool|product>] [--release <path>] [--detail] [--help]\n",
        " Script apply the dependency rules provides via rule-file and print out any\n",
	" voilation of those rules in BuildFiles. One can use --package <package> and\n",
	" --dependency <tool|product> option to find out why <package> is being link\n",
	" against <tool|product>.\n";
  exit 0;
}

sub checkDependency()
{
  my ($prod,$pack,$cache)=@_;
  my $fprod="${pack}/${prod}";
  if ($prod eq $pack) { $fprod=$pack;}
  my %rules=();
  foreach my $type ("pos","nag")
  {
    foreach my $sec (keys %{$cache->{RULES}{$type}})
    {
      $rules{$type}{$sec}=[];
      foreach my $r (@{$cache->{RULES}{$type}{$sec}})
      {
        foreach my $x (@$r)
        {
          if ($fprod=~/^$x$/){push @{$rules{$type}{$sec}},$r;last;}
        }
      }
    }
  }
  my @allowed=();
  my $msg=0;
  foreach my $dep (keys %{$cache->{DEPS}{$prod}{ALL}})
  {
    my $isTool=0;
    if (exists $cache->{TOOLS}{$dep})
    {
      $isTool=1;
      my $found="";
      foreach my $a (@allowed){if (exists $cache->{DEPS}{$a}{ALL}{$dep}){$found=$a;last;}}
      if ($found){push @allowed,$dep; next;}
    }
    my $stat=&applyFilter($fprod,$dep,\%rules,"allowed-uses");
    if ($stat == -1)
    {
      $stat=&applyFilter($dep,$fprod,\%rules,"allowed-usedby");
      if ($stat == -1)
      {
        $stat=&applyFilter($fprod,$dep,\%rules,"not-allowed-uses");
        if ($stat == -1)
	{
	  $stat=&applyFilter($dep,$fprod,\%rules,"not-allowed-usedby");
	}
	if ($stat==1){$stat=0;}
	else{$stat=1;}
      }
    }
    if ($stat == 0)
    {
      if (!$msg){print ">> Checking dependency for $fprod\n"; $msg=1;}
      my $type="indirect";
      if (exists $cache->{DEPS}{$prod}{DIRECT}{$dep}){$type="direct"}
      print "  ****ERROR: Dependency violation ($type): $fprod $dep\n";
      if ($detail){&searchDeps($prod,$dep,$cache,"  ");}
    }
    elsif ($isTool){push @allowed,$dep;}
  }
  if ($msg){print ">> Done Checking dependency for $fprod\n";}
}

sub searchDeps()
{
  my ($p,$d,$cache,$tab)=@_;
  print "${tab}*",&product2Package($p,$cache)," ($p)\n";
  if ($p eq $d){return;}
  if (exists $cache->{DEPS}{$p}{ALL}{$d})
  {
    foreach my $u (keys %{$cache->{DEPS}{$p}{DIRECT}})
    {
      if ($u eq $d){print "${tab}  *$u (",$cache->{PACKS}{$u}[0],")\n";}
      if (!exists $cache->{PACKS}{$u}){next;}
      $u=$cache->{PACKS}{$u}[0];
      if (exists $cache->{DEPS}{$u}{ALL}{$d}){&searchDeps($u,$d,$cache,"$tab  ");}
    }
  }
}


#############################################################
#Check for dependnecy rules and returns
#-1: If there is no rule matches this dependnecy
#0:  If dependency in negated for this section
#1: If dependency is positive for this section
#############################################################

sub applyFilter()
{
  my ($left,$right,$rules,$sec)=@_;
  my $l=""; my $r="";
  foreach my $rx (@{$rules->{nag}{$sec}})
  {
    $l=$rx->[0]; $r=$rx->[1];
    if ($left=~/^$l$/)
    {
      if ($right=~/^$r$/){return 0;}    
    }
  }
  foreach my $rx (@{$rules->{pos}{$sec}})
  {
    $l=$rx->[0]; $r=$rx->[1];
    if ($left=~/^$l$/)
    {
      if ($right=~/^$r$/){return 1;}
    }
  }
  return -1;
}

#################################
# Read Project Dependency Rules #
#################################
sub readRules ()
{
  my ($file)=@_;
  my $cache={}; my $config={};
  foreach my $t ("pos","nag")
  {
    foreach my $x ("allowed","not-allowed")
    {
      foreach my $y ("uses","usedby"){$cache->{$t}{"${x}-${y}"}=[];}
    }
  }
  my $ref;
  open($ref,$file) || die "ERROR: Can not open file for reading: $file\n";
  my $sec="";
  while(my $l=<$ref>)
  {
    chomp $l;
    if ($l=~/^\s*(#.*|)$/o){next;}
    if ($l=~/^\s*\[\s*([^\s\]]+)\s*]\s*/o)
    {
      $sec=lc($1);
      if (($sec ne "config") && (!exists $cache->{pos}{$sec})){$cache->{pos}{$sec}=[];$cache->{nag}{$sec}=[];}
    }
    elsif ($sec ne "")
    {
      my ($pks,$deps)=split(":",$l,2);
      if ($sec eq "config")
      {
        $pks=~s/\s+//g; $deps=~s/\s+//g;
	if (!exists $config->{$pks}){$config->{$pks}={};}
	foreach my $p (split(/,/,$deps)){$config->{$pks}{$p}=1;}
	next;
      }
      my $type="pos";
      $pks=~s/^\s*//o; $pks=~s/\s*$//o;
      if ($pks=~/^!(.+)/o){$pks=$1; $type="nag";}
      foreach my $dep (split ",",$deps)
      {
        $dep=~s/^\s*//o;$dep=~s/\s*$//o;
	push @{$cache->{$type}{$sec}},[$pks,$dep];
      }
    }
  }
  close($ref);
  return ($cache, $config);
}

##########################################################################
# Read Tools and Project cache of all externals and SCRAM-based projects #
##########################################################################
sub readBuildFileCache()
{
  my ($release,$arch,$cache)=@_;
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
      foreach my $d (keys %{$cache->{Caches}{$t}{BUILDTREE}})
      {
        if ($ignore_test && ($cache->{Caches}{$t}{BUILDTREE}{$d}{CLASS} eq "TEST")){next;}
        &readPkgInfo($d,$t,$cache);
      }
      if ($t eq "self")
      {
        my $releaseTop=&SCRAMGenUtils::getFromEnvironmentFile("RELEASETOP",$release);
	if ($releaseTop ne "")
	{
	  $cache->{Caches}{$t}=&SCRAMGenUtils::readCache("${releaseTop}/.SCRAM/${arch}/ProjectCache.db.gz");
	  foreach my $d (keys %{$cache->{Caches}{$t}{BUILDTREE}})
	  {
	    if ($ignore_test && ($cache->{Caches}{$t}{BUILDTREE}{$d}{CLASS} eq "TEST")){next;}
            &readPkgInfo($d,$t,$cache);
	  }
	}
      }
      delete $cache->{Caches}{$t};
    }
    else{&readToolsInfo(lc($t),$cache);}
  }
  delete $cache->{Caches};
}

sub initCache()
{
  my ($release,$arch)=@_;
  my $cache={};
  if (!defined $source_only){&readBuildFileCache($release,$arch,$cache);}
  &updateSourceDeps($cache);
  foreach my $p (keys %{$cache->{PRODS}}){&allDeps($p,$cache);}
  foreach my $k (keys %$cache){if ($k!~/^(DEPS|TOOLS|PACKS)$/){delete $cache->{$k};}}
  return $cache;
}

sub allDeps()
{
  my ($prod,$cache)=@_;
  if(exists $cache->{DEPS}{$prod}){return;}
  $cache->{DEPS}{$prod}{ALL}={};
  $cache->{DEPS}{$prod}{DIRECT}={};
  $cache->{DEPS}{$prod}{PACK}=$cache->{PRODS}{$prod}{PACK};
  $cache->{DEPS}{$prod}{TOOL}=$cache->{PRODS}{$prod}{TOOL};
  foreach my $d (keys %{$cache->{PRODS}{$prod}{DEPS}})
  {
    if (!exists $cache->{PACKS}{$d})
    {
      if (!exists $cache->{WARNS}{$d})
      {
        print "WARNING: UNKNOWN PACKAGE $d (DEPS OF $prod)\n";
        $cache->{WARNS}{$d}=1;
      }
    }
    else
    {
      my $p=$cache->{PACKS}{$d}[0];
      &allDeps($p,$cache);
      $cache->{DEPS}{$prod}{ALL}{$d}=1;
      $cache->{DEPS}{$prod}{DIRECT}{$d}=1;
      foreach my $u (keys %{$cache->{DEPS}{$p}{ALL}}){$cache->{DEPS}{$prod}{ALL}{$u}=1;}
    }
  }
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
  $d=~s/^LCG\///;
  if($class=~/^(LIBRARY|CLASSLIB|SEAL_PLATFORM)$/o){&addProd($t,$name,dirname($d),$cache,$c1);}
  elsif($class eq "PACKAGE"){&addProd($t,$name,$d,$cache,$c1);}
  elsif($class=~/^(TEST|BIN|PLUGINS|BINARY)$/o){&addProds($t,$d,$cache,$c1);}
  elsif($class=~/^(PYTHON|SUBSYSTEM|DATA_INSTALL|SCRIPTS|PROJECT|IVS)$/o){return;}
  else{print STDERR "WARNING: UNKNOW TYPE $class in $t/$d\n";}
}

sub addProd()
{
  my ($tool,$prod,$pack,$cache,$c)=@_;
  if (exists $cache->{PRODS}{$prod}){return;}
  if (defined $c)
  {
    &initProd($cache,$prod,$tool,$pack);
    if(exists $c->{USE}){&addDirectDeps($c->{USE},$cache->{PRODS}{$prod}{DEPS},$cache);}
    if((exists $c->{EXPORT}) && (exists $c->{EXPORT}{USE})){&addDirectDeps($c->{EXPORT}{USE},$cache->{PRODS}{$prod}{DEPS},$cache);}
  }
}

sub addProds()
{
  my ($tool,$pack,$cache,$c)=@_;
  if (exists $cache->{PACKS}{$pack}){return;}
  if (!defined $c){return;}
  my $cuse=[];
  if (exists $c->{USE}){$cuse=$c->{USE};}
  if (exists $c->{BUILDPRODUCTS})
  {
    foreach my $t (keys %{$c->{BUILDPRODUCTS}})
    {
      foreach my $p (keys %{$c->{BUILDPRODUCTS}{$t}})
      {
        my $c1=$c->{BUILDPRODUCTS}{$t}{$p}{content};
	&addProd($tool,$p,$pack,$cache,$c->{BUILDPRODUCTS}{$t}{$p}{content});
	&addDirectDeps($cuse,$cache->{PRODS}{$p}{DEPS},$cache);
      }
    }
  }
}

sub addDirectDeps()
{
  my ($uses,$c,$cache)=@_;
  foreach my $u (@{$uses})
  {
    $u=~s/^LCG\///;
    my $p=$u;
    my $t=lc($u);
    if (exists $cache->{Caches}{TOOLS}{SETUP}{$t}){$p=$t;}
    $c->{$p}=1;
  }
}

sub readToolsInfo()
{
  my ($t,$cache)=@_;
  if (exists $cache->{PRODS}{$t}){return;}
  $cache->{TOOLS}{$t}=1;
  $cache->{PRODS}{$t}{TOOL}=$t;
  $cache->{PRODS}{$t}{PACK}=$t;
  $cache->{PRODS}{$t}{DEPS}={};
  $cache->{PACKS}{$t}[0]=$t;
  if (exists $cache->{Caches}{TOOLS}{SETUP}{$t}{USE})
  {&addDirectDeps($cache->{Caches}{TOOLS}{SETUP}{$t}{USE},$cache->{PRODS}{$t}{DEPS},$cache);}
}

sub product2Package()
{
  my ($p,$cache)=@_;
  if(exists $cache->{PROD2PACK}{$p}){return $cache->{PROD2PACK}{$p};}
  my $pk="";
  if (exists $cache->{DEPS}{$p}){$pk=$cache->{DEPS}{$p}{PACK};}
  $cache->{PROD2PACK}{$p}=$pk;
  return $pk;
}

sub updateSourceDeps ()
{
  my ($cache)=@_;
  $cache->{FILES}={};
  if (!&readUses($cache,$ENV{CMSSW_BASE})){&readCompilerDeps($cache,$ENV{CMSSW_BASE},$ENV{CMSSW_RELEASE_BASE});}
  if ($ENV{CMSSW_RELEASE_BASE}){&readUses($cache,$ENV{CMSSW_RELEASE_BASE});}
  delete $cache->{FILES};
}

sub initProd ()
{
  my ($cache,$prod,$tool,$pack)=@_;
  if (exists $cache->{PRODS}{$prod}){return;}
  $cache->{PRODS}{$prod}{TOOL}=$tool;
  $cache->{PRODS}{$prod}{PACK}=$pack;
  $cache->{PRODS}{$prod}{DEPS}={};
  if (!exists $cache->{PACKS}{$pack}){$cache->{PACKS}{$pack}=[];}
  push @{$cache->{PACKS}{$pack}},$prod;
}

sub readCompilerDeps()
{
  my ($cache,$path,$rbase)=@_;
  if (!-d "${path}/tmp/$ENV{SCRAM_ARCH}/src"){return;}
  my $ref;
  foreach my $file (`find ${path}/tmp/$ENV{SCRAM_ARCH}/src -name '*.dep' -type f`)
  {
    chomp $file;
    if ($ignore_test && $file=~/\/src\/$ignore_test/){next;}
    my $srcpack=""; my $prod="";
    open($ref,$file) || die "Can not open file for reading: $file\n";
    while(my $line=<$ref>)
    {
      chomp $line;
      if ($line=~/:/o){$prod=""; next;}
      if ($line=~/^\s*$/o){next;}
      if ($rbase){$line=~s/^\s*${rbase}\///o;}
      if ($line=~/^\//){next;}
      $line=~s/^\s*src\///o; $line=~s/\s*\\//o;
      my $pack=$line; $pack=~s/([^\/]+\/[^\/]+)\/.+/$1/;
      if ($prod eq "")
      {
        $srcpack=$pack;
        $prod=$line;
        $cache->{FILES}{$prod}=1;
        &initProd($cache,$prod,"self",$prod);
        next;
      }
      elsif ($srcpack ne $pack){&initProd($cache,$line,"self",$line); $cache->{PRODS}{$prod}{DEPS}{$line}=1;}
    }
    close($ref);
  }
}

sub readUses()
{
  my ($cache,$path)=@_;
  my $uses="${path}/etc/dependencies/uses.out.gz";
  if (-f $uses)
  {
    my $ref;
    open($ref,"gunzip -c $uses |") || die "Can not open file for reading: $uses\n";
    while  (my $line=<$ref>)
    { 
      chomp($line);
      my ($src,$line)=split(' ',$line,2);
      if ($src ne "")
      {
        if ($ignore_test && $src=~/^$ignore_test/){next;}
        if (exists $cache->{FILES}{$src}){next;}
        my $pack=$src;
        $pack=~s/([^\/]+\/[^\/]+)\/.+/$1/;
        my $prod=$src;
        &initProd($cache,$prod,"self",$prod);
        foreach my $p (split(' ',$line))
        {
          my $x=$p;
          $p=~s/([^\/]+\/[^\/]+)\/.+/$1/;
          if ($p ne $pack){&initProd($cache,$x,"self",$x); $cache->{PRODS}{$prod}{DEPS}{$x}=1;}
        }
      }
    }
    close($ref);
    return 1;
  }
  return 0;
}
