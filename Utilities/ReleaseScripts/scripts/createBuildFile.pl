#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use Getopt::Long;
use SCRAMGenUtils;
use FileHandle;
use IPC::Open2;

$|=1;

#get the command-line options
my $xmlbf="BuildFile";
if(&GetOptions(
	       "--dir=s",\$dir,
	       "--order=s",\@packs,
	       "--config=s",\$configfile,
	       "--redo=s",\@prod,
	       "--detail",\$detail,
	       "--xml",\$xml,
	       "--help",\$help,
              ) eq ""){print STDERR "#Wrong arguments.\n"; &usage_msg();}

if(defined $help){&usage_msg();}
if(defined $detail){$detail="--detail";}
if(defined $xml){$xml=1; $xmlbf="BuildFile.xml";}
else{$xml=0;}
if(defined $configfile){$configfile="--config $configfile";}

my $sdir=dirname($0);
my $pwd=`/bin/pwd`; chomp $pwd; $pwd=&SCRAMGenUtils::fixPath($pwd);

if((!defined $dir) || ($dir=~/^\s*$/)){print "ERROR: Missing SCRAM-based project release path.\n"; exit 1;}
if($dir!~/^\//){$dir="${pwd}/${dir}";}
$dir=&SCRAMGenUtils::fixPath($dir);
my $release=&SCRAMGenUtils::scramReleaseTop($dir);
if(!-d "${release}/.SCRAM"){print STDERR "ERROR: $dir is not under a SCRAM-based project.\n"; exit 1;}
my $scram_ver=&SCRAMGenUtils::scramVersion($release);
if($scram_ver=~/^V1_0_/)
{
  print STDERR "ERROR: This version of script will only work with SCRAM versions V1_1* and above.\n";
  print STDERR "\"$release\" is based on SCRAM version $scram_ver.\n";
  exit 1;
}
my $project=lc(&SCRAMGenUtils::getFromEnvironmentFile("SCRAM_PROJECTNAME",$release));
if($project eq ""){print STDERR "ERROR: Can not find SCRAM_PROJECTNAME in ${release}.SCRAM/Environment file.\n"; exit 1;}

my $devarea="${release}/tmp/AutoBuildFile";
if($pwd!~/^$release(\/.*|)$/){$devarea="${pwd}/AutoBuildFile";}
if(!-f "${devarea}/.SCRAM/Environment")
{
  $devarea=&SCRAMGenUtils::createTmpReleaseArea($release,1,$devarea);
  system("cd $devarea; $SCRAMGenUtils::SCRAM_CMD b -r echo_CXX 2>&1 > /dev/null");
}
&SCRAMGenUtils::init ($devarea);

my $scramarch=&SCRAMGenUtils::getScramArch();
my $cache={};
my $pcache={};
my $projcache={};
my $cachedir="${devarea}/bfcache/${scramarch}";
my $bferrordir="${cachedir}/bferrordir";
my $cfile="${cachedir}/product.cache";
my $pcfile="${cachedir}/project.cache";
my $inccachefile="${cachedir}/include_chace.txt";

if(!-d "$cachedir"){system("mkdir -p $cachedir");}
if(-f $inccachefile){system("rm -f $inccachefile");}
&initCache($dir);
print "DONE\n";

foreach my $p (@prod)
{
  if($p=~/^all$/i)
  {
    foreach $x (keys %{$cache}){delete $cache->{$x}{done};}
    @prod=();
    last;
  }
  elsif(exists $cache->{$p}){delete $cache->{$p}{done};}
}
foreach my $p (keys %$cache)
{
  if((exists $pcache->{prod}{$p}) && (exists $pcache->{prod}{$p}{dir}))
  {
    my $d=&SCRAMGenUtils::fixPath($pcache->{prod}{$p}{dir});
    my $d1="${release}/src/${d}";
    if($d1!~/^$dir(\/.*|)$/){$cache->{$p}{skip}=1;}
    else{delete $cache->{$p}{skip};}
  }
}
&SCRAMGenUtils::writeHashCache($cache,$cfile);

foreach my $p (@packs)
{
  foreach my $p1 (split /\s*,\s*/,$p)
  {
    $p1=&run_func("safename",$project,"${release}/src/${p1}");
    print "Working on $p1\n";
    if($p1){&processProd($p1);}
  }
}
if(scalar(@prod)==0)
{foreach my $f (keys %$cache){&processProd($f);}}
else{foreach my $p (@prod){&processProd($p);}}
exit 0;

sub initCache ()
{
  my $dir=shift || "";
  if((-f $cfile) && (-f $pcfile))
  {
    $cache=&SCRAMGenUtils::readHashCache($cfile);
    $pcache=&SCRAMGenUtils::readHashCache($pcfile);
    foreach my $p (keys %$cache)
    {if((exists $cache->{$p}{done}) && ($cache->{$p}{done}==0)){delete $cache->{$p}{done};}}
  }
  else
  {
    my $cf=&SCRAMGenUtils::fixCacheFileName("${release}/.SCRAM/${scramarch}/ProjectCache.db");
    if(-f $cf)
    {
      $projcache=&SCRAMGenUtils::readCache($cf);
      foreach my $d (keys %{$projcache->{BUILDTREE}}){&updateProd($d);}
      $projcache={};
      &SCRAMGenUtils::writeHashCache($cache,$cfile);
      &SCRAMGenUtils::writeHashCache($pcache,$pcfile);
    }
    else{print STDERR "$cf file does not exists. Script need this to be available.\n"; exit 1;}
  }
}

sub updateProd ()
{
  my $p=shift;
  if(exists $projcache->{BUILDTREE}{$p}{CLASS} && (exists $projcache->{BUILDTREE}{$p}{RAWDATA}{content}))
  {
    my $suffix=$projcache->{BUILDTREE}{$p}{SUFFIX};
    if($suffix ne ""){return 0;}
    my $class=$projcache->{BUILDTREE}{$p}{CLASS};
    my $c=$projcache->{BUILDTREE}{$p}{RAWDATA}{content};
    if($class eq "LIBRARY"){return &addPack($c,dirname($p));}
    elsif($class eq "PACKAGE"){return &addPack($c,$p);}
    elsif($class=~/^(TEST|BIN|PLUGINS|BINARY)$/){return &addProds($c,$p);}
  }
  return 0;
}

sub addProds ()
{
  my $c=shift;
  my $p=shift;
  if(exists $pcache->{dir}{$p}){return 1;}
  $pcache->{dir}{$p}=1;
  my $bf1="${release}/src/${p}/BuildFile.xml";
  if(!-f $bf1){$bf1="${release}/src/${p}/BuildFile";}
  if(!-f $bf1){return 0}
  &addProdDep($c,$p,1);
  my $bf=undef;
  foreach my $t (keys %{$c->{BUILDPRODUCTS}})
  {
    foreach my $prod (keys %{$c->{BUILDPRODUCTS}{$t}})
    {
      if($prod=~/^\s*$/){next;}
      my $xname=basename($prod);
      my $name=$xname;
      my $type=lc($t);
      if(exists $pcache->{prod}{$name})
      {
	$name="DPN_${xname}";
	my $i=0;
	while(exists $pcache->{prod}{$name}){$name="DPN${i}_${xname}";$i++;}
	my $pbf=$pcache->{prod}{$xname}{dir}."/".$pcache->{prod}{$xname}{bf};
	print STDERR "WARNING: \"$bf1\" has a product \"$xname\" which is already defined in \"$pbf\". Going to change it to \"$name\".\n";
      }
      $pcache->{prod}{$name}{dir}=$p;
      $pcache->{prod}{$name}{type}=$type;
      $pcache->{prod}{$name}{bf}=basename($bf1);
      if (!defined $bf){$bf=&SCRAMGenUtils::readBuildFile($bf1);}
      if((exists $bf->{$type}{$xname}) && (exists $bf->{$type}{$xname}{file}))
      {
        my $files="";
	foreach my $f (@{$bf->{$type}{$xname}{file}}){$files.="$f,";}
	$files=~s/\,$//;
	if($files ne ""){$pcache->{prod}{$name}{file}=$files;}
      }
      if(exists $c->{BUILDPRODUCTS}{$t}{$prod}{content})
      {&addProdDep($c->{BUILDPRODUCTS}{$t}{$prod}{content},$name,1);}
      $cache->{$name}={};
    }
  }
}

sub addPack ()
{
  my $c=shift;
  my $p=shift;
  if(exists $pcache->{dir}{$p}){return 1;}
  $pcache->{dir}{$p}=1;
  my $prod=&run_func("safename",$project,"${release}/src/${p}");
  if($prod eq ""){print STDERR "ERROR: Script is not ready for $project SCRAM-based project.\n"; exit 1;}
  my $bf="${release}/src/${p}/BuildFile.xml";
  if(!-f $bf){$bf="${release}/src/${p}/BuildFile";}
  if(!-f $bf){return 0;}
  $pcache->{prod}{$prod}{dir}=$p;
  $pcache->{prod}{$prod}{bf}=basename($bf);
  $cache->{$prod}={};
  &addProdDep($c,$p);
  return 1;
}

sub addProdDep ()
{
  my $c=shift;
  my $p=shift;
  my $d=shift || 0;
  foreach my $u (@{$c->{USE}})
  {
    if(($u ne "") && (!exists $pcache->{deps}{$p}{$u}))
    {if(&updateProd($u)){&addInDirectPackDep($p,$u,$d,3);}}
  }
}

sub addInDirectPackDep ()
{ 
  my $p=shift;
  my $u=shift;
  my $d=shift || 0;
  my $level=shift || return;
  $pcache->{deps}{$p}{$u}=$level;
  if(!$d){$pcache->{rdeps}{$u}{$p}=$level;}
  if($level==1){return;}
  if(exists $pcache->{deps}{$u})
  {
    foreach my $x (keys %{$pcache->{deps}{$u}})
    {&addInDirectPackDep($p,$x,$d,$level-1);}
  }
}

sub processProd ()
{
  my $prod=shift;
  if ((!exists $cache->{$prod}) || (exists $cache->{$prod}{skip}) || (exists $cache->{$prod}{done})){return;}
  $cache->{$prod}{done}=0;
  my $pack=$pcache->{prod}{$prod}{dir};
  my $bfn=$pcache->{prod}{$prod}{bf};
  if ($pack eq ""){return 0;}
  if(exists $pcache->{deps}{$pack})
  {
    my %luse=();
    foreach my $u (keys %{$pcache->{deps}{$pack}}){$luse{$u}=1;}
    if(exists $pcache->{deps}{$prod})
    {foreach my $u (keys %{$pcache->{deps}{$prod}}){$luse{$u}=1;}}
    foreach my $u (keys %luse)
    {
      if(exists $pcache->{dir}{$u})
      {
        my $u1=&run_func("safename",$project,"${release}/src/${u}");
        if($u1 ne ""){&processProd($u1);}
      }
    }
  }
  my $nexport="${cachedir}/${prod}no-export";
  if(exists $pcache->{rdeps}{$pack})
  {
    my @nuse=();
    if(exists $pcache->{rdeps}{$pack})
    {foreach my $u (keys %{$pcache->{rdeps}{$pack}}){push @nuse,$u;}}
    if(scalar(@nuse) > 0)
    {
      my $nfile;
      open($nfile, ">$nexport") || die "Can not open file \"$nexport\" for writing.";
      foreach my $u (@nuse){print $nfile "$u\n";}
      close($nfile);
    }
  }
  elsif(-f "$nexport"){system("rm -f $nexport");}
  my $bfsrcdir="${devarea}/src/${pack}";
  if((exists $pcache->{prod}{$prod}{type}) || (exists $pcache->{prod}{$prod}{file})){$bfsrcdir="";}
  my $bfdir="${devarea}/newBuildFile/src/${pack}";
  system("mkdir -p $bfdir $bfsrcdir");
  my $nfile="${bfdir}/${bfn}.auto";
  my $ptype="";my $pname="";my $pfiles=""; my $xargs="";
  if ($xml){$nfile="${bfdir}/${xmlbf}.auto"; $xargs.=" --xml";}
  if(exists $pcache->{prod}{$prod}{type})
  {
    $ptype="--prodtype ".$pcache->{prod}{$prod}{type};
    $pname="--prodname $prod --files '".$pcache->{prod}{$prod}{file}."'";
    $nfile="${bfdir}/${prod}${bfn}.auto";
    if ($xml){$nfile="${bfdir}/${prod}${xmlbf}.auto";}
  }
  my $cmd="${sdir}/_createBuildFile.pl $xargs $configfile --dir ${release}/src/${pack} --tmprelease $devarea --buildfile $nfile $ptype $pname $pfiles $detail --chksym";
  print "$cmd\n";
  my $reader; my $writer;
  my $pid=open2($reader, $writer,"$cmd 2>&1");
  $writer->autoflush();
  while(my $line=<$reader>)
  {
    chomp $line;
    print "$line\n";
    if($line=~/^PLEASE_PROCESS_FIRST:(.+?)$/)
    {
      my $u=$1;
      my $u1=&run_func("safename",$project,"${release}/src/${u}");
      if(($u1 ne "") && (exists $cache->{$u1}))
      {
        print "New Dependency added: $pack => $u\n";
	$pcache->{deps}{$pack}{$u}=1;
	$pcache->{rdeps}{$u}{$pack}=1;
	foreach my $x (keys %{$pcache->{rdeps}{$pack}}){$pcache->{rdeps}{$u}{$x}=1;}
	&processProd($u1);
      }
      print $writer "PROCESSED:$u\n";
    }
  }
  close($reader); close($writer);
  waitpid $pid,0;
  if(-f $nfile)
  {
    $cache->{$prod}{done}=1;
    &SCRAMGenUtils::writeHashCache($cache,$cfile);
  }
  if(-f "$nexport"){system("rm -f $nexport");}
  print "##########################################################################\n";
}

#####################################
# Run a tool specific func
####################################
sub run_func ()
{
  my $func=shift || return "";
  my $tool=shift || return "";
  if($tool eq "self"){$tool=$project;}
  $tool=lc($tool);
  $func.="_${tool}";
  if(exists &$func){return &$func(@_);}
  return "";
}
#############################################
# generating library safe name for a package
#############################################
sub safename_pool ()
{return "lcg_".basename(shift);}
sub safename_seal ()
{return "lcg_".basename(shift);}
sub safename_coral ()
{return "lcg_".basename(shift);}

sub safename_ignominy ()
{return &safename_cms1(shift);}
sub safename_iguana ()
{return &safename_cms1(shift);}
sub safename_cmssw ()
{return &safename_cms2(shift);}

sub safename_cms1 ()
{
  my $dir=shift;
  if($dir=~/^${release}\/src\/([^\/]+?)\/([^\/]+)$/){return "${2}";}
  else{return "";}
}
sub safename_cms2 ()
{
  my $dir=shift;
  if($dir=~/^${release}\/src\/([^\/]+?)\/([^\/]+)$/){return "${1}${2}";}
  else{return "";}
}

sub usage_msg()
{
  my $script=basename($0);
  print "Usage: $script --dir <path> [--xml] [--order <pack>[--order <pack> [...]]] [--detail]\n",
        "        [--redo <prod|all> [--redo <prod> [...]]] [--config <file>]\n\n",
        "e.g.\n",
        "  $script --dir /path/to/a/project/release/area\n\n",
        "--dir <path>    Directory path for which you want to generate BuildFile(s).\n",
	"--order <pack>  Packages order in which script should process them\n",
	"--redo <pack>   Re-process an already done package\n",
	"--config <file> Extra configuration file\n",
        "--detail        To get a detail processing log info\n",
	"--xml           To generate xml BuildFiles i.e. BuildFile.xml.auto\n\n",
        "This script will generate all the BuildFile(s) for your <path>. Generated BuildFile.auto will be available under\n",
	"AutoBuildFile/newBuildFile if you are not in a dev area otherwise in <devarea>/tmp/AutoBuildFile/newBuildFile.\n",
	"Do not forget to run \"mergeProdBuildFiles.pl --dir <dir>/AutoBuildFile/newBuildFile\n",
	"after running this script so that it can merge multiple products BuuildFiles in to one.\n",
	"Once BuildFile.auto are generated then you can copy all newBuilsFilew/*/BuildFile.auto in to your src/*/BuildFile.\n";
  exit 0;
}
