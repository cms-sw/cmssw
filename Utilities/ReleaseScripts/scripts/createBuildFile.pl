#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use Getopt::Long;
use SCRAMGenUtils;
use FileHandle;
use IPC::Open2;

$|=1;

#get the command-line options
if(&GetOptions(
	       "--dir=s",\$dir,
	       "--order=s",\@packs,
	       "--config=s",\$configfile,
	       "--redo=s",\@prod,
	       "--jobs=i",\$jobs,
	       "--detail",\$detail,
	       "--xml",\$xml,
	       "--merge",\$merge,
	       "--help",\$help,
              ) eq ""){print STDERR "#Wrong arguments.\n"; &usage_msg();}

my $sdir=dirname($0);
my $pwd=`/bin/pwd`; chomp $pwd; $pwd=&SCRAMGenUtils::fixPath($pwd);
if ($sdir!~/^\//){$sdir=&SCRAMGenUtils::fixPath("${pwd}/${sdir}");}

my $xconfig="";
my $xjobs=10;
if(defined $help){&usage_msg();}
if(defined $detail){$detail="--detail";}
if(defined $xml){$xml=1;}
else{$xml=0;}
if(defined $jobs){$xjobs=$jobs; $jobs="--jobs $jobs";}
if(defined $configfile){$xconfig=$configfile;$configfile="--config $xconfig";}
elsif(-f "${sdir}/createBuildFile.conf"){$xconfig="${sdir}/createBuildFile.conf";$configfile="--config $xconfig";}

if((!defined $dir) || ($dir=~/^\s*$/)){print "ERROR: Missing SCRAM-based project release path.\n"; exit 1;}
if($dir!~/^\//){$dir="${pwd}/${dir}";}
$dir=&SCRAMGenUtils::fixPath($dir);
my $release=&SCRAMGenUtils::scramReleaseTop($dir);
if($release eq ""){print STDERR "ERROR: $dir is not under a SCRAM-based project.\n"; exit 1;}
&SCRAMGenUtils::init ($release);
my $scram_ver=&SCRAMGenUtils::scramVersion();
if($scram_ver=~/^V1_0_/)
{
  print STDERR "ERROR: This version of script will only work with SCRAM versions V1_1* and above.\n";
  print STDERR "\"$release\" is based on SCRAM version $scram_ver.\n";
  exit 1;
}
my $project=lc(&SCRAMGenUtils::getFromEnvironmentFile("SCRAM_PROJECTNAME",$release));
my $releasetop=&SCRAMGenUtils::getFromEnvironmentFile("RELEASETOP",$release);
if ($releasetop ne ""){system($SCRAMGenUtils::SCRAM_CMD." build -r echo_CXX 2>&1 >/dev/null");}
if($project eq ""){print STDERR "ERROR: Can not find SCRAM_PROJECTNAME in ${release}.SCRAM/Environment file.\n"; exit 1;}

my $scramarch=&SCRAMGenUtils::getScramArch();
my $tmpdir="${release}/tmp/AutoBuildFile";
my $symboldir="${release}/tmp/Symbols/${scramarch}";
if($pwd!~/^$release(\/.*|)$/)
{
  $tmpdir="${pwd}/AutoBuildFile";
  $symboldir="${tmpdir}/Symbols/${scramarch}";
}

my $data={};
my $cache={};
my $pcache={};
my $projcache={};
my $locprojcache={};
my $cachedir="${tmpdir}/${scramarch}";
my $depsdir="${cachedir}/new_pack_deps";
my $bferrordir="${cachedir}/bferrordir";
my $cfile="${cachedir}/product.cache";
my $pcfile="${cachedir}/project.cache";
my $xmlbf="BuildFile.xml";

&SCRAMGenUtils::updateConfigFileData ($xconfig,$data);

system("mkdir -p $symboldir $depsdir");
&initCache($dir);

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
    my $d="${release}/src/".&SCRAMGenUtils::fixPath($pcache->{prod}{$p}{dir});
    if (!-d $d){$cache->{$p}{skip}=1;next;}
    if($d!~/^$dir(\/.*|)$/){$cache->{$p}{skip}=1;}
    else{delete $cache->{$p}{skip};}
  }
}
&SCRAMGenUtils::writeHashCache($cache,$cfile);

foreach my $p (@packs)
{
  foreach my $p1 (split /\s*,\s*/,$p)
  {
    if (exists $pcache->{packmap}{$p1}){$p1=$pcache->{packmap}{$p1};}
    else{$p1=&run_func("safename",$project,"${release}/src/${p1}");}
    if($p1){&processProd($p1);}
  }
}
if(scalar(@prod)==0){foreach my $f (keys %$cache){&processProd($f);}}
else{foreach my $p (@prod){&processProd($p);}}
if(!defined $merge){system("${sdir}/mergeProdBuildFiles.pl --release $release --dir ${tmpdir}/newBuildFile/src");}
print STDERR "Path of newly generated BuildFile(s): ${tmpdir}/newBuildFile/src\n";
exit 0;

sub updateRevDeps ()
{
  my $pack=shift;
  if (!defined $pack)
  {
    foreach my $dir (keys %{$pcache->{dir}}){&updateRevDeps($dir);}
    foreach my $dir (keys %{$pcache->{dir}}){delete $pcache->{dir}{$dir}{rdepsdone};delete $pcache->{dir}{$dir}{first};}
  }
  elsif ((!exists $pcache->{dir}{$pack}{rdepsdone}) && (-d "${release}/src/${pack}"))
  {
    $pcache->{dir}{$pack}{rdeps}={};
    $pcache->{dir}{$pack}{rdepsdone}=1;
    if (exists $pcache->{dir}{$pack}{deps})
    {
      foreach my $dep (keys %{$pcache->{dir}{$pack}{deps}})
      {
        &updateRevDeps($dep);
	if (exists $pcache->{dir}{$dep}{rdeps}){$pcache->{dir}{$dep}{rdeps}{$pack}=1;}
      }
    }
  }
}

sub initCache ()
{
  my $dir=shift || "";
  if((-f $cfile) && (-f $pcfile) && (-f "${symboldir}.SymbolsCache.db"))
  {
    $cache=&SCRAMGenUtils::readHashCache($cfile);
    $pcache=&SCRAMGenUtils::readHashCache($pcfile);
    if (-d $depsdir)
    {
      my $ref;
      my $dirty=0;
      opendir ($ref,$depsdir) || die "Can not open directory for reading:$depsdir\n";
      foreach my $prod (readdir($ref))
      {
        if (($prod=~/^\./) || (!exists $pcache->{prod}{$prod})){next;}
	my $pk=$pcache->{prod}{$prod}{dir};
	$prod="${depsdir}/${prod}";
	my $c=&SCRAMGenUtils::readHashCache($prod);
	delete $c->{rdeps};delete $c->{rdepsdone};delete $c->{first};
	if (!exists $pcache->{dir}{$pk}{first}){$pcache->{dir}{$pk}=$c;$pcache->{dir}{$pk}{first}=1;}
	else
	{
	  foreach my $k (keys %$c)
	  {
	    my $v=$c->{$k};
	    if (ref($v) eq "HASH"){foreach my $k1 (keys %$v){$pcache->{dir}{$pk}{$k}{$k1}=$v->{$k1};}}
	    else{$pcache->{dir}{$pk}{$k}=$v;}
	  }
	}
	$dirty=1;
	unlink $prod;
      }
      closedir($ref);
      if($dirty)
      {
        &updateRevDeps();
	&SCRAMGenUtils::writeHashCache($pcache,$pcfile);
      }
    }
    $data->{ALL_SYMBOLS}=&SCRAMGenUtils::readHashCache("${symboldir}.SymbolsCache.db");
    foreach my $p (keys %$cache){if((exists $cache->{$p}{done}) && ($cache->{$p}{done}==0)){delete $cache->{$p}{done};}}
  }
  else
  {
    print STDERR "Initializing project caches and shared libs symbols information. It will take few mins.\n";
    my $tcache={};
    $cf=&SCRAMGenUtils::fixCacheFileName("${release}/.SCRAM/${scramarch}/ToolCache.db");
    if (-f $cf)
    {
      $tcache=&SCRAMGenUtils::readCache($cf);
      &addToolDep($tcache);
    }
    else{print STDERR "$cf file does not exists. Script need this to be available.\n"; exit 1;}
    
    if ($releasetop ne "")
    {
      my $cf=&SCRAMGenUtils::fixCacheFileName("${releasetop}/.SCRAM/${scramarch}/ProjectCache.db");
      if (-f $cf){$projcache=&SCRAMGenUtils::readCache($cf);}
      else{print STDERR "$cf file does not exists. Script need this to be available.\n"; exit 1;}
    }
    
    my $cf=&SCRAMGenUtils::fixCacheFileName("${release}/.SCRAM/${scramarch}/ProjectCache.db");
    if (!-f $cf){system("cd $release; scram b -r echo_CXX xfast 2>&1 >/dev/null");}
    $locprojcache=&SCRAMGenUtils::readCache($cf);
    foreach my $d (keys %{$locprojcache->{BUILDTREE}}){$projcache->{BUILDTREE}{$d}=$locprojcache->{BUILDTREE}{$d};}
    &addPackDep ($projcache);
    &updateRevDeps();
    
    &SCRAMGenUtils::scramToolSymbolCache($tcache,"self",$symboldir,$xjobs,$pcache->{packmap});
    &SCRAMGenUtils::waitForChild();
    print STDERR "\n";
    $data->{ALL_SYMBOLS}=&SCRAMGenUtils::mergeSymbols($symboldir,"${symboldir}.SymbolsCache.db");
    foreach my $d (reverse sort keys %{$projcache->{BUILDTREE}}){&updateProd($d);}
    $projcache={};
    $locprojcache={};
    &SCRAMGenUtils::writeHashCache($cache,$cfile);
    &SCRAMGenUtils::writeHashCache($pcache,$pcfile);
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
    if($class=~/^(LIBRARY|CLASSLIB|SEAL_PLATFORM)$/){return &addPack($projcache->{BUILDTREE}{$p}{NAME},dirname($p));}
    elsif($class eq "PACKAGE"){return &addPack($projcache->{BUILDTREE}{$p}{NAME},$p);}
    elsif($class=~/^(TEST|BIN|PLUGINS|BINARY)$/){return &addProds($c,$p);}
  }
  return 0;
}

sub addProds ()
{
  my $c=shift;
  my $p=shift;
  if (exists $pcache->{dir}{$p}{bf}){return;}
  my $bf1=&getBuildFile($p);
  $pcache->{dir}{$p}{bf}=basename($bf1);
  if($bf1 eq ""){return 0}
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
	my $d1=$pcache->{prod}{$xname}{dir};
	my $pbf="${d1}/".$pcache->{dir}{$d1}{bf};
	print STDERR "WARNING: \"$bf1\" has a product \"$xname\" which is already defined in \"$pbf\". Going to change it to \"$name\".\n";
      }
      $pcache->{prod}{$name}{dir}=$p;
      $pcache->{prod}{$name}{type}=$type;
      if (!defined $bf){$bf=&SCRAMGenUtils::readBuildFile($bf1);}
      if((exists $bf->{$type}{$xname}) && (exists $bf->{$type}{$xname}{file}))
      {
        my $files="";
	foreach my $f (@{$bf->{$type}{$xname}{file}}){$files.="$f,";}
	$files=~s/\,$//;
	if($files ne ""){$pcache->{prod}{$name}{file}=$files;}
      }
      $cache->{$name}={};
    }
  }
}

sub getBuildFile ()
{
  my $pack=shift;
  my $bf="${release}/src/${pack}/BuildFile.xml";
  if(!-f $bf){$bf="${release}/src/${pack}/BuildFile";}
  if(!-f $bf){$bf="";}
  return $bf;
}

sub addPack ()
{
  my $prod=shift;
  my $dir=shift;
  if (exists $pcache->{dir}{$dir}{bf}){return;}
  $pcache->{dir}{$dir}{bf}=basename(&getBuildFile($dir));
  $pcache->{prod}{$prod}{dir}=$dir;
  $cache->{$prod}={};
  return 1;
}

sub addToolDep ()
{
  my $tools=shift;
  my $t=shift;
  if (!defined $t)
  {
    foreach $t (&SCRAMGenUtils::getOrderedTools($tools)){&addToolDep($tools,$t);}
    return;
  }
  if (exists $pcache->{tools}{$t}{deps}){return;}
  $pcache->{tools}{$t}{deps}={};
  my $c=$pcache->{tools}{$t}{deps};
  if (($t ne "self") && ((exists $tools->{SETUP}{$t}{SCRAM_PROJECT}) && ($tools->{SETUP}{$t}{SCRAM_PROJECT}==1)))
  {
    my $base=uc("${t}_BASE");
    if(exists $tools->{SETUP}{$t}{$base}){$base=$tools->{SETUP}{$t}{$base};}
    if(-d $base)
    {
      my $cf=&SCRAMGenUtils::fixCacheFileName("${base}/.SCRAM/${scramarch}/ProjectCache.db");
      if (-f $cf){&addPackDep(&SCRAMGenUtils::readCache($cf));}
    }
    &SCRAMGenUtils::scramToolSymbolCache($tools,$t,$symboldir,$xjobs,$pcache->{packmap});
  }
  else{&SCRAMGenUtils::toolSymbolCache($tools,$t,$symboldir,$xjobs);}
  if (!exists $tools->{SETUP}{$t}{USE}){return;}
  foreach my $u (@{$tools->{SETUP}{$t}{USE}})
  {
    $u=lc($u);
    if(exists $tools->{SETUP}{$u})
    {
      &addToolDep($tools,$u);
      $c->{$u}=1;
      foreach my $k (keys %{$pcache->{tools}{$u}{deps}}){$c->{$k}=1;}
    }
  }
}

sub addPackDep ()
{
  my $cache=shift;
  my $p=shift;
  if (!defined $p)
  {
    my @packs=();
    foreach $p (keys %{$cache->{BUILDTREE}})
    {
      my $c=$cache->{BUILDTREE}{$p};
      my $suffix=$c->{SUFFIX};
      if($suffix ne ""){next;}
      my $class=$c->{CLASS};
      if($class=~/^(LIBRARY|CLASSLIB|SEAL_PLATFORM)$/)
      {
        my $pack=$c->{PARENT};
	$pcache->{packmap}{$pack}=$c->{NAME};
	&addPackDep($cache,$c,$pack);
      }
      elsif($class eq "PACKAGE"){push @packs,$p;}
      elsif($class=~/^(TEST|BIN|PLUGINS|BINARY)$/){&addPackDep($cache,$c,$p);}
    }
    foreach my $p (@packs){&addPackDep($cache,$cache->{BUILDTREE}{$p},$p);}
    return;
  }
  my $n=shift;
  if (exists $pcache->{dir}{$n}{deps}){return;}
  $pcache->{dir}{$n}{deps}={};
  if((!exists $p->{RAWDATA}) || (!exists $p->{RAWDATA}{DEPENDENCIES})){return;}
  my $c=$pcache->{dir}{$n}{deps};
  foreach my $u (keys %{$p->{RAWDATA}{DEPENDENCIES}})
  {
    my $t=lc($u);
    if ((exists $pcache->{tools}{$t}) && (exists $pcache->{tools}{$t}{deps}))
    {
      $c->{$t}=1;
      foreach my $k (keys %{$pcache->{tools}{$t}{deps}}){$c->{$k}=1;}
    }
    else
    {
      if (exists $cache->{BUILDTREE}{"${u}/src"}){&addPackDep($cache,$cache->{BUILDTREE}{"${u}/src"},$u);}
      elsif(exists $cache->{BUILDTREE}{$u}){&addPackDep($cache,$cache->{BUILDTREE}{$u},$u);}
      $c->{$u}=1;
      foreach my $k (keys %{$pcache->{dir}{$u}{deps}}){$c->{$k}=1;}
    }
  }
}

sub processProd ()
{
  my $prod=shift;
  if ((!exists $cache->{$prod}) || (exists $cache->{$prod}{skip}) || (exists $cache->{$prod}{done})){return;}
  $cache->{$prod}{done}=0;
  my $pack=$pcache->{prod}{$prod}{dir};
  if ($pack eq ""){return 0;}
  my $bfn=$pcache->{dir}{$pack}{bf};
  my $pc=$pcache->{dir}{$pack};
  if(exists $pc->{deps})
  {
    foreach my $u (keys %{$pc->{deps}})
    {
      if(exists $pcache->{dir}{$u})
      {
        my $u1="";
	if (exists $pcache->{packmap}{$u}){$u1=$pcache->{packmap}{$u};}
	else{$u1=&run_func("safename",$project,"${release}/src/${u}");}
        if($u1 ne ""){&processProd($u1);}
      }
    }
  }
  my $nexport="${cachedir}/${prod}no-export";
  if(exists $pc->{rdeps})
  {
    my @nuse=();
    foreach my $u (keys %{$pc->{rdeps}}){push @nuse,$u;}
    if(scalar(@nuse) > 0)
    {
      my $nfile;
      open($nfile, ">$nexport") || die "Can not open file \"$nexport\" for writing.";
      foreach my $u (@nuse){print $nfile "$u\n";}
      close($nfile);
    }
  }
  elsif(-f "$nexport"){system("rm -f $nexport");}

  my $bfdir="${tmpdir}/newBuildFile/src/${pack}";
  system("mkdir -p $bfdir");
  my $nfile="${bfdir}/${bfn}.auto";
  my $ptype="";my $pname="";my $pfiles=""; my $xargs="";
  if ($xml || ($bfn=~/\.xml$/)){$nfile="${bfdir}/${xmlbf}.auto"; $xargs.=" --xml";}
  if(exists $pcache->{prod}{$prod}{type})
  {
    $ptype="--prodtype ".$pcache->{prod}{$prod}{type};
    $pname="--prodname $prod --files '".$pcache->{prod}{$prod}{file}."'";
    $nfile="${bfdir}/${prod}${bfn}.auto";
    if ($xml){$nfile="${bfdir}/${prod}${xmlbf}.auto";}
  }
  else{$pname="--prodname $prod";}
  my $cmd="${sdir}/_createBuildFile.pl $xargs $configfile --dir ${release}/src/${pack} --tmpdir $tmpdir $jobs --buildfile $nfile $ptype $pname $pfiles $detail";
  &processCMD($pack,$cmd);
  if(-f $nfile)
  {
    $cache->{$prod}{done}=1;
    &updateActualPackDeps($prod,$pack,$nfile);
    &SCRAMGenUtils::writeHashCache($cache,$cfile);
  }
  if(-f "$nexport"){system("rm -f $nexport");}
  print STDERR "##########################################################################\n";
}

sub updateActualPackDeps ()
{
  my $prod=shift;
  my $pack=shift;
  my $d={};
  $d->{prodname}=$prod;
  if ((exists $pcache->{prod}{$prod}) && (exists $pcache->{prod}{$prod}{type})){$d->{prodtype}=$pcache->{prod}{$prod}{type};}
  my $prev="";
  if (!exists $pcache->{dir}{$pack}{first})
  {
    $pcache->{dir}{$pack}{deps}={};
    delete $pcache->{dir}{$pack}{rdeps};
  }
  else{$prev=$pcache->{dir}{$pack}{first};}
  my $bf=&SCRAMGenUtils::readBuildFile(shift);
  my $flags=&SCRAMGenUtils::findBuildFileTag($d,$bf,"flags");
  my $edm="";
  foreach my $a (keys %$flags)
  {
    foreach my $c (@{$flags->{$a}})
    {
      if (exists $c->{flags}{EDM_PLUGIN}){foreach my $v (@{$c->{flags}{EDM_PLUGIN}}){if($v->{v}==1){$edm="edm";last;}}}
      if ($edm){last;}
    }
    if ($edm){last;}
  }
  if ($edm){$pcache->{dir}{$pack}{plugin}=$edm;}
  my $use=&SCRAMGenUtils::findBuildFileTag($d,$bf,"use");
  foreach my $a (keys %$use)
  {
    foreach my $c (@{$use->{$a}})
    {
      foreach my $d (keys %{$c->{use}})
      { 
        $pcache->{dir}{$pack}{deps}{$d}=1;
	if((exists $pcache->{tools}{$d}) && (exists $pcache->{tools}{$d}{deps}))
	{foreach my $d1 (keys %{$pcache->{tools}{$d}{deps}}){$pcache->{dir}{$pack}{deps}{$d1}=1;}}
	elsif((exists $pcache->{dir}{$d}) && (exists $pcache->{dir}{$d}{deps}))
	{foreach my $d1 (keys %{$pcache->{dir}{$d}{deps}}){$pcache->{dir}{$pack}{deps}{$d1}=1;}}
      }
    }
  }
  delete $pcache->{dir}{$pack}{first};
  &SCRAMGenUtils::writeHashCache($pcache->{dir}{$pack},"${depsdir}/${prod}");
  if (($prev) && (-f "${depsdir}/${prev}")){unlink "${depsdir}/${prev}";}
  $pcache->{dir}{$pack}{first}=$prod;
}

sub processCMD ()
{
  my $pack=shift;
  my $cmd=shift;
  print "$cmd\n";
  my $reader; my $writer;
  my $pid=open2($reader, $writer,"$cmd 2>&1");
  $writer->autoflush();
  while(my $line=<$reader>)
  {
    chomp $line;
    if ($line=~/^([^:]+):(.*)$/)
    {
      my $req=$1;
      my $func="processRequest_${req}";
      if(exists &$func){print $writer "${req}_DONE:",&$func($2,$pack),"\n";}
      else{print "$line\n";}
    }
    else{print "$line\n";}
  }
  close($reader); close($writer);
  waitpid $pid,0;
}

sub processRequest_EXIT ()
{
  print STDERR "EXIT Requested\n";
  exit 0;
}

sub processRequest_HAVE_DEPS ()
{
  my $line=shift;
  my $rep="NO";
  if($line=~/^([^:]+):(.+)$/)
  {
    my $packs=$1; my $tool=$2;
    my %tools=();
    foreach my $pack (split /,/,$packs)
    {
      if (($pack eq "") || ($pack eq $tool)){next;}
      $tools{$pack}=1;
    }
    $packs=&hasDependency($tool,\%tools);
    if($packs){$rep="YES:$packs";}
  }
  return $rep;
}

sub hasDependency ()
{
  my $tool=shift;
  my $tools=shift;
  foreach my $d (keys %$tools)
  {
    if ($d eq $tool){next;}
    if((exists $pcache->{tools}{$d}) && (exists $pcache->{tools}{$d}{deps}{$tool})){return $d;}
    elsif(exists $pcache->{dir}{$d})
    {
      if((exists $pcache->{dir}{$d}{plugin}) && ($pcache->{dir}{$d}{plugin} eq "edm")){next;}
      elsif(exists $pcache->{dir}{$d}{deps}{$tool}){return $d;}
    }
  }
  return "";
}

sub processRequest_PLEASE_PROCESS_FIRST ()
{
  my $u=shift;
  print STDERR "REQUEST: Process package $u\n";
  my $pack=shift;
  my $u1="";
  if (exists $pcache->{packmap}{$u}){$u1=$pcache->{packmap}{$u};}
  else{$u1=&run_func("safename",$project,"${release}/src/${u}");}
  if ($u1 ne "")
  {
    if ((!exists $cache->{$u1}) && (-d "${release}/src/${u}"))
    {
      $cache->{$u1}={};
      $pcache->{prod}{$u1}{dir}=$u;
      my $bf="BuildFile";
      if (-f "${release}/src/${u}/${bf}.xml"){$bf="BuildFile.xml";}
      $pcache->{dir}{$u}{bf}=$bf;
    }
  }
  if(($u1 ne "") && (exists $cache->{$u1}))
  {
    my $pc=$pcache->{dir}{$pack};
    print "New Dependency added: $pack => $u\n";
    $pc->{deps}{$u}=1;
    $pcache->{dir}{$u}{rdeps}{$pack}=1;
    foreach my $x (keys %{$pc->{rdeps}}){$pcache->{dir}{$u}{rdeps}{$x}=1;}
    &processProd($u1);
  }
  return $u;
}

sub processRequest_PRODUCT_INFO ()
{
  my $pack=shift;
  my $rep="PROCESS";
  if (exists $pcache->{tools}{$pack}){$rep="NOT_EXISTS";}
  elsif (exists $pcache->{packmap}{$pack})
  {
    my $prod=$pcache->{packmap}{$pack};
    if(!exists $cache->{$prod}){$rep="NOT_EXISTS";}
    elsif(exists $cache->{$prod}{done}){$rep="DONE";}
    elsif(exists $cache->{$prod}{skip}){$rep="SKIP";}
  }
  return $rep;
}

sub processRequest_SYMBOL_CHECK_REQUEST ()
{
  my $prod=shift;
  print STDERR "REQUEST: Symbol check $prod\n";
  my $sym={};
  if (-f $prod){$sym=&SCRAMGenUtils::getLibSymbols($prod,'^U$');}
  elsif ((exists $cache->{$prod}) && (exists $pcache->{prod}{$prod}) && (exists $pcache->{prod}{$prod}{dir}))
  {
    my $d=$pcache->{prod}{$prod}{dir};
    if (-d "${release}/tmp/${scramarch}/src/${d}")
    {
      my $objdir=`find ${release}/tmp/${scramarch}/src/${d} -name $prod -type d`; chomp $objdir;
      my $ref;
      if ($objdir && opendir($ref,$objdir))
      {
        foreach my $file (readdir($ref))
	{
	  $file="${objdir}/${file}";
	  if (($file=~/\.o$/) && (-f $file))
	  {
	    my $sx=&SCRAMGenUtils::getObjectSymbols($file,'^U$');
	    foreach my $s (keys %$sx){$sym->{$s}=$sx->{$s};}
	  }
	}
        closedir($ref);
      }
    }
  }
  my %ts=();
  my $allsyms=$data->{ALL_SYMBOLS};
  foreach my $s (keys %$sym)
  {
    if (exists $allsyms->{$s})
    {
      my $s1=$allsyms->{$s};
      my $hassystem=0;
      foreach my $t (keys %$s1)
      {
	if ($t eq "system"){$ts{$s}={};}
	if (!$hassystem){foreach my $l (keys %{$s1->{$t}}){$ts{$s}{$t}=$l;}}
	if ($t eq "system"){$hassystem=1;}
      }
    }
  }
  my $tmpfile=&SCRAMGenUtils::getTmpFile ($cachedir);
  &SCRAMGenUtils::writeHashCache(\%ts,$tmpfile);
  return $tmpfile;
}

sub processRequest_PACKAGE_TYPE ()
{
  my $pack=shift;
  my $reply="UNKNOWN";
  if (exists $pcache->{tools}{$pack}){$reply="TOOL";}
  elsif (exists $pcache->{dir}{$pack})
  {
    if(exists $pcache->{dir}{$pack}{plugin}){$reply=uc($pcache->{dir}{$pack}{plugin});}
    else{$reply="PACK";}
  }
  return $reply;
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
  print "Usage: $script --dir <path> [--xml] [--order <pack>[--order <pack> [...]]] [--detail] [--merge]\n",
        "        [--redo <prod|all> [--redo <prod> [...]]] [--jobs <jobs>] [--config <file>]\n\n",
        "e.g.\n",
        "  $script --dir /path/to/a/project/release/area\n\n",
        "--dir <path>    Directory path for which you want to generate BuildFile(s).\n",
	"--order <pack>  Packages order in which script should process them\n",
	"--redo <pack>   Re-process an already done package\n",
	"--config <file> Extra configuration file\n",
	"--jobs <jobs>   Number of parallel jobs\n",
        "--detail        To get a detail processing log info\n",
	"--xml           To generate xml BuildFiles i.e. BuildFile.xml.auto\n",
	"--merge         Do not merge generated BuildFiles at the end\n\n",
        "This script will generate all the BuildFile(s) for your <path>. Generated BuildFile.auto will be available under\n",
	"AutoBuildFile/newBuildFile if you are not in a dev area otherwise in <devarea>/tmp/AutoBuildFile/newBuildFile.\n",
	"Do not forget to run \"mergeProdBuildFiles.pl --dir <dir>/AutoBuildFile/newBuildFile\n",
	"after running this script so that it can merge multiple products BuuildFiles in to one.\n",
	"Once BuildFile.auto are generated then you can copy all newBuilsFilew/*/BuildFile.auto in to your src/*/BuildFile.\n";
  exit 0;
}
