#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use SCRAMGenUtils;
use Getopt::Long;

my $pwd=`/bin/pwd`; chomp $pwd; $pwd=&SCRAMGenUtils::fixPath($pwd);
my $scriptDir=dirname($0);
if ($scriptDir!~/^\//){$scriptDir="${pwd}/${scriptDir}";}

#get the command-line options
if(&GetOptions(
	       "--release=s",\$dir,
	       "--tool=s",\@itools,
	       "--product=s",\@iproducts,
	       "--detail",\$detail,
	       "--dependency=s",\$dependency,
	       "--clean",\$clean,
	       "--recursive",\$full,
	       "--cppfilt",\$cppfilt,
	       "--graph",\$graph,
	       "--help",\$help,
              ) eq ""){print STDERR "#Wrong arguments.\n"; &usage_msg();}

my %systemsym=();
$systemsym{"__cxa_.+"}=1;
$systemsym{"__gxx_.+"}=1;
$systemsym{_fini}=1;
$systemsym{_init}=1;
$systemsym{fclose}=1;
$systemsym{fopen}=1;
$systemsym{"pthread_cond_.+"}=1;
$systemsym{regexec}=1;
$systemsym{".+\@\@GLIBCXX.*"}=1;
$systemsym{".+\@\@GCC.*"}=1;
$systemsym{".+\@\@CXXABI.*"}=1;
$systemsym{__fxstat64}=1;
$systemsym{__lxstat64}=1;
$systemsym{__xstat64}=1;
$systemsym{_dl_argv}=1;
$systemsym{_r_debug}=1;
$systemsym{getrlimit64}=1;
$systemsym{pthread_create}=1;
$systemsym{readdir64}=1;

my $symDefine="T|R|V|W|B|D";
my $symUnDefine="U";

if(defined $help){&usage_msg();}

if(defined  $detail){$detail=1;}
else{$detail=0;}

if(defined  $cppfilt){$cppfilt=1;}
else{$cppfilt=0;}

if(defined  $full){$full=1;}
else{$full=0;}

if(defined  $graph){$graph=1;}
else{$graph=0;}

if (!defined $dir){$dir=$pwd;}
my $release=&SCRAMGenUtils::scramReleaseTop($dir);
if($release eq ""){print STDERR "ERROR: Please run this script from a SCRAM-based area.\n"; exit 1;}
&SCRAMGenUtils::init ($release);
my $releaseTop=&SCRAMGenUtils::getFromEnvironmentFile("RELEASETOP",$release);
$SCRAMGenUtils::CacheType=0;

my $scramarch=&SCRAMGenUtils::getScramArch();
my $cachedir="${pwd}/Cache";
my $symboldir="${cachedir}/Symbols";
my $data={};
my $xjobs=10;

my %utools=();
if ($dependency eq "")
{
  foreach my $t (@itools)
  {
    my $lt=lc($t);
    if (-f "${release}/.SCRAM/${scramarch}/timestamps/${lt}"){$utools{TOOLS}{$lt}=1;}
    else{$utools{TOOLS}{$t}=1;}
  }
  foreach my $t (@iproducts){$utools{SELF}{$t}=1;}
  if (scalar(keys %utools)==0){print STDERR "Missing tool/product name.\n"; &usage_msg();}
}

if (defined  $clean){system("rm -rf $symboldir");}
system("mkdir -p $symboldir ${cachedir}/Dot");

print "Reading external tools symbols info ....";
system("mkdir -p ${symboldir}/TOOLS ${symboldir}/SYSTEM");
my $cf="${release}/.SCRAM/${scramarch}/ToolCache.db.gz";
if (-f $cf){&addToolDep(&SCRAMGenUtils::readCache($cf),"TOOLS");}
&SCRAMGenUtils::waitForChild();
print STDERR "\n";
$data->{SYSTEM_SYMBOLS_DEF}=&SCRAMGenUtils::mergeSymbols("${symboldir}/SYSTEM","",$symDefine);

$data->{paths}{0}="lib/${scramarch}";
$data->{paths}{1}="bin/${scramarch}";
$data->{paths}{2}="test/${scramarch}";

my $prodfile="${release}/src/ReleaseProducts.list";
if (!-f $prodfile)
{
  my $rdir=$release;
  if ($releaseTop ne "")
  {
    $prodfile="${release}/src/ReleaseProducts.list";
    $rdir=$releaseTop;
  }
  if (!-f $prodfile)
  {
    $prodfile="${cachedir}/ReleaseProducts.list";
    if (!-f $prodfile){system("${scriptDir}/RelProducts.pl $rdir > $prodfile");}
  }
}

foreach my $line (`cat $prodfile`)
{
  chomp $line;
  my ($pack,$prods)=split(":",$line,2);
  $data->{pack}{$pack}={};
  my @types=split('\\|',$prods,4);
  for(my $i=0;$i<1;$i++)
  {
    foreach my $prod (split(",",$types[$i]))
    {
      if ($prod eq ""){next;}
      $data->{pack}{$pack}{$prod}=1;
      $data->{prod}{$prod}{pack}=$pack;
      $data->{prod}{$prod}{path}=$i;
    }
  }
}

print "Reading local libraries symbols info ....";
system("mkdir -p ${symboldir}/SELF");
foreach my $prod (keys %{$data->{prod}})
{
  my $p=$data->{paths}{$data->{prod}{$prod}{path}};
  my $f="${release}/${p}/${prod}";
  if ((!-f $f) && ($releaseTop ne "")){$f="${releaseTop}/${p}/${prod}";}
  if ($prod=~/^lib.+\.so$/){$data->{PRODUCT_MAP}{$prod}=$data->{prod}{$prod}{pack};}
  if (-f $f)
  {
    &SCRAMGenUtils::symbolCacheFork($f,"cmssw","${symboldir}/SELF",$xjobs);
    print STDERR ".";
  }
  else{print STDERR "ERROR: Missing file: $f\n";}
}
&SCRAMGenUtils::waitForChild();
print STDERR "\n";
&SCRAMGenUtils::writeHashCache($data->{PRODUCT_MAP},"${symboldir}/Lib2Package.db");
&SCRAMGenUtils::writeHashCache($data->{TOOL_TYPE},"${symboldir}/LibTypes.db");

if ($dependency ne "")
{
  $data->{PRODUCT_MAP} = &SCRAMGenUtils::readHashCache("${symboldir}/Lib2Package.db");
  $data->{TOOL_TYPE} = &SCRAMGenUtils::readHashCache("${symboldir}/LibTypes.db");
  $data->{SELF_SYMBOLS_DEF}=&SCRAMGenUtils::mergeSymbols("${symboldir}/SELF","",$symDefine);
  $data->{TOOLS_SYMBOLS_DEF}=&SCRAMGenUtils::mergeSymbols("${symboldir}/TOOLS","",$symDefine);
  &doDependency($dependency);
  exit 0;
}
else
{
  delete $data->{SELF_SYMBOLS_DEF};
  delete $data->{TOOLS_SYMBOLS_DEF};
  delete $data->{PRODUCT_MAP};
  delete $data->{TOOL_TYPE}; 
}

$data->{SELF_SYMBOLS_UNDEF}=&SCRAMGenUtils::mergeSymbols("${symboldir}/SELF","",$symUnDefine);

my $toolsym={};
&findDefinedSyms(\%utools,$symDefine,$toolsym);

my %deps=();
$deps{direct}=&findSymbolDependency($toolsym);
if($detail){&detailDepInfo("direct",\%deps);}
if($full){&processSelfDependency("direct",\%deps);}

my %packs=();
foreach my $lib (keys %deps)
{
  if (exists $data->{prod}{$lib}){$packs{$data->{prod}{$lib}{pack}}{$lib}=1;}
  foreach my $dep (keys %{$deps{$lib}})
  {
    if(exists $data->{prod}{$dep}){$packs{$data->{prod}{$dep}{pack}}{$dep}=1;}
  }
}

print "\n##################################\nALL DEPS\n";
foreach my $pk (sort keys %packs)
{
  my $direct="";
  foreach my $x (keys %{$packs{$pk}})
  {if(exists  $deps{direct}{$x}){$direct="*"; last;}}
  print "${direct}${pk}:\n";
  foreach my $p (sort keys %{$packs{$pk}}){print "    $p\n";}
  print "\n";
}

sub usage_msg()
{
  my $s=basename($0);
  print "Usage: \n$s --release <path> --tool <tool>|--product <product>|--dependnecy <library>\n",
        "            [--recursive] [--cppfilt] [--clean] [--detail] [--help]\n\n",
        " Script can find out actual binary dependency of a cmssw products\n",
	" (lib/exe/plugin) OR it can be used to find out which cmssw packages/products\n",
	" are dependening on an external tool or cmssw product e.g.\n",
	" * To find out the actual binary dependency of sub-system/package/individual\n";
	"   products(lib/exe/plugin)\n",
	"    $s --rel <path> --dependnecy FWCore\n",
	"    $s --rel <path> --dependnecy FWCore/Framework\n",
	"    $s --rel <path> --dependnecy pluginFWCorePrescaleServicePlugin.so\n",
	"    $s --rel <path> --dependnecy libFWCoreFramework.so\n\n",
	" * To find out which product(s) of CMSSW actually need a tool/product\n",
	"    $s --rel <path> --tool oracle --tool oracleocci --tool oracle\n",
	"    $s --rel <path> --tool xdaq --full\n",
	"    $s --rel <path> --product libFWCoreFramework.so [--recursive]\n",
	"   when --recursive option is specified then it will show all the products which\n",
	"   directly/in-directly depending on this tool/product\n";
  exit 0;
}

##############################################################################

sub findDefinedSyms()
{
  my ($c,$type,$syms)=@_;
  foreach my $x (keys %$c)
  {
    foreach my $t (keys %{$c->{$x}})
    {
      my $fil="lib*.$t";
      if ($x eq "SELF"){$fil="$t.cmssw";}
      foreach my $f (`find ${symboldir}/${x} -name '$fil' -type f`)
      {
        chomp $f;
	&accSymbolsFromFile($f,$syms,$type);
      }
    }
  }
  foreach my $s (keys %$syms){if (&isSystemSymbol($s)==1){delete $syms->{$s};}}
}

sub accSymbolsFromFile()
{
  my ($file,$syms,$type,$system)=@_;
  print "Reading File: $file\n";
  my $s=&SCRAMGenUtils::readHashCache($file);
  foreach my $x (keys %$s)
  {
    foreach my $l (keys %{$s->{$x}})
    {
      foreach my $ty (keys %{$s->{$x}{$l}})
      {
        if ($ty=~/^$type$/){foreach my $s (keys %{$s->{$x}{$l}{$ty}}){$syms->{$s}=1;}}
      }
    }
  }
  if (defined $system){foreach my $s (keys %$syms){if (&isSystemSymbol($s)==1){delete $syms->{$s};}}}
}

sub detailDepInfo()
{
  my $prod=shift;
  my $cache=shift;
  print "DEPS:$prod\n";
  foreach my $x (sort keys %{$cache->{$prod}})
  {
    print "  $x\n";
    foreach my $s (sort keys %{$cache->{$prod}{$x}})
    {
      if ($cppfilt){$s=&SCRAMGenUtils::cppFilt($s);}
      print "    $s\n";
    }
  }
  print "\n";
}

sub processSelfDependency()
{
  my $prod=shift;
  my $cache=shift;
  foreach my $d (keys %{$cache->{$prod}})
  {
    if (exists $cache->{$d}){next;}
    $cache->{$d}={};
    if ($d!~/^lib.+\.so$/){next;}
    my $c="${symboldir}/SELF/${d}.cmssw";
    if (!-f $c){print STDERR "ERROR: Missing symbol cache file: $c\n";}
    else
    {
      my $syms={};
      &accSymbolsFromFile($c,$syms,$symDefine,1);
      $cache->{$d}=&findSymbolDependency($syms);
      if ($detail){&detailDepInfo($d,$cache);}
      &processSelfDependency($d,$cache);
    }
  }
}

sub findSymbolDependency()
{
  my ($syms)=@_;
  my $d={};
  foreach my $sym (keys %$syms)
  {
    if (exists $data->{SELF_SYMBOLS_UNDEF}{$sym})
    {
      foreach my $l (keys %{$data->{SELF_SYMBOLS_UNDEF}{$sym}{cmssw}}){$d->{$l}{$sym}=1;}
    }
  }
  return $d;
}

sub getLibMap()
{
  my $cache=shift;
  my $map={};
  foreach my $dir (keys %{$cache->{BUILDTREE}})
  {
    my $c=$cache->{BUILDTREE}{$dir};
    my $suffix=$c->{SUFFIX};
    if($suffix ne ""){next;}
    my $class=$c->{CLASS};
    my $name=$c->{NAME};
    if($class=~/^(LIBRARY|CLASSLIB|SEAL_PLATFORM)$/){$map->{$c->{PARENT}}=$c->{NAME};}
  }
  return $map;
}

sub addToolDep ()
{
  my ($tools,$symdir,$t)=@_;
  if (!defined $t)
  {
    foreach $t (&SCRAMGenUtils::getOrderedTools($tools)){&addToolDep($tools,$symdir,$t);}
    return;
  }
  if (exists $tools->{TOOLSDONE}{$t}{deps}){return;}
  $tools->{TOOLSDONE}{$t}{deps}={};
  my $c=$tools->{TOOLSDONE}{$t}{deps};
  if ($t ne "self")
  {
    if ($tools->{SETUP}{$t}{SCRAM_PROJECT} == 1)
    {
      my $bv=uc($t)."_BASE";
      my $sbase=$tools->{SETUP}{$t}{$bv};
      my $spfile="${sbase}/.SCRAM/${scramarch}/ProjectCache.db.gz";
      if (-f $spfile)
      {
        my $libmap=&getLibMap(&SCRAMGenUtils::readCache($spfile));
        &SCRAMGenUtils::scramToolSymbolCache($tools,$t,"${symboldir}/${symdir}",$xjobs,$libmap);
	foreach my $p (keys %$libmap)
	{
	  my $l=$libmap->{$p};
	  $data->{PRODUCT_MAP}{"lib${l}.so"}=$p;
	  $data->{TOOL_TYPE}{"lib${l}.so"}="TOOLS";
	}
      }
      next;
    }
    elsif ($t eq "cxxcompiler")
    {
      my $sbase=$tools->{SETUP}{$t}{GCC_BASE};
      if (-d "${sbase}/lib")
      {
        foreach my $l ("stdc++","gcc_s","gfortran","ssp")
	{
	  my $lib="${sbase}/lib/lib${l}.so";
          if(-f $lib)
	  {
	    $data->{PRODUCT_MAP}{"lib${l}.so"}="system";
	    &SCRAMGenUtils::symbolCacheFork($lib,"cxxcompiler","${symboldir}/SYSTEM",$xjobs);
	  }
	  else{print STDERR "No such file: $lib\n";}
	}
      }
      foreach my $l ("m","util","rt")
      {
	my $f=0;
	foreach my $d ("/lib","/usr/lib")
        {
	  my $lib="${d}/lib${l}.so";
          if(-f $lib)
	  {
	    &SCRAMGenUtils::symbolCacheFork($lib,"system","${symboldir}/SYSTEM",$xjobs);
	    $data->{PRODUCT_MAP}{"lib${l}.so"}="system";
	    $f=1;
	    last;
	  }
	}
	if (!$f){print STDERR "No such file: lib$l.so\n";}
      }
      foreach my $lib ("/lib/libpthread.so.0","/lib/libc.so.6")
      {
	if(-f $lib)
	{
	  &SCRAMGenUtils::symbolCacheFork($lib,"system","${symboldir}/SYSTEM",$xjobs);
	  my $l=basename($lib);
	  $data->{PRODUCT_MAP}{"lib${l}.so"}="system";
	}
	else{print STDERR "No such file: $lib\n";}
      }
    }
    else
    {
      &SCRAMGenUtils::toolSymbolCache($tools,$t,"${symboldir}/TOOLS",$xjobs);
      if (exists $tools->{SETUP}{$t}{LIB})
      {
        foreach my $l (@{$tools->{SETUP}{$t}{LIB}})
	{
	  $data->{PRODUCT_MAP}{"lib${l}.so"}=$t;
	  $data->{TOOL_TYPE}{"lib${l}.so"}="TOOLS";
	}
      }
    }
    if (!exists $tools->{SETUP}{$t}{USE}){return;}
    foreach my $u (@{$tools->{SETUP}{$t}{USE}})
    {
      $u=lc($u);
      if(exists $tools->{SETUP}{$u})
      {
        &addToolDep($tools,$u);
        $c->{$u}=1;
        foreach my $k (keys %{$tools->{TOOLSDONE}{$t}{deps}}){$c->{$k}=1;}
      }
    }
  }
}

sub isSystemSymbol()
{
  my $sym=shift;
  if (exists $data->{SYSTEM_SYMBOLS_DEF}{$sym}){return 1;}
  foreach my $r (keys %systemsym){if($sym=~/^$r$/){return 1;}}
  return 0;
}

##############################################################
# Library Dependecy
##############################################################
sub readSyms()
{
  my ($lib,$type)=@_;
  my $sfile="${symboldir}/${type}";
  my $tool="cmssw";
  if (($type ne "SELF") && (exists $data->{PRODUCT_MAP}{$lib})){$tool=$data->{PRODUCT_MAP}{$lib};}
  my $sfile="${symboldir}/${type}/${lib}.${tool}";
  if (-f $sfile)
  {
    my $c=&SCRAMGenUtils::readHashCache($sfile);
    if (exists $c->{$tool}{$lib}){return $c->{$tool}{$lib};}
  }
  print STDERR "ERROR: Symbols not found for library: $lib\n";
  return {};
}

sub sym2Lib()
{
  my $lib=shift;
  my $graph=shift;
  my $cache=shift || {};
  my $tab=shift || "";
  my $parent=shift || "";
  my $depth=1+(length($tab)/2);
  if (exists $cache->{$lib}){return;}
  $cache->{$lib}=1;
  my $ltype="SELF";
  if (exists $data->{TOOL_TYPE}{$lib}){$ltype=$data->{TOOL_TYPE}{$lib};}
  my %syms=();
  my $deps={};
  if (!exists $data->{LIBDEPS}{$lib})
  {
    my $xsyms=&readSyms($lib,$ltype);
    foreach my $sym (keys %{$xsyms->{U}})
    {
      if (&isSystemSymbol($sym)==1){next;}
      $syms{$sym}=1;
    }
    foreach my $sym (keys %syms)
    {
      my $c=undef;
      if ($ltype eq "SELF")
      {
        if (exists $data->{SELF_SYMBOLS_DEF}{$sym}){$c=$data->{SELF_SYMBOLS_DEF}{$sym};}
        elsif(exists $data->{TOOLS_SYMBOLS_DEF}{$sym}){$c=$data->{TOOLS_SYMBOLS_DEF}{$sym};}
      }
      elsif(exists $data->{TOOLS_SYMBOLS_DEF}{$sym}){$c=$data->{TOOLS_SYMBOLS_DEF}{$sym};}
      if (defined $c)
      {
        foreach my $t (keys %{$c})
        {
          foreach my $l (keys %{$c->{$t}})
	  {
	    my $p=$l;
	    if (exists $data->{PRODUCT_MAP}{$l}){$p=$data->{PRODUCT_MAP}{$l};}
	    if (!exists $deps->{$p}){$deps->{$p}{c}=0;}
	    $deps->{$p}{c}+=1;
	    $deps->{$p}{l}{$l}=1;
	  }
        }
      }
      elsif ($ltype eq "SELF"){$data->{UNKNOWN_SYM}{$sym}=1;}
    }
    $data->{LIBDEPS}{$lib}=$deps;
    if ($full)
    {
      foreach my $p (sort keys %$deps)
      {
        foreach my $l (keys %{$deps->{$p}{l}}){if (!exists $data->{LIBDEPS}{$l}){&sym2Lib($l,$graph);}}
      }
    }
  }
  else{$deps=$data->{LIBDEPS}{$lib};}
  my $mp=$lib;
  if ($depth>1){$mp=$data->{PRODUCT_MAP}{$lib};}
  if (($detail) && ($depth==1)){print "#######################################\n$lib:\n";}
  if ($depth<10){$depth="0$depth";}
  foreach my $p (sort keys %$deps)
  {
    if ($graph){$cache->{GRAPH}{DEPS}{$mp}{$p}=1;}
    if ($p eq "system"){next;}
    if (($detail) && ($p ne $parent))
    {
      my $ls=join(",",sort keys %{$deps->{$p}{l}});
      print "$depth.${tab}  $p (",$deps->{$p}{c},": $ls)\n";
    }
    foreach my $l (keys %{$deps->{$p}{l}}){&sym2Lib($l,$graph,$cache,"$tab  ",$p);}
  }
  if ($depth==1)
  {
    if (!$detail){print "\n###########################\n$lib:\n";}
    my %p=();
    my $pk="";
    foreach my $l (keys %$cache)
    {
      if (exists $data->{PRODUCT_MAP}{$l})
      {
        $pk=$data->{PRODUCT_MAP}{$l};
	if ($pk ne "system"){$p{$pk}=1;}
      }
    }
    if (exists $data->{PRODUCT_MAP}{$lib}){delete $p{$data->{PRODUCT_MAP}{$lib}};}
    my @ds=sort keys %p;
    if ($graph)
    {
      $cache->{GRAPH}{NODES}{$lib}=1;
      foreach my $x (@ds){$cache->{GRAPH}{NODES}{$x}=1;}
      &generateGraph($lib,$cache->{GRAPH});
    }
    print " * Direct dependencies:\n";
    foreach my $x (@ds){if (exists $deps->{$x}){print "  $x\n";}}
    print " * Indirect dependencies:\n";
    foreach my $x (@ds){if (!exists $deps->{$x}){print "  $x\n";}}
  }
}

sub generateGraph()
{
  my $l=shift;
  my $data=shift;
  my $ref;
  if (open($ref,">${cachedir}/Dot/${l}.dot"))
  {
    print $ref "digraph Dependencies {\n",
               '  fontname="Helvetica"; fontsize=12; center=true; ratio=auto;concentrate=true;',"\n",
               '  label="\n',$l,'\n"',"\n",
               '  node [shape=ellipse, fontname="Helvetica-Bold", fontsize=12, style=filled, color="0.9 1.0 1.0", fontcolor="0 0 1" ]',"\n",
               '  edge [fontname="Helvetica", fontsize=12 ]',"\n\n";
    foreach my $n (keys %{$data->{NODES}}){print $ref "  \"$n\" []\n";}
    foreach my $n (keys %{$data->{DEPS}})
    {
      foreach my $d (keys %{$data->{DEPS}{$n}}) {print $ref "  \"$n\" -> \"$d\" []\n";}
    }
    print $ref "}\n";
    close($ref);
  }
  else{print STDERR "ERROR: Can not open file for writing: ${l}.dot\n";}
}

sub doDependency()
{
  my $dependency=shift;
  my @prods=();
  if (exists $data->{prod}{$dependency}){$prods[0]=$dependency;}
  else
  {
    foreach my $pk (sort keys %{$data->{pack}})
    {
      if ($pk=~/^$dependency(\/.+|)$/o){foreach my $p (keys %{$data->{pack}{$pk}}){push @prods,$p;}}
    }
  }
  foreach my $lib (@prods){&sym2Lib($lib,$graph);}
  if ($detail && (exists $data->{UNKNOWN_SYM}))
  {
    print "SYMBOLS UNKNOWN:\n";
    foreach my $s (sort keys %{$data->{UNKNOWN_SYM}})
    {
      if ($cppfilt){$s=&SCRAMGenUtils::cppFilt($s);}
      print "$s\n";
    }
  }
}
