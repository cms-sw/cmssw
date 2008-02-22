#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use Getopt::Long;
use SCRAMGenUtils;
$|=1;

my $curdir=`/bin/pwd`; chomp $curdir;
my $scriptdir=dirname($0);
my $dir=shift;

if($dir eq ""){$dir=$curdir;}
if($dir ne "INTERNAL_RUNNING_OF_SCRIPT")
{
  if(!-d $dir){die "No such directory: $dir\n";}
  system("$0 INTERNAL_RUNNING_OF_SCRIPT $dir @ARGV | grep '^DATA:' | sed 's|^DATA:||'");
  exit $?;
}
else{$dir=shift;}

if($dir=~/^[^\/]/){$dir="${curdir}/${dir}";}
$dir=&SCRAMGenUtils::fixPath($dir);
my $arch=shift || "";
my $scram_list_skip=5;
my $flags="";
my $tmprel="";

my $release=&SCRAMGenUtils::scramReleaseTop($dir);
if($release eq ""){die "\"$dir\" is not SCRAM-base project area.\n";}

my $headerext="\\.(h|hh|hpp)\$";
my $srcext="\\.(cc|cpp|c|cxx)\$";
my $lexparseext="_(lex\\.l|parse\\.y)\$";
my $config=&SCRAMGenUtils::getFromEnvironmentFile("SCRAM_CONFIGDIR","$release") || "config";
my $src=&SCRAMGenUtils::getFromEnvironmentFile("SCRAM_SOURCEDIR","$release") || "src";
my $releasetop=&SCRAMGenUtils::getFromEnvironmentFile("RELEASETOP","$release") || $release;
my $project=&SCRAMGenUtils::getFromEnvironmentFile("SCRAM_PROJECTNAME","$release");
if($release eq $dir){$dir.="/${src}";}

my $packdir=$dir;
if($packdir=~/^${release}\/${src}(\/.+|)$/){$packdir=$1;$packdir=~s/^\/*//;}
else{die "Wrong subsystem/package directory \"$dir\".\n";}
my $actualdir=$packdir;
if($packdir eq ""){$packdir=".+";}
else{$packdir="^$packdir(\/.+|)\$";}

if (!-f "${release}/${config}/scram_version"){die "${release}/${config}/scram_version file does not exist.\n"; exit 1;}
if (!-d "${release}/${src}"){die "${release}/${src} directory does not exist. No sources to work on.\n"; exit 1;}

my $SCRAM_CMD="scram";
my $SCRAM_VER=`cat ${release}/${config}/scram_version`; chomp $SCRAM_VER;
if($SCRAM_VER=~/^\s*V1_/){$SCRAM_CMD="scramv1";}
elsif($SCRAM_VER=~/^\s*V0_/){$SCRAM_CMD="scram";}
if($arch eq ""){$arch=`$SCRAM_CMD arch`; chomp $arch;}
$SCRAM_CMD="$SCRAM_CMD -arch $arch";
$SCRAMGenUtils::SCRAM_CMD=$SCRAM_CMD;

print 'DATA:OWNHEADER=^(.+)\\/[^\\/]+\\/([^\\.]+)\\.(cc|CC|cpp|C|c|CPP|cxx|CXX)$:"$1\\/.+?\\/$2\\.(h|hh|hpp|H|HH|HPP)\\$"',"\n";
print "DATA:BASE_DIR=${release}/${src}\n";
$flags="-I${release}/${src}";

my $cache={};
if($releasetop ne $release)
{
  print "DATA:BASE_DIR=${releasetop}/${src}\n";
  $flags.=" -I${releasetop}/${src}";
  my $doscramb=1;
  if((-f "${release}/tmp/${arch}/Makefile") && (-f "${release}/.SCRAM/${arch}/ProjectCache.db"))
  {
    $cache->{cache}=&SCRAMGenUtils::readCache("${release}/.SCRAM/${arch}/ProjectCache.db");
    my %dirstocheck=();
    if($actualdir=~/\//){$dirstocheck{$actualdir}=1;}
    else
    {
      if($actualdir ne ""){$actualdir.="/";}
      foreach my $d (&SCRAMGenUtils::readDir("${release}/${src}/${actualdir}",1))
      {
        $d="${actualdir}${d}";
        if($d!~/\//)
        {foreach my $d1 (&SCRAMGenUtils::readDir("${release}/${src}/${d}",1)){$dirstocheck{"${d}/${d1}"}=1;}}
        else{$dirstocheck{$d}=1;}
      }
    }
    my $dircount=scalar(keys %dirstocheck);
    foreach my $d (keys %{$cache->{cache}{BUILDTREE}})
    {
      if(exists $dirstocheck{$d})
      {
        delete $dirstocheck{$d};
	$dircount--;
	if($dircount == 0){$doscramb=0;last;}
      }
    }
  }
  if($doscramb)
  {
    system("cd $release; $SCRAM_CMD b -r echo_CXX 2>&1 > /dev/null");
    $cache->{cache}=&SCRAMGenUtils::readCache("${release}/.SCRAM/${arch}/ProjectCache.db");
  }
}

my $tmprel=$release;
if(!-f "${release}/tmp/${arch}/Makefile")
{
  my $dev=0;
  if($releasetop ne $release){$dev=1;}
  $tmprel=&SCRAMGenUtils::createTmpReleaseArea($releasetop,$dev);
  system("rm -rf ${tmprel}/src; cp -r ${release}/src ${tmprel}");
  system("cd $tmprel; $SCRAM_CMD b -r echo_CXX 2>&1 > /dev/null");
  foreach my $f ("tmp/${arch}/Makefile",".SCRAM/${arch}/ProjectCache.db",".SCRAM/${arch}/ToolCache.db")
  {
    open(MAKEIN,"${tmprel}/${f}") || die "can not open ${tmprel}/${f} file for reading";
    open(MAKEOUT,">${tmprel}/${f}.new") || die "can not open ${tmprel}/${f}.new file for reading";
    while(my $line=<MAKEIN>)
    {
      chomp $line;
      $line=~s/$tmprel/$release/g;
      print MAKEOUT "$line\n";
    }
    close(MAKEIN);
    close(MAKEOUT);
    system("mv ${tmprel}/${f}.new ${tmprel}/${f}");
  }
  $cache->{cache}=&SCRAMGenUtils::readCache("${tmprel}/.SCRAM/${arch}/ProjectCache.db");
}
$cache->{tools}=&SCRAMGenUtils::readCache("${tmprel}/.SCRAM/${arch}/ToolCache.db");

my $cxx="/usr/bin/c++";
my $cxx_ver="";
if(exists $cache->{tools}{SETUP}{cxxcompiler}{CXX}){$cxx=$cache->{tools}{SETUP}{cxxcompiler}{CXX};}
else{$cxx=&SCRAMGenUtils::getBuildVariable($tmprel,"CXX") || $cxx;}
if(-x $cxx){$cxx_ver=`$cxx --version | head -1 | awk '{print \$3}'`;chomp $cxx_ver;}

print "DATA:COMPILER=$cxx\n";
print "DATA:COMPILER_VERSION=$cxx_ver\n";

&genConfig ($packdir,$cache);
&updateOrderedTools($cache);
my $toolcount=scalar(@{$cache->{orderedtool}});
$cache->{uniqinc}={};
for(my $i=$toolcount;$i>0;$i--)
{
  my $tool=$cache->{orderedtool}[$i-1];
  if(exists $cache->{tools}{SETUP}{$tool}{INCLUDE})
  {
    foreach my $inc (@{$cache->{tools}{SETUP}{$tool}{INCLUDE}})
    {
      if(!exists $cache->{uniqinc}{$inc})
      {$flags.=" -I$inc";$cache->{uniqinc}{$inc}=1;}
    }
  }
}

foreach my $inc (split /\s+/, &SCRAMGenUtils::getBuildVariable($tmprel,"CPPDEFINES"))
{$flags="$flags ".postProcessFlag ("-D$inc");}
$flags="$flags ".&postProcessFlag(&SCRAMGenUtils::getBuildVariable($tmprel,"CPPFLAGS"));
$flags="$flags ".&postProcessFlag(&SCRAMGenUtils::getBuildVariable($tmprel,"CXXFLAGS"));
print "DATA:COMPILER_FLAGS=$flags\n";
my $tmpl_compile_support=&checkTemplateCompilationSupport ();
my $def_compile_support=&checkDefineCompilationSupport ();
if(($tmpl_compile_support==0) || ($def_compile_support==0)){&genSkip ($cache);}
print "DATA:SKIP_INCLUDES=^(Ig|Vis).*?/interface/config\.h:classlib/sysapi/system\\.h\$\n";
print "DATA:SKIP_INCLUDES=.+?:Utilities/Testing/interface/CppUnit_testdriver\\.icpp\$\n";
print "DATA:SKIP_INCLUDES=.+?:FWCore/ParameterSet/test/ConfigTestMain\\.h\$\n";
print "DATA:SKIP_INCLUDE_INDIRECT_ADD=.+?:(Ig|Vis).*?/interface/config\\.h\$\n";
print "DATA:SKIP_AND_ADD_REMOVED_INCLUDES=.*?/classes\\.h\$\n";
print "DATA:SKIP_AND_ADD_REMOVED_INCLUDES=.*?/([^/]*)LinkDef\\.h\$\n";
print "DATA:SOURCE_EXT=$lexparseext\n";

foreach my $d ("$curdir","$scriptdir")
{
  if(-f "${d}/${project}_IncludeChecker.conf")
  {
    print "DATA:#Extra Include Checker Configuration: ${d}/${project}_IncludeChecker.conf\n";
    foreach my $l (`cat ${d}/${project}_IncludeChecker.conf`){print "DATA:$l";}
  }
}
&final_exit(0);

sub final_exit ()
{
  my $code=shift || 0;
  if(($tmprel ne "") && ($tmprel ne $release)){system("rm -rf $tmprel");}
  exit $code;
}

sub updateOrderedTools ()
{
  my $cache=shift;
  my $tool=shift || "";
  if($tool eq "")
  {
    if(exists $cache->{orderedtool}){return;}
    $cache->{tooldone}={}; $cache->{orderedtool}=[];
    foreach my $t (keys %{$cache->{tools}{SETUP}}){&updateOrderedTools($cache,$t);}
  }
  elsif(!exists $cache->{tooldone}{$tool})
  {
    $cache->{tooldone}{$tool}=1;
    if(exists $cache->{tools}{SETUP}{$tool}{USE})
    {
      foreach my $t (@{$cache->{tools}{SETUP}{$tool}{USE}})
      {
	$t=lc($t);
	&updateOrderedTools($cache,$t);
      }
    }
    push @{$cache->{orderedtool}},$tool;
  }
}

sub genSkip ()
{
  my $data=shift;
  foreach my $prod (keys %{$data->{prods}})
  {
    foreach my $f (@{$data->{prods}{$prod}{files}})
    {
      
      if($f!~/.+?$headerext/i){next;}
      if($f=~/.+?FWD$headerext/i){print "DATA:SKIP_FILES=^$f\$\n";next;}
      if(-f "${release}/${src}/${f}")
      {
        foreach my $line (`cat ${release}/${src}/${f}`)
        {
          chomp $line;
	  if(($def_compile_support==0) && ($line=~/^\s*\#\s*define\s+.+?\\$/))
	  {print "DATA:SKIP_FILES=^$f\$\n";last;}
	  if(($tmpl_compile_support==0) && ($line=~/^\s*template\s*<.+/))
	  {print "DATA:SKIP_FILES=^$f\$\n";last;}
        }
      }
    }
  }
}

sub genConfig ()
{
  my $dir=shift;
  my $data=shift;
  my $cache=$data->{cache};
  $data->{prods}={};
  foreach my $d (keys %{$cache->{BUILDTREE}})
  {
    if($d!~/$dir/){next;}
    if((exists $cache->{BUILDTREE}{$d}{CLASS}) && (exists $cache->{BUILDTREE}{$d}{RAWDATA}{content}))
    {
      my $suffix=$cache->{BUILDTREE}{$d}{SUFFIX};
      if($suffix ne ""){next;}
      my $class=$cache->{BUILDTREE}{$d}{CLASS};
      my $c=$cache->{BUILDTREE}{$d}{RAWDATA}{content};
      if($class eq "PACKAGE"){&addPack($data,$c,$d);}
      elsif($class=~/^(TEST|BIN|PLUGINS|BINARY)$/){&addProds($data,$c,$d);}
    }
  }
  &getFlags($data->{prods});
  foreach my $prod (sort keys %{$data->{prods}})
  {
    print "DATA:COMPILER_FLAGS=".$data->{prods}{$prod}{flags}."\n";
    foreach my $file (@{$data->{prods}{$prod}{files}})
    {
      print "DATA:FILES=$file\n";
      if($file=~/$lexparseext/){print "DATA:SKIP_AND_ADD_REMOVED_INCLUDES=^$file\$\n";}
    }
  }
}

sub addPack()
{
  my $data=shift;
  my $cache=shift;
  my $d=&SCRAMGenUtils::fixPath(shift);
  my $prod=&run_func("safename",$project,"${release}/${src}/${d}");
  if($prod eq ""){die "ERROR: Script is not ready for $project SCRAM-based project.\n";}
  $data->{prods}{$prod}{files}=[];
  my $pkinterface=&run_func("pkg_interface",$project,$d);
  my $pksrc=&run_func("pkg_src",$project,$d);
  foreach my $d1 ($pkinterface,$pksrc)
  {
    my $dir="${release}/${src}/${d}/${d1}";
    if(-d "$dir")
    {
      foreach my $f (&SCRAMGenUtils::readDir($dir,2))
      {if(($f=~/$headerext/i) || ($f=~/$srcext/i) || ($f=~/$lexparseext/)){push @{$data->{prods}{$prod}{files}},"${d}/${d1}/${f}";}}
    }
  }
  if(scalar(@{$data->{prods}{$prod}{files}})==0){delete $data->{prods}{$prod};}
}

sub addProds ()
{
  my $data=shift;
  my $cache=shift;
  my $d=&SCRAMGenUtils::fixPath(shift);
  my $bf1="${release}/src/${d}/BuildFile";
  my $bf=undef;
  foreach my $t (keys %{$cache->{BUILDPRODUCTS}})
  {
    foreach my $prod (keys %{$cache->{BUILDPRODUCTS}{$t}})
    {
      if($prod=~/^\s*$/){next;}
      my $name=basename($prod);
      $data->{prods}{$name}{files}=[];
      my $type=lc($t);
      if(!defined $bf){$bf=&SCRAMGenUtils::readBuildFile($bf1);}
      if((exists $bf->{$type}{$name}) && (exists $bf->{$type}{$name}{file}))
      {
	foreach my $f (@{$bf->{$type}{$name}{file}})
	{
	  foreach my $f1 (split /\,/,$f)
	  {
	    foreach my $f2 (`ls ${release}/src/${d}/$f1 2> /dev/null`)
	    {
	      chomp $f2;
	      $f2=~s/^${release}\/src\///;
	      $f2=&SCRAMGenUtils::fixPath($f2);
	      my $fd=dirname($f2);
	      push @{$data->{prods}{$name}{files}},$f2;
	    }
	  }
	}
      }
      if(scalar(@{$data->{prods}{$name}{files}})==0){delete $data->{prods}{$prod};}
    }
  }
}

sub getFlags ()
{
  my $cache=shift;
  my @prods=keys %{$cache};
  my $s=0;
  my $t=scalar(@prods);
  my $e=$t;
  my $m=200;
  while($e>0)
  {
    if($e>$m){$e=$m;}
    my $rules="";
    my $prod="";
    while($e>0)
    {
      $prod=$prods[$s++];$e--;
      foreach my $f ("CPPFLAGS", "CXXFLAGS"){$rules="$rules echo_${prod}_${f}";}
    }
    $e=$t-$s;
    if($rules eq ""){next;}
    foreach my $output (`cd $tmprel; $SCRAM_CMD b -f $rules 2>&1`)
    {
      chomp $output;
      if ($output=~/^\s*((.+?)_(CPPFLAGS|CXXFLAGS))\s+=\s+(.*)/)
      {
        $prod=$2;
        my $val=$4;
        if (exists $cache->{$prod})
        {
	  my $oldval=$cache->{$prod}{flags};
	  $cache->{$prod}{flags}="$oldval ".&postProcessFlag($val);
        }
      }
    }
  }
}

sub postProcessFlag ()
{
  my $l=shift;
  my $new="";
  while($l=~/^(.*?\s+\-D[^\s]+=)"(.*)$/)
  {
    $new="$new $1'\"";
    my $rest=$2;
    my $esc=0;
    my $done=0;
    foreach my $ch (split //, $rest)
    {
      if($done){$l="$l$ch";}
      elsif($ecs){$esc=0;$new="$new$ch";}
      elsif($ch eq '\\'){$esc=1;$new="$new$ch";}
      elsif($ch eq '"'){$done=1;$new="$new$ch'";$l="";}
      else{$new="$new$ch";}
    }
  }
  return "$new $l";
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
{return "lcg_coral_".basename(shift);}

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
#############################################
# getting interface file directory name
#############################################
sub pkg_interface_pool ()
{return basename(shift);}
sub pkg_interface_seal ()
{return basename(shift);}
sub pkg_interface_coral ()
{return basename(shift);}

sub pkg_interface_ignominy ()
{return "interface";}
sub pkg_interface_iguana ()
{return "interface";}
sub pkg_interface_cmssw ()
{return "interface";}

sub pkg_src_pool ()
{return "src";}
sub pkg_src_seal ()
{return "src";}
sub pkg_src_coral ()
{return "src";}

sub pkg_src_ignominy ()
{return "src";}
sub pkg_src_iguana ()
{return "src";}
sub pkg_src_cmssw ()
{return "src";}
########################################################################

sub checkTemplateCompilationSupport ()
{
  return 0;
  my $dir=&SCRAMGenUtils::getTmpDir();
  system("echo \"template <class  T>class A{A(T t){std::cout <<std::endl;}};\" > ${dir}/test.cc");
  my $data=`cd ${dir}; $cxx -c -fsyntax-only test.cc 2>&1`;
  my $tmpl=$?;
  system("rm -rf $dir");
  if($tmpl != 0){$tmpl=1;}
  return $tmpl;
}

sub checkDefineCompilationSupport ()
{
  return 0;
}
