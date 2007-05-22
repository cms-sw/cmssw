#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use Getopt::Long;
use SCRAMGenUtils;
$|=1;

my $curdir=`/bin/pwd`; chomp $curdir;
my $dir=shift || $curdir;
if($dir=~/^[^\/]/){$dir=&SCRAMGenUtils::fixPath("${curdir}/${dir}");}
my $arch=shift || "";
my $scram_list_skip=5;
my $flags="";
my $tmprel="";

my $release=&SCRAMGenUtils::scramReleaseTop($dir);
if($release eq ""){print "\"$dir\" is not SCRAM-base project area.\n";exit 1;}
if($release eq $dir){$dir.="/src";}

my $config=&SCRAMGenUtils::getFromEnvironmentFile("SCRAM_CONFIGDIR","$release") || "config";
my $src=&SCRAMGenUtils::getFromEnvironmentFile("SCRAM_SOURCEDIR","$release") || "src";
my $releasetop=&SCRAMGenUtils::getFromEnvironmentFile("RELEASETOP","$release") ||$release;
my $project=&SCRAMGenUtils::getFromEnvironmentFile("SCRAM_PROJECTNAME","$release");
if($release eq $dir){$dir.="/${src}";}

if (!-f "${release}/${config}/scram_version"){print "${release}/${config}/scram_version file does not exist.\n"; exit 1;}
if (!-d "${release}/${src}"){print "${release}/${src} directory does not exist. No sources to work on.\n"; exit 1;}

my $SCRAM_CMD="scram";
my $SCRAM_VER=`cat ${release}/${config}/scram_version`; chomp $SCRAM_VER;
if($SCRAM_VER=~/^\s*V1_/){$SCRAM_CMD="scramv1";}
elsif($SCRAM_VER=~/^\s*V0_/){$SCRAM_CMD="scram";}
if($arch eq ""){$arch=`$SCRAM_CMD arch`; chomp $arch;}
$SCRAM_CMD="$SCRAM_CMD -arch $arch";
$SCRAMGenUtils::SCRAM_CMD=$SCRAM_CMD;

print "DATA:OWNHEADER=^(.+?)\\/src\\/(.+?)\\.[^\\.]+:\"\$1/.+?/\$2\.h\"\n";
print "DATA:BASE_DIR=${release}/${src}\n";
$flags="-I${release}/${src}";
if($releasetop ne $release){print "DATA:BASE_DIR=${releasetop}/${src}\n";$flags.=" -I${releasetop}/${src}";}

my $tmprel=&SCRAMGenUtils::createTmpReleaseArea($releasetop);
system("cd $tmprel; $SCRAM_CMD b -r echo_CXX 2>&1 > /dev/null");
if(-f "${release}/tmp/${arch}/Makefile"){system("cp ${release}/tmp/${arch}/Makefile ${tmprel}/tmp/${arch}");}
my $cxx=&SCRAMGenUtils::getBuildVariable($tmprel,"CXX") || "/usr/bin/c++";
my $cxx_ver="";
if(-x $cxx){$cxx_ver=`$cxx --version | head -1 | awk '{print \$3}'`;chomp $cxx_ver;}

print "DATA:COMPILER=$cxx\n";
print "DATA:COMPILER_VERSION=$cxx_ver\n";
  
&genConfig ($dir);

foreach my $t (`cd $tmprel; $SCRAM_CMD tool list | tail +$scram_list_skip | awk '{print \$1}'`)
{
  chomp $t;
  if($t=~/^\s*$/){next;}
  foreach my $line (`cd $tmprel; $SCRAM_CMD tool info $t 2>&1`)
  {
    chomp $line;
    if ($line=~/^\s*INCLUDE\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $l (split /\s+/, $line){$flags.=" -I$l";}
    }
  }
}

foreach my $inc (split /\s+/, &SCRAMGenUtils::getBuildVariable($tmprel,"CPPDEFINES"))
{$flags="$flags ".postProcessFlag ("-D$inc");}
$flags="$flags ".&postProcessFlag(&SCRAMGenUtils::getBuildVariable($tmprel,"CPPFLAGS"));
$flags="$flags ".&postProcessFlag(&SCRAMGenUtils::getBuildVariable($tmprel,"CXXFLAGS"));
print "DATA:COMPILER_FLAGS=$flags\n";
&genSkip ("${release}/${src}");
&final_exit(0);

sub final_exit ()
{
  my $code=shift || 0;
  if($tmprel ne ""){system("rm -rf $tmprel");}
  exit $code;
}

sub genSkip ()
{
  my $dir=shift;
  my $dref;
  if(!opendir($dref,$dir))
  {
    print STDERR  "Can not open directory \"$dir\" for reading.\n";
    &final_exit(1);
  }
  my @files=readdir($dref);
  closedir($dref);
  foreach my $file (@files)
  {
    if($file=~/^\./){next;}
    if($file=~/^(CVS|data|html|doc|admin|scripts|java|perl|BuildFile|bin|test|python)$/){next;}
    $file="${dir}/${file}";
    if(-d "$file"){&genSkip ($file);next;}
    if(!-f $file){next;}
    if($file!~/^.+?\/(src|interface)\/[^\/]+$/){next;}
    if($file!~/^.+?\/[^\.]+?\.(h|hpp|hh)$/){next;}
    if($file=~/.+?[Ff][Ww][Dd]\.h$/)
    {
      $file=~s/${release}\/${src}\///;
      print "DATA:SKIP_FILES=$file\n";
    }
    elsif($file=~/.*\.(h|hpp|hh)$/)
    {
      my $def=0;
      my $tmpl=0;
      if(($cxx_ver=~/^3\.[4-9]/) || ($cxx_ver=~/^[4-9]\.\d+/))
      {$tmpl=1;}
      foreach my $line (`cat $file`)
      {
        chomp $line;
	if(($def==0) && ($line=~/^\s*\#\s*define\s+.+?\\$/))
	{
	  $file=~s/${release}\/${src}\///;
	  print "DATA:SKIP_FILES=$file\n";
	  last;
	}
	if(($tmpl==0) && ($line=~/^\s*template\s*<.+/))
	{
	  $file=~s/${release}\/${src}\///;
	  print "DATA:SKIP_FILES=$file\n";
	  last;
	}
      }
    }
  }
}

sub genConfig ()
{
  my $dir=shift;
  my @srcs=();
  &getAllSource ($dir, \@srcs);
  my %dirs=();
  foreach my $s (@srcs)
  {
    $dir=dirname($s);
    if(basename($dir) eq "interface"){next;}
    $dirs{$dir}{FILES}{$s}=1;
    $dirs{$dir}{FLAGS}="";
  }
  foreach my $s (@srcs)
  {
    $dir=dirname($s);
    my $dir1=basename($dir);
    if($dir1 eq "src"){next;}
    if($dir1 eq "interface")
    {$dir=dirname($dir)."/src";}
    if(exists $dirs{$dir})
    {
      $dirs{$dir}{FILES}{$s}=1;
      $dirs{$dir}{FLAGS}="";
    }
  }
  &getFlags (\%dirs);
  foreach my $dir (keys %dirs)
  {
    my $dir1=basename($dir);
    print "DATA:COMPILER_FLAGS=".$dirs{$dir}{FLAGS}."\n";
    foreach my $f (keys %{$dirs{$dir}{FILES}})
    {print "DATA:FILES=$f\n";}
  }
}

sub getFlags ()
{
  my $cache=shift;
  my @dirs=sort keys %$cache;
  my $s=0;
  my $t=scalar(@dirs);
  my $e=$t;
  my $m=200;
  while($e>0)
  {
    if($e>$m){$e=$m;}
    my $rules="";
    my %c1=();
    while($e>0)
    {
      my $dir=$dirs[$s++];$e--;
      my $func="safename_".$project;
      my $rule=&$func($dir);
      foreach my $f ("CPPFLAGS", "CXXFLAGS")
      {
        my $flag="${rule}_${f}";
        $c1{$flag}=$dir;
        $rules="$rules echo_${flag}";
      }
    }
    $e=$t-$s;
    foreach my $output (`cd $tmprel; $SCRAM_CMD b -f $rules 2>&1`)
    {
      chomp $output;
      my $var="";
      my $val="";
      if ($output=~/^\s*([^\s]+?)\s+=\s+(.*)/)
      {
        my $var=$1;
        my $val=$2;
        if (exists $c1{$var})
        {
	  my $oldval=$cache->{$c1{$var}}{FLAGS};
	  $cache->{$c1{$var}}{FLAGS}="$oldval ".&postProcessFlag($val);
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

sub getAllSource ()
{
  my $dir=shift;
  my $data=shift;
  my $dref;
  if(!opendir($dref,$dir)){print STDERR "Can not open directory \"$dir\" for reading.\n"; &final_exit(1);}
  my @files=readdir($dref);
  closedir($dref);
  foreach my $f (@files)
  {
    if(($f=~/^\./) || ($f=~/^(CVS|domain|doc|html|admin|bin|test|tests|scripts|plugins|plugin|data)$/)){next;}
    my $f="${dir}/$f";
    if(-d $f){&getAllSource ($f,$data);}
    elsif((-f $f) && ($f=~/^.+?\/(src|interface)\/([^\/\.]+?)\.(h|hh|cc|cpp|CC|CPP)$/))
    {
      $f=~s/^${release}\/${src}\/(.+)/$1/;
      push @{$data}, $f;
    }
  }
}

sub safename_CMSSW ()
{return &safename_based_on_subsystem_package(shift);}

sub safename_IGUANA ()
{return &safename_based_on_package(shift);}

sub safename_IGNOMINY ()
{return &safename_based_on_package(shift);}

sub safename_based_on_package ()
{
  my $dir=shift;
  if($dir=~/^([^\/]+?)\/([^\/]+?)\/.+$/)
  {return $2;}
  return "";
}

sub safename_based_on_subsystem_package ()
{
  my $dir=shift;
  if($dir=~/^([^\/]+?)\/([^\/]+?)\/.+$/)
  {return "${1}${2}";}
  return "";
}


