#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use Getopt::Long;
use File::Path;
use SCRAMGenUtils;

if(-t STDIN){die "Please run createBuildFile.pl script. $0 is not suppose to run directly.\n";}

$|=1;
#get the command-line options
if(&GetOptions(
	       "--files=s",\@files,
	       "--dir=s",\$dir,
	       "--prodname=s",\$prodname,
	       "--prodtype=s",\$prodtype,
	       "--use=s",\@use,
	       "--no-use=s",\@nuse,
	       "--export=s",\@export,
	       "--no-export=s",\@nexport,
	       "--replace=s",\%replace,
	       "--flag=s",\@flags,
	       "--ref-bf=s",\$refbf,
	       "--dictclasses=s",\$dictclasses,
	       "--linkdef=s",\$linkdef,
	       "--iglet=s",\$iglet,
	       "--tmpdir=s",\$tmpdir,
	       "--buildfile=s",\$buildfilename,
	       "--config=s",\$configfile,
	       "--jobs=i",\$jobs,
	       "--help",\$help,
	       "--plugin",\$plugin,
	       "--clean",\$clean,
	       "--xml",\$xml,
	       "--detail",\$detail,
              ) eq ""){print STDERR "#Wrong arguments.\n"; &usage_msg();}

my $compiler="cxxcompiler";
my $srcext="cpp|cc|c|cxx|f|f77";
my $hdext="h|hh|inc|ii|i|icpp|icc";
my $pwd=`/bin/pwd`; chomp $pwd;
my $cache={};
$SCRAMGenUtils::InternalCache=$cache;
$SCRAMGenUtils::CacheType=1;

if(defined $help){&usage_msg();}
if(defined $plugin){$plugin=1;}
else{$plugin=0;}
if(defined $xml){$xml="--xml";}
else{$xml="";}
if(defined $detail){$detail=1;}
else{$detail=0;}
if (! defined $jobs){$jobs=10;}
$SCRAMGenUtils::DEBUG=$detail;

if((!defined $prodname) || ($prodname=~/^\s*$/)){$prodname="";}
else{$prodname=~s/^\s*//;$prodname=~s/\s*$//;}
if((!defined $dictclasses) || ($dictclasses=~/^\s*$/)){$dictclasses='^.+?\/classes\.h$';}
if((!defined $linkdef) || ($linkdef=~/^\s*$/)){$linkdef='LinkDef\.h$';}
if((!defined $iglet) || ($iglet=~/^\s*$/)){$iglet='iglet\.cc$';}
if((defined $prodtype) && ($prodtype!~/^(library|bin)$/)){print STDERR "Product type could only be \"library\" or \"bin\".\n"; exit 1;}

if((!defined $dir) || ($dir=~/^\s$/)){print STDERR "Please use --dir <dir> for which you want to find the dependency information.\n"; exit 1;}
if($dir!~/^\//){$dir="${pwd}/${dir}";}
$dir=&SCRAMGenUtils::fixPath($dir);
my $release=&SCRAMGenUtils::scramReleaseTop($dir);
if($release eq ""){print STDERR "ERROR: Directory \"$dir\" does not exist under a SCRAM-based project area.\n";exit 1;}
my $releasetop=&SCRAMGenUtils::getFromEnvironmentFile("RELEASETOP",$release);
my $envcache=&SCRAMGenUtils::getEnvironmentFileCache($release);
&SCRAMGenUtils::init($release);

if((defined $refbf) && ($refbf!~/^\s*$/))
{if(!-f $refbf){print STDERR "ERROR: Reference BuildFile \"$refbf\" does not exist.\n"; exit 1;}}
else{$refbf="";}
if ($refbf eq "")
{
  my $bf="${dir}/BuildFile.xml";
  if(!-f $bf){$bf="${dir}/BuildFile";}
  if (-f $bf){$refbf=$bf;}
}
if ($refbf=~/\.xml$/){$xml="--xml";}

if((!defined $buildfilename) || ($buildfilename=~/^\s*$/)){$buildfilename="";}
elsif($buildfilename!~/^\//){$buildfilename="${pwd}/${buildfilename}";}
if($buildfilename)
{
  $buildfilename=&SCRAMGenUtils::fixPath($buildfilename);
  my $d=dirname($buildfilename);
  if(!-d $d){system("mkdir -p $d");}
}

my $project=&SCRAMGenUtils::getFromEnvironmentFile("SCRAM_PROJECTNAME","$release");
$project=lc($project);

my $pkgsrcdir=&run_func("pkg_src",$project,$dir);
if ($pkgsrcdir eq ""){print STDERR "ERROR: Script not ready yet to work for \"$project\" SCRAM-based project.\n"; exit 1;}
my $pkginterfacedir=&run_func("pkg_interface",$project,$dir);

my $scramarch=&SCRAMGenUtils::getScramArch();
if((!defined $tmpdir) || ($tmpdir=~/^\s*$/))
{
  $tmpdir="${release}/tmp/AutoBuildFile";
  if($pwd!~/^$release(\/.*|)$/){$tmpdir="${pwd}/AutoBuildFile";}
}

my $ccfiles=0;
my $isPackage=0;
my $PackageName="";
my $filestr="";
if(scalar(@files)==0)
{
  if(-d "${dir}/${pkgsrcdir}")
  {
    foreach my $f (&SCRAMGenUtils::readDir("${dir}/${pkgsrcdir}",2))
    {
      if($f=~/\.($srcext)$/i){push @files,"${dir}/${pkgsrcdir}/${f}";$ccfiles++;}
      elsif($f=~/\.($hdext)$/i){push @files,"${dir}/${pkgsrcdir}/${f}";}
    }
  }
  if(-d "${dir}/${pkginterfacedir}")
  {
    foreach my $f (&SCRAMGenUtils::readDir("${dir}/${pkginterfacedir}",2))
    {
      if($f=~/\.($hdext)$/i){push @files,"${dir}/${pkginterfacedir}/${f}";}
      elsif($f=~/\.($srcext)$/i){push @files,"${dir}/${pkginterfacedir}/${f}";}
    }
  }
  if(scalar(@files)==0)
  {
    if ($refbf && $buildfilename)
    {
      my $bf=basename($refbf);
      if ($buildfilename=~/\/$bf\.auto$/){system("cp $refbf $buildfilename");}
      else{&SCRAMGenUtils::convert2XMLBuildFile($refbf,$buildfilename);}
    }
    exit 0;
  }
  else
  {
    $isPackage=1;
    $PackageName=$dir; $PackageName=~s/^$release\/src\///;
    if ($prodname eq ""){$prodname=&run_func("safename",$project,$dir);}
  }
}
else
{
  my @xf=@files;
  @files=();
  foreach my $f (@xf)
  {
    foreach my $f1 (split /\s*,\s*/,$f)
    {foreach my $f2 (split /\s+/,$f1){if($f2 ne ""){push @files,$f2;}}}
  }
  $filestr=join(",",@files);
  @xf=();
  foreach my $f (@files)
  {
    my @fs=();
    foreach my $f1 (`ls ${dir}/$f`)
    {
      chomp $f1;
      if(-f $f1)
      {
        $f1=&SCRAMGenUtils::fixPath($f1);
        push @xf,$f1;
        push @fs,$f1;
      }
    }
    if(scalar(@fs)==0)
    {
      my $found=0;
      if($f!~/\*/)
      {
        if($f=~/^(.+?)\.([^\.]+)$/)
	{
	  my $f1=$1;
	  my $e=$2;
	  foreach my $x ("cc","cpp","cxx","C")
	  {
	    if($x eq $e){next;}
	    if(-f "${dir}/${f1}.${x}")
	    {
	      $f1="${dir}/${f1}.${x}";
	      push @xf,&SCRAMGenUtils::fixPath($f1);
	      $found=1;
	      print STDERR "FILE REPLACE: $f => $f1\n";
	      last;
	    }
	  }
	}
      }
      if(!$found){print STDERR "WARNING: No such file \"$f\" under directory \"$dir\".\n";}
    }
  }
  @files=();
  @files=@xf;
  if(scalar(@files)>0)
  {
    foreach my $f (@files)
    {
      if($f!~/\.($srcext)$/i)
      {print STDERR "ERROR: Only files with extensions \"$srcext\" should be passed with --files command-line arg.\n";exit 1;}
    }
  }
  $ccfiles++;
  if ($prodname eq "")
  {
    if(scalar(@files)==1){$prodname=basename($files[0]);$prodname=~s/\.[^\.]+$//;}
    else{print STDERR "You should also use \"--name <product_name>\" when \"--files <file>\" command-line arguments are used.\n";exit 1;}
  }
}

my $data={};
$data->{files}={};
$data->{filter}=$dir;

foreach my $f ("EDM_PLUGIN", "SEALPLUGIN", "SEAL_PLUGIN_NAME", "NO_LIB_CHECKING", "GENREFLEX_FAILES_ON_WARNS", "ROOTMAP", "ADD_SUBDIR", "CODEGENPATH"){$data->{sflags}{$f}=1;}
foreach my $f ("CPPDEFINES"){$data->{keyflags}{$f}=1;}

$data->{bfflags}=[];
foreach my $f (@flags){push @{$data->{bfflags}},$f;}

my $cachedir="${tmpdir}/${scramarch}";
my $incdir="${cachedir}/includes";
if(!-d $incdir){system("mkdir -p $incdir");}
my $cachefile="${cachedir}/toolcache";
my $inccache={};
if((!defined $clean) && (-f "$cachefile"))
{
  print STDERR "Reading previously save internal cache $cachefile.\n";
  $cache=&SCRAMGenUtils::readHashCache("$cachefile");
  $SCRAMGenUtils::InternalCache=$cache;
}
else
{
  print STDERR "Reading project's SCRAM cache\n";
  &update_project_cache($cache);
  $cache->{IGNORE_BASETOOLS}{rootrflx}=1;
  $cache->{IGNORE_BASETOOLS}{boost_python}=1;
  &save_toolcache();
}

if(exists $cache->{COMPILER})
{
  $data->{compilecmd}=$cache->{COMPILER};
  $data->{compileflags}=$cache->{CXXFLAGS}." ".$cache->{CXXINCLUDE};
}
else{print STDERR "#WARNING: No compiler found. So script is not going to parse the file for seal plugin macros.\n";}

$configfile=&SCRAMGenUtils::updateConfigFileData($configfile,$data,$caahe);

my $igletfile="";
my $srcplugin="";
my $srcedmplugin="";
my $fortrancode=0;
my $castor=0;
my $cacherefbf=undef;
my $exflags="";
$data->{prodtype}=$prodtype;
$data->{prodname}=$prodname;
$data->{extra_include_path}={};
if($refbf && (-f $refbf))
{
  $cacherefbf=&SCRAMGenUtils::readBuildFile($refbf);
  my $f=&SCRAMGenUtils::findBuildFileTag($data,$cacherefbf,"flags");
  foreach my $a (keys %$f)
  {
    foreach my $c (@{$f->{$a}})
    {
      foreach my $f1 (keys %{$c->{flags}})
      {
        my $v=$c->{flags}{$f1};
	if($f1 eq "CPPDEFINES"){foreach my $fv (@$v){$exflags .=" -D".&replaceVariables($fv->{v});}}
	elsif($f1 eq "CXXFLAGS"){foreach my $fv (@$v){$exflags.="   ".&replaceVariables($fv->{v});}}
	elsif($f1 eq "CPPFLAGS"){foreach my $fv (@$v){$exflags.="   ".&replaceVariables($fv->{v});}}
	elsif($f1 eq "LCG_DICT_HEADER")
	{
	  my $lcgreg="";
	  foreach my $fv (@$v)
	  {
	    my $files=$fv->{v};
	    $files=~s/\,/ /g;
	    foreach my $fn (split /\s+/, $files)
	    {
	      if($lcgreg){$lcgreg.="|$fn";}
	      else{$lcgreg="$fn";}
	    }
	  }
	  $dictclasses="^.+?\/(".$lcgreg.")\$";
	}
      }
    }
  }
  $f=&SCRAMGenUtils::findBuildFileTag($data,$cacherefbf,"include_path");
  foreach my $a (keys %$f)
  {
    foreach my $c (@{$f->{$a}})
    {
      foreach my $f1 (keys %{$c->{include_path}})
      {
        my $f2=&replaceVariables($f1);
	if(($f2!~/^\s*$/) && (-d $f2)){$exflags.=" -I$f2";$data->{extra_include_path}{$f2}=1;}
      }
    }
  }
}

if($isPackage){$prodtype="";}
if($isPackage || ($prodtype eq "library"))
{
  delete $data->{PROD_TYPE_SEARCH_RULES}{main};
  my $d=$dir;
  if($isPackage){$d="${dir}/${pkgsrcdir}";}
  elsif(scalar(@files)>0)
  {
    my $f=$files[0];
    if($f=~/^(.+?)\/[^\/]+$/){$d=$1;}
  }
  if(-d $d)
  {
    foreach my $f (&SCRAMGenUtils::readDir($d,2))
    {
      $f=&SCRAMGenUtils::fixPath("${d}/${f}");
      if($f=~/$dictclasses/)
      {
        $plugin=-1;
	$data->{deps}{src}{rootrflx}=1;
	$ccfiles++;
	my $found=0;
	foreach my $f1 (@files){if($f1 eq $f){$found=1;last;}}
	if(!$found){push @files,$f;}
      }
      elsif($f=~/$linkdef/){$data->{deps}{src}{rootcintex}=1;$ccfiles++;}
      elsif($f=~/$iglet/){$igletfile=$f;}
    }
  }
}
my $fullpath=&findProductInRelease($prodname);
print STDERR "Reading source files\n";
my $srcfiles=[];
foreach my $file (@files)
{
  if ($file=~/\.($srcext)$/i)
  {
    if($file!~/\.(f|f77)$/i){push @$srcfiles,$file;}
    elsif(!$fortrancode){$fortrancode=1;print STDERR "FORTRAN FILE:$file\n";}
  }
}
if (scalar(@$srcfiles)>0)
{
  unlink "${cachedir}/searchPreprocessedInfo";
  my $pid=&SCRAMGenUtils::forkProcess($jobs);
  if($pid==0)
  {
    &SCRAMGenUtils::searchPreprocessedFile($srcfiles,$data,$exflags);
    &SCRAMGenUtils::writeHashCache($data->{PROD_TYPE_SEARCH_RULES},"${cachedir}/searchPreprocessedInfo");
    exit 0;
  }
}
foreach my $file (@files){&process_cxx_file ($file,$data);}
if(!$detail){print STDERR "\n";}
my $bindeps=&getBinaryDependency($prodname,$fullpath);
my $tid=&SCRAMGenUtils::startTimer ();
&SCRAMGenUtils::waitForChild();
if($detail){print STDERR "WAIT TIME:",&SCRAMGenUtils::stopTimer($tid),"\n";}
if ((scalar(@$srcfiles)>0) && (-f "${cachedir}/searchPreprocessedInfo"))
{
  $data->{PROD_TYPE_SEARCH_RULES}=&SCRAMGenUtils::readHashCache("${cachedir}/searchPreprocessedInfo");
  if((exists $data->{PROD_TYPE_SEARCH_RULES}{sealplugin}) && (exists $data->{PROD_TYPE_SEARCH_RULES}{sealplugin}{file}))
  {
    $srcplugin=$data->{PROD_TYPE_SEARCH_RULES}{sealplugin}{file};
    delete $data->{PROD_TYPE_SEARCH_RULES}{sealplugin};
  }
  if((exists $data->{PROD_TYPE_SEARCH_RULES}{edmplugin}) && (exists $data->{PROD_TYPE_SEARCH_RULES}{edmplugin}{file}))
  {
    $srcedmplugin=$data->{PROD_TYPE_SEARCH_RULES}{edmplugin}{file};
    delete $data->{PROD_TYPE_SEARCH_RULES}{edmplugin};
  }
  if((exists $data->{PROD_TYPE_SEARCH_RULES}{main}) && (exists $data->{PROD_TYPE_SEARCH_RULES}{main}{file}))
  {
    my $f=$data->{PROD_TYPE_SEARCH_RULES}{main}{file};
    if(($prodtype ne "") && ($prodtype ne "bin"))
    {print STDERR "\"$prodname\" seemed like a \"bin\" product because there is \"main()\" exists in \"f\" file.\n";}
    else{$prodtype="bin";}
    if($detail){print STDERR "Executable:$prodname:$f\n";}
    delete $data->{PROD_TYPE_SEARCH_RULES}{main};
  }
  if((exists $data->{PROD_TYPE_SEARCH_RULES}{castor}) && (exists $data->{PROD_TYPE_SEARCH_RULES}{castor}{file}))
  {
    my $f=$data->{PROD_TYPE_SEARCH_RULES}{castor}{file};
    if($detail){print STDERR "Castor Dependency:$prodname:$f\n";}
    $castor=1;
    delete $data->{PROD_TYPE_SEARCH_RULES}{castor};
  }
}
print STDERR "....\n";
$data->{ccfiles}=$ccfiles;
$data->{isPackage}=$isPackage;
$data->{filestr}=$filestr;

my $pack=$dir;
$pack=~s/^$release\/src\///;
if($detail && $srcplugin){print STDERR "SealPlugin:$prodname:$srcplugin\n";}
if($detail && $srcedmplugin){print STDERR "EDMPlugin:$prodname:$srcedmplugin\n";}
if ($fullpath && ($fullpath=~/\/plugin.+\.so$/) && (($srcedmplugin eq "") || ($srcplugin)))
{
  $srcedmplugin="(forced to be an EDM plugin as its build as EDM plugin in release)";
  $srcplugin="";
}

if($fortrancode){foreach my $t ("pythia6","genser"){if (exists $cache->{TOOLS}{$t}){$data->{deps}{src}{$t}=1;}}}
if($plugin==-1)
{
  my $err=0;
  if (($srcplugin ne "") || ($srcedmplugin ne ""))
  {
    print STDERR "#WARNING: You have LCG disctionaries file(s) matching regexp \"$dictclasses\" in your package \"$pack\" and also you have SEAL/EDM PLUGIN MODULE macro defined.\n";
    $err=1;
  }
  if($err)
  {
    if($srcplugin ne ""){print STDERR "#WARNING: Seal Plugin File: $srcplugin\n";}
    if($srcedmplugin ne ""){print STDERR "#WARNING: EDM Plugin File : $srcedmplugin\n";}
    print STDERR "#WARNING: Packages which generate LCG disctionaries are not suppose to be seal/edm plugins. So please fix your package.\n";
  }
}
elsif(($srcplugin ne "") && ($srcedmplugin ne ""))
{
  print STDERR "#WARNING: Your package \"$pack\" has macros for both seal and edm plugins. You are suppose to have macros for only one plugin, please fix this.\n";
  print STDERR "#WARNING: Seal Plugin File: $srcplugin\n";
  print STDERR "#WARNING: EDM Plugin File : $srcedmplugin\n";
  $plugin=-1;
}
elsif(($srcplugin ne "") || ($srcedmplugin ne "")){$plugin=1;}

my $defaultplugintype=uc(&run_func("defaultplugin",$project));
if(($dir=~/\/sealplugins$/) || ($dir=~/\/plugins$/))
{
  if($plugin <= 0){push @{$data->{bfflags}},"${defaultplugintype}PLUGIN=0";}
  elsif($srcedmplugin ne ""){push @{$data->{bfflags}},"EDM_PLUGIN=1";}
  elsif($srcplugin ne ""){push @{$data->{bfflags}},"SEALPLUGIN=1";}
}
elsif($plugin>0)
{
  if($srcplugin ne ""){push @{$data->{bfflags}},"SEALPLUGIN=1";}
  elsif($srcedmplugin ne ""){push @{$data->{bfflags}},"EDM_PLUGIN=1";}
}

#read ref BuildFile for extra flags
&SCRAMGenUtils::updateFromRefBuildFile($cacherefbf,$data);

# Extra flags
foreach my $f (@{$data->{bfflags}})
{
  my ($n,$v)=split /=/,$f,2;
  if(($f=~/^$n=/) && ($n!~/^\s*$/))
  {
    $v=~s/^\s*//;$v=~s/\s*$//;
    $n=~s/^\s*//;$n=~s/\s*$//;
    $n=uc($n);
    if(exists $data->{sflags}{$n})
    {
      if($n eq "SEAL_PLUGIN_NAME"){$n="SEALPLUGIN", $v=1;}
      if(($n eq "SEALPLUGIN") && ($plugin<0)){next;}
      if(($n eq "EDM_PLUGIN") && ($plugin<0)){next;}
      $data->{flags}{$n}=$v;
    }
    else
    {
      if(!exists $data->{flags}{$n}){$data->{flags}{$n}=[];}
      push @{$data->{flags}{$n}},$v;
    }
  }
}

# Extra tool/package to export
foreach my $x (&commaSepDeps(\@export,$cache,"${prodname}export")){$data->{deps}{src}{$x}=1;}

# Extra tool/package not to export
foreach my $x (&commaSepDeps(\@nexport,$cache,"${prodname}no-export"))
{
  if(exists $data->{deps}{src}{$x}){print STDERR "MSG:Removed dependency on \"$x\" due to no-export arguments.\n";}
  delete $data->{deps}{src}{$x};
  $data->{NO_EXPORT}{$x}=1;
}

# Extra tool/package to use
foreach my $x (&commaSepDeps(\@use,$cache,"${prodname}use")){$data->{deps}{src}{$x}=1;}

# Extra tool/package not to use
foreach my $x (&commaSepDeps(\@nuse,$cache,"${prodname}no-use"))
{
  if(exists $data->{deps}{src}{$x}){print STDERR "MSG:Removed dependency on \"$x\" due to no-use arguments.\n";}
  delete $data->{deps}{src}{$x};
   $data->{NO_USE}{$x}=1;
}

# Extra tool/package replacement
foreach my $x (keys %replace)
{
  my $v=$replace{$x};
  my $lx=lc($x); my $lv=lc($v);
  if(exists $cache->{TOOLS}{$lx}){$x=$lx;}
  if(exists $cache->{TOOLS}{$lv}){$v=$lv;}
  $data->{replace}{$x}=$v;
}

foreach my $dep (keys %{$data->{deps}{src}})
{if(exists $data->{replace}{$dep}){delete $data->{deps}{src}{$dep}; $data->{deps}{src}{$data->{replace}{$dep}}=1;}}

foreach my $u (keys %{$data->{deps}{src}})
{
  my $rep=&parentCommunication("PRODUCT_INFO",$u);
  my $exprocess=0;
  if ($rep ne "NOT_EXISTS") 
  {
    if ($rep eq "PROCESS"){&parentCommunication("PLEASE_PROCESS_FIRST",$u);$exprocess=1;}
  }
  elsif((-d "${release}/src/${u}/${pkgsrcdir}") || (-d "${release}/src/${u}/${pkginterfacedir}")){&parentCommunication("PLEASE_PROCESS_FIRST",$u);$exprocess=1;}
  if($detail && $exprocess){print STDERR "###### Back to processing of $prodname  #######\n";}
  my $rep=&parentCommunication("PACKAGE_TYPE",$u);
  if ($rep!~/^(TOOL|PACK)$/)
  {
    delete $data->{deps}{src}{$u};
    if ($detail){print STDERR "Deleting dependency \"$u\" due to its type \"$rep\"\n";}
  }
}

if(exists $data->{deps}{src}{castor}){$castor=0;}
&symbolCheck($bindeps);
&extraProcessing($prodname,$dir);
&SCRAMGenUtils::removeExtraLib ($cache,$data);
&SCRAMGenUtils::printBuildFile($data, "$buildfilename");
&final_exit("",0);
#########################################
#
sub findProductInRelease ()
{
  my $prod=shift;
  foreach my $dir ($release,$releasetop)
  {
    if($dir eq ""){next;}
    foreach my $xd (keys %{$data->{PRODUCT_SEARCH_PATHS}{PACK}})
    {
      foreach my $d ("${xd}/${scramarch}","${scramarch}/${xd}")
      {
        if (!-d "${dir}/${d}"){next;}
	if(-f "${dir}/${d}/lib${prod}.so"){return "${dir}/${d}/lib${prod}.so";}
        elsif(-f "${dir}/${d}/plugin${prod}.so"){return "${dir}/${d}/plugin${prod}.so";}
      }
    }
    foreach my $xd (keys %{$data->{PRODUCT_SEARCH_PATHS}{PROD}})
    {
      foreach my $d ("${xd}/${scramarch}","${scramarch}/${xd}")
      {
        if (!-d "${dir}/${d}"){next;}
	if(-f "${dir}/${d}/${prod}"){return "${dir}/${d}/${prod}";}
      }
    }
  }
  return "";
}

sub extraProcessing ()
{
  my $p=shift;
  my $dir=shift;
  if (exists $data->{EXTRA_TOOLS}{PRODUCTS})
  {
    my $c=$data->{EXTRA_TOOLS}{PRODUCTS};
    foreach my $exp (keys %$c)
    {if ($p=~/$exp/){foreach my $t (keys %{$c->{$exp}}){if (!exists $data->{deps}{src}{$t}){$data->{deps}{src}{$t}=1;print STDERR "Added dependency(forced):$t\n";}}}}
  }
  if (exists $data->{EXTRA_TOOLS}{PATHS})
  {
    my $c=$data->{EXTRA_TOOLS}{PATHS};
    foreach my $exp (keys %$c)
    {if ($dir=~/$exp/){foreach my $t (keys %{$c->{$exp}}){if (!exists $data->{deps}{src}{$t}){$data->{deps}{src}{$t}=1;print STDERR "Added dependency(forced):$t\n";}}}}
  }
  if (exists $data->{deps}{src})
  {
    if (exists $data->{EXTRA_TOOLS}{HASDEPS})
    {
      my $c=$data->{EXTRA_TOOLS}{HASDEPS};
      foreach my $t (keys %$c)
      {
        if(exists $data->{deps}{src}{$t})
	{
	  foreach my $d (keys %{$c->{$t}})
	  {
	    if((!exists $data->{deps}{src}{$d}) && (!$isPackage || ($PackageName ne $d))){$data->{deps}{src}{$d}=1;print STDERR "Added dependency(forced):$d (due to $t)\n";}
	  }
	}
      }	
    }
    if (exists $data->{REMOVE_TOOLS}{PRODUCTS})
    {
      my $c=$data->{REMOVE_TOOLS}{PRODUCTS};
      foreach my $exp (keys %$c)
      {if ($p=~/$exp/){foreach my $t (keys %{$c->{$exp}}){if (exists $data->{deps}{src}{$t}){delete $data->{deps}{src}{$t};print STDERR "Removed dependency(forced):$t\n";}}}}
    }
    if (exists $data->{REMOVE_TOOLS}{PATHS})
    {
      my $c=$data->{REMOVE_TOOLS}{PATHS};
      foreach my $exp (keys %$c)
      {if ($dir=~/$exp/){foreach my $t (keys %{$c->{$exp}}){if (exists $data->{deps}{src}{$t}){delete $data->{deps}{src}{$t};print STDERR "Removed dependency(forced):$t\n";}}}}
    }
  }
  if(exists $data->{flags})
  {
    if (exists $data->{REMOVE_FLAGS}{PRODUCTS})
    {
      my $c=$data->{REMOVE_FLAGS}{PRODUCTS};
      foreach my $exp (keys %$c)
      {if ($p=~/$exp/){foreach my $f (keys %{$c->{$exp}}){if (exists $data->{flags}{$f}){delete $data->{flags}{$f};print STDERR "Removed flag(forced):$f\n";}}}}
    }
    if (exists $data->{REMOVE_FLAGS}{PATHS})
    {
      my $c=$data->{REMOVE_FLAGS}{PATHS};
      foreach my $exp (keys %$c)
      {if ($dir=~/$exp/){foreach my $f (keys %{$c->{$exp}}){if (exists $data->{flags}{$f}){delete $data->{flags}{$f};print STDERR "Removed flag(forced):$f\n";}}}}
    }
  }
}

sub getBinaryDependency()
{
  my $p1=shift;
  my $p=shift;
  my $ts={};
  if ($p eq ""){print STDERR "WARNING: Could not find product \"$p1\". Going to seach for generated object files.\n";$p=$p1;}
  my $res=&parentCommunication("SYMBOL_CHECK_REQUEST",$p);
  if ($res ne "")
  {
    $ts=&SCRAMGenUtils::readHashCache($res);
    unlink $res;
  }
  return $ts;
}

sub processBinaryDeps ()
{
  my $ts=shift;
  my $depstr=join(",",keys %{$data->{deps}{src}});
  my %done=();
  my %btools=();
  foreach my $s (keys %$ts)
  {
    foreach my $t (keys %{$ts->{$s}})
    {
      if (exists $done{$t}){if ($done{$t}==1){delete $ts->{$s};last;}}
      my $d=0;
      if ((exists $data->{deps}{src}{$t}) || (exists $data->{IGNORE_SYMBOL_TOOLS}{$t})){$d=1;}
      else
      {
        my $skip=0;
	if (exists $cache->{BASETOOLS}{$t})
	{
	  foreach my $t1 (keys %{$cache->{BASETOOLS}{$t}})
	  {
	    if(exists $data->{deps}{src}{$t1}){$skip=1;$btools{$t1}=1;last;}
	  }
	}
	if(!$skip)
	{
	  my $res=&parentCommunication("HAVE_DEPS","$depstr:$t");
          if ($res=~/^YES:(.+)/){print "DELETING Indirectly exists via:$t ($1)\n";$d=1;}
	}
      }
      $done{$t}=$d;
      if($d){delete $ts->{$s};last;}
    }
  }
  foreach my $t (keys %btools){delete $data->{deps}{src}{$t};print STDERR "DELETED BASE TOOL:$t\n";}
  my $symcount=keys %$ts;
  my $tsx={};
  foreach my $s (keys %$ts)
  {
    my @t=keys %{$ts->{$s}};
    if(scalar(@t)==1)
    {
      $tsx->{$t[0]}{$s}=$ts->{$s}{$t[0]};
      delete $ts->{$s};
      $symcount--;
    }
  }
  if ($symcount)
  {
    foreach my $t (keys %$tsx){foreach my $s (keys %$ts){if(exists $ts->{$s}{$t}){delete $ts->{$s};$symcount--;}}}
    if ($symcount)
    {
      foreach my $s (keys %$ts)
      {
        foreach my $t (keys %{$data->{SAME_LIB_TOOL}})
	{
	  if(exists $ts->{$s}{$t})
	  {
	    foreach my $t1 (keys %{$data->{SAME_LIB_TOOL}{$t}}){if(exists $ts->{$s}{$t1}){delete $ts->{$s}{$t1};}}
	    if(scalar(keys %{$ts->{$s}})==1){$tsx->{$t}{$s}=$ts->{$s}{$t};delete $ts->{$s};$symcount--;last;}
	  }
	}
      }
    }
    if ($symcount && (defined $cacherefbf) && (exists $cacherefbf->{use}))
    {
      foreach my $s (keys %$ts)
      {
        foreach my $t (keys %{$ts->{$s}}){if(exists $cacherefbf->{use}{$t}){$tsx->{$t}{$s}=$ts->{$s}{$t};delete $ts->{$s};$symcount--;last;}}
      }
    }
  }
  my $depstr=join(",",keys %$tsx);
  foreach my $t (keys %$tsx)
  {
    my $res=&parentCommunication("HAVE_DEPS","$depstr:$t");
    if ($res=~/^YES:(.+)/){delete $tsx->{$t};print STDERR "1:DELETING Indirectly exists via:$t ($1)\n";}
  }
  if ($symcount)
  {
    print STDERR "WARNING: Following symbols are defined in multiple tools/packages\n";
    foreach my $s (keys %$ts)
    {
      my $s1=&SCRAMGenUtils::cppFilt ($s);
      print STDERR "  Symbol:$s1\n";
      foreach my $t (keys %{$ts->{$s}}){print STDERR "    Tool:$t\n";}
    }
  }
  return $tsx;
}

sub symbolCheck()
{
  my $tsx=&processBinaryDeps(shift);
  foreach my $t (keys %$tsx)
  {
    if($t eq "system"){next;}
    if (exists $data->{deps}{src}{$t}){next;}
    elsif(exists $data->{NO_USE}{$t}){next;}
    elsif(exists $data->{NO_EXPORT}{$t}){next;}
    $data->{deps}{src}{$t}=1;
    print STDERR "EXTRA TOOL due to missing symbols:$t\n";
    print STDERR "  SYMBOLS FOUND:\n";
    foreach my $s (keys %{$tsx->{$t}})
    {
      my $s1=&SCRAMGenUtils::cppFilt ($s);
      print STDERR "    $s1 =>$tsx->{$t}{$s}\n";
    }
  }
}

sub parentCommunication ()
{
  my $req=shift;
  my $msg=shift;
  print "$req:$msg\n";
  $req.="_DONE";
  my $rep=<STDIN>;chomp $rep;
  if($rep=~/^$req:\s*(.*)$/){$rep=$1;}
  else{print STDERR "$req FAILED\n$rep\n";$rep="";}
  return $rep;
}

##############################################################
#
sub process_cxx_file ()
{
  my $file=shift;
  my $data=shift;
  if(exists $data->{files}{$file}){return;}
  $data->{files}{$file}={};
  if($detail){print STDERR "Working on $file\n";}
  else{print STDERR ".";}
  &SCRAMGenUtils::searchIncFilesCXX($file,$data->{files}{$file});
  if(exists $data->{files}{$file}{includes})
  {
    my $selfdir=dirname($file);
    my $filter=$data->{filter};
    foreach my $inc (keys %{$data->{files}{$file}{includes}})
    {
      my $ainc=$inc;
      if($inc=~/^\./)
      {
	$inc=&SCRAMGenUtils::fixPath("${selfdir}/${inc}");
	if($inc=~/^$release\/src\/(.+)$/){$inc=$1;}
      }
      else{$inc=&SCRAMGenUtils::fixPath($inc);}
      if(exists $inccache->{UNKNOWN}{$inc}){next;}
      elsif(-f "${selfdir}/${inc}"){&process_cxx_file ("${selfdir}/${inc}",$data);}
      else
      {
	my $id="";
	my $info = &find_inc_file_path($inc);
	my $fpath=$info->{fullpath};
	if($fpath ne "")
	{
	  if ($fpath=~/^${filter}\/.+$/){&process_cxx_file ($fpath,$data);}
	  else
	  {
	    foreach my $pack (keys %{$info->{pack}})
	    {
	      if(("$pack" ne "$project") && ("$pack" ne "self"))
	      {
	        if($isPackage)
	        {
	          if($file=~/^${filter}\/${pkginterfacedir}\//){$data->{deps}{src}{$pack}=1;}
	          elsif($file=~/^${filter}\/${pkgsrcdir}\//){$data->{deps}{src}{$pack}=1;}
	        }
	        else{$data->{deps}{src}{$pack}=1;}
	      }
	      if($detail && ($info->{new}==1)){$info->{new}=2; print STDERR "#$ainc=>$pack\n";}
	    }
	  }
	}
	elsif($detail && ($info->{new}==1)){$info->{new}=2; print STDERR "#$ainc:UNKNOWN (might be from system directories)\n";}
      }
    }
  }
}

sub read_inc_cache()
{
  my $inc=shift;
  if (-z "${incdir}/${inc}/.info"){$inccache->{UNKNOWN}{$inc}=1;$inccache->{INC}{$inc}={};}
  else
  {
    my $ref;
    open($ref,"${incdir}/${inc}/.info") || die "Can not open file for reading:${incdir}/${inc}/.info\n";
    my $line=<$ref>; chomp $line;
    foreach my $p (split /:/,$line){$inccache->{INC}{$inc}{pack}{$p}=1;}
    $line=<$ref>; chomp $line;
    $inccache->{INC}{$inc}{fullpath}=$line;
    close($ref);
  }
  return $inccache->{INC}{$inc};
}

sub write_inc_cache()
{
  my $inc=shift;
  my $ref;
  open($ref,">${incdir}/${inc}/.info") || die "Can not open file for writing:${incdir}/${inc}/.info\n";
  if (!exists $inccache->{UNKNOWN}{$inc})
  {
    print $ref "",join(":",keys %{$inccache->{INC}{$inc}{pack}}),"\n";
    print $ref "",$inccache->{INC}{$inc}{fullpath},"\n";
  }
  close($ref);
}

sub find_inc_file_path ()
{
  my $inc=shift;
  if (exists $inccache->{INC}{$inc}){return $inccache->{INC}{$inc};}
  elsif(-f "${incdir}/${inc}/.info"){return &read_inc_cache($inc);}
  $inccache->{INC}{$inc}{new}=1;
  my $c=$inccache->{INC}{$inc};
  if(exists $data->{extra_include_path})
  {
    foreach my $d (keys %{$data->{extra_include_path}})
    {
      if(-f "${d}/${inc}")
      {
	my $pack="";
	$c->{fullpath}="${d}/${inc}";
	$pack=&run_func("check_inc","self",$inc);
	if($pack ne ""){$c->{pack}{$pack}=1;}
	return $c;
      }
    }
  }
  foreach my $t (keys %{$data->{EXTRA_TOOL_INFO}})
  {
    foreach my $exp (keys %{$data->{EXTRA_TOOL_INFO}{$t}{PRETOOL_INCLUDE_SEARCH_REGEXP}})
    {
      if($inc=~/$exp/)
      {
        my @incdirs=();
	if (exists $data->{EXTRA_TOOL_INFO}{$t}{INCLUDE}){@incdirs=keys %{$data->{EXTRA_TOOL_INFO}{$t}{INCLUDE}};}
	elsif((exists $cache->{TOOLS}{$t}) &&(exists $cache->{TOOLS}{$t}{INCLUDE})){@incdirs=@{$cache->{TOOLS}{$t}{INCLUDE}};}
        foreach my $d (@incdirs)
	{
	  if(-f "${d}/${inc}")
	  {
	    $c->{pack}{$t}=1;
	    $c->{fullpath}="${d}/${inc}";
	    return $c;
	  }
	}
      }
    }
  }
  foreach my $t (@{$cache->{OTOOLS}})
  {
    my $tool=$t;
    if($tool eq $project){next;}
    foreach my $d (@{$cache->{TOOLS}{$tool}{INCLUDE}})
    {
      if(!-f "${d}/${inc}"){next;}
      my $pack="";
      $c->{fullpath}="${d}/${inc}";
      $pack=&run_func("check_inc",$tool,$inc);
      if($pack ne "")
      {
	my $base=$cache->{TOOLS}{$tool}{BASE};
	if($tool eq "self"){$base=$release;}
	if(-f "${base}/src/${pack}/${inc}"){$c->{fullpath}="${base}/src/${pack}/${inc}";}
        $c->{pack}{$pack}=1;
      }
      else
      {
        if(exists $cache->{BASETOOLS}{$tool})
        {foreach my $p (keys %{$cache->{BASETOOLS}{$tool}}){$c->{pack}{$p}=1;}}
        else{$c->{pack}{$tool}=1;}
      }
      return $c;
    }
  }
  foreach my $t (keys %{$data->{EXTRA_TOOL_INFO}})
  {
    foreach my $exp (keys %{$data->{EXTRA_TOOL_INFO}{$t}{INCLUDE_SEARCH_REGEXP}})
    {
      if($inc=~/$exp/)
      {
        my @incdirs=();
	if (exists $data->{EXTRA_TOOL_INFO}{$t}{INCLUDE}){@incdirs=keys %{$data->{EXTRA_TOOL_INFO}{$t}{INCLUDE}};}
	elsif((exists $cache->{TOOLS}{$t}) &&(exists $cache->{TOOLS}{$t}{INCLUDE})){@incdirs=@{$cache->{TOOLS}{$t}{INCLUDE}};}
	foreach my $d (@incdirs)
	{
	  if(-f "${d}/${inc}")
	  {
	    $c->{pack}{$t}=1;
	    $c->{fullpath}="${d}/${inc}";
	    $c->{new}=1;
	    return $c;
	  }
	}
      }
    }
  }
  $inccache->{UNKNOWN}{$inc}=1;
  return $c;
}

sub final_exit()
{
  my $msg=shift;
  my $code=shift || 0;
  &save_toolcache();
  print STDERR "$msg\n";
  exit $code;
}

sub save_toolcache()
{
  my $f=$cache->{dirty};
  delete $cache->{dirty};
  if($f)
  {
    my $dir=dirname($cachefile);
    system("mkdir -p $dir");
    &SCRAMGenUtils::writeHashCache($cache,"$cachefile");
  }
  my %dirs=();
  my %newinc=();
  my $ndir=0;
  foreach my $inc (keys %{$inccache->{INC}})
  {
    if (exists $inccache->{INC}{$inc}{new})
    {
      $newinc{$inc}=1;
      my $d=\%dirs;
      $ndir=1;
      foreach my $x (split /\//,$inc)
      {
	if($x eq ""){next;}
	$d->{$x} ||={};
	$d=$d->{$x};
      }
    }
  }
  if ($ndir){mkpath(&SCRAMGenUtils::findUniqDirs(\%dirs,$incdir),0,0755);}
  foreach my $inc (keys %newinc){&write_inc_cache($inc);}
}

sub commaSepDeps ()
{
  my $d=shift;
  my $c=shift;
  my $type=shift;
  my @t=();
  my $xd="";
  if(-f "${cachedir}/${type}")
  {
    open(IFILE, "${cachedir}/${type}") || die "Can not open file \"${cachedir}/${type}\" for reading.";
    while(my $l=<IFILE>)
    {
      $l=~s/^\s*(.+?)\s*$/$1/;
      if(($l=~/^\s*$/) || ($l=~/^\s*#/)){next;}
      $xd.="$l,";
    }
    close(IFILE);
  }
  foreach my $x1 (@$d,$xd)
  {
    if($x1=~/^\s*([^\s]+)\s*$/)
    {
      $x1=$1;
      foreach my $x (split /\s*,\s*/, $x1)
      {
        if($x eq ""){next;}
	my $lx=lc($x);
        if(exists $c->{TOOLS}{$lx}){$x=$lx;}
	push @t,$x;
      }
    }
  }
  return @t;
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
  my $func1="${func}_${tool}";
  if(exists &$func1){return &$func1(@_);}
  elsif($tool ne "default"){return run_func ($func,"default",@_);}
  return "";
}

sub exists_func()
{
  my $func=shift || return "";
  my $tool=shift || return "";
  if($tool eq "self"){$tool=$project;}
  $tool=lc($tool);
  $func.="_${tool}";
  if(exists &$func){return 1;}
  return 0;
}
######################################################################
# Finding packages for a header file
######################################################################
sub check_inc_extra_pack_info ()
{
  my $f=shift;
  my $tool=shift;
  if((exists $data->{EXTRA_TOOL_INFO}{$tool}) && (exists $data->{EXTRA_TOOL_INFO}{$tool}{FILES_PACKAGE_MAP}))
  {
    foreach my $exp (keys %{$data->{EXTRA_TOOL_INFO}{$tool}{FILES_PACKAGE_MAP}})
    {if($f=~/$exp/){return $data->{EXTRA_TOOL_INFO}{$tool}{FILES_PACKAGE_MAP}{$exp};}}
  }
  return "";
}

sub check_inc_seal ()
{return &check_lcg2levels(shift,"seal");}

sub check_inc_pool ()
{return &check_lcg1level(shift,"pool");}

sub check_inc_coral ()
{return &check_lcg1level(shift,"coral");}

sub check_lcg1level ()
{
  my $f=shift;
  my $tool=shift;
  my $base="";
  my $x=&check_inc_extra_pack_info($f,$tool);
  if($x){return $x;}
  if(!exists $cache->{TOOLS}{$tool})
  {$tool="self";$base=$release;}
  else{$base=$cache->{TOOLS}{$tool}{BASE};}
  $base.="/src";
  if(!exists $cache->{TOOLS}{$tool}{PACKS})
  {
    foreach my $pack (&SCRAMGenUtils::readDir($base,1)){$cache->{TOOLS}{$tool}{PACKS}{$pack}=1;}
    $cache->{dirty}=1;
  }
  foreach my $pack (keys %{$cache->{TOOLS}{$tool}{PACKS}})
  {if(-f "${base}/${pack}/${f}"){return $pack;}}
  foreach my $pack (keys %{$cache->{TOOLS}{$tool}{PACKS}})
  {if($f=~/^$pack\/src\//){return $pack;}}
  return "";
}

sub check_lcg2levels()
{
  my $f=shift;
  my $tool=shift;
  my $base="";
  my $x=&check_inc_extra_pack_info($f,$tool);
  if($x){return $x;}
  if(!exists $cache->{TOOLS}{$tool})
  {$tool="self";$base=$release;}
  else{$base=$cache->{TOOLS}{$tool}{BASE};}
  $base.="/src";
  if(!exists $cache->{TOOLS}{$tool}{PACKS})
  {
    foreach my $subsys (&SCRAMGenUtils::readDir($base,1))
    {
      if($subsys=~/^(CVS|config|src|doc|admin|html|cmt|doxygen|qmtest|scram)$/){next;}
      foreach my $pack (&SCRAMGenUtils::readDir("${base}/${subsys}",1))
      {
        if($pack=~/^(CVS|config|src|doc|admin|html|cmt|doxygen|qmtest|scram)$/){next;}
	$cache->{TOOLS}{$tool}{PACKS}{"${subsys}/${pack}"}=1;
      }
    }
    $cache->{dirty}=1;
  }
  foreach my $pack (keys %{$cache->{TOOLS}{$tool}{PACKS}})
  {if(-f "${base}/${pack}/${f}"){return $pack;}}
  foreach my $pack (keys %{$cache->{TOOLS}{$tool}{PACKS}})
  {if($f=~/^$pack\/src\//){return $pack;}}
  return "";
}

sub check_inc_self ()
{return &check_cms_scram (shift,$project);}

sub check_inc_iguana ()
{return &check_cms_scram (shift,"iguana");}

sub check_inc_ignominy ()
{return &check_cms_scram (shift,"ignominy");}

sub check_inc_cmssw ()
{return &check_cms_scram (shift,"cmssw");}

sub check_cms_scram ()
{
  my $f=shift;
  my $tool=shift;
  my $x=&check_inc_extra_pack_info($f,$tool);
  if($x){return $x;}
  if($f=~/^(.+?)\/(interface|src)\/.+$/){return $1;}
  return "";
}

sub check_inc_rootrflx ()
{return &check_inc_rootcore(shift);}
sub check_inc_rootmath ()
{return &check_inc_rootcore(shift);}
sub check_inc_rootminuit2 ()
{return &check_inc_rootcore(shift);}
sub check_inc_rootcintex ()
{return &check_inc_rootcore(shift);}
sub check_inc_rootcore ()
{
  my $f=shift;
  my $x=&check_inc_extra_pack_info($f,$project) || &check_inc_extra_pack_info($f,"rootcore") || "rootcore";
  return $x;
}

sub check_inc_boost_filesystem ()
{return &check_inc_boost(shift);}
sub check_inc_boost_program_options ()
{return &check_inc_boost(shift);}
sub check_inc_boost_regex ()
{return &check_inc_boost(shift);}
sub check_inc_boost_python ()
{return &check_inc_boost(shift);}

sub check_inc_boost ()
{
  my $f=shift;
  my $x=&check_inc_extra_pack_info($f,"boost") || "boost";
  return $x;
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
sub safename_self ()
{return &safename_cms2($project);}

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

sub defaultplugin_seal (){return "seal";}
sub defaultplugin_pool (){return "seal";}
sub defaultplugin_coral (){return "seal";}
sub defaultplugin_ignominy (){return "seal";}
sub defaultplugin_iguana (){return "seal";}
sub defaultplugin_cmssw (){return "edm_";}
sub defaultplugin_default (){return "seal";}

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

##########################################################
sub update_project_cache ()
{
  my $cache=shift;
  my $cf=&SCRAMGenUtils::fixCacheFileName("${release}/.SCRAM/${scramarch}/ToolCache.db");
  if (!-f $cf){die "Can not find file for reading: $cf\n";}
  my $c=&SCRAMGenUtils::readCache($cf);
  my %allinc=();
  my $allincstr="";
  my $flags="";
  $cache->{OTOOLS}=[];
  foreach my $t (&SCRAMGenUtils::getOrderedTools($c))
  {
    push @{$cache->{OTOOLS}},$t;
    my $bt=uc($t)."_BASE";
    my $sproj=$c->{SETUP}{$t}{SCRAM_PROJECT} || 0;
    $cache->{TOOLS}{$t}{SCRAM_PROJECT}=$sproj;
    if($t eq "self"){$bt=uc($project)."_BASE";$sproj=1;$bt=$release;}
    elsif(exists $c->{SETUP}{$t}{$bt}){$cache->{VARS}{$bt}=$c->{SETUP}{$t}{$bt};$bt=$c->{SETUP}{$t}{$bt};$cache->{TOOLS}{$t}{BASE}=$bt;}
    if($t eq $compiler){if((exists $c->{SETUP}{$t}{CXX}) && (-x $c->{SETUP}{$t}{CXX})){$cache->{COMPILER}=$c->{SETUP}{$t}{CXX};}}
    if(exists $c->{SETUP}{$t}{FLAGS})
    {
      if(exists $c->{SETUP}{$t}{FLAGS}{CXXFLAGS}){$flags.=" ".join(" ",@{$c->{SETUP}{$t}{FLAGS}{CXXFLAGS}});}
      if(exists $c->{SETUP}{$t}{FLAGS}{CPPDEFINES}){$flags.=" -D".join(" -D",@{$c->{SETUP}{$t}{FLAGS}{CPPDEFINES}});}
    }
    foreach my $f ("INCLUDE", "LIBDIR", "LIB", "USE")
    {
      if(exists $c->{SETUP}{$t}{$f})
      {
        my %tmp=();
	foreach my $k (@{$c->{SETUP}{$t}{$f}})
        {
          if($f eq "USE"){$k=lc($k);}
          if(!exists $tmp{$k})
	  {
	    if(!exists $cache->{TOOLS}{$t}{$f}){$cache->{TOOLS}{$t}{$f}=[];}
	    push @{$cache->{TOOLS}{$t}{$f}},$k;
	    $tmp{$k}=1;
	    if(($f eq "INCLUDE") && ($k!~/^\s*$/) && (!exists $allinc{$k}) && (-d $k))
	    { 
	      if($t eq "self")
	      {
	        my $sdir="${release}/src";
		if(!exists $allinc{$sdir}){push @{$cache->{TOOLS}{$t}{$f}},$sdir;$allinc{$sdir}=1;}
	      }
	      $allinc{$k}=1;$allincstr.=" -I$k";
	    }
	  }
	}
      }
    }
    if((exists $cache->{TOOLS}{$t}) && (exists $cache->{TOOLS}{$t}{LIB}) && (!exists $cache->{TOOLS}{$t}{LIBDIR}))
    {
      if(&exists_func("check_inc",$t)){$cache->{IGNORE_BASETOOLS}{$t}=1;}
      if(!exists $cache->{IGNORE_BASETOOLS}{$t})
      {
        foreach my $l (@{$cache->{TOOLS}{$t}{LIB}})
        {
          my $ts=&find_lib_tool($c,$l,$t);
	  foreach my $t1 (keys %$ts)
	  {
	    my $file=&SCRAMGenUtils::findActualPath($ts->{$t1});
	    $cache->{BASETOOLS}{$t}{$t1}=1;
	    $cache->{XBASETOOLS}{$t1}{$t}=1;
	  }
        }
      }
    }
  }
  $cache->{CXXINCLUDE}=$allincstr;
  $cache->{CXXFLAGS}=$flags;
  $cache->{dirty}=1;
}

sub replaceVariables {
  my $key=shift || return "";
  while($key=~/(.*)\$\(([^\$\(\)]*)\)(.*)/){
    my $subkey=$2;
    if("$subkey" ne ""){
      if($subkey eq "LOCALTOP"){$subkey="$release";}
      elsif(exists $envcache->{$subkey}){$subkey=$envcache->{$subkey};}
      elsif(exists $cache->{VARS}{$subkey}){$subkey=$cache->{VARS}{$subkey};}
      else{$subkey=$ENV{$subkey};}
    }
    $key="${1}${subkey}${3}";
  }
  return $key;
}

sub find_lib_tool ()
{
  my $c=shift;
  my $lib=shift;
  my $t=shift;
  my $tools={};
  if(exists $c->{SETUP}{$t}{USE})
  {
    foreach my $t1 (@{$c->{SETUP}{$t}{USE}})
    {
      $t1=lc($t1);
      if(exists $c->{SETUP}{$t1}{LIBDIR})
      {
        foreach my $d (@{$c->{SETUP}{$t1}{LIBDIR}})
	{
	  if(-f "${d}/lib$lib.so"){$tools->{$t1}="${d}/lib$lib.so";}
	  elsif(-f "${d}/lib$lib.a"){$tools->{$t1}="${d}/lib$lib.a";}
	}
      }
    }
    if(scalar(keys %$tools)>0){return $tools;}
    foreach my $t1 (@{$c->{SETUP}{$t}{USE}})
    {
      my $t2=&find_lib_tool($c,$lib,lc($t1));
      if(scalar(keys %$t2)>0){return $t2;}
    }
  }
  return $tools;
}

#########################################################################
# Help message
#########################################################################

sub usage_msg()
{
  my $script=basename($0);
  print "Usage: $script --dir <path>\n";
  print "\t[--files <files>]      [--prodname <name>] [--prodtype <library|bin>]\n";
  print "\t[--use <tool/pack>]    [--no-use <tool/pack>]\n";
  print "\t[--export <tool/pack>] [--no-export <tool/pack>] [--flag <flags>]\n";
  print "\t[--replace <oldtool/oldpackage>=<newdtool/newpackage>]\n";
  print "\t[--ref-bf <BuildFile>] [--jobs <jobs>] [--tmpdir <dir>] [--buildfile <name>]\n";
  print "\t[--plugin] [--detail] [--help]\n\n";
  print "Where\n";
  print "  --dir <path>         Path of package or other product area area for which you want to\n";
  print "                       generate BuildFile.\n";
  print "  --files <files>      Comma separated list of files relative path w.t.r the dirtecoty\n";
  print "                       provided via --dir <path> command-line argument. Only\n";
  print "                       files with extensions \"$srcext\" are allowed.\n";
  print "                       You can add this command-line argument multiple times.\n";
  print "  --prodname <name>    Name of the product. By default is there is only one source file\n";
  print "                       provided via --files command-line argument then name will be drived\n";
  print "                       from that by removing the file extensions (.[$srcext]).\n";
  print "  --prodtype <type>    Name of the product type. Only valid values are \"library\" or \"bin\".\n";
  print "  --use <tool/pack>    To add an extra dependency on tool/package(for non-export section only)\n";
  print "                       You can add this command-line argument multiple times.\n";
  print "  --no-use <tool/pack> To remove dependency on tool/package(for non-export section only)\n";
  print "                       You can add this command-line argument multiple times.\n";
  print "  --export <tool/pack> To add an extra dependency on tool/package(for both export and non-export sections)\n";
  print "                       You can add this command-line argument multiple times.\n";
  print "  --no-export <tool/pack>\n";
  print "                       To remove dependency on tool/package(for both export and non-export sections)\n";
  print "                       You can add this command-line argument multiple times.\n";
  print "  --replace <oldtool/oldpack>=<newtool/newpack>\n";
  print "                       To replace a oldtool/oldpack with a newtool/newpack.\n";
  print "                       You can add this command-line argument multiple times.\n";
  print "  --ref-bf <buildfile> Provide a reference BuildFile to search for extra flags. By default is uses the\n";
  print "                       SubSystem/Package/BuildFile for a package directory.\n";
  print "  --jobs <jobs>        Number of parallel processes\n";
  print "  --tmpdir <path>      Path of a temporary area where this script should save its internal caches and put newly\n";
  print "                       generated BuildFile(s).\n";
  print "  --buildfile <name>   Path for auto generated BuildFile. By default it will be printed on strandard output.\n";
  print "  --plugin             Generate BuildFile with plugin flag in it.\n";
  print "                       NOTE: If package contains classes.h then this flag will not be added.\n";
  print "  --clean              Reset the internal tool cache and start from scratch.\n";
  print "  --xml                To generate XML-based BuildFiles i.e. BuildFile.xml.auto\n";
  print "  --detail             Run in debug mode i.e. prints more debug output.\n";
  print "  --help               Print this help message.\n\n";
  print "E.g. running something following from your project top level directory\n\n";
  print "  $script --dir src/FWCore/Services --flag CPPDEFINES=\"DEBUG=1\" --flag CPPDEFINES='USER=\"name\"'\n\n";
  print "means print a BuildFile for FWCore/Services package and add two extra flags.\n";
  print "NOTE: If there is already a <path>/BuildFile exists then script will read that BuildFile\n";
  print "and add all the flags, makefile fragments, extra libs etc. in the newly generated BuildFile too.\n";
  exit 0;
}
