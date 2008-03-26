#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use Getopt::Long;
use SCRAMGenUtils;

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
	       "--tmprelease=s",\$tmprel,
	       "--buildfile=s",\$buildfilename,
	       "--config=s",\$configfile,
	       "--chksym",\$chksym,
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

if(defined $help){&usage_msg();}
if(defined $plugin){$plugin=1;}
else{$plugin=0;}
if(defined $xml){$xml="--xml";}
else{$xml="";}
if(defined $chksym){$chksym=1;}
else{$chksym=0;}
if(defined $detail){$detail=1;}
else{$detail=0;}
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

if((defined $refbf) && ($refbf!~/^\s*$/))
{if(!-f $refbf){print STDERR "ERROR: Reference BuildFile \"$refbf\" does not exist.\n"; exit 1;}}
else{$refbf="";}

$SCRAMGenUtils::DEBUG=$detail;
my $project=&SCRAMGenUtils::getFromEnvironmentFile("SCRAM_PROJECTNAME","$release");
my $baserelease=&SCRAMGenUtils::getFromEnvironmentFile("RELEASETOP","$release");
$project=lc($project);

my $pkgsrcdir=&run_func("pkg_src",$project,$dir);
if ($pkgsrcdir eq ""){print STDERR "ERROR: Script not ready yet to work for \"$project\" SCRAM-based project.\n"; exit 1;}
my $pkginterfacedir=&run_func("pkg_interface",$project,$dir);

my $ccfiles=0;
my $isPackage=0;
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
  if(scalar(@files)==0){exit 0;}
  else
  {
    $isPackage=1;
    $prodname=&run_func("safename",$project,$dir);
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
$data->{EXTRA_SYMBOL_TOOLS}{castor}=1;
$data->{EXTRA_SYMBOL_TOOLS}{herwig}=1;
$data->{EXTRA_SYMBOL_TOOLS}{lhapdf}=1;
$data->{EXTRA_SYMBOL_TOOLS}{pythia6}=1;
$data->{EXTRA_SYMBOL_TOOLS}{pythia}=1;
$data->{EXTRA_SYMBOL_TOOLS}{jimmy}=1;

$data->{EXTRA_TOOL_INFO}{opengl}{INCLUDE}{"/usr"}=1;
$data->{EXTRA_TOOL_INFO}{opengl}{INCLUDE}{"/usr/include"}=1;
$data->{EXTRA_TOOL_INFO}{opengl}{INCLUDE_SEARCH_REGEXP}{"^GL\/.+"}=1;

$data->{EXTRA_TOOL_INFO}{iguana}{FILES_PACKAGE_MAP}{'^classlib\/.+'}="Iguana/Utilities";
$data->{EXTRA_TOOL_INFO}{iguana}{FILES_PACKAGE_MAP}{'^gl2ps.h$'}="Iguana/GL2PS";
$data->{EXTRA_TOOL_INFO}{cmssw}{FILES_PACKAGE_MAP}{'^classlib\/.+'}="Iguana/Utilities";
$data->{EXTRA_TOOL_INFO}{cmssw}{FILES_PACKAGE_MAP}{'^gl2ps.h$'}="Iguana/GL2PS";
$data->{EXTRA_TOOL_INFO}{rootcore}{FILES_PACKAGE_MAP}{'^Reflex\/'}="rootrflx";
$data->{EXTRA_TOOL_INFO}{rootcore}{FILES_PACKAGE_MAP}{'Math\/'}="rootmath";
$data->{EXTRA_TOOL_INFO}{rootcore}{FILES_PACKAGE_MAP}{'^Minuit2\/'}="rootminuit2";
$data->{EXTRA_TOOL_INFO}{rootcore}{FILES_PACKAGE_MAP}{'^Cintex\/'}="rootcintex";
$data->{EXTRA_TOOL_INFO}{boost}{FILES_PACKAGE_MAP}{'^boost\/filesystem(\/|\.).+'}="boost_filesystem";
$data->{EXTRA_TOOL_INFO}{boost}{FILES_PACKAGE_MAP}{'^boost\/program_options(\/|\.).+'}="boost_program_options";
$data->{EXTRA_TOOL_INFO}{boost}{FILES_PACKAGE_MAP}{'^boost\/regex(\/|[\.\_]).+'}="boost_regex";
$data->{EXTRA_TOOL_INFO}{boost}{FILES_PACKAGE_MAP}{'^boost\/python(\/|\.).+'}="boost_python";

if(($configfile ne "") && (-f "$configfile"))
{
  foreach my $line (`cat $configfile`)
  {
    chomp $line;
    my @x=split /:/,$line;
    my $count=scalar(@x);
    if($count<3){next;}
    my $c=undef;
    if($x[0] eq "DATA"){$c=$data;}
    elsif($x[0] eq "CACHE"){$c=$cache;}
    if(defined $c)
    {
      if($detail){print STDERR "#Configuring=>$line\n";}
      my $i=0;
      for($i=1;$i<$count-2;$i++)
      {
        my $y=$x[$i];
	if(!exists $c->{$y}){$c->{$y}={};}
	$c=$c->{$y};
      }
      $c->{$x[$i]}=$x[$i+1];
    }
  }
}
else{$configfile="";}
$data->{bfflags}=[];
foreach my $f (@flags){push @{$data->{bfflags}},$f;}

my $scramarch=&SCRAMGenUtils::getScramArch();
if((!defined $tmprel) || ($tmprel=~/^\s*$/))
{
  $tmprel="${release}/tmp/AutoBuildFile";
  if($pwd!~/^$release(\/.*|)$/){$tmprel="${pwd}/AutoBuildFile";}
}
&SCRAMGenUtils::init($tmprel);
my $cachedir="${tmprel}/bfcache/${scramarch}";
if(!-d $cachedir){system("mkdir -p $cachedir");}
my $cachefile="${cachedir}/toolcache";
my $symbolfile="${cachedir}/symbols.txt";
my $unincfile="${cachedir}/unknow_inc.txt";
my $prodfile="${cachedir}/product.cache";
my $inccache={};
my $inccache_dirty=0;
if((!defined $clean) && (-f "$cachefile"))
{
  print STDERR "Reading previously save internal cache $cachefile.\n";
  $cache=&SCRAMGenUtils::readHashCache("$cachefile");
  $SCRAMGenUtils::InternalCache=$cache;
  if($chksym)
  {
    if(-f $symbolfile){$data->{symbols}=&SCRAMGenUtils::readHashCache($symbolfile);}
    else
    {
      print STDERR "ERROR: There is no symbols file \"$symbolfile\". May be you have delete it by hand.\n";
      print STDERR "Please try running with --clean command-line option to regenerate this file.\n";
      exit 1;
    }
  }
}
else
{
  print STDERR "Reading project's SCRAM cache\n";
  &update_project_cache($cache);
  $cache->{IGNORE_BASETOOLS}{rootrflx}=1;
  $cache->{IGNORE_BASETOOLS}{boost_python}=1;
  if(exists $cache->{SYMBOLS})
  {&SCRAMGenUtils::writeHashCache($cache->{SYMBOLS},$symbolfile);}
  else{system("touch $symbolfile");}
  if($chksym){$data->{symbols}=$cache->{SYMBOLS};}
  else{delete $cache->{SYMBOLS};}
  &save_toolcache();
}

if(-f $unincfile){$inccache=&SCRAMGenUtils::readHashCache($unincfile);}

if(exists $cache->{COMPILER}){$data->{compilecmd}=$cache->{COMPILER}." ".$cache->{CXXFLAGS}." ".$cache->{CXXINCLUDE};}
else{print STDERR "#WARNING: No compiler found. So script is not going to parse the file for seal plugin macros.\n";}

if ($refbf eq "")
{
  my $bf="${dir}/BuildFile.xml";
  if(!-f $bf){$bf="${dir}/BuildFile";}
  if (-f $bf){$refbf=$bf;}
}

foreach my $f ("EDM_PLUGIN", "SEALPLUGIN", "SEAL_PLUGIN_NAME", "NO_LIB_CHECKING", "GENREFLEX_FAILES_ON_WARNS", "ROOTMAP", "ADD_SUBDIR", "CODEGENPATH")
{$data->{sflags}{$f}=1;}
foreach my $f ("CPPDEFINES")
{$data->{keyflags}{$f}=1;}
$data->{filter}=$dir;
$data->{files}={};
$data->{searchPreprocessed}{sealplugin}{filter}='\s+seal\:\:ModuleDef\s+\*SEAL_MODULE\s+';
$data->{searchPreprocessed}{edmplugin}{filter}='^\s*static\s+.+?\:\:PMaker\<.+\>\s+s_maker\d+\s+\(';
$data->{searchPreprocessed}{main}{filter}='\bmain\s*\(';
$data->{searchPreprocessed}{castor}{filter}='\brfio_[a-z]+';
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
	if($f1 eq "CPPDEFINES"){foreach my $fv (@$v){$exflags.=" -D".$fv->{v};}}
	elsif($f1 eq "CXXFLAGS"){foreach my $fv (@$v){$exflags.=" ".$fv->{v};}}
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
  delete $data->{searchPreprocessed}{main};
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
        $plugin=-1;$data->{deps}{src}{rootrflx}=1;$data->{deps}{src}{rootrflx}=1;$ccfiles++;
	my $found=0;
	foreach my $f1 (@files){if($f1 eq $f){$found=1;last;}}
	if(!$found){push @files,$f;}
      }
      elsif($f=~/$linkdef/){$data->{deps}{src}{rootcintex}=1;$ccfiles++;}
      elsif($f=~/$iglet/){$igletfile=$f;}
    }
  }
}

print STDERR "Reading source files\n";
foreach my $file (@files)
{
  &process_cxx_file ($file,$data);
  if($file=~/\.($srcext)$/i)
  {
    if($file!~/\.(f|f77)$/i)
    {&SCRAMGenUtils::searchPreprocessedFile($file,$data,$exflags);}
    else{$fortrancode=1;print "FORTRAN FILE:$file\n";}
    if((exists $data->{searchPreprocessed}{sealplugin}) && (exists $data->{searchPreprocessed}{sealplugin}{file}))
    {$srcplugin=$data->{searchPreprocessed}{sealplugin}{file};delete $data->{searchPreprocessed}{sealplugin};}
    if((exists $data->{searchPreprocessed}{edmplugin}) && (exists $data->{searchPreprocessed}{edmplugin}{file}))
    {$srcedmplugin=$data->{searchPreprocessed}{edmplugin}{file};delete $data->{searchPreprocessed}{edmplugin};}
    if((exists $data->{searchPreprocessed}{main}) && (exists $data->{searchPreprocessed}{main}{file}))
    {
      my $f=$data->{searchPreprocessed}{main}{file};
      if(($prodtype ne "") && ($prodtype ne "bin"))
      {print STDERR "\"$prodname\" seemed like a \"bin\" product because there is \"main()\" exists in \"f\" file.\n";}
      else{$prodtype="bin";}
      if($detail){print STDERR "Executable:$prodname:$f\n";}
      delete $data->{searchPreprocessed}{main};
    }
    if((exists $data->{searchPreprocessed}{castor}) && (exists $data->{searchPreprocessed}{castor}{file}))
    {
      my $f=$data->{searchPreprocessed}{castor}{file};
      if($detail){print STDERR "Castor Dependency:$prodname:$f\n";}
      $castor=1;
      delete $data->{searchPreprocessed}{castor};
    }
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

if($fortrancode){$data->{deps}{src}{pythia6}=1;$data->{deps}{src}{genser}=1;}
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
{if($plugin <= 0){push @{$data->{bfflags}},"${defaultplugintype}PLUGIN=0";}}
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
foreach my $x (&commaSepDeps(\@export,$cache,"${prodname}export"))
{$data->{deps}{src}{$x}=1;}

# Extra tool/package not to export
foreach my $x (&commaSepDeps(\@nexport,$cache,"${prodname}no-export"))
{
  if(exists $data->{deps}{src}{$x})
  {print STDERR "MSG:Removed dependency on \"$x\" due to no-export arguments.\n";}
  delete $data->{deps}{src}{$x};
}

# Extra tool/package to use
foreach my $x (&commaSepDeps(\@use,$cache,"${prodname}use"))
{$data->{deps}{src}{$x}=1;}

# Extra tool/package not to use
foreach my $x (&commaSepDeps(\@nuse,$cache,"${prodname}no-use"))
{
  if(exists $data->{deps}{src}{$x}){print STDERR "MSG:Removed dependency on \"$x\" due to no-use arguments.\n";}
  delete $data->{deps}{src}{$x};
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

my $xtools=0;
foreach my $bt (keys %{$cache->{XBASETOOLS}})
{
  if(exists $data->{deps}{src}{$bt})
  {foreach my $bt1 (keys %{$cache->{XBASETOOLS}{$bt}}){if(!exists $data->{deps}{src}{$bt1}){$xtools++;}}}
}

if((-f $prodfile) && (!-d STDIN))
{
  my $c=&SCRAMGenUtils::readHashCache($prodfile);
  foreach my $u (keys %{$data->{deps}{src}})
  {
    my $u1=&run_func("safename",$project,"${release}/src/${u}");
    if ($u1 ne "")
    {
      if(exists $c->{$u1})
      {
        if ((!exists $c->{$u1}{done}) && (!exists $c->{$u1}{skip}))
	{
          print "PLEASE_PROCESS_FIRST:$u\n";
          my $res=<STDIN>;chomp $res;
          if($res eq "PROCESSED:$u"){print "OK PROCESSING DONE FOR:$u\n";}
          else{print "PARENT RESPONCE:$res\n"}
	}
      }
      elsif((-d "${release}/src/${u}/${pkgsrcdir}") || (-d "${release}/src/${u}/${pkginterfacedir}"))
      {
	my $nsbfile="${tmprel}/src/${u}/BuildFile.xml";
	if(!-f $nsbfile){$nsbfile="${tmprel}/src/${u}/BuildFile";}
	if(!-f $nsbfile)
	{
	  my $nbfile="${tmprel}/newBuildFile/src/${u}/BuildFile.auto";
	  if ($xml){$nbfile="${tmprel}/newBuildFile/src/${u}/BuildFile.xml.auto";}
	  if(!-f $nbfile)
	  {
	    my $cmd="$0 --dir ${release}/src/${u} --buildfile $nbfile $xml ";
	    if($configfile ne ""){$cmd.=" --config $configfile";}
	    if($tmprel ne ""){$cmd.=" --tmprelease $tmprel";}
	    if($detail){$cmd.=" --detail";}
	    if($chksym){$cmd.=" --chksym";}
	    print STDERR "MSG: Running $cmd\n";
	    system("cd $pwd; $cmd");
	    if(!-f $nbfile)
	    {
	      system("mkdir ${tmprel}/newBuildFile/src/${u}");
	      if($xml){system("echo \"<export>\n  <flags DummyFlagToAvoidWarning=\"0\">\n</export>\" > $nbfile");}
	      else{system("echo \"<export>\n  <flags DummyFlagToAvoidWarning=\"0\"/>\n</export>\" > $nbfile");}
	    }
	  }
	  if(!-f $nsbfile)
	  {
	    my $nf="BuildFile";
	    if($xml){$nf="BuildFile.xml";}
	    system("mkdir ${tmprel}/src/${u}; cp $nbfile ${tmprel}/src/${u}/${nf}");
	  }
	}
      }
    }
  }
}

if(exists $data->{deps}{src}{castor}){$castor=0;}

if($chksym && ($xtools || $castor))
{
  if(!-f "${tmprel}/.SCRAM/Environment")
  {
    $tmprel=&SCRAMGenUtils::createTmpReleaseArea($release,1,$tmprel);
    system("cd $tmprel; $SCRAM_CMD b -r echo_CXX 2>&1 > /dev/null");
  }
  my $reldir=$dir; $reldir=~s/^$release\///;
  my $bdir=dirname($reldir);
  $bdir="${tmprel}/${bdir}";
  my $bname=basename($dir);
  my $tmppdir="${tmprel}/tmp/${scramarch}/${reldir}/${prodname}";
  if($isPackage){$tmppdir="${tmprel}/tmp/${scramarch}/${reldir}/${pkgsrcdir}/${prodname}";}
  if(-d "$tmppdir"){system("rm -rf $tmppdir");}
  if(-d "${tmprel}/${reldir}"){system("rm -rf ${tmprel}/${reldir}");}
  system("mkdir -p $bdir; cp -rpf $dir ${bdir}/${bname}");
  if(-d "${bdir}/${bname}/data"){system("rm -rf ${bdir}/${bname}/data");}
  $bdir.="/${bname}";

  my $bfile="${bdir}/BuildFile";
  if($xml){$bfile="${bdir}/BuildFile.xml";}
  my $pflagval="";
  if(exists $data->{flags}{NO_LIB_CHECKING}){$pflagval=$data->{flags}{NO_LIB_CHECKING};}
  $data->{flags}{NO_LIB_CHECKING}=1;
  &SCRAMGenUtils::printBuildFile($data,$bfile);
  if($detail){print "#MSG:Going to compile $prodname\n";}
  system("cd ${bdir}; $SCRAMGenUtils::SCRAM_CMD b -v -r -j 4 $prodname");
  my $err=$?;
  if($pflagval ne ""){$data->{flags}{NO_LIB_CHECKING}=$pflagval;}
  else{delete $data->{flags}{NO_LIB_CHECKING};}
  
  my $tmpprod="${tmppdir}/lib${prodname}.so";
  if($prodtype eq "bin"){$tmpprod="${tmppdir}/${prodname}";}
  if(($prodtype ne "bin") && (-f $tmpprod))
  {
    my $missing=&SCRAMGenUtils::symbolChecking($tmpprod);
    if($missing != 0)
    {
      my $sym=&SCRAMGenUtils::getLibSymbols($tmpprod);
      &addToolFromSymbols($data,$sym);
    }
  }
  elsif(($prodtype eq "bin") && ($err!=0))
  {
    my $objs=[];
    foreach my $f (&SCRAMGenUtils::readDir($tmppdir,2))
    {if($f=~/\.o$/){push @$objs,"${tmppdir}/${f}";}}
    my $sym=&SCRAMGenUtils::getObjectFileSymbols($objs);
    &addToolFromSymbols($data,$sym);
  }
}

if((!defined $buildfilename) || ($buildfilename=~/^\s*$/)){$buildfilename="";}
elsif($buildfilename!~/^\//){$buildfilename="${pwd}/${buildfilename}";}
if($buildfilename)
{
  $buildfilename=&SCRAMGenUtils::fixPath($buildfilename);
  my $d=dirname($buildfilename);
  if(!-d $d){system("mkdir -p $d");}
}
&SCRAMGenUtils::removeExtraLib ($cache,$data);
&SCRAMGenUtils::printBuildFile($data, "$buildfilename");
&final_exit("",0);
#########################################
#
sub addToolFromSymbols ()
{
  my $data=shift;
  my $sym=shift;
  my %ts=();
  foreach my $s (keys %$sym)
  {
    if($sym->{$s} ne "U"){next;}
    foreach my $f (keys %{$data->{symbols}})
    {
      my $lib=basename($f);
      if((exists $data->{symbols}{$f}{sym}{$s}) && ($data->{symbols}{$f}{sym}{$s}=~/^(T|V)$/))
      {foreach my $t (keys %{$data->{symbols}{$f}{tool}}){$ts{$t}{$s}=$lib;}}
    }
  }
  foreach my $t (keys %ts)
  {
    $data->{deps}{src}{$t}=1;
    print STDERR "EXTRA TOOL due to missing symbols:$t\n";
    print "  SYMBOLS FOUND:\n";
    foreach my $s (keys %{$ts{$t}})
    {
      my $s1=`c++filt $s`; chomp $s1;
      print "    $s1 =>$ts{$t}{$s}\n";
    }
  }
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
      if(exists $inccache->{$inc}){next;}
      elsif(-f "${selfdir}/${inc}"){&process_cxx_file ("${selfdir}/${inc}",$data);}
      else
      {
	my %info = &find_inc_file_path($inc);
	my $fpath=$info{fullpath};
	#print STDERR "MSG:$inc:$fpath:$info{pack}\n";
	if($fpath ne "")
	{
	  if ($fpath=~/^${filter}\/.+$/){&process_cxx_file ($fpath,$data);}
	  else
	  {
	    foreach my $pack (keys %{$info{pack}})
	    {
	      #print STDERR "#$inc:$fpath:$pack\n";
	      if(("$pack" ne "$project") && ("$pack" ne "self"))
	      {
	        if($isPackage)
	        {
	          if($file=~/^${filter}\/${pkginterfacedir}\//){$data->{deps}{src}{$pack}=1;}
	          elsif($file=~/^${filter}\/${pkgsrcdir}\//){$data->{deps}{src}{$pack}=1;}
	        }
	        else{$data->{deps}{src}{$pack}=1;}
	      }
	      if($detail){print STDERR "#$ainc=>$pack\n";}
	    }
	  }
	}
	elsif($detail){print STDERR "#$ainc:UNKNOWN (might be from system directories)\n";$inccache->{$inc}=1;$inccache_dirty=1;}
      }
    }
  }
}

sub find_inc_file_path ()
{
  my $inc=shift;
  my %info;
  if(exists $data->{extra_include_path})
  {
    foreach my $d (keys %{$data->{extra_include_path}})
    {
      if(-f "${d}/${inc}")
      {
        my $pack="";
	$info{fullpath}="${d}/${inc}";
	$pack=&run_func("check_inc","self",$inc);
	if($pack ne "")
	{$info{pack}{$pack}=1;}
	return %info;
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
      $info{fullpath}="${d}/${inc}";
      $pack=&run_func("check_inc",$tool,$inc);
      if($pack ne "")
      {
	my $base=$cache->{TOOLS}{$tool}{BASE};
	if($tool eq "self"){$base=$release;}
	if(-f "${base}/src/${pack}/${inc}"){$info{fullpath}="${base}/src/${pack}/${inc}";}
        $info{pack}{$pack}=1;
      }
      else
      {
        if(exists $cache->{BASETOOLS}{$tool})
        {foreach my $p (keys %{$cache->{BASETOOLS}{$tool}}){$info{pack}{$p}=1;}}
        else{$info{pack}{$tool}=1;}
      }
      return %info;
    }
  }
  foreach my $t (keys %{$data->{EXTRA_TOOL_INFO}})
  {
    foreach my $exp (keys %{$data->{EXTRA_TOOL_INFO}{$t}{INCLUDE_SEARCH_REGEXP}})
    {
      if($inc=~/$exp/)
      {
        foreach my $d (keys %{$data->{EXTRA_TOOL_INFO}{$t}{INCLUDE}})
	{
	  if(-f "${d}/${inc}")
	  {
	    $info{pack}{$t}=1;
	    $info{fullpath}="${d}/${inc}";
	    return %info;
	  }
	}
      }
    }
  }
  return %info;
}

sub final_exit()
{
  my $msg=shift;
  my $code=shift || 0;
  &save_toolcache();
  print "$msg\n";
  exit $code;
}

sub save_toolcache()
{
  my $f=$cache->{dirty};
  my $sym=undef;
  if(exists $cache->{SYMBOLS})
  {$sym=$cache->{SYMBOLS};}
  delete $cache->{SYMBOLS};
  delete $cache->{dirty};
  if($f)
  {
    my $dir=dirname($cachefile);
    system("mkdir -p $dir");
    &SCRAMGenUtils::writeHashCache($cache,"$cachefile");
  }
  if($inccache_dirty)
  {&SCRAMGenUtils::writeHashCache($inccache,$unincfile);}
  $inccache_dirty=0;
  if(defined $sym){$cache->{SYMBOLS}=$sym;}
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
  if (!-f $cf){system("cd $release; $SCRAMGenUtils::SCRAM_CMD build -r echo_CXX 2>&1");}
  my $c=&SCRAMGenUtils::readCache($cf);
  my %allinc=();
  my $allincstr="";
  my $flags="";
  foreach my $t (keys %{$c->{SETUP}})
  {
    my $bt=uc($t)."_BASE";
    if($t eq "self"){$bt=uc($project)."_BASE";}
    if(exists $c->{SETUP}{$t}{$bt})
    {$cache->{TOOLS}{$t}{BASE}=$c->{SETUP}{$t}{$bt};}
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
	    if(!exists $cache->{SYMBOLS}{$file})
	    {$cache->{SYMBOLS}{$file}{sym}=&SCRAMGenUtils::getLibSymbols($file);}
	    $cache->{SYMBOLS}{$file}{tool}{$t}=1;
	  }
        }
      }
    }
    elsif(exists $data->{EXTRA_SYMBOL_TOOLS}{$t})
    {
      if((exists $cache->{TOOLS}{$t}) && ($cache->{TOOLS}{$t}{LIBDIR}))
      {
        if(exists $cache->{TOOLS}{$t}{LIB})
	{
	  foreach my $l (@{$cache->{TOOLS}{$t}{LIB}})
	  {
	    foreach my $d (@{$cache->{TOOLS}{$t}{LIBDIR}})
	    {
	      my $file="${d}/lib${l}.so";
	      if(!-f $file){$file="${d}/lib${l}.a";}
	      if(-f $file)
	      {
	        $file=&SCRAMGenUtils::findActualPath($file);
		if(!exists $cache->{SYMBOLS}{$file})
		{$cache->{SYMBOLS}{$file}{sym}=&SCRAMGenUtils::getLibSymbols($file);}
		$cache->{SYMBOLS}{$file}{tool}{$t}=1;
	      }
	    }
	  }
	}
      }
    }
  }
  $cache->{CXXINCLUDE}=$allincstr;
  $cache->{CXXFLAGS}=$flags;
  $cache->{OTOOLS}=[];
  if(exists $cache->{TOOLS}{self}){push @{$cache->{OTOOLS}},"self";}
  my @order=();
  foreach my $t (keys %{$c->{SELECTED}})
  {
    my $index=$c->{SELECTED}{$t};
    if($index=~/^\d+$/)
    {
      if(!defined $order[$index]){$order[$index]=[];}
      push @{$order[$index]},$t;
    }
  }
  foreach my $t (@order)
  {
    if(defined $t)
    {foreach my $t1 (@$t){if((defined $t1) && ($t1!~/^\s*$/) && (exists $cache->{TOOLS}{$t1})){push @{$cache->{OTOOLS}},$t1;}}}
  }
  $cache->{dirty}=1;
}

sub replaceVariables {
  my $key=shift || return "";
  my $ecache=&SCRAMGenUtils::getEnvironmentFileCache($release);
  while($key=~/(.*)\$\(([^\$\(\)]*)\)(.*)/){
    my $subkey=$2;
    if("$subkey" ne ""){
      if($subkey eq "LOCALTOP"){$subkey="$release";}
      elsif(exists $ecache->{$subkey}){$subkey=$ecache->{$subkey};}
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
  print "\t[--ref-bf <BuildFile>] [--tmprelease <release>] [--buildfile <name>]\n";
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
  print "  --tmprelease <path>  Path of a temporary release area which this script can re-use to test the newly\n";
  print "                       generated BuildFile.\n";
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
