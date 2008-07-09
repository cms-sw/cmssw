package SCRAMGenUtils;
use File::Basename;

our $SCRAM_CMD="scramv1";
our $SCRAM_VERSION="";
local $SCRAM_ARCH="";
local $DEBUG=0;
local $InternalCache={};

sub init ()
{
  my $dir=shift;
  &scramVersion ($dir);
  my $arch=&getScramArch();
  if (!-f "${dir}/tmp/${arch}/Makefile"){system("cd ${dir}; $SCRAM_CMD b -r echo_CXX ufast>/dev/null 2>&1");}
  my $scram_home=dirname(dirname(&SCRAMGenUtils::getBuildVariable($dir,"SCRAM","ufast")));
  if (!-d "$scram_home"){die "ERROR: Failed to find SCRAM.\n";}
  $ENV{SCRAM_HOME}=$scram_home;
  unshift @INC,"${scram_home}/src";
  unshift @INC,"${dir}/config";
}

sub fixPath ()
{
  my $dir=shift;
  my @parts=();
  my $p="/";
  if($dir!~/^\//){$p="";}
  foreach my $part (split /\//, $dir)
  {
    if($part eq ".."){pop @parts;}
    elsif(($part ne "") && ($part ne ".")){push @parts, $part;}
  }
  return "$p".join("/",@parts);
}

sub findActualPath ()
{
  my $file=shift;
  if(-l $file)
  {
    my $dir=dirname($file);
    $file=readlink($file);
    if($file!~/^\//){$file="${dir}/${file}";}
    return &findActualPath($file);
  }
  return $file;
}

sub readDir ()
{
  my $dir=shift;
  my $type=shift || 0;
  my @data=();
  opendir(DIR,$dir) || die "Can not open directory $dir for reading.";
  foreach my $f (readdir(DIR))
  {
    if($f=~/^\./){next;}
    if($type == 0){push @data,$f;}
    elsif(($type == 1) && (-d "${dir}/${f}")){push @data,$f;}
    elsif(($type == 2) && (-f "${dir}/${f}")){push @data,$f;}
  }
  closedir(DIR);
  return @data;
}

sub getTmpFile ()
{
  my $dir=shift || &getTmpDir ();
  my $index=0;
  my $tmp="${dir}/f${index}";
  while(-f $tmp)
  {$index++;$tmp="${dir}/f${index}";}
  system("touch $tmp");
  return $tmp;
}

sub getTmpDir ()
{
  my $dir=shift;
  if(!defined $dir)
  {
    if((exists $ENV{GEN_UTILS_TMPDIR}) && (-d $ENV{GEN_UTILS_TMPDIR}))
    {$dir="$ENV{GEN_UTILS_TMPDIR}";}
    else{$dir="/tmp";}
    $dir="${dir}/delete_me_$$";
  }
  my $index=0;
  my $tmp="${dir}_${index}";
  while(-d $tmp)
  {$index++;$tmp="${dir}_d${index}";}
  system("mkdir -p $tmp");
  return $tmp;
}

#########################################################################
# Reading Project Cache DB
#########################################################################

sub scramVersion ()
{
  my $rel=shift;
  if ($SCRAM_VERSION eq "")
  {
    my $ref;
    if(open($ref,"${rel}/config/scram_version"))
    {
      $SCRAM_VERSION=<$ref>; chomp $SCRAM_VERSION;
      close($ref);
    }
  }
  return $SCRAM_VERSION;
}

sub fixCacheFileName ()
{
  my $file=shift;
  my $gz="";
  if ($SCRAM_VERSION=~/^V[2-9]/){if($file!~/\.gz$/){$gz=".gz";}}
  return "$file$gz";
}

sub readCache()
{
  eval ("use Cache::CacheUtilities");
  if(!$@){return &Cache::CacheUtilities::read(shift);}
  else{die "Unable to find Cache/CacheUtilities.pm PERL module.";}
}

sub getScramArch ()
{
  if($SCRAM_ARCH eq "")
  {
    if(exists $ENV{SCRAM_ARCH}){$SCRAM_ARCH=$ENV{SCRAM_ARCH};}
    else{$SCRAM_ARCH=`$SCRAM_CMD arch`;chomp $SCRAM_ARCH;}
  }
  return $SCRAM_ARCH;
}

sub getFromEnvironmentFile ()
{
  my $var=shift;
  my $rel=shift;
  if(!exists $InternalCache->{$rel}{EnvironmentFile}){&getEnvironmentFileCache($rel);}
  if(exists $InternalCache->{$rel}{EnvironmentFile}{$var}){return $InternalCache->{$rel}{EnvironmentFile}{$var};}
  return "";
}

sub getEnvironmentFileCache ()
{
  my $rel=shift;
  if(!exists $InternalCache->{$rel}{EnvironmentFile})
  {
    my $ref;
    $InternalCache->{$rel}{EnvironmentFile}={};
    my $arch=&getScramArch();
    foreach my $f ("Environment","${arch}/Environment")
    {
      if (-f "${rel}/.SCRAM/${f}")
      {
        open($ref,"${rel}/.SCRAM/${f}") || die "Can not open ${rel}/.SCRAM/${f} file for reading.";
        while(my $line=<$ref>)
        {
          if(($line=~/^\s*$/) || ($line=~/^\s*#/)){next;}
          if($line=~/^\s*([^=\s]+?)\s*=\s*(.+)$/)
          {$InternalCache->{$rel}{EnvironmentFile}{$1}=$2;}
        }
        close($ref);
        $InternalCache->{dirty}=1;
      }
    }
  }
  return $InternalCache->{$rel}{EnvironmentFile};
}

sub createTmpReleaseArea ()
{
  my $rel=shift;
  my $dev=shift;
  my $dir=shift || &getTmpDir ();
  system("mkdir -p $dir");
  if($SCRAM_ARCH eq ""){&getScramArch ();}
  my $cf=&fixCacheFileName("${rel}/.SCRAM/${SCRAM_ARCH}/ProjectCache.db");
  if(!-f $cf){system("cd $rel; $SCRAM_CMD b -r echo_CXX ufast 2>&1 >/dev/null");}
  foreach my $sdir (".SCRAM", "config")
  {
    if(-d "${dir}/${sdir}"){system("rm -rf ${dir}/${sdir}");}
    system("cp -rpf ${rel}/${sdir} $dir");
  }
  my $setup=0;
  my $envfile="${dir}/.SCRAM/Environment";
  if ($SCRAM_VERSION=~/^V[2-9]/){$envfile="${dir}/.SCRAM/${SCRAM_ARCH}/Environment";}
  if($dev)
  {
    my $rtop=&getFromEnvironmentFile("RELEASETOP",$rel);
    if($rtop eq "")
    {
      system("touch $envfile; echo \"RELEASETOP=$rel\" >> $envfile");
      $setup=1;
    }
  }
  else
  {
    system("chmod -R u+w $dir");
    if ($SCRAM_VERSION=~/^V[2-9]/){system("rm -f $envfile");}
    else
    {
      system("grep -v \"RELEASETOP=\" $envfile  > ${envfile}.new");
      system("mv ${envfile}.new $envfile");
    }
    $setup=1;
  }
  my $prn="projectrename";
  if ($SCRAM_VERSION=~/^V1_0_/){$prn="ProjectRename";}
  system("cd $dir; $SCRAM_CMD build -r $prn >/dev/null 2>&1");
  if($setup){system("cd $dir; $SCRAM_CMD setup self >/dev/null 2>&1");}
  return $dir;
}

sub getBuildVariable ()
{
  my $dir=shift;
  my $var=shift || return "";
  my $xrule="";
  if($SCRAM_VERSION=~/^V[1-9]\d*_[1-9]\d*_/){$xrule=shift;}
  my $val=`cd $dir; $SCRAM_CMD b -f echo_${var} $xrule 2>&1 | grep "$var *="`; chomp $val;
  $val=~s/^\s*$var\s+=\s+//;
  return $val;
}

sub getOrderedTools ()
{
  my $cache=shift;
  my $tool=shift || "";
  my $data=shift || {};
  if($tool eq "self"){return;}
  if($tool eq "")
  {
    $data->{tooldone}={}; $data->{orderedtool}=[]; $data->{scrambased}={};
    foreach my $t (keys %{$cache->{SETUP}}){&getOrderedTools($cache,$t,$data);}
    foreach my $stool ("seal","coral","pool","iguana","ignominy","cmssw")
    {
      if(exists $data->{scrambased}{$stool})
      {unshift @{$data->{orderedtool}},$stool;delete $data->{scrambased}{$stool};}
    }
    foreach my $stool (keys %{$data->{scrambased}})
    {unshift @{$data->{orderedtool}},$stool;}
    unshift @{$data->{orderedtool}},"self";
    delete $data->{scrambased};
    delete $data->{tooldone};
    return $data->{orderedtool};
  }
  elsif((!exists $data->{tooldone}{$tool}) && (exists $cache->{SETUP}{$tool}))
  {
    $data->{tooldone}{$tool}=1;
    if ((exists $cache->{SETUP}{$tool}{SCRAM_PROJECT}) && ($cache->{SETUP}{$tool}{SCRAM_PROJECT} == 1)){$data->{scrambased}{$tool}=1;return;}
    if(exists $cache->{SETUP}{$tool}{USE})
    {
      foreach my $t (@{$cache->{SETUP}{$tool}{USE}})
      {
	$t=lc($t);
	&getOrderedTools($cache,$t,$data);
      }
    }
    unshift @{$data->{orderedtool}},$tool;
  }
}

#################################################################
# Reading writing cache files
#################################################################
sub writeHashCache ()
{
  my $cache=shift;
  my $file=shift;
  use Data::Dumper;
  my $cachefh;
  if (open($cachefh,">$file"))
  {
    $Data::Dumper::Varname='cache';
    $Data::Dumper::Purity = 1;
    print $cachefh Dumper($cache);
    close $cachefh;
  }
  else{die "can not open file $file for writing.";}
}

sub readHashCache ()
{
  my $file=shift;
  my $cache=undef;
  my $cachefh;
  if (open($cachefh,$file))
  {
    my @cacheitems = <$cachefh>;
    close $cachefh;
    $cache = eval "@cacheitems";
    die "Cache load error: ",$EVAL_ERROR,"\n", if ($EVAL_ERROR);
  }
  else{die "can not open file $file for reading.";}
  return $cache;
}
################################################
# Find SCRAM based release area
################################################
sub scramReleaseTop()
{return &checkWhileSubdirFound(shift,".SCRAM");}

sub checkWhileSubdirFound()
{
  my $dir=shift;
  my $subdir=shift;
  while((!-d "${dir}/${subdir}") && ($dir ne "/")){$dir=dirname($dir);}
  if(-d "${dir}/${subdir}"){return $dir;}
  return "";
}
#################################################
# Shared lib functions
#################################################

sub getLibSymbols ()
{
  my $file=&findActualPath(shift);
  my $cache={};
  if(($file ne "") && (-f $file))
  {
    my $islib=`file $file | sed 's|.*:||' | grep -i "shared object"`; chomp $islib;
    if($islib ne "")
    {
      foreach my $line (`nm -D $file`)
      {
        chomp $line;
	if($line=~/^([0-9A-Fa-f]+|)\s+([A-Za-z])\s+([^\s]+)\s*$/)
	{$cache->{$3}=$2;}
      }
    }
  }
  return $cache;
}

#################################################
# Object Files symbols
#################################################

sub getObjectFileSymbols ()
{
  my $files=shift;
  my $cache={};
  foreach my $file (@{$files})
  {
    if(($file ne "") && (-f $file))
    {
      my $isobj=`file $file | sed 's|.*:||' | grep -i "SB relocatable"`; chomp $isobj;
      if($isobj ne "")
      {
        foreach my $line (`nm $file`)
        {
          chomp $line;
	  if($line=~/^([0-9A-Fa-f]+|)\s+([A-Za-z])\s+([^\s]+)\s*$/)
	  {
	    my $s=$3; my $t=$2;
	    if(!exists $cache->{$s}){$cache->{$s}=$t;}
	    elsif($t eq "T"){$cache->{$s}=$t;}
	  }
        }
      }
    }
  }
  return $cache;
}


######################################################
#SCRAM BuildFile

##########################################################
# Read BuildFile
##########################################################
sub XML2DATA ()
{
  my $xml=shift;
  my $data=shift || {};
  foreach my $c (@{$xml->{child}})
  {
    if(exists $c->{name})
    {
      my $n=$c->{name};
      if($n=~/^(environment)$/){&XML2DATA($c,$data);}
      elsif($n=~/^(library|bin)$/)
      {
        my $fl=$c->{attrib}{file};
	my $p=$c->{attrib}{name};
	if($p ne ""){$data->{$n}{$p}{file}=[];}
	foreach my $f (split /\s+/,$fl)
	{
	  if($p eq "")
	  {
	    $p=basename($f); $p=~s/\.[^.]+$//;
	    $data->{$n}{$p}{file}=[];
	  }
	  push @{$data->{$n}{$p}{file}},"$f";
	}
	if ($p ne "")
	{
	  $data->{$n}{$p}{deps}={};
	  &XML2DATA($c,$data->{$n}{$p}{deps});
	}
      }
      elsif($n=~/^(use|lib)$/){$data->{$n}{$c->{attrib}{name}}=1;}
      elsif($n=~/^(flags)$/)
      {
	my @fs=keys %{$c->{attrib}};
	my $f=uc($fs[0]);
	my $v=$c->{attrib}{$fs[0]};
	if(!exists $data->{$n}{$f}){$data->{$n}{$f}=[];}
	my $i=scalar(@{$data->{flags}{$n}});
	$data->{$n}{$f}[$i]{v}=$v;
      }
      elsif($n=~/^(architecture)$/)
      {
        my $a=$c->{attrib}{name};
	if(!exists $data->{arch}{$a}){$data->{arch}{$a}={};}
	&XML2DATA($c,$data->{arch}{$a});
      }
      elsif($n=~/^(export)$/)
      {
        $data->{$n}={};
	&XML2DATA($c,$data->{$n});
      }
      elsif($n=~/^(include_path)$/)
      {
	$data->{$n}{$c->{attrib}{path}}=1;
      }
      elsif($n=~/^(makefile)$/)
      {
        if(!exists $data->{$n}){$data->{$n}=[];}
	foreach my $d (@{$c->{cdata}}){push @{$data->{$n}},"$d\n";}
      }
    }
  }
  return $data;
}

sub readBuildFile ()
{
  my $bfile=shift;
  my $bfn=basename($bfile);
  eval ("use SCRAM::Plugins::DocParser");
  if($@){die "Can not local SCRAM/Plugins/DocParser.pm perl module for reading $bfile.\n";}
  my $input=undef;
  if ($bfn!~/BuildFile\.xml/)
  {
    eval ("use SCRAM::Plugins::Doc2XML");
    if($@){die "Can not local SCRAM/Plugins/DocParser.pm perl module for reading $bfile.\n";}
    my $doc2xml = SCRAM::Plugins::Doc2XML->new(0);
    my $xml=$doc2xml->convert($bfile);
    $input = join("",@$xml);
  }
  my $xml = SCRAM::Plugins::DocParser->new();
  $xml->parse($bfile,$input);
  return &XML2DATA($xml->{output});
}

sub findBuildFileTag ()
{
  my $data=shift;
  my $bf=shift;
  my $tag=shift;
  my $d=shift || {};
  my $arch=shift || "FORALLARCH";
  my $pt=$data->{prodtype};
  my $pn=$data->{prodname};
  if(exists $bf->{$tag})
  {
    if(!exists $d->{$arch}){$d->{$arch}=[];}
    push @{$d->{$arch}},$bf;
  }
  if($pt && (exists $bf->{$pt}) && (exists $bf->{$pt}{$pn}) && (exists $bf->{$pt}{$pn}{deps}))
  {&findBuildFileTag($data,$bf->{$pt}{$pn}{deps},$tag,$d,$arch);}
  if(exists $bf->{arch})
  {
    if($SCRAM_ARCH eq ""){&getScramArch ();}
    foreach my $arch (keys %{$bf->{arch}})
    {if($SCRAM_ARCH=~/$arch/){&findBuildFileTag($data,$bf->{arch}{$arch},$tag,$d,$arch);}}
  }
  return $d;
}

sub updateFromRefBuildFile ()
{
  my $cacherefbf = shift;
  my $data  = shift;
  if(defined $cacherefbf)
  {
    my $f=&findBuildFileTag($data,$cacherefbf,"flags");
    my $l=&findBuildFileTag($data,$cacherefbf,"lib");
    my $m=&findBuildFileTag($data,$cacherefbf,"makefile");
    my $i=&findBuildFileTag($data,$cacherefbf,"include_path");
    my $ix={};
    if(exists $cacherefbf->{export})
    {$ix=&findBuildFileTag($data,$cacherefbf->{export},"include_path");}
    foreach my $a (keys %$f)
    {
      foreach my $c (@{$f->{$a}})
      {
        foreach my $f1 (keys %{$c->{flags}})
        {
	  if(($f1 eq "SEAL_PLUGIN_NAME") || ($f1 eq "SEALPLUGIN") || ($f1 eq "EDM_PLUGIN")){next;}
          foreach my $fv (@{$c->{flags}{$f1}}){push @{$data->{bfflags}},"$f1=".$fv->{v};}
        }
      }
    }
    foreach my $a (keys %$l)
    {
      foreach my $c (@{$l->{$a}})
      {foreach my $f1 (keys %{$c->{lib}}){$data->{lib}{$a}{$f1}=1;}}
    }
    foreach my $a (keys %$i)
    {
      foreach my $c (@{$i->{$a}})
      {foreach my $f1 (keys %{$c->{include_path}}){$data->{include_path}{$a}{$f1}=1;}}
    }
    foreach my $a (keys %$ix)
    {
      foreach my $c (@{$ix->{$a}})
      {foreach my $f1 (keys %{$c->{include_path}}){$data->{export}{include_path}{$a}{$f1}=1;}}
    }
    foreach my $a (keys %$m)
    {
      foreach my $c (@{$m->{$a}})
      {
        my $c1=$c->{makefile};
        if(scalar(@$c1)>0)
        {
          if(!exists $data->{makefile}{$a}){$data->{makefile}{$a}=[];}
	  foreach my $f1 (@$c1){push @{$data->{makefile}{$a}},$f1;}
        }
      }
    }
  }
}

sub _xmlendtag()
{
  my $xml=shift;
  if($xml){return "/";}
  return "";
}

sub printBuildFile ()
{
  my $data=shift;
  my $file=shift;
  my $tab="";
  my $isPackage=$data->{isPackage};
  my $prodtype=$data->{prodtype};
  my $prodname=$data->{prodname};
  my $filestr=$data->{filestr};
  my $ccfiles=$data->{ccfiles};
  my $outfile=STDOUT;
  my $closefile=0;
  my $bfn=basename($file);
  my $xml=0;
  if($bfn=~/BuildFile\.xml/){$xml=1;}
  if($file ne "")
  {
    $outfile="";
    if(!open($outfile,">$file"))
    {
      print STDERR "Can not open file \"$file\" for writing. Going to print output on STDOUT.\n";
      $outfile=STDOUT;
    }
    else{$closefile=1;}
  }
  if(!$isPackage)
  {
    print $outfile "<$prodtype name=\"$prodname\" file=\"$filestr\">\n";
    $tab=" ";
  }
  my $edmplugin=0;
  if($ccfiles>0)
  {
    if(exists $data->{deps}{src})
    {
      foreach my $dep (sort keys %{$data->{deps}{src}})
      {print $outfile "$tab<use name=\"$dep\"",&_xmlendtag($xml),">\n";}
    }
    foreach my $f (sort keys %{$data->{flags}})
    {
      if($f eq "EDM_PLUGIN"){$edmplugin=$data->{flags}{$f};}
      if(exists $data->{sflags}{$f})
      {
        my $v=$data->{flags}{$f};
        print $outfile "$tab<flags $f=\"$v\"",&_xmlendtag($xml),">\n";
      }
      else
      {
        foreach my $v (@{$data->{flags}{$f}})
        {
          if(exists $data->{keyflags}{$f})
	  {
	    my ($n,$v1)=split /=/,$v,2;
	    if($v=~/^$n=/)
	    {
	      if($v1=~/^\"(.*?)\"$/){print $outfile "$tab<flags ${f}='${n}=\\\"$1\\\"'";}
	      else{print $outfile "$tab<flags ${f}='${n}=${v1}'";}
	    }
	    else{print $outfile "$tab<flags ${f}=\"${n}\"";}
	  }
	  else{print $outfile "$tab<flags $f=\"$v\"";}
	  print $outfile &_xmlendtag($xml),">\n";
        }
      }
    }
  }
  my %allarch=();
  foreach my $f (keys %{$data->{include_path}}){$allarch{$f}{include_path}=1;}
  foreach my $f (keys %{$data->{lib}}){$allarch{$f}{lib}=1;}
  foreach my $f (keys %{$data->{makefile}}){$allarch{$f}{makefile}=1;}
  foreach my $a (sort keys %allarch)
  {
    if($a ne "FORALLARCH"){print $outfile "$tab<architecture name=\"$a\">\n";$tab="$tab  ";}
    if(exists $allarch{$a}{include_path})
    {foreach my $f (sort keys %{$data->{include_path}{$a}}){print $outfile "$tab<include_path path=\"$f\"",&_xmlendtag($xml),">\n";}}
    if(exists $allarch{$a}{lib})
    {foreach my $f (sort keys %{$data->{lib}{$a}}){print $outfile "$tab<lib name=\"$f\"",&_xmlendtag($xml),">\n";}}
    if(exists $allarch{$a}{makefile})
    {
      print $outfile "$tab<makefile>\n";
      foreach my $f (@{$data->{makefile}{$a}}){print $outfile "$f";}
      print $outfile "$tab</makefile>\n";
    }
    if($a ne "FORALLARCH"){$tab=~s/  $//;print $outfile "$tab</architecture>\n";}
  }
  if(!$isPackage){print $outfile "</$prodtype>\n";$tab="";}
  elsif(!$edmplugin)
  {
    print $outfile "<export>\n";
    my $hasexport=0;
    if(exists $data->{export})
    {
      my %allarch=();
      foreach my $a (keys %{$data->{export}{include_path}}){$allarch{$a}{include_path}=1;}
      foreach my $a (sort keys %allarch)
      {
        if($a ne "FORALLARCH"){print $outfile "  <architecture name=\"$a\">\n";$tab="  ";$hasexport=1;}
        if(exists $allarch{$a}{include_path})
        {foreach my $f (sort keys %{$data->{export}{include_path}{$a}}){print $outfile "$tab  <include_path path=\"$f\"",&_xmlendtag($xml),">\n";$hasexport=1;}}
        if($a ne "FORALLARCH"){$tab="";print $outfile "  </architecture>\n";}
      }
    }
    if (!$xml)
    {
      if(exists $data->{deps}{src})
      {
        my @packs=sort keys %{$data->{deps}{src}};
        foreach my $dep (@packs)
        {print $outfile "  <use name=\"$dep\">\n";$hasexport=1;}
      }
    }
    if(($ccfiles>0) && ($edmplugin==0))
    {
      if($xml){print $outfile "  <lib name=\"1\"/>\n";}
      else{print $outfile "  <lib name=\"$prodname\">\n";}
      $hasexport=1;
    }
    if(!$hasexport){print $outfile "  <flags DummyFlagToAvoidWarning=\"0\"",&_xmlendtag($xml),">\n";}
    print $outfile "</export>\n";
  }
  if($closefile){close($outfile);}
}

sub removeDuplicateTools ()
{
  my $cache=shift;
  my $data=shift;
  foreach my $x ("src", "interface")
  {
    foreach my $t (keys %{$data->{deps}{$x}})
    {
      if((exists $cache->{TOOLS}{$t}) && (exists $cache->{TOOLS}{$t}{USE}))
      {foreach my $t1 (@{$cache->{TOOLS}{$t}{USE}}){delete $data->{deps}{$x}{$t1};}}
    }
  }
}

sub removeExtraLib ()
{
  my $cache=shift;
  my $data=shift;
  foreach my $a (keys %{$data->{lib}})
  {
    foreach my $lib (keys %{$data->{lib}{$a}})
    {
      foreach my $t (keys %{$data->{deps}{src}})
      {if(&isLibInTool($lib,$t,$cache)){delete $data->{lib}{$a}{$lib};last;}}
    }
    if(scalar(keys %{$data->{lib}{$a}})==0){delete $data->{lib}{$a};}
  }
}

sub isLibInTool ()
{
  my $lib=shift;
  my $tool=shift;
  my $cache=shift;
  if((exists $cache->{TOOLS}) && (exists $cache->{TOOLS}{$tool}))
  {
    if(exists $cache->{TOOLS}{$tool}{LIB})
    {
      foreach my $l (@{$cache->{TOOLS}{$tool}{LIB}})
      {if($lib eq $l){return 1;}}
    }
    if(exists $cache->{TOOLS}{$tool}{USE})
    {
      foreach my $t (@{$cache->{TOOLS}{$tool}{USE}})
      {if(&isLibInTool($lib,$t,$cache)){return 1;}}
    }
  }
  return 0;
}

#########################################################################
# Read C/C++ file
#########################################################################
sub searchIncFilesCXX ()
{
  my $file=shift;
  my $data=shift;
  $data->{includes}={};
  my $cache=&readCXXFile($file);
  if(!exists $cache->{lines}){return;}
  my $total_lines=scalar(@{$cache->{lines}});
  for(my $i=0;$i<$total_lines;$i++)
  {
    my $line=$cache->{lines}[$i];
    while($line=~/\\\//){$line=~s/\\\//\//;}
    if ($line=~/^\s*#\s*include\s*([\"<](.+?)[\">])\s*/)
    {$data->{includes}{$2}=1;}
  }
}

sub readCXXFile ()
{
  my $file=shift;
  my $cache=shift || {};
  my $fref=0;
  if (!open ($fref, "$file"))
  {print STDERR "ERROR: Can not open file \"$file\" for reading.\n";return $cache;}
  $cache->{comment_type}=0;
  $cache->{string_started}=0;
  $cache->{lines}=[];
  $cache->{line_numbers}=[];
  $cache->{comments_lines}=0;
  $cache->{empty_lines}=0;
  $cache->{total_lines}=0;
  $cache->{code_lines}=0;
  while(my $line=<$fref>)
  {
    chomp $line;$line=~s/\r$//;
    &incData(\$cache->{total_lines});
    
    #check for empty line
    if ($line=~/^\s*$/){&incData(\$cache->{empty_lines});next;}
    
    #combine all lines which ends with /
    $cache->{tmp}{lines}=[];
    $cache->{tmp}{line_nums}=[];
    $cache->{tmp}{comments}=[];
    if ($line=~/^(.*?)\\$/)
    {
      my $pre=$1;
      
      #check for empty line
      if ($pre=~/^\s*$/){&incData(\$cache->{empty_lines});next;}
      while($line=<$fref>)
      {
	chomp $line;$line=~s/\r$//;
	&incData(\$cache->{total_lines});
	if ($line=~/^(.*?)\\$/)
	{
	  $line=$1;
	  if ($line=~/^\s*$/){&incData(\$cache->{empty_lines});$pre="${pre}${line}";}
	  else
	  {
	    push @{$cache->{tmp}{lines}}, $pre;
	    push @{$cache->{tmp}{line_nums}}, $cache->{total_lines}-1;
	    $pre=$line;
	  }
	}
	else
	{
	  if ($line=~/^\s*$/){&incData(\$cache->{empty_lines});$line="${pre}${line}";}
	  else
	  {
	    push @{$cache->{tmp}{lines}}, $pre;
	    push @{$cache->{tmp}{line_nums}}, $cache->{total_lines}-1;
	  }
	  push @{$cache->{tmp}{lines}}, $line;
	  push @{$cache->{tmp}{line_nums}}, $cache->{total_lines};
	  last;
	}
      }
    }
    else
    {
      push @{$cache->{tmp}{lines}}, $line;
      push @{$cache->{tmp}{line_nums}}, $cache->{total_lines};
    }
    &removeCommentCXX ($cache);
  }
  close ($fref);
  delete $cache->{tmp};
  delete $cache->{comment_type};
  delete $cache->{string_started};
  $cache->{code_lines}=scalar(@{$cache->{lines}});
  return $cache;
}

sub removeCommentCXX ()
{
  my $cache=shift;
  my $e=scalar(@{$cache->{tmp}{lines}});
  if ($cache->{comment_type}==2){$cache->{comment_type}=0;}
  for(my $i=0; $i < $e; $i++)
  {
    my $line=$cache->{tmp}{lines}[$i];
    if ($cache->{comment_type} == 2)
    {
      $cache->{tmp}{lines}[$i]="";
      $cache->{string_started}=0;
      if ($line=~/^\s*$/){&incData(\$cache->{empty_lines});$cache->{tmp}{comments}[$i]=0;}
      else{$cache->{tmp}{comments}[$i]=1;}
    }
    elsif ($cache->{comment_type} == 1)
    {
      $cache->{string_started}=0;
      if ($line=~/^\s*$/){&incData(\$cache->{empty_lines});$cache->{tmp}{comments}[$i]=0;}
      elsif ($line=~/^(.*?)\*\/(.*)$/)
      {
	my $x=$1;
	$line=$2;
	$cache->{comment_type}=0;
	$cache->{tmp}{lines}[$i]=$line;
	&adjustCommentType1CXX($cache,$i,$line,$x);
        if ($line!~/^\s*$/){$i--;}
      }
      else{$cache->{tmp}{lines}[$i]="";$cache->{tmp}{comments}[$i]=1;}
    }
    else
    {
      &removeStringCXX ($cache);
      $line=$cache->{tmp}{lines}[$i];
      my $x1=undef; my $x2=undef;
      if ($line=~/^(.*?)\/\/(.*)$/){$line=$1;$x2=$2;}
      if ($line=~/^(.*?)\/\*(.*)$/)
      {
        $line=$1;
	$x1=$2;
	if(defined $x2){$x1.="//${x2}";$x2=undef;}
      }
      if(defined $x1){$i=removeCommentType1CXX ($cache,$i,$line,$x1);}
      elsif(defined $x2){if($x2!~/\s*INCLUDECHECKER\s*:\s*SKIP/i){$i=removeCommentType2CXX ($cache,$i,$line,$x2);}}
      elsif($cache->{tmp}{comments}[$i] eq ""){$cache->{tmp}{comments}[$i]=0;}
    }
  }
  for(my $i=0; $i < $e; $i++)
  {
    my $line=$cache->{tmp}{lines}[$i];
    $cache->{comments_lines}=$cache->{comments_lines}+$cache->{tmp}{comments}[$i];
    if ($line!~/^\s*$/)
    {
      push @{$cache->{lines}}, $line;
      push @{$cache->{line_numbers}}, $cache->{tmp}{line_nums}[$i];
    }
  }
}

sub removeCommentType2CXX ()
{
  my $cache=shift;
  my $i=shift;
  my $line=shift;
  my $x=shift;
  $cache->{tmp}{lines}[$i]=$line;
  $cache->{comment_type}=2;
  $cache->{string_started}=0;
  &adjustCommentType1CXX($cache,$i,$line,$x);
  return $i;
}

sub removeCommentType1CXX ()
{
  my $cache=shift;
  my $i=shift;
  my $line=shift;
  my $x=shift;
  $cache->{string_started}=0;
  my $ni=$i;
  if ($x=~s/^(.*?)\*\///)
  {
    $line="${line}${x}";
    $x=$1;
    if ($line!~/^\s*$/){$ni--;}
  }
  else{$cache->{comment_type}=1;}
  &adjustCommentType1CXX($cache,$i,$line,$x);
  $cache->{tmp}{lines}[$i]=$line;
  return $ni;
}

sub adjustCommentType1CXX
{
  my $cache=shift;
  my $i=shift;
  my $line=shift;
  my $x=shift;
  if ($x=~/[^\s]/){$cache->{tmp}{comments}[$i]=1;}
  if ($line=~/^\s*$/)
  {if(!$cache->{tmp}{comments}[$i]){&incData(\$cache->{empty_lines});$cache->{tmp}{comments}[$i]=0;}}
  elsif($cache->{tmp}{comments}[$i] eq ""){$cache->{tmp}{comments}[$i]=0;}
}

sub removeStringCXX ()
{
  my $cache=shift;
  my $lines=$cache->{tmp}{lines};
  my $str_started=$cache->{string_started};
  my $esc=0;
  my $e=scalar(@{$lines});
  for(my $i=0; $i < $e; $i++)
  {
    my $line=$lines->[$i];
    my $x1=length($line);
    if($line=~/^(.*?)\/\*(.*)$/){$x1=length($1);}
    if($line=~/^(.*?)\/\/(.*)$/)
    {
      my $x2=length($1);
      if ($x2 < $x1){$x1=$x2;}
    }
    my $nl="";
    my $j=-1;
    foreach my $ch (split //, $line)
    {
      $j++;
      if ($str_started)
      {
        if ($esc){$esc=0;}
        elsif($ch eq "\\"){$esc=1;}
        elsif ($ch eq '"'){$str_started=0;}
	elsif ($ch eq '/'){$nl="${nl}\\";}
	elsif ($ch eq '*'){$nl="${nl}\\";}
      }
      elsif(($ch eq '"') && ($j < $x1)){$str_started=1;}
      $nl="${nl}${ch}";
    }
    $lines->[$i]=$nl;
  }
  $cache->{string_started}=$str_started;
}

sub skipIfDirectiveCXX ()
{
  my $data=shift;
  my $s=shift;
  my $e=shift;
  my $i=$s;
  for(;$i<$e;$i++)
  {
    my $line=$data->[$i];
    if ($line=~/^\s*#\s*if(n|\s+|)def(ined|\s+|)/)
    {$i=&skipIfDirectiveCXX ($data, $i+1, $e);}
    elsif($line=~/^\s*#\s*endif\s*/){last;}
  }
  return $i;
}

sub searchPreprocessedFile ()
{
  my $file=shift;
  my $data=shift;
  my $xflags=shift || "";
  my $ofile=shift || "";
  my %search=();
  my $hasfilter=0;
  my $delfile=0;
  foreach my $k (keys %{$data->{searchPreprocessed}})
  {if(!exists $data->{searchPreprocessed}{$k}{file}){$search{$k}=1;$hasfilter=1;}}
  if(!$hasfilter){return;}
  if($ofile eq ""){$ofile=&generatePreprocessedCXX($file,$data,$xflags);$delfile=1;}
  if($ofile eq ""){return;}
  if(!open(OFILE,"$ofile"))
  {
    print STDERR "Can not open file \"$ofile\" for reading.";
    if($delfile){my $d=dirname($ofile);system("rm -rf $d");}
    exit 0;
  }
  while(my $line=<OFILE>)
  {
    chomp $line;
    foreach my $k (keys %search)
    {
      my $f=$data->{searchPreprocessed}{$k}{filter};
      if($line=~/$f/)
      {
	$data->{searchPreprocessed}{$k}{file}=$file;
	delete $search{$k};
      }
    }
    if(scalar(keys %search)==0){last;}
  }
  close(OFILE);
  if($delfile){my $d=dirname($ofile);system("rm -rf $d");}
}

sub generatePreprocessedCXX ()
{
  my $file=shift;
  my $data=shift;
  my $xflags=shift || "";
  my $compilecmd=$data->{compilecmd};
  if($compilecmd ne "")
  {
    $compilecmd.=" $xflags";
    my $tmpdir=&getTmpDir();
    my $ofile="${tmpdir}/preprocessed.$$";
    if(system("$compilecmd -E -o $ofile $file")==0){return $ofile;}
    system("rm -rf $tmpdir");
  }
  return "";
}

sub symbolChecking ()
{
  my $lib=shift || return 0;
  my $d=dirname($lib);
  my $rel=&scramReleaseTop($d);
  if($rel eq ""){return 1;}
  my $cxx="";
  my $ld_path="";
  foreach my $f ("CXX", "CXXFLAGS", "CXXSHAREDOBJECTFLAGS", "LDFLAGS", "LD_LIBRARY_PATH")
  {
    my $val="";
    if(exists $InternalCache->{$rel}{BuildVariables}{$f})
    {$val=$InternalCache->{$rel}{BuildVariables}{$f};}
    else
    {
      $val=&getBuildVariable($d,$f);
      $InternalCache->{$rel}{BuildVariables}{$f}=$val;
      $InternalCache->{dirty}=1;
    }
    if($f eq "LD_LIBRARY_PATH"){$ld_path=$val;}
    else{$cxx.=" $val";}
  }
  if($ld_path ne ""){$cxx="LD_LIBRARY_PATH=$ld_path; export LD_LIBRARY_PATH; $cxx";}
  $cxx.=" -L$d";
  my $l=basename($lib); $l=~s/^lib(.+?)\.so$/$1/;
  my $tmpd=&getTmpDir();
  my $tmpf="${tmpd}/$l.cpp";
  system("echo \"int main(){}\" > $tmpf");
  print ">> Checking for missing symbols.\n";
  if($DEBUG){print "$cxx -o ${tmpf}.out -l$l $tmpf\n";}
  my @lines=`$cxx -o ${tmpf}.out -l$l $tmpf 2>&1`;
  my $ret=$?;
  system("rm -rf $tmpd");
  if($ret != 0)
  {
    $ret=0;
    foreach my $line (@lines)
    {
      chomp $line;
      if($line=~/\/lib${l}\.so:\s+undefined reference to\s+/){print "$line\n";$ret=1;}
    }
  }
  return $ret;
}
###################################################
sub leftAdjust {
  my $i;
  my $data=shift;
  my $width=shift;
  if(length($data)<$width){
    for($i=length($data);$i<$width;$i++){
      $data="$data ";
    }
  }  
  return $data;
}

sub rightAdjust {
  my $i;
  my $data=shift;
  my $width=shift;
  if(length($data)<$width){
    for($i=length($data);$i<$width;$i++){
      $data=" $data";
    }
  }  
  return $data;
}

sub setPrecision {
  my $num=shift;
  my $prec=shift;
  if($num=~/[^.]+\./){
    if($num=~s/([^.]+\.(\d{0,$prec}))\d*/$1/){ $l=length($2);}
  }
  else { $num="$num."; $l=0;}  
  for($i=$l;$i<$prec;$i++){$num="${num}0";}
  chomp $num;
  return $num;
}

sub incData ()
{
  my $data=shift;
  $$data=$$data+(shift || 1);
}

1;
