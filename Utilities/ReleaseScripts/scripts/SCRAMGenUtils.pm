package SCRAMGenUtils;
use File::Basename;
use Storable qw(nstore retrieve);

our $SCRAM_CMD="scram";
our $SCRAM_VERSION="";
local $SCRAM_ARCH="";
local $DEBUG=0;
local $InternalCache={};
local $Cache={};
local $CacheType=1;

sub init ()
{
  my $dir=shift;
  $CacheType=1;
  &scramVersion ($dir);
  &getScramArch();
  unshift @INC,"$ENV{SCRAM_HOME}/src";
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
  my $tmp="${dir}/f${index}.$$";
  while(-f $tmp)
  {$index++;$tmp="${dir}/f${index}.$$";}
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

sub updateConfigFileData ()
{
  my $file=shift;
  if (($file eq "") || (!-f "$file")){return "";}
  my $data=shift || return $file;
  my $cache=shift || {};
  my $r;
  open ($r,$file) || die "Can not open file for reading:$file\n";
  while(my $line=<$r>)
  {
    chomp $line;
    if(($line=~/^\s*#/) || ($line=~/^\s*$/)){next;}
    my @x=split /:/,$line;
    my $count=scalar(@x);
    for(my $i=1;$i<$count;$i++)
    {
      my $y=$x[$i];
      if ($y=~/^\s*$/)
      {
        for(my $j=$i;$j<$count-1;$j++){$x[$j]=$x[$j+1];}
	pop @x;
	$count--;
	$i--;
	next;
      }
      my $py=$x[$i-1];
      if ($py=~/\\$/){$x[$i-1]="$py:$y";$x[$i]="";$i--;}
    }
    my $c=undef;
    my $remove="";
    if($x[0]=~/^(X|)DATA$/){$c=$data;$remove=$1;}
    elsif($x[0]=~/^(X|)CACHE$/){$c=$cache;$remove=$1;}
    if ($remove){$count++;}
    if($count<3){next;}
    if(defined $c)
    {
      #if($DEBUG){print STDERR "#Configuring=>$line\n";}
      my $i=0;
      for($i=1;$i<$count-2;$i++)
      {
        my $y=$x[$i];
	if(!exists $c->{$y}){$c->{$y}={};}
	$c=$c->{$y};
      }
      if ($remove){delete $c->{$x[$i]};}
      else{$c->{$x[$i]}=$x[$i+1];}
    }
  }
  close($r);
  return $file;
}

sub findUniqDirs ()
{
  my $dirs=shift;
  my $dir=shift;
  my $uniq = shift || [];
  my $c=0;
  foreach my $d (keys %$dirs){&findUniqDirs($dirs->{$d},"${dir}/${d}",$uniq);$c++;}
  if ($c == 0){push @$uniq,$dir;}
  return $uniq;
}

#########################################################################
# Reading Project Cache DB
#########################################################################

sub scramVersion ()
{
  my $rel=shift;
  if ($SCRAM_VERSION eq "")
  {
    if (exists $ENV{SCRAM_HOME}){$SCRAM_VERSION=basename($ENV{SCRAM_HOME}); return $SCRAM_VERSION;}
    my $scram=`which $SCRAM_CMD 2>&1`; chomp $scram;
    if ($scram!~/\/$SCRAM_CMD\s*$/){die "can not find $SCRAM_CMD command.\n";}
    my $dir=`cd $rel; sh -v $scram --help 2>&1 | grep SCRAMV1_ROOT=`; chomp $dir;
    $dir=~s/SCRAMV1_ROOT=["']//; $dir=~s/['"].*//;
    if (!-d $dir){die "Can not find scram installation path.\n";}
    my $sver=basename($dir);
    my $sdir=dirname($dir);
    $sver=~s/^(V\d+_\d+_).+$/$1/;
    my $dref;
    if (opendir($dref,$sdir))
    {
      my %vers=();
      foreach my $ver (readdir($dref)){if($ver=~/^$sver/){push @vers,$ver;}}
      closedir($dref);
      my $c=scalar(@vers);
      if($c)
      {
        @vers=sort @vers;
        $dir="${sdir}/".$vers[$c-1];
      }
    }
    $SCRAM_VERSION=basename($dir);
    $ENV{SCRAM_HOME}=$dir;
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

sub writeCache()
{
  eval ("use Cache::CacheUtilities");
  if(!$@){return &Cache::CacheUtilities::write(shift,shift);}
  else{die "Unable to find Cache/CacheUtilities.pm PERL module.";}
}

sub getScramArch ()
{
  if($SCRAM_ARCH eq "")
  {
    if(exists $ENV{SCRAM_ARCH}){$SCRAM_ARCH=$ENV{SCRAM_ARCH};}
    else{$SCRAM_ARCH=`$SCRAM_CMD arch`;chomp $SCRAM_ARCH;$ENV{SCRAM_ARCH}=$SCRAM_ARCH;}
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
  my $prn="projectrename";
  if ($SCRAM_VERSION=~/^V1_0_/){$prn="ProjectRename";}
  system("cd $dir; $SCRAM_CMD build -r $prn >/dev/null 2>&1");
  my $setup=0;
  my $envfile="${dir}/.SCRAM/${SCRAM_ARCH}/Environment";
  if ($SCRAM_VERSION=~/^V1/){$envfile="${dir}/.SCRAM/Environment";}
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
    if ($SCRAM_VERSION=~/^V1_/)
    {
      system("grep -v \"RELEASETOP=\" $envfile  > ${envfile}.new");
      system("mv ${envfile}.new $envfile");
    }
    else{system("rm -f $envfile");}
    $setup=1;
  }
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
  my $rev=shift || 0;
  my $tools=$cache->{SETUP};
  my $c={};
  $c->{done}={};
  $c->{scram}={};
  $c->{data}=[];
  $c->{cache}=$tools;
  foreach my $t (sort keys %$tools)
  {
    if ($t eq "self"){next;}
    if ((exists $tools->{$t}{SCRAM_PROJECT}) && ($tools->{$t}{SCRAM_PROJECT}==1)){$c->{scram}{$t}=1;next;}
    &_getOrderedTools($c,$t);
  }
  foreach my $t (keys %{$c->{scram}}){&_getOrderedSTools($c,$t);}
  my @odata=();
  foreach my $d (@{$c->{data}})
  {if (ref($d) eq "ARRAY"){foreach my $t (@$d) {push @odata,$t;}}}
  if (exists $tools->{self}){push @odata,$tools->{self};}
  my @otools =();
  my @ctools=();
  foreach my $t ( reverse @odata )
  {
    if ((exists $t->{SCRAM_COMPILER}) && ($t->{SCRAM_COMPILER}==1)){push @ctools,$t->{TOOLNAME}; next;}
    push @otools,$t->{TOOLNAME};
  }
  push @otools,@ctools;
  if ($rev){@otools=reverse @otools;}
  return @otools;
}

sub _getOrderedSTools ()
{
   my $c=shift;
   my $tool=shift;
   my $order=-1;
   if(exists $c->{done}{$tool}){return $c->{done}{$tool};}
   $c->{done}{$tool}=$order;
   if(!exists $c->{scram}{$tool}){return $order;}
   if(!exists $c->{cache}{$tool}){return $order;}
   my $base=uc($tool)."_BASE";
   if(!exists $c->{cache}{$tool}{$base}){return $order;}
   $base=$c->{cache}{$tool}{$base};
   if(!-d $base){print STDERR "ERROR: Release area \"$base\" for \"$tool\" is not available.\n"; return $order;}
   my $cfile=&fixCacheFileName("${base}/.SCRAM/${SCRAM_ARCH}/ToolCache.db");
   if (!-f $cfile){print STDERR "ERROR: Tools cache file for release area \"$base\" is not available.\n";return $order;}
   my $cache=&readCache($cfile);
   my $tools=$cache->{SETUP};
   my $order=scalar(@{$c->{data}})-1;
   foreach my $t (keys %$tools)
   {
     if($t eq "self"){next;}
     if((exists $tools->{$t}{SCRAM_PROJECT}) && ($tools->{$t}{SCRAM_PROJECT}==1))
     {
       my $o=&_getOrderedSTools($c,$t);
       if ($o>$order){$order=$o;}
     }
   }
   $order++;
   $c->{done}{$tool}=$order;
   if(!defined $c->{data}[$order]){$c->{data}[$order]=[];}
   push @{$c->{data}[$order]},$c->{cache}{$tool};
   $c->{done}{$tool}=$order;
   return $order;
}

sub _getOrderedTools()
{
  my $c    = shift;
  my $tool = shift;
  my $order=-1;
  if(exists $c->{done}{$tool}){return $c->{done}{$tool};}
  $c->{done}{$tool}=$order;
  if (exists $c->{cache}{$tool})
  {
    if (exists $c->{cache}{$tool}{USE})
    {
      foreach my $use (@{$c->{cache}{$tool}{USE}})
      {
	my $o=&_getOrderedTools($c,lc($use));
	if ($o>$order){$order=$o;}
      }
    }
    $order++;
    if(!defined $c->{data}[$order]){$c->{data}[$order]=[];}
    push @{$c->{data}[$order]},$c->{cache}{$tool};
    $c->{done}{$tool}=$order;
  }
  return $order;
}

#################################################################
# Reading writing cache files
#################################################################
sub writeHashCache ()
{
  my $cache=shift;
  my $file=shift;
  my $binary=shift || undef;
  if (!defined $binary){$binary=$CacheType;}
  if ($binary)
  {
   eval {nstore($cache,$file);};
   die "Cache write error: ",$EVAL_ERROR,"\n", if ($EVAL_ERROR);
  }
  else
  {
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
}

sub readHashCache ()
{
  my $file=shift;
  my $binary=shift || undef;
  if (!defined $binary){$binary=$CacheType;}
  my $cache=undef;
  if ($binary)
  {
   $cache = eval "retrieve(\"$file\")";
   die "Cache load error: ",$@,"\n", if ($@);
  }
  else
  {
    my $cachefh;
    if (open($cachefh,$file))
    {
      my @cacheitems = <$cachefh>;
      close $cachefh;
      $cache = eval "@cacheitems";
      die "Cache load error: ",$EVAL_ERROR,"\n", if ($EVAL_ERROR);
    }
    else{die "can not open file $file for reading.";}
  }
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
  while((!-d "${dir}/${subdir}") && ($dir!~/^[\.\/]$/)){$dir=dirname($dir);}
  if(-d "${dir}/${subdir}"){return $dir;}
  return "";
}
#################################################
# Shared lib functions
#################################################

sub getLibSymbols ()
{
  my $file=&findActualPath(shift);
  my $filter=shift || ".+";
  my $cache={};
  if(($file ne "") && (-f $file))
  {
    foreach my $line (`nm -D $file`)
    {
      chomp $line;
      if($line=~/^([0-9A-Fa-f]+|)\s+([A-Za-z])\s+([^\s]+)\s*$/)
      {
        my $s=$3; my $type=$2;
	if ($type=~/$filter/){$cache->{$s}=$type;}
      }
    }
  }
  return $cache;
}

sub getObjectSymbols ()
{
  my $file=&findActualPath(shift);
  my $filter=shift || ".+";
  my $cache={};
  if(($file ne "") && (-f $file))
  {
    foreach my $line (`nm $file`)
    {
      chomp $line;
      if($line=~/^([0-9A-Fa-f]+|)\s+([A-Za-z])\s+([^\s]+)\s*$/)
      {
        my $s=$3; my $type=$2;
	if ($type=~/$filter/){$cache->{$s}=$type;}
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
	my $i=scalar(@{$data->{$n}{$f}});
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
  my $raw=shift || 0;
  my $bfn=basename($bfile);
  eval ("use SCRAM::Plugins::DocParser");
  if($@){die "Can not locate SCRAM/Plugins/DocParser.pm perl module for reading $bfile.\n";}
  my $input=undef;
  if ($bfn!~/BuildFile\.xml(.auto|)/)
  {
    eval ("use SCRAM::Plugins::Doc2XML");
    if($@){die "Can not locate SCRAM/Plugins/Doc2XML.pm perl module for reading $bfile.\n";}
    my $doc2xml = SCRAM::Plugins::Doc2XML->new(0);
    my $xml=$doc2xml->convert($bfile);
    $input = join("",@$xml);
  }
  else
  {
    my $ref;
    if(open($ref,$bfile))
    {
      while(my $l=<$ref>)
      {
        chomp $l;
        if ($l=~/^\s*(#.*|)$/){next;}
        $input.="$l ";
      }
    }
  }
  my $xml = SCRAM::Plugins::DocParser->new();
  $xml->parse($bfile,$input);
  if ($raw){return $xml->{output};}
  return &XML2DATA($xml->{output});
}

sub convert2XMLBuildFile ()
{
  &dumpXMLBuildFile(&readBuildFile(shift,1),shift);
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

sub dumpXMLBuildFile ()
{
  my $xml=shift;
  my $outfile=shift;
  my $tab=shift || "";
  my $ref=undef;
  if (!ref($outfile)){open($ref,">$outfile") || die "CAn not open file for writing:$outfile\n";}
  else{$ref=$outfile;}
  foreach my $c (@{$xml->{child}})
  {
    if(exists $c->{name})
    {
      my $n=$c->{name};
      if($n=~/^(environment)$/){print $ref "${tab}<$n>\n";&dumpXMLBuildFile($c,$ref,"$tab  ");print $ref "${tab}</$n>\n";}
      elsif($n=~/^(library|bin)$/)
      {
        my $fl=$c->{attrib}{file};
	my $p=$c->{attrib}{name};
	my @fs=();
	foreach my $f (split /\s+/,$fl)
	{
	  if($p eq ""){$p=basename($f); $p=~s/\.[^.]+$//;}
	  push @fs,$f;
	}
	if ($p ne "")
	{
	  print $ref "${tab}<$n name=\"$p\" file=\"",join(",",@fs),"\">\n";
	  &dumpXMLBuildFile($c,$ref,"$tab  ");
	  print $ref "${tab}</$n>\n";
	}
      }
      elsif($n=~/^(use|lib)$/){print $ref "${tab}<$n name=\"",$c->{attrib}{name},"\"/>\n";}
      elsif($n=~/^(flags)$/)
      {
	my @fs=keys %{$c->{attrib}};
	my $f=uc($fs[0]);
	my $v=$c->{attrib}{$fs[0]};
	print $ref "${tab}<$n $f=\"$v\"/>\n";
      }
      elsif($n=~/^(architecture)$/)
      {
        my $a=$c->{attrib}{name};
	print $ref "${tab}<$n name=\"",$c->{attrib}{name},"\">\n";
	&dumpXMLBuildFile($c,$ref,"$tab  ");
	print $ref "${tab}</$n>\n";
      }
      elsif($n=~/^(export)$/)
      {
        print $ref "${tab}<$n>\n";
	&dumpXMLBuildFile($c,$ref,"$tab  ");
	print $ref "${tab}</$n>\n";
      }
      elsif($n=~/^(include_path)$/)
      {
	 print $ref "${tab}<$n path=\"",$c->{attrib}{path},"\"/>\n";
      }
      elsif($n=~/^(makefile)$/)
      {
        print $ref "${tab}<$n>\n";
	if(!exists $data->{$n}){$data->{$n}=[];}
	foreach my $d (@{$c->{cdata}}){print $ref "$d\n";}
	print $ref "${tab}</$n>\n";
      }
    }
  }
  if (!ref($outfile)){close($ref);}
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
  if(($ccfiles>0) || ($isPackage))
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
	      if($v1=~/^\"(.*?)\"$/){print $outfile "$tab<flags ${f}=\"${n}=\\\"$1\\\"\"";}
	      else{print $outfile "$tab<flags ${f}=\"${n}=${v1}\"";}
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
    if(($ccfiles>0) && ($edmplugin==0))
    {
      print $outfile "  <lib name=\"1\"",&_xmlendtag($xml),">\n";
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
  foreach my $k (keys %{$data->{PROD_TYPE_SEARCH_RULES}})
  {if(!exists $data->{PROD_TYPE_SEARCH_RULES}{$k}{file}){$search{$k}=1;$hasfilter=1;}}
  if(!$hasfilter){return;}
  if($ofile eq ""){$ofile=&generatePreprocessedCXX($file,$data,$xflags);$delfile=1;}
  if($ofile eq ""){return;}
  if(!open(OFILE,"$ofile"))
  {
    print STDERR "Can not open file \"$ofile\" for reading.";
    if($delfile){my $d=dirname($ofile);system("rm -rf $d");}
    exit 0;
  }
  my $ref=0;
  if(ref($file) eq "ARRAY"){$ref=1;}
  while(my $line=<OFILE>)
  {
    chomp $line;
    if ($ref && ($line=~/^const char\* CreateBuildFileScriptVariable_$$\d+=\"([^"]+)";$/)){$file=$1;next;}
    foreach my $k (keys %search)
    {
      foreach my $f (keys %{$data->{PROD_TYPE_SEARCH_RULES}{$k}{filter}})
      {
        if($line=~/$f/)
        {
	  $data->{PROD_TYPE_SEARCH_RULES}{$k}{file}=$file;
	  delete $search{$k};
	  last;
        }
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
    my $cflags=$data->{compileflags}." ".$xflags;
    my $tmpdir=&getTmpDir();
    my $ofile="${tmpdir}/preprocessed.$$";
    my $fname="${ofile}.cc";
    my $xincs={};
    if (ref($file) eq "ARRAY")
    {
      foreach my $f (@$file){$xincs->{dirname($f)}=1;}
      $xincs=join(" -I",keys %$xincs);
      system("touch $fname; x=0; for f in ".join(" ",@$file)."; do echo \"const char* CreateBuildFileScriptVariable_$$\$x=\\\"\$f\\\";\" >> $fname; cat \$f >> $fname; x=`expr \$x + 1`; done");
    }
    else{system("cp $file $fname");}
    my @output=`$compilecmd -I$xincs $cflags -E -o $ofile $fname 2>&1`;
    my $err=$?;
    if ($err==0){return $ofile;}
    my %incs=();
    foreach my $l (@output)
    {
      chomp $l;
      print STDERR "$l\n";
      if ($l=~/:\s*([^\s:]+)\s*:\s*No such file or directory\s*$/i){$incs{$1}=1;}
    }
    if (scalar(keys %incs)>0)
    {
      my $iref;my $oref;
      if (open($iref,$fname))
      {
        if (open($oref,">${fname}.new"))
        {
          while(my $line=<$iref>)
	  {
	    chomp $line;
	    if ($line=~/^\s*#\s*include\s*(<|")([^>"]+)(>|")/)
	    {
	      if (exists $incs{$2})
	      {
	        $line="//$line";
		print STDERR "Commecting out: $2\n";
	      }
	    }
	    print $oref "$line\n";
	  }
          close($oref);
        }
        close($iref);
	if (-f "${fname}.new")
	{
	  system("cp ${fname}.new $fname");
	  if(system("$compilecmd -I$xincs $cflags -E -o $ofile $fname")==0){return $ofile;}
	}
      }
    }
    return $fname;
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
    print @lines;
    foreach my $line (@lines)
    {
      chomp $line;
      if($line=~/\/lib${l}\.so:\s+undefined reference to\s+/){return 1;}
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
###################################################
sub startTimer ()
{
  my $msg=shift;
  my $info=shift || 0;
  my $time=&getTime();
  my $id=0;
  while(exists $Cache->{TIMERS}{"${time}.$$.${id}"}){$id++;}
  $id="${time}.$$.${id}";
  $Cache->{TIMERS}{$id}{start}=$time;
  $Cache->{TIMERS}{$id}{info}=$info;
  if($info)
  {
    $Cache->{TIMERS}{$id}{msg}=$msg;
    print STDERR "TIMER STARTED($id):$msg\n";
  }
  return $id;
}

sub stopTimer()
{
  my $id=shift;
  my $time=undef;
  if (exists $Cache->{TIMERS}{$id})
  {
    $time=&getTime() - $Cache->{TIMERS}{$id}{start};
    if ($Cache->{TIMERS}{$id}{info})
    {
      my $msg=shift || $Cache->{TIMERS}{$id}{msg};
      print STDERR "TIMER STOPED ($id):$msg:$time\n";
    }
    delete $Cache->{TIMERS}{$id};
  }
  return $time;
}

sub timePassed ()
{
  my $id=shift;
  my $time=undef;
  if (exists $Cache->{TIMERS}{$id}){$time=&getTime() - $Cache->{TIMERS}{$id}{start};}
  return $time;
}

sub getTime ()
{
  if(!exists $Cache->{HiResLoaded})
  {
    eval "require Time::HiRes";
    if(!$@){$Cache->{HiResLoaded}=1;}
    else{$Cache->{HiResLoaded}=0;}
  }
  my $time=undef;
  if ($Cache->{HiResLoaded}){$time=Time::HiRes::gettimeofday();}
  else{$time=`date +\%s.\%N`; chomp $time;}
  return $time;
}
###################################################
sub toolSymbolCache ()
{
  my $cache=shift;
  my $tool=shift;
  if (!exists $cache->{SETUP}{$tool}){return;}
  my $dir=shift;
  my $jobs=shift || 1;
  print STDERR ".";
  if(exists $cache->{SETUP}{$tool}{LIB})
  {
    my $dirs=&searchBaseToolPaths($cache,$tool,"LIBDIR");
    foreach my $l (@{$cache->{SETUP}{$tool}{LIB}})
    {
      foreach my $d (@$dirs)
      {
        my $lib="${d}/lib${l}.so";
        if(!-f $lib){$lib="${d}/lib${l}.a";}
        if(-f $lib)
        {
	  &symbolCacheFork($lib,$tool,$dir,$jobs);
	  last;
        }
      }
    }
  }
  elsif($tool eq "cxxcompiler")
  {
    my $base=$cache->{SETUP}{$tool}{GCC_BASE} || $cache->{SETUP}{$tool}{CXXCOMPILER_BASE};
    if (($base ne "") && (-f "${base}/lib/libstdc++.so"))
    {
      &symbolCacheFork("${base}/lib/libstdc++.so","system",$dir,$jobs);
      foreach my $ldd (`ldd ${base}/lib/libstdc++.so`)
      {
        chomp $ldd;
	if ($ldd=~/\=\>\s+([^\s]+)\s+\(0x[0-9a-f]+\)\s*$/)
	{
	  $ldd=$1;
	  if (-f $ldd){&symbolCacheFork($ldd,"system",$dir,$jobs);}
	}
      }
    }
  }
}

sub scramToolSymbolCache ()
{
  my $cache=shift;
  my $tool=shift;
  if (!exists $cache->{SETUP}{$tool}){return;}
  my $dir=shift;
  my $jobs=shift || 1;
  my $libmap=shift;
  my $count=-1;
  foreach my $p (keys %$libmap)
  {
    my $l=$libmap->{$p};
    foreach my $d (@{$cache->{SETUP}{$tool}{LIBDIR}})
    {
      my $lib="${d}/lib${l}.so";
      if(!-f $lib){$lib="${d}/lib${l}.a";}
      if(-f $lib)
      {
	$count++;
	if (($count%100)==0){print STDERR ".";}
	&symbolCacheFork($lib,$p,$dir,$jobs);
	last;
      }
    }
  }
}

sub searchBaseToolPaths ()
{
  my $cache=shift;
  my $tool=shift;
  my $var=shift;
  my $paths=shift || [];
  if(exists $cache->{SETUP}{$tool})
  {
    my $c=$cache->{SETUP}{$tool};
    if(exists $c->{$var}){foreach my $d (@{$c->{$var}}){push @$paths,$d;}}
    if(exists $c->{USE}){foreach my $u (@{$c->{USE}}){&searchBaseToolPaths($cache,lc($u),$var,$paths);}}
    if (($var eq "LIBDIR") && (scalar(@$paths)==0) && ((!exists $c->{SCRAM_COMPILER}) || ($c->{SCRAM_COMPILER}==0)))
    {push @$paths,"/lib64","/usr/lib64","/lib","/usr/lib";}
  }
  return $paths;
}

sub symbolCacheFork ()
{
  my $lib=shift;
  my $t1=(stat($lib))[9];
  my $tool=shift;
  my $dir=shift;
  my $lname=basename($lib);
  my $pk=$tool;$pk=~s/\///g;
  my $cfile="${dir}/${lname}.${pk}";
  if ((stat($cfile)) && ((stat(_))[9] == $t1)){$Cache->{SYMBOL_CACHE_UPDATED}=$t1; return 0;}
  my $jobs=shift;
  if ($jobs > 1)
  {
    my $pid=&forkProcess($jobs);
    if($pid==0){&_symbolCache($lib,$cfile,$t1,$tool,$lname);exit 0;}
  }
  else{&_symbolCache($lib,$cfile,$t1,$tool,$lname);}
  $Cache->{SYMBOL_CACHE_UPDATED}=$t1;
  return 1;
}

sub _symbolCache ()
{
  my $lib=shift;
  my $cfile=shift;
  my $time=shift;
  my $tool=shift;
  my $lname=shift;
  my $shared="";
  if($lib=~/\.so$/){$shared="-D";}
  my $c={};
  my $fil="[A-Za-z]";
  foreach my $line (`nm $shared $lib`)
  {
    chomp $line;
    if($line=~/\s+($fil)\s+(.+)$/){$c->{$tool}{$lname}{$1}{$2}=1;}
  }
  &writeHashCache($c,$cfile);
  utime($time,$time,$cfile);
}

sub mergeSymbols ()
{
  my $dir=shift;
  my $file=shift || "";
  my $filter=shift || "T|R|V|W|B|D";
  my $cache={};
  if(-d $dir)
  {
    print STDERR "Merging symbols $dir ($filter) ....";
    my $ltime=0;
    if ($file ne "")
    {
      $ltime=$Cache->{SYMBOL_CACHE_UPDATED} || time;
      if ((stat($file)) && ((stat(_))[9] == $ltime)){print STDERR "\n";return &readHashCache($file);}
    }
    my $count=0;
    my $r;
    opendir($r,$dir) || die "Can not open directory for reading: $dir\n";
    foreach my $f (readdir($r))
    {
      if ($f=~/^\./){next;}
      $count++;
      if(($count%100)==0){print STDERR ".";}
      my $c=&readHashCache("${dir}/${f}");
      foreach my $t (keys %$c)
      {
        foreach my $l (keys %{$c->{$t}})
	{
          foreach my $x (keys %{$c->{$t}{$l}})
	  {
	    if ($x=~/^$filter$/)
	    {
	      foreach my $s (keys %{$c->{$t}{$l}{$x}}){$cache->{$s}{$t}{$l}=$x;}
	    }
	  }
	}
      }
    }
    if ($file ne "")
    {
      &writeHashCache($cache,"$file");
      utime($ltime,$ltime,$file);
    }
  }
  print STDERR "\n";
  return $cache;
}

sub cppFilt ()
{
  my $s=shift;
  if (exists $Cache->{CPPFLIT}{$s}){return $Cache->{CPPFLIT}{$s};}
  my $s1=`c++filt $s`; chomp $s1;
  $Cache->{CPPFLIT}{$s}=$s1;
  return $s1;
}

sub _symDir ()
{
  my $sym=shift;
  my $d="";
  my $c=1;
  while($sym=~s/^(.{$c})//){$d.="/$1";if($c<8){$c++;}}
  return "${d}/${sym}";
}
######################################################
sub forkProcess ()
{
  my $limit=shift || 1;
  &waitForChild($limit-1);
  my $pid=0;
  my $err=0;
  do
  {
    $pid = fork ();
    if (!defined $pid)
    {
      $err++;
       print STDERR "WARNING: Can not fork a new process:$err: $@\n";
       if ($err > 10 ){die "ERROR: Exiting due to fork() failure.\n";}
    }
  } while (!defined $pid);
  if ($pid>0){$Cache->{FORK}{pids}{$pid}=1;$Cache->{FORK}{running}=$Cache->{FORK}{running}+1;}
  return $pid;
}

sub waitForChild ()
{
  use POSIX ":sys_wait_h";
  my $limit=shift || 0;
  my $running=$Cache->{FORK}{running} || 0;
  while ($running>$limit)
  {
    my $pid=-1;
    do
    {
      $pid = waitpid(-1, WNOHANG);
      if (exists $Cache->{FORK}{pids}{$pid}) { $running--; delete $Cache->{FORK}{pids}{$pid};}
    } while ($pid > 0);
    if ($running>$limit){sleep 1;}
  }
  $Cache->{FORK}{running}=$running;
}

######################################################
sub makeRequest ()
{
  my $dir=shift;
  my $msg=shift;
  my $file=&getTmpFile($dir);
  unlink $file;
  &writeMsg("${file}.REQUEST",$msg);
  my $reply="";
  while (1)
  {
    if (!-f "${file}.REPLY.DONE"){next;}
    $reply=&readMsg("${file}.REPLY");
    last;
  }
  return $reply;
}

sub readRequests()
{
  my $dir=shift;
  my $req={};
  my $ref;
  opendir ($ref,$dir) || die "Can not open directory for reading: $dir\n";
  foreach my $f (readdir($ref)){if ($f=~/^((.+)\.REQUEST)\.DONE$/){$req->{"${dir}/${2}.REPLY"}=&readMsg("${dir}/${1}");}}
  closedir($ref);
  return $req;
}

sub readMsg ()
{
  my $file=shift;
  my $ref;
  open($ref,$file) || die "Can not open file for reading:$file\n";
  my $input=<$ref>; chomp $input;
  close($ref);
  unlink $file;
  unlink "$file.DONE";
  return $input;
}

sub writeMsg ()
{
  my $file=shift;
  my $msg=shift;
  my $ref;
  open($ref,">$file") || die "Can not open file for writing:$file\n";
  print $ref "$msg\n";
  close($ref);
  open($ref,">$file.DONE") || die "Can not open file for writing:$file.DONE\n";
  close($ref);
}

sub writeJson()
{
  my ($obj,$tab)=@_;
  my $str="";
  my $ref=ref($obj);
  my $indent=&_indent($tab);
  if ($ref eq "HASH")
  {
    $str="{";
    foreach my $k (sort keys %$obj){$str.="\n${indent}  \"$k\": ".&writeJson($obj->{$k},$tab+length($k)+6);}
    chomp($str);
    $str=~s/, *$//;
    $str.="\n${indent}},";
  }
  elsif($ref eq "ARRAY")
  {
    $str.="[";
    foreach my $i (@$obj){$str.="\n${indent}  ".&writeJson($i,$tab+2);chomp($str);}
    chomp($str);
    $str=~s/, *$//;
    $str.="\n${indent}],";
  }
  else{$str.="\"$obj\",";}
  return $str;
}

sub _indent()
{
  my $l=shift;
  my $s="";
  for(my $i=0;$i<$l;$i++){$s.=" ";}
  return $s;
}

1;
