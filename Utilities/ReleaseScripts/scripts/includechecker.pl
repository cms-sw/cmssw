#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use Getopt::Long;
use Digest::MD5  qw(md5_hex);
use SCRAMGenUtils;
$|=1;

my $INSTALL_PATH = dirname($0);
my $curdir=`/bin/pwd`; chomp $curdir;
my $tmp_inc="includechecker/src";
my $dummy_inc="dummy_include";
my $config_cache={};
my $cache_file="config_cache";
my %extra_dummy_file;
$extra_dummy_file{string}{"bits/stringfwd.h"}=1;
$extra_dummy_file{stdexcept}{"exception"}=1;
$config_cache->{COMPILER}="c++";
$config_cache->{COMPILER_FLAGS}[0]="";
$config_cache->{INCLUDE_FILTER}=".+";
$config_cache->{OWNHEADER}[0]='^(.*?)\\.(cc|CC|cpp|C|c|CPP|cxx|CXX)$:"${1}\\.(h|hh|hpp|H|HH|HPP)\\$"';
$config_cache->{FWDHEADER}{'^.*?(\\/|)[^\\/]*[Ff][Ww][Dd].h$'}=1;
$config_cache->{FWDHEADER}{'^iosfwd$'}=1;
$config_cache->{FWDHEADER}{'^bits\/stringfwd.h$'}=1;
$config_cache->{HEADER_EXT}{"\\.(h||hh|hpp|H|HH|HPP)\$"}=1;
$config_cache->{SOURCE_EXT}{"\\.(cc|CC|cpp|C|c|CPP|cxx|CXX)\$"}=1;
$config_cache->{SKIP_INCLUDE_INDIRECT_ADD}={};
$config_cache->{SKIP_INCLUDES}={};
$config_cache->{LOCAL_HEADERS}{'^(.+)/[^/]+$:"$1/.+"'}=1;
$SCRAMGenUtils::CacheType=1;

if(&GetOptions(
               "--config=s",\$config,
	       "--tmpdir=s",\$tmp_dir,
	       "--filter=s",\$filefilter,
	       "--redo=s",\$redo,
	       "--redoerr",\$redoerr,
	       "--keep",\$keep,
               "--detail",\$detail,
               "--recursive",\$recursive,
	       "--includeall",\$includeall,
               "--unique",\$unique,
	       "--sysheader",\$system_header_skip,
	       "--local",\$local_header_skip,
	       "--skipheaders",\$skipheaders,
               "--help",\$help,
              ) eq ""){print "ERROR: Wrong arguments.\n"; &usage_msg();}

if(defined $help){&usage_msg();}

if ((!defined $config) || ($config=~/^\s*$/) || (!-f $config))
{
  print "Error: Missing config file.\n";
  &usage_msg();
}

if(defined $unique){$unique=1;}
else{$unique=0;}

if(defined $redoerr){$redoerr=1;}
else{$redoerr=0;}

if(defined $system_header_skip){$system_header_skip=0;}
else{$system_header_skip=1;}

if(defined $local_header_skip){$local_header_skip=0;}
else{$local_header_skip=1;}

if(defined $includeall){$includeall=1;}
else{$includeall=0;}

if(defined $detail){$detail=1;}
else{$detail=0;}

if(defined $skipheaders){$skipheaders=1;}
else{$skipheaders=0;}

if(defined $recursive){$recursive=1;}
else{$recursive=0;}

if((!defined $filefilter) || ($filefilter=~/^\s*$/)){$filefilter="";}

if((!defined $redo) || ($redo=~/^\s*$/)){$redo="";}

if(!defined $keep){$keep=0;}
else{$keep=1;}

if ((!defined $tmp_dir) || ($tmp_dir=~/^\s*$/)) {$tmp_dir="/tmp/delete_me_includechecker_$$";}
else
{
  if($tmp_dir=~/^[^\/]/){$tmp_dir=&SCRAMGenUtils::fixPath("${curdir}/${tmp_dir}");}
  $keep=1;
}

system("mkdir -p ${tmp_dir}/backup/files ${tmp_dir}/backup/ids");
print "TMP directory:$tmp_dir\n";
&init ($config);
chdir($tmp_dir);
if($filefilter eq ""){$filefilter=".+";}
print "MSG: Skipping system headers check:$system_header_skip\n";

foreach my $f (sort(keys %{$config_cache->{FILES}}))
{
  &check_file($f);
  if (-f "${tmp_dir}/quit"){system("rm -f ${tmp_dir}/quit; touch ${tmp_dir}/chkquit");last;}
}
&final_exit();

sub find_file ()
{
  my $file=shift;
  foreach my $b (@{$config_cache->{BASE_DIR_ORDERED}})
  {if (-f "${b}/${file}"){return $b;}}
  return "";
}

sub check_includes ()
{
  my $srcfile=shift;
  my $cache=shift;
  my $origfile=$cache->{original};
  my $base_dir=$config_cache->{FILES}{$origfile}{BASE_DIR};
  my $origrel_dir=&SCRAMGenUtils::fixPath(dirname($origfile));
  my $orig_dir="${base_dir}/${origrel_dir}";
  my $filter=$config_cache->{INCLUDE_FILTER};
  &read_file ("${base_dir}/${origfile}", $cache);
  
  my $total_inc=scalar(@{$cache->{includes}});
  my $inc_added=0;
  my $actual_inc_added=0;
  my $inc_removed=0;

  my $skip=1;
  if(($total_inc+$cache->{incsafe}) < $cache->{code_lines}){$skip=&is_skipped($origfile);}
  $config_cache->{FILES}{$origfile}{INTERNAL_SKIP}=$skip;
  
  for(my $i=0; $i<$total_inc; $i++)
  {$config_cache->{FILES}{$origfile}{INCLUDES}{$cache->{includes}[$i]}=$cache->{includes_line}[$i];}
  
  my $inc_type="ALL_INCLUDES_REMOVED";
  my $skip_add=&match_data($origfile,"SKIP_AND_ADD_REMOVED_INCLUDES");
  my $skip_add_mod=0;
  if($includeall && !$skip_add){$inc_type="ALL_INCLUDES";}
  my $otype ="${inc_type}_ORDERED";
  $cache->{msglog}=[];
  for(my $i=0; $i<$total_inc; $i++)
  {
    my $inc_file=$cache->{includes}[$i];
    my $inc_line=$cache->{includes_line}[$i];
    my $b="";
    if (exists $config_cache->{FILES}{$inc_file})
    {$b=$config_cache->{FILES}{$inc_file}{BASE_DIR};}
    elsif(-f "${orig_dir}/${inc_file}")
    {
      $b=$base_dir;
      if($origrel_dir ne ".")
      {
        my $pinc=$inc_file;
	$inc_file=&SCRAMGenUtils::fixPath("${origrel_dir}/${pinc}");
	$inc_line=~s/$pinc/${inc_file}/;
	$cache->{includes}[$i]=$inc_file;
	$cache->{includes_line}[$i]=$inc_line;
	$config_cache->{FILES}{$origfile}{INCLUDES}{$inc_file}=$inc_line;
	delete $config_cache->{FILES}{$origfile}{INCLUDES}{$pinc};
	if($detail){push @{$cache->{msglog}},"MSG: Private header $pinc in $origfile file.($inc_file : $inc_line)\n";}
      }
    }
    else{$b=&find_file ($inc_file);}
    
    my $inc_skip = $skipheaders || &is_skipped($inc_file);
    if(!$inc_skip)
    {
      if($inc_file!~/$filter/){$inc_skip=1;}
      elsif(!$recursive && !exists $config_cache->{FILES}{$inc_file})
      {
        my $fd=dirname($inc_file);
	if($fd ne $origrel_dir){$inc_skip=1;}
      }
    }
    if ($b ne "")
    {
      if (!exists $config_cache->{FILES}{$inc_file})
      {
	if(($inc_skip==0) && &should_skip("${b}/${inc_file}")){$config_cache->{SKIP_FILES}{$inc_file}=1;$inc_skip=1;}
	$config_cache->{FILES}{$inc_file}{COMPILER_FLAGS_INDEX}=$config_cache->{FILES}{$origfile}{COMPILER_FLAGS_INDEX};
	$config_cache->{FILES}{$inc_file}{BASE_DIR}=$b;
      }
      $config_cache->{FILES}{$inc_file}{INTERNAL_SKIP}=$inc_skip;
      &check_file($inc_file);
      $inc_skip=$config_cache->{FILES}{$inc_file}{INTERNAL_SKIP};
      my $num=$cache->{includes_line_number}[$i];
      my $cur_total = scalar(@{$cache->{includes}});
      if($includeall)
      {
        foreach my $inc (@{$config_cache->{FILES}{$inc_file}{ALL_INCLUDES_ORDERED}})
        {
	  if(&is_skipped_inc_add($origfile,$inc)){next;}
	  my $l=$config_cache->{FILES}{$inc_file}{ALL_INCLUDES}{$inc};
	  if(!exists $config_cache->{FILES}{$origfile}{ALL_INCLUDES}{$inc})
	  {
	    push @{$config_cache->{FILES}{$origfile}{ALL_INCLUDES_ORDERED}}, $inc;
	    $config_cache->{FILES}{$origfile}{ALL_INCLUDES}{$inc}=$l;
	  }
        }
        if(!exists $config_cache->{FILES}{$origfile}{ALL_INCLUDES}{$inc_file})
        {
          push @{$config_cache->{FILES}{$origfile}{ALL_INCLUDES_ORDERED}}, $inc_file;
          $config_cache->{FILES}{$origfile}{ALL_INCLUDES}{$inc_file}=$inc_line;
        }
      }
      if($skip)
      {
        foreach my $inc (@{$config_cache->{FILES}{$inc_file}{ALL_INCLUDES_REMOVED_ORDERED}})
	{
	  if(&is_skipped_inc_add($origfile,$inc)){next;}
	  my $l=$config_cache->{FILES}{$inc_file}{ALL_INCLUDES_REMOVED}{$inc};
	  if(!exists $config_cache->{FILES}{$origfile}{ALL_INCLUDES_REMOVED}{$inc})
	  {
	    push @{$config_cache->{FILES}{$origfile}{ALL_INCLUDES_REMOVED_ORDERED}}, $inc;
	    $config_cache->{FILES}{$origfile}{ALL_INCLUDES_REMOVED}{$inc}=$l;
	  }
	}
	if(!$skip_add){next;}
      }
      foreach my $inc (@{$config_cache->{FILES}{$inc_file}{$otype}})
      {
	if(&is_skipped_inc_add($origfile,$inc,)){next;}
	my $l=$config_cache->{FILES}{$inc_file}{$inc_type}{$inc};
	if(exists $config_cache->{FILES}{$inc_file}{ALL_INCLUDES_REMOVED}{$inc})
	{$config_cache->{FILES}{$origfile}{INCLUDE_REMOVED_INDIRECT}{$inc}=1;}
	    
	if(exists $config_cache->{FILES}{$origfile}{INCLUDES}{$inc}){next;}
	for(my $j=$i+1; $j<$total_inc; $j++)
	{$cache->{includes_line_number}[$j]++;}
	push @{$cache->{includes}}, $inc;
	push @{$cache->{includes_line}}, $l;
	if ($inc_skip)
	{
	  push @{$cache->{includes_line_number}}, $num;
	  $cache->{includes_line_number}[$i]++;
	  &add_include ($srcfile, $num-1, $l);
	}
	else
	{
	  push @{$cache->{includes_line_number}}, $num+1;
	  &add_include ($srcfile, $num, $l);
	}
	if($detail){push @{$cache->{msglog}},"Added \"$inc\" in \"$origfile\". Removed/included in \"$inc_file\"\n";}
	$skip_add_mod=1;
	$num++;
	$cur_total++;
	$config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}{$inc}=1;
	$config_cache->{FILES}{$origfile}{INCLUDES}{$inc}=$l;
	$inc_added++;
      }
    }
    elsif($includeall && !exists $config_cache->{FILES}{$origfile}{ALL_INCLUDES}{$inc_file})
    {
      push @{$config_cache->{FILES}{$origfile}{ALL_INCLUDES_ORDERED}}, $inc_file;
      $config_cache->{FILES}{$origfile}{ALL_INCLUDES}{$inc_file}=$inc_line;
    }
  }
  if($detail){foreach my $f (@{$cache->{msglog}}){print "$f";}}
  delete $cache->{msglog};
  print "#################################################\n";
  print "File: $origfile\n";
  print "  Total lines            : ".$cache->{total_lines}."\n";
  print "  Code lines             : ".$cache->{code_lines}."\n";
  print "  Commented lines        : ".$cache->{comments_lines}."\n";
  print "  Empty lines            : ".$cache->{empty_lines}."\n";
  print "  Number of includes     : ".$total_inc."\n";
  print "  Additional Includes    : ".$inc_added."\n\n";
  $config_cache->{FILES}{$origfile}{ERROR}=0;
  if($skip)
  {
    if($detail){print "SKIPPED:$origfile\n";}
    if($skip_add && $skip_add_mod){&movenewfile($srcfile, $origfile,$orig_dir);}
    return;
  }
  
  my $stime=time;
  my $oflags=$config_cache->{COMPILER_FLAGS}[$config_cache->{FILES}{$origfile}{COMPILER_FLAGS_INDEX}];
  my $compiler=$config_cache->{COMPILER};
  my $xincshash={};
  my $error=0;
  my $flags="-shared -fsyntax-only -I${orig_dir} $oflags";
  my @origwarns=`$compiler -MMD $flags -o ${srcfile}.o $srcfile 2>&1`;
  if ($? != 0){$error=1;}
  elsif(-f "${srcfile}.d")
  {
    my $inc_cache=&read_mmd("${srcfile}.d");
    foreach my $inc (@$inc_cache)
    {
      if($inc=~/^\/.+$/)
      {
        my $d=dirname($inc);
        if(!exists $xincshash->{$d}){$xincshash->{$d}=1;}
      }
    }
  }
  my $origwarns_count=scalar(@origwarns);
  foreach my $warn (@origwarns)
  {chomp $warn;$warn=~s/$srcfile/${base_dir}\/${origfile}/g;}

  $total_inc=scalar(@{$cache->{includes}});
  if($detail && (exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}))
  {
    if($includeall)
    {print "Following files are added (because those were removed or indirectly added from included headers) ";}
    else{print "Following files are added (because those were removed from included headers)";}
    print "inorder to make the compilation work.\n";
    foreach my $f (keys %{$config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}})
    {print "  $f\n";}
    print "\n";
  }
  if ($error)
  {
    print "File should be compiled without errors.\n";
    print "Compilation errors are:\n";
    foreach my $w (@origwarns){print "$w\n";}
    print "\nCompilation might be failed due to the fact that this file\n";
    print "is not self parsed";
    if (exists $config_cache->{FILES}{$origfile}{INCLUDE_REMOVED_INDIRECT})
    {
      print " OR due to removal of some header from other files.\n";
      print "Header files removed from other headers are:\n";
      foreach my $f (keys %{$config_cache->{FILES}{$origfile}{INCLUDE_REMOVED_INDIRECT}})
      {print "  $f\n";}
      if (exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT})
      {
        print "\nBut following headers are included again in $origfile:\n";
	foreach my $f (keys %{$config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}})
	{print "  $f\n";}
	print "It might be possible that header file include order is causing the problem.\n";
	print "Or there is really an error in your $origfile file (might be not self parsed).\n";
      }
    }
    else{print ".\n";}
    delete $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT};
    delete $config_cache->{FILES}{$origfile}{INCLUDE_REMOVED_INDIRECT};
    $config_cache->{FILES}{$origfile}{ERROR}=1;
    return;
  }
  if(-f "${srcfile}.o"){system("rm -f ${srcfile}.o");}
  delete $config_cache->{FILES}{$origfile}{INCLUDE_REMOVED_INDIRECT};

  my $own_header="";
  if (!$cache->{isheader})
  {
    foreach my $exp (@{$config_cache->{OWNHEADER}})
    {
      my ($fil,$fil2)=split /:/,$exp;
      my $h=$origfile;
      if($h=~/$fil/)
      {$h=eval $fil2;}
      for(my $i=0; $i < $total_inc; $i++)
      {
        my $f=$cache->{includes}[$i];
	if ($f=~/$h/)
	{$own_header=$f;last;}
      }
      if($own_header){print "OWNHEADER: $origfile => $own_header\n";last;}
    }
  }
  if($detail && $origwarns_count>0)
  {
    print "---------------- $origfile: ORIGINAL WARNINGS -------------------\n";
    foreach my $w (@origwarns){print "$w\n";}
    print "---------------- $origfile: ORIGINAL WARNINGS : DONE ------------\n";
  }
  
  my $num = -1;
  my $fwdcheck=0; 
  `$compiler -M $flags -o ${srcfile}.d $srcfile 2>&1`;
  my $origincpath=&find_incs_in_deps("${srcfile}.d",$config_cache->{FILES}{$origfile}{INCLUDES});
  my $minc={};
  while(1)
  {
    my $i=&find_next_index ($cache, $num, $fwdcheck);
    if ($i==-1)
    {
      if($fwdcheck==1){last;}
      else{$fwdcheck=1;$num=-1;next;}
    }
    $num = $cache->{includes_line_number}[$i];
    my $inc_file=$cache->{includes}[$i];
    my $inc_line=$cache->{includes_line}[$i];
    
    if ($inc_file eq $own_header)
    {if($detail){print "  Skipped checking of \"$inc_file\" (Assumption: .cc always needs its own .h)\n";} next;}
    if ((!exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}) ||
        (!exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}{$inc_file}))
    {
      if (exists $config_cache->{SKIP_ORIG_INCLUDES})
      {
        my $next=0;
        foreach my $exp (keys %{$config_cache->{SKIP_ORIG_INCLUDES}})
        {
          if($origfile=~/$exp/)
	  {
	    if($detail){print "  Skipped checking of \"$inc_file\" in \"$origfile\" due to SKIP_ORIG_INCLUDES flag in the config file\n";}
	    $next=1;
	    last;
	  }
        }
        if($next){next;}
      }
      if (&skip_self_include($origfile,$inc_file))
      {if($detail){print "  Skipped checking of \"$inc_file\" in \"$origfile\" due to SKIP_SELF_INCLUDES flag in the config file\n";} next;}
    }
    if(&skip_include($origfile,$inc_file))
    {if($detail){print "  Skipped checking of \"$inc_file\" in \"$origfile\" due to SKIP_INCLUDES flag in the config file\n";} next;}
    
    my $force_inc_remove=0;
    my $exists_in_own_header=0;
    my $is_sys_header=0;
    my $inc_fpath="";
    if (($own_header ne "") && 
        (exists $config_cache->{FILES}{$own_header}) &&
	(exists $config_cache->{FILES}{$own_header}{INCLUDES}{$inc_file}))
    {$exists_in_own_header=1;$force_inc_remove=1;}
    else
    {
      $inc_fpath=$origincpath->{$inc_file};
      if(!$inc_fpath){print "ERROR: Could not find full include path for \"$inc_file\" from \"$origfile\"\n";}
      $is_sys_header=&is_system_header($inc_fpath,$xincshash);
      if ($is_sys_header)
      {
        if($system_header_skip)
        {
	  if(!exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT} ||
	     !exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}{$inc_file})
	  {if($detail){print "  Skipped checking of \"$inc_file\" in \"$origfile\" due to SYSTEM HEADERS\n";} next;}
	  else{$force_inc_remove=1; print "  FORCED REMOVED(System header indirectly added):$origfile:$inc_file\n";}
        }
      }
      elsif ($local_header_skip)
      {
	if(&is_local_header($origfile,$inc_file))
	{
	  if(!exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT} ||
	     !exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}{$inc_file})
	  {if($detail){print "  Skipped checking of \"$inc_file\" in \"$origfile\" due to LOCAL HEADERS SKIP\n";} next;}
	}
      }
    }
    
    &comment_line ($srcfile, $num);
    if (exists $minc->{$inc_file})
    {
      print "  Skipped checking of \"$inc_file\" in \"$origfile\" due to MULTIPLE INCLUDE STATEMENTS\n";
      $force_inc_remove=1;
    }
    else{$minc->{$inc_file}=1;}
    my $loop=2;
    if($force_inc_remove){$loop=0;}
    elsif($unique){$loop=1;}
    my @warns=();
    my $inc_req=0;
    my $process_flag=0;
    for(my $x=0;$x<$loop;$x++)
    {
      @warns=();
      system("rm -f ${srcfile}.o ${srcfile}.d");
      $flags="-MD -shared -fsyntax-only -I${tmp_dir}/${dummy_inc} -I${orig_dir} $oflags";
      @warns=`$compiler $flags -o ${srcfile}.o $srcfile 2>&1`;
      my $ret_val=$?;
      if($x==1)
      {
	my $c=&find_inc_in_deps("${srcfile}.d",$inc_file);
	if ($detail){print "INCLUDE PICK UP $x: $inc_file:",join(",",keys %$c),"\n";}
      }
      foreach my $w (@warns)
      {chomp $w;$w=~s/$srcfile/${base_dir}\/${origfile}/;}
      my $nwcount=scalar(@warns);
      if($detail)
      {
        print "---------- $origfile : ACTUAL WARN/ERRORS AFTER REMOVING $inc_file (iteration: $x) ----------\n";
        foreach my $w (@warns){print "$w\n";}
        print "---------- $origfile : ACTUAL WARN/ERRORS AFTER REMOVING $inc_file (iteration: $x) DONE -----\n";
      }
      if ($ret_val != 0)
      {
	if($x==0){$inc_req=1;}
	else
	{
	  my $sf="${base_dir}/${origfile}";
	  foreach my $w (@warns)
	  {
	    foreach my $ow (@origwarns){if("$ow" eq "$w"){$w="";$ow="";last;}}
	    if($w)
	    {
	      if($w=~/^\s*$sf:\d+:\s*/)
	      {
	        if($w!~/\s+instantiated from here\s*$/)
	        {$inc_req=1; last;}
	      }
	      elsif($w=~/^.+?:\d+:\s+confused by earlier errors/)
	      {$inc_req=1;last;}
	    }
	  }
	}
      }
      elsif ($origwarns_count == scalar(@warns))
      {
        for(my $j=0; $j<$origwarns_count; $j++)
        {if ("$warns[$j]" ne "$origwarns[$j]"){$inc_req=1;last;}}
      }
      else{$inc_req=1;}
      if($inc_req || $unique){last;}
      elsif($x==0)
      {
	my $c=&find_inc_in_deps("${srcfile}.d",$inc_file);
	my $icount=scalar(keys %$c);
        if($icount==0){last;}
	if ($detail){print "INCLUDE PICK UP $x: $inc_file:",join(",",keys %$c),"\n";}
	if ($is_sys_header){&createdummyfile($config_cache->{FILES}{$origfile},$inc_file);}
	else{&createbackup($inc_fpath);}
      }
    }
    if($is_sys_header){&cleanupdummydir();}
    else{&recoverbackup($inc_fpath);}
    
    if($inc_req==0)
    {
      if($unique){delete $minc->{$inc_file};}
      system("rm -f ${srcfile}.backup");
      $inc_removed++;
      if($detail || !exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT} ||
         !exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}{$inc_file})
      {
        print "  Checking \"$inc_file\" at line number \"$num\" ..... [ Not required ";
	if ($exists_in_own_header)
	{print "(Exists in its own header) ]\n";}
	else{print "]\n";}
      }
      
      if (!exists $config_cache->{FILES}{$origfile}{ALL_INCLUDES_REMOVED}{$inc_file})
      {
	push @{$config_cache->{FILES}{$origfile}{ALL_INCLUDES_REMOVED_ORDERED}}, $inc_file;
	$config_cache->{FILES}{$origfile}{ALL_INCLUDES_REMOVED}{$inc_file}=$inc_line;
      }
      delete $config_cache->{FILES}{$origfile}{INCLUDES}{$inc_file};
    }
    else
    {
      system("mv ${srcfile}.backup $srcfile");
      my $extra=0;
      if (exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT})
      {
        if(exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}{$inc_file})
	{$extra=1;}
      }
      if ($extra)
      {print "  Checking \"$inc_file\" at line number \"$num\" ..... [ Required (ADD) ]\n";$actual_inc_added++;}
      elsif($detail)
      {print "  Checking \"$inc_file\" at line number \"$num\" ..... [ Required ]\n";}
    }
    system("rm -f ${srcfile}.o ${srcfile}.d");
  }
  system("grep -v \"//INCLUDECHECKER: Removed this line:.*//INCLUDECHECKER: Added this line\" $srcfile > ${srcfile}.new");
  my $diff=`diff ${srcfile}.new ${base_dir}/${origfile}`; chomp $diff;
  if ($diff ne ""){&movenewfile("${srcfile}.new", $origfile,$orig_dir);}
  print "\n";
  print "  Include added          : ".$inc_added."\n";
  print "  Include removed        : ".$inc_removed."\n";
  print "  Actual include added   : ".$actual_inc_added."\n";
  print "  Actual include removed : ",$inc_removed-$inc_added+$actual_inc_added,"\n";
  my $dtime=time-$stime;
  print "  Processing time(sec)   : ".$dtime."\n";
  delete $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT};
}

sub movenewfile ()
{
  my $nfile=shift;
  my $file=shift;
  my $odir=shift;
  my $dir=dirname($file);
  my $name=basename($file);
  $dir="${tmp_dir}/${tmp_inc}/${dir}";
  system("mkdir -p $dir; rm -f ${dir}/${name}; cp $nfile ${dir}/${name}");
  if(!-f "${odir}/${name}.original_file_wo_incchk_changes"){system("cp ${odir}/${name} ${odir}/${name}.original_file_wo_incchk_changes");}
  system("mv $nfile ${odir}/${name}");
}

sub  islocalinc ()
{
  my $cache=shift;
  my $inc=shift;
  if(!exists $cache->{INCLUDE_ADDED_INDIRECT} ||
     !exists $cache->{INCLUDE_ADDED_INDIRECT}{$inc})
  {return 1;}
  return 0;
}

sub createdummyfile ()
{
  my $cache=shift;
  my $file=shift;
  my $extra_only=shift || 0;
  if(!$extra_only)
  {&createdummyfile1 ($file);}
  if(exists $extra_dummy_file{$file})
  {
    foreach my $f (keys %{$extra_dummy_file{$file}})
    {
      if(!exists $cache->{INCLUDES}{$f})
      {&createdummyfile ($cache,$f);}
    }
  }
}

sub createbackup()
{
  my $file=shift;
  my $id=&md5_hex ($file);
  if (-f "${tmp_dir}/backup/files/${id}"){return;}
  system("echo \"$file\" > ${tmp_dir}/backup/ids/$id && mv $file ${tmp_dir}/backup/files/${id} && touch $file");
}

sub recoverbackup ()
{
  my $file=shift;
  my $id=shift || &md5_hex ($file);
  if (-f "${tmp_dir}/backup/files/${id}")
  {system("mv ${tmp_dir}/backup/files/${id} $file");}
  system("rm -f ${tmp_dir}/backup/ids/$id ${tmp_dir}/backup/files/${id}");
}

sub createdummyfile1 ()
{
  my $file=shift;
  my $dir="${tmp_dir}/${dummy_inc}/".dirname($file);
  my $file1=basename($file);
  system("mkdir -p $dir; touch ${dir}/${file1}");
}

sub cleanupdummydir()
{
  system("rm -rf ${tmp_dir}/${dummy_inc}/* 2>&1 > /dev/null");
}

sub is_system_header ()
{
  my $file=shift;
  my $cache=shift;
  if($file=~/^${tmp_dir}\/${tmp_inc}\/.+/){return 0;}
  foreach my $d (@{$config_cache->{BASE_DIR_ORDERED}})
  {if($file=~/^$d\/.+/){return 0;}}
  foreach my $d (keys %$cache)
  {
    my $d1=$d;
    $d1=~s/\+/\\\+/g;
    if($file=~/^$d1\/.+/){return 0;}
  }
  return 1;
}

sub find_inc_in_deps ()
{
  my $file=shift;
  my $inc=shift;
  my $paths={};
  if(!-f "$file"){return $paths;}
  $inc=~s/\+/\\\+/g;
  my $lines=&read_mmd($file);
  foreach my $line (@$lines)
  {if($line=~/^(.+?\/$inc)$/){$paths->{$1}=1;}}
  return $paths;
}

sub find_incs_in_deps ()
{
  my $file=shift;
  my $cache=shift;
  my $paths={};
  if(!-f "$file"){return $paths;}
  my %incs=();
  my $hasinc=0;
  foreach my $inc (keys %$cache)
  {
    my $xinc=$inc;
    $xinc=~s/\+/\\\+/g;
    $incs{$xinc}=$inc;
    $hasinc++;
  }
  if(!$hasinc){return $paths;}
  my $lines=&read_mmd($file);
  foreach my $line (@$lines)
  {
    foreach my $xinc (keys %incs)
    {
      if($line=~/^\s*(.+?\/$xinc)$/)
      {$paths->{$incs{$xinc}}=$line;delete $incs{$xinc};$hasinc--;last;}
    }
    if(!$hasinc){return $paths;}
  }
  if($hasinc)
  {
    print "MSG: Could not find the following includes:\n";
    foreach my $inc (keys %incs){print "  ",$incs{$inc},"\n";}
  }
  return $paths;
}

sub find_next_index ()
{
  my $cache=shift;
  my $prev=shift;
  my $fwdcheck=shift || 0;
  if(!defined $prev){$prev=-1;}
  my $count=scalar(@{$cache->{includes_line_number}});
  my $next=345435434534;
  my $index=-1;
  for(my $i=0; $i<$count;$i++)
  {
    my $inc=$cache->{includes}[$i];
    my $isfwd=0;
    foreach my $fwd (keys %{$config_cache->{FWDHEADER}})
    {if($inc=~/$fwd/){$isfwd=1; last;}}
    if($fwdcheck==1)
    {if(!$isfwd){next;}}
    elsif($isfwd){next;}
    my $j=$cache->{includes_line_number}[$i];
    if (($j>$prev) && ($j<$next)){$next=$j;$index=$i;}
  }
  return $index;
}

sub read_mmd()
{
  my $file=shift;
  my $cache=[];
  if(!open(MMDFILE,$file)){die "Can not open \"$file\" for reading.";}
  while(my $line=<MMDFILE>)
  {
    chomp $line;
    $line=~s/\s*(\\|)$//;$xline=~s/^\s*//;
    if($line=~/:$/){next;}
    foreach my $l (split /\s+/,$line)
    {push @$cache,$l;}
  }
  close(MMDFILE);
  return $cache;
}

sub check_cyclic ()
{
  my $file=shift;
  my $key=shift || "INC_LIST";
  my $msg="";
  my $depth = scalar(@{$config_cache->{$key}});
  if ($depth > 0)
  {
    for(my $i=0; $i<$depth; $i++)
    {
      if($config_cache->{$key}[$i] eq $file)
      {
        my $msg="";
	for(my $j=$i; $j<$depth;$j++)
	{$msg.=$config_cache->{$key}[$j]." -> ";}
	$msg.=$file;
      }
    }
  }
  return $msg;
}

sub should_skip ()
{
  my $file=shift;
  if(-f "$file")
  {
    foreach my $line (`cat $file`)
    {
     chomp $line;
     if($line=~/^\s*\#\s*define\s+.+?\\$/){return 1;}
     if($line=~/^\s*template\s*<.+/){return 1;}
   }
 }
 return 0;
}

sub check_file ()
{
  my $file=shift;
  my $depth = scalar(@{$config_cache->{INC_LIST}});
  my $cymsg=&check_cyclic($file);
  if($cymsg){print "WARNING: Cyclic includes:\n"; print "$cymsg\n";}
  if (exists $config_cache->{FILES}{$file}{DONE}){return;}
  $depth++;
  $config_cache->{FILES}{$file}{DONE}=1;
  $config_cache->{REDONE}{$file}=1;
  delete $config_cache->{FILES}{$file}{ERROR};
  delete $config_cache->{FILES}{$file}{FINAL_DONE};
  push @{$config_cache->{INC_LIST}}, $file;
  
  my $base=$config_cache->{FILES}{$file}{BASE_DIR};
  my $check=1;
  my $tmpfile="";
  my %cache=();

  my $dir="${tmp_dir}/tmp_${depth}";
  system("mkdir -p $dir");
  
  $cache{$tmpfile}{isheader}=-1;
  foreach my $ext (keys %{$config_cache->{HEADER_EXT}})
  {
    if(($ext!~/^\s*$/) && ($file=~/$ext/))
    {
      $tmpfile=rearrangePath ("${dir}/".basename($file).".cc");
      $cache{$tmpfile}{isheader}=1;
      if ($skipheaders){$config_cache->{FILES}{$file}{INTERNAL_SKIP}=1;}
    }
  }
  if($cache{$tmpfile}{isheader} == -1)
  {
    foreach my $ext (keys %{$config_cache->{SOURCE_EXT}})
    {
      if(($ext!~/^\s*$/) && ($file=~/$ext/))
      {
        $tmpfile=rearrangePath ("${dir}/".basename($file));
        $cache{$tmpfile}{isheader}=0;
      }
    }
  }
  if($cache{$tmpfile}{isheader} == -1)
  {
    print "WARNING: $file does match any of the following src/header extension regexp.\n";
    foreach my $ext (keys %{$config_cache->{HEADER_EXT}},keys %{$config_cache->{SOURCE_EXT}})
    {if($ext!~/^\s*$/){print "  $ext\n";}}
    $config_cache->{FILES}{$file}{FINAL_DONE}=1;
  }
  $cache{$tmpfile}{incsafe}=0;
  
  if ($tmpfile ne "")
  {
     system("cp -pf ${base}/${file} $tmpfile; chmod u+w $tmpfile");
     $cache{$tmpfile}{original}=$file;
     &check_includes ($tmpfile, $cache{$tmpfile});
     if($config_cache->{FILES}{$file}{ERROR} == 0)
     {$config_cache->{FILES}{$file}{FINAL_DONE}=1;}
     if($keep)
     {
       my $d=dirname("${tmp_dir}/cache/files/$file");
       if(!-d $d){system("mkdir -p $d");}
       &SCRAMGenUtils::writeHashCache($config_cache->{FILES}{$file}, "${tmp_dir}/cache/files/$file");
     }
  }
  system("rm -rf $dir");
  pop @{$config_cache->{INC_LIST}};
}

sub comment_line ()
{
  my $file=shift;
  my $line_number=shift;
  my $nfile="${file}.new_$$";
  if (!open (OLDFILE, "$file")){die "Can not open file $file for reading(FROM: comment_line:1)\n";}
  if (!open (NEWFILE, ">$nfile")){die "Can not open file $nfile for writing(FROM: comment_line:2)\n";}
  my $l=0;
  while(my $line=<OLDFILE>)
  {
    $l++;
    if ($l == $line_number)
    {print NEWFILE "//INCLUDECHECKER: Removed this line: $line";}
    else{print NEWFILE "$line";}
  }
  close(OLDFILE);
  close(NEWFILE);
  system("mv $file ${file}.backup");
  system("mv $nfile $file");
}

sub add_include ()
{
  my $file=shift;
  my $line_number=shift;
  my $add=shift;
  if($add!~/^.+?\s+\/\/INCLUDECHECKER: Added this line\s*$/)
  {$add="$add //INCLUDECHECKER: Added this line";}
  my $nfile="${file}.new_$$";
  if (!open (OLDFILE, "$file")){die "Can not open file $file for reading (From: add_include1)\n";}
  if (!open (NEWFILE, ">$nfile")){die "Can not open file $nfile for writing (From: add_include1)\n";}
  my $l=0;
  my $printed=0;
  while(my $line=<OLDFILE>)
  {
    if ($l == $line_number)
    {print NEWFILE "$add\n";$printed=1;}
    print NEWFILE "$line";
    $l++;
  }
  if (!$printed)
  {print NEWFILE "$add\n";$printed=1;}
  close(OLDFILE);
  close(NEWFILE);
  system("mv $nfile $file");
}

sub read_file ()
{
  my $file=shift;
  my $cache=shift;
  my $origfile=$cache->{original};
  
  $config_cache->{FILES}{$origfile}{INCLUDES}={};
  $config_cache->{FILES}{$origfile}{ALL_INCLUDES_REMOVED}={};
  $config_cache->{FILES}{$origfile}{ALL_INCLUDES_REMOVED_ORDERED}=[];
  delete $config_cache->{FILES}{$origfile}{INCLUDE_REMOVED_INDIRECT};
  delete $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT};
  delete $config_cache->{FILES}{$origfile}{ALL_INCLUDES};
  delete $config_cache->{FILES}{$origfile}{ALL_INCLUDES_ORDERED};
  if($includeall)
  {
    $config_cache->{FILES}{$origfile}{ALL_INCLUDES}={};
    $config_cache->{FILES}{$origfile}{ALL_INCLUDES_ORDERED}=[];
  }
  
  &SCRAMGenUtils::readCXXFile($file,$cache);
  if(!exists $cache->{lines}){return;}
  
  $cache->{includes}=[];
  $cache->{includes_line_number}=[];
  $cache->{incsafe} = 0;

  my $total_lines=scalar(@{$cache->{lines}});
  my $first_ifndef=0;
  my $extern=0;
  for(my $i=0;$i<$total_lines;$i++)
  {
    my $line=$cache->{lines}[$i];
    if(($i==0) && ($line=~/^\s*#\s*ifndef\s+([^\s]+)\s*$/))
    {
      my $def=$1;
      my $nline=$cache->{lines}[$i+1];
      if($nline=~/^\s*#\s*define\s+$def\s*/){$i++;$first_ifndef=1;$cache->{incsafe}=3;next;}
    }
    if (!$first_ifndef && $cache->{isheader} && ($line=~/^\s*#\s*ifndef\s+/)){$first_ifndef=1; $cache->{incsafe}=3; next;}
    if($line=~/^\s*#\s*if(n|\s+|)def(ined|\s+|)/)
    {$i=&SCRAMGenUtils::skipIfDirectiveCXX ($cache->{lines}, $i+1, $total_lines);next;}
    
    while($line=~/\\\//){$line=~s/\\\//\//;}
    if($line=~/^\s*extern\s+\"C(\+\+|)\"\s+\{\s*$/){$extern=1;next;}
    if($extern && $line=~/^\s*\}.*/){$extern=0; next;}
    if ($line=~/^\s*#\s*include\s*([\"<](.+?)[\">])\s*(.*)$/)
    {
      my $inc_file=$2;
      my $comment=$3;
      my $num=$cache->{line_numbers}[$i];
      if($inc_file!~/^\./)
      {
        my $inc_filex=&SCRAMGenUtils::fixPath($inc_file);
	if($inc_file ne $inc_filex)
	{
	  $line=~s/$inc_file/$inc_filex/;
	  print "WARNING: Please fix the include statement on line NO. $num of file \"$origfile\"\n";
	  print "         by replacing \"$inc_file\" with \"$inc_file1\".\n";
	  $inc_file=$inc_file1;
	}
      }
      else{print "WARNING: Include statement \"$inc_file\" on line NO. $num of file \"$origfile\" has a relative path. Please fix this.\n";}
      if($comment=~/\/\/\s*INCLUDECHECKER\s*:\s*SKIP/i){print "  Skipped checking of \"$inc_file\" in \"$origfile\"(Developer forced skipping)\n";next;}
      if($extern){print "  Skipped checking of \"$inc_file\" in \"$origfile\"(\"extern\" section include)\n";next;}
      push @{$cache->{includes}}, $inc_file;
      push @{$cache->{includes_line_number}}, $num;
      push @{$cache->{includes_line}}, $line;
    }
  }
}

sub updateFromCachedFiles ()
{
  my $dir=shift || "${tmp_dir}/cache/files";
  my $bdir=shift || $dir;
  foreach my $f (&SCRAMGenUtils::readDir($dir))
  {
    my $fp="${dir}/${f}";
    if(-d $fp){&updateFromCachedFiles($fp,$bdir);}
    elsif(-f $fp)
    {
      my $file=$fp;
      $file=~s/^$bdir\///;
      $config_cache->{FILES}{$file}=&SCRAMGenUtils::readHashCache($fp);
    }
  }
}

sub resotre_backup ()
{
  my $ids="${tmp_dir}/backup/ids";
  foreach my $id (&SCRAMGenUtils::readDir($ids,2))
  {
    my $ref;
    if(!open($ref,"${ids}/${id}")){die "Can not open file for reading: ${ids}/${id}";}
    my $file=<$ref>; chomp $file;
    close($ref);
    &recoverbackup($file,$id);
    print "Recovered backup: $file\n";
  }
}

sub init ()
{
  my $config=shift;
  if ("$tmp_dir" ne "/")
  {
    &resotre_backup ();
    system("mkdir -p ${tmp_dir}/${tmp_inc} ${tmp_dir}/${dummy_inc} ${tmp_dir}/cache");
    system("/bin/rm -rf ${tmp_dir}/tmp_* 2>&1");
    system("/bin/rm -rf ${tmp_dir}/${dummy_inc}/* 2>&1");
    if (-f "${tmp_dir}/cache/${cache_file}")
    {
      $config_cache=&SCRAMGenUtils::readHashCache("${tmp_dir}/cache/${cache_file}");
      $config_cache->{REDONE}={};
      if(-d "${tmp_dir}/cache/files"){&updateFromCachedFiles("${tmp_dir}/cache/files");}
      foreach my $f (keys %{$config_cache->{FILES}})
      {
	if($redo && ($f=~/$redo/)){delete $config_cache->{FILES}{$f}{DONE};delete $config_cache->{FILES}{$f}{INTERNAL_SKIP};next;}
        if(exists $config_cache->{FILES}{$f}{DONE})
        {
          if(($config_cache->{FILES}{$f}{FINAL_DONE}==1) || 
	      (($config_cache->{FILES}{$f}{ERROR}==1) && ($redoerr==0))){$config_cache->{FILES}{$f}{DONE}=1;}
	  else{delete $config_cache->{FILES}{$f}{DONE};}
        }
      }
      
      if(exists $config_cache->{INCLUDEALL})
      {
        if($config_cache->{INCLUDEALL} != $includeall)
	{
	  my $msg="with";
	  if($includeall){$msg="without";}
	  print "WARNING: Previously you had run includechecker.pl ",$msg," --includeall command-line option.\n";
	  print "WARNING: Using the previous value of includeall\n";
	  $includeall=$config_cache->{INCLUDEALL};
	}
      }
      else{$config_cache->{INCLUDEALL}=$includeall;}
      if(exists $config_cache->{SYSTEM_HEADER_SKIP})
      {
        if($config_cache->{SYSTEM_HEADER_SKIP} != $system_header_skip)
	{
	  my $msg="without";
	  if($system_header_skip){$msg="with";}
	  print "WARNING: Previously you had run includechecker.pl ",$msg," --sysheader command-line option.\n";
	  print "WARNING: Using the previous value of sysheader\n";
	  $system_header_skip=$config_cache->{SYSTEM_HEADER_SKIP};
	}
      }
      else{$config_cache->{SYSTEM_HEADER_SKIP}=$system_header_skip;}
      if(exists $config_cache->{SKIP_HEADERS})
      {
        if($config_cache->{SKIP_HEADERS} != $skipheaders)
	{
	  my $msg="without";
	  if($skipheaders){$msg="with";}
	  print "WARNING: Previously you had run includechecker.pl ",$msg," --skipheaders command-line option.\n";
	  print "WARNING: Using the previous value of sysheader\n";
	  $skipheaders=$config_cache->{SKIP_HEADERS};
	}
      }
      else{$config_cache->{SKIP_HEADERS}=$skipheaders;}
      if(exists $config_cache->{LOCAL_HEADER_SKIP})
      {
        if($config_cache->{LOCAL_HEADER_SKIP} != $local_header_skip)
	{
	  my $msg="without";
	  if($local_header_skip){$msg="with";}
	  print "WARNING: Previously you had run includechecker.pl ",$msg," --local command-line option.\n";
	  print "WARNING: Using the previous value of local\n";
	  $local_header_skip=$config_cache->{LOCAL_HEADER_SKIP};
	}
      }
      else{$config_cache->{LOCAL_HEADER_SKIP}=$local_header_skip;}
      my $pfil=$config_cache->{FILEFILTER};
      if(($pfil ne $filefilter) && ($filefilter ne ""))
      {
        print "WARNING: You have tried to change the file filter used for previous run. Script is going to use the previous value.\n";
	print "         New filter:\"$filefilter\"\n";
	print "         Old filter:\"$pfil\"\n";
      }
      $filefilter=$pfil;
      delete $config_cache->{INC_LIST};
      if (!-f "${tmp_dir}/cache/${cache_file}.append"){&SCRAMGenUtils::writeHashCache($config_cache, "${tmp_dir}/cache/${cache_file}");return;}
      else{system("rm -f ${tmp_dir}/cache/${cache_file}.append");}
    }
  }
  &read_config ($config);
  $config_cache->{INCLUDEALL}=$includeall;
  $config_cache->{FILEFILTER}=$filefilter;
  $config_cache->{SYSTEM_HEADER_SKIP}=$system_header_skip;
  $config_cache->{LOCAL_HEADER_SKIP}=$local_header_skip;
  $config_cache->{SKIP_HEADERS}=$skipheaders;
  if($keep)
  {
    &SCRAMGenUtils::writeHashCache($config_cache, "${tmp_dir}/cache/${cache_file}");
    #&init($config);
  }
}

sub final_exit ()
{
  my $code=shift || 0;
  if ("$tmp_dir" ne "/")
  {
    if ($keep)
    {
      system("/bin/rm -rf ${tmp_dir}/tmp_* 2>&1");
      print "Tmp directory to reuse: $tmp_dir\n";
    }
    else{system("/bin/rm -rf $tmp_dir");}
  }
  exit $code;
}

sub rearrangePath ()
{
  my $d=shift || return "";
  if($d=~/^[^\/]/){$d="${curdir}/${d}";}
  return &SCRAMGenUtils::fixPath ($d);
}

sub read_config ()
{
  my $file=shift;
  my $flag_index=scalar(@{$config_cache->{COMPILER_FLAGS}})-1;
  foreach my $line (`cat $file`)
  {
    chomp $line;
    if($line=~/^\s*#/){next;}
    if($line=~/^\s*$/){next;}
    if($line=~/^\s*COMPILER\s*=\s*(.+)$/)
    {
      $config_cache->{COMPILER}=$1;
    }
    elsif($line=~/^\s*COMPILER_FLAGS\s*=\s*(.+)$/)
    {
      push @{$config_cache->{COMPILER_FLAGS}}, "$1";
      $flag_index++;
    }
    elsif($line=~/^\s*DEFAULT_COMPILER_FLAGS\s*=\s*(.+)$/)
    {
      $config_cache->{COMPILER_FLAGS}[0]=$1;
    }
    elsif($line=~/^\s*BASE_DIR\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $dir (split /\s+/,$line)
      {
        if (!-d $dir){print "BASE_DIR \"$dir\" does not exist.\n"; next;}
	if(!exists $config_cache->{BASE_DIR_ORDERED}){$config_cache->{BASE_DIR_ORDERED}=[];}
        if(!exists $config_cache->{BASE_DIR}{$dir})
        {
          $config_cache->{BASE_DIR}{$dir}=1;
	  $config_cache->{NON_SYSTEM_INCLUDE}{$dir}=1;
          push @{$config_cache->{BASE_DIR_ORDERED}}, $dir;
        }
      }
    }
    elsif($line=~/^\s*HEADER_EXT\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $exp (split /\s+/,$line)
      {$config_cache->{HEADER_EXT}{$exp}=1;}
    }
    elsif($line=~/^\s*SOURCE_EXT\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $exp (split /\s+/,$line)
      {$config_cache->{SOURCE_EXT}{$exp}=1;}
    }
    elsif($line=~/^\s*INCLUDE_FILTER\s*=\s*(.+)$/)
    {
      $line=$1;
      $config_cache->{INCLUDE_FILTER}=$line;
    }
    elsif($line=~/^\s*SKIP_FILES\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $file (split /\s+/,$line){$config_cache->{SKIP_FILES}{$file}=1;}
    }
    elsif($line=~/^\s*SKIP_AND_ADD_REMOVED_INCLUDES\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $file (split /\s+/,$line){$config_cache->{SKIP_AND_ADD_REMOVED_INCLUDES}{$file}=1;}
    }
    elsif($line=~/^\s*FWDHEADER\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $file (split /\s+/,$line){$config_cache->{FWDHEADER}{$file}=1;}
    }
    elsif($line=~/^\s*SKIP_INCLUDES\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $exp (split /\s+/,$line)
      {$config_cache->{SKIP_INCLUDES}{$exp}=1;}
    }
    elsif($line=~/^\s*SKIP_SELF_INCLUDES\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $exp (split /\s+/,$line)
      {$config_cache->{SKIP_SELF_INCLUDES}{$exp}=1;}
    }
    elsif($line=~/^\s*SKIP_ORIG_INCLUDES\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $exp (split /\s+/,$line)
      {$config_cache->{SKIP_ORIG_INCLUDES}{$exp}=1;}
    }
    elsif($line=~/^\s*LOCAL_HEADERS\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $exp (split /\s+/,$line)
      {$config_cache->{LOCAL_HEADERS}{$exp}=1;}
    }
    elsif($line=~/^\s*SKIP_INCLUDE_INDIRECT_ADD\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $exp (split /\s+/,$line)
      {$config_cache->{SKIP_INCLUDE_INDIRECT_ADD}{$exp}=1;}
    }
    elsif($line=~/^\s*FILES\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $file (split /\s+/,$line)
      {if(!exists $config_cache->{FILES}{$file}){$config_cache->{FILES}{$file}{COMPILER_FLAGS_INDEX}=$flag_index;}}
    }
    elsif($line=~/^\s*OWNHEADER\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $x (split /\s+/,$line)
      {
        my $found=0;
	foreach my $p (@{$config_cache->{OWNHEADER}})
	{if($p eq $x){$found=1;last;}}
	if(!$found){push @{$config_cache->{OWNHEADER}},$x;}
      }
    }
  }
  my $fil=$filefilter;
  if($fil eq ""){$fil=".+";}
  foreach my $f (sort keys %{$config_cache->{FILES}})
  {
    my $b=&find_file($f);
    if($b){$config_cache->{FILES}{$f}{BASE_DIR}=$b;}
    else{delete $config_cache->{FILES}{$f}; print "$f does not exist in any base directory.\n";}
  }
}

sub check_skip ()
{
  my $file=shift || return 0;
  my $inc=shift || return 0;
  my $key=shift || return 0;
  foreach my $exp (keys %{$config_cache->{$key}})
  {
    my ($fexp,$val)=split /:/,$exp;
    if($fexp && $val)
    {
      if($file=~/$fexp/)
      {
        my $x=eval $val;
        if($inc=~/$x/){return 1;}
      }
    }
  }
  return 0;
}

sub is_local_header ()
{return &check_skip(shift,shift,"LOCAL_HEADERS");}  

sub skip_include ()
{
  my $file=shift || return 0;
  my $inc=shift || return 0;
  my $ccfile=0;
  if ($inc=~/^.+(\.[^\.]+)$/)
  {
    my $fext=$1;
    $ccfile=1;
    foreach my $exp (keys %{$config_cache->{HEADER_EXT}}){if($inc=~/$exp/){$ccfile=0;last;}}
  }
  return $ccfile || &check_skip($file,$inc,"SKIP_INCLUDES");
}

sub skip_self_include ()
{return &check_skip(shift,shift,"SKIP_SELF_INCLUDES");}

sub is_skipped_inc_add ()
{return &check_skip(shift,shift,"SKIP_INCLUDE_INDIRECT_ADD");}

sub is_skipped()
{
  my $file=shift;
  if($file!~/$filefilter/){return 1;}
  if((exists $config_cache->{FILES}{$file}) && (exists $config_cache->{FILES}{$file}{INTERNAL_SKIP}))
  {return $config_cache->{FILES}{$file}{INTERNAL_SKIP};}
  foreach my $type ("SKIP_FILES","FWDHEADER","SKIP_AND_ADD_REMOVED_INCLUDES")
  {if(&match_data($file,$type)){return 1;}}
  return 0;
}

sub match_data ()
{
  my $data=shift;
  my $type=shift;
  if(!exists $config_cache->{$type}){return 0;}
  foreach my $exp (keys %{$config_cache->{$type}}){if ($data=~/$exp/){return 1;}}
  return 0;
}

sub usage_msg()
{
  print "Usage: \n$0 \\\n\t[--config <file>] [--tmpdir <path>] [--filter <redexp>] [--redo <regexp>]\\\n";
  print "   [--redoerr] [--keep] [--recursive] [--unique] [--includeall] [--sysheader] [--local]\\\n";
  print "    [--detail] [--help]\n\n";
  print "  --config <file>    File which contains the list of files to check.\n";
  print "                     File format is:\n";
  print "                       COMPILER=<compiler path> #Optional: Default is \"".$config_cache->{COMPILER}."\".\n";
  print "                       COMPILER_FLAGS=<flags>   #Optional: Default are \"".$config_cache->{COMPILER_FLAGS}[0]."\".\n";
  print "                       HEADER_EXT=<regexp>      #Optional: Default is \"".join(", ",keys %{$config_cache->{HEADER_EXT}})."\".\n";
  print "                       SOURCE_EXT=<regexp>      #Optional: Default is \"".join(", ",keys %{$config_cache->{SOURCE_EXT}})."\".\n";
  print "                       INCLUDE_FILTER=<regexp>  #Optional: Default is \"".$config_cache->{INCLUDE_FILTER}."\"\n";
  print "                         #This filter is use to find the included files.\n";
  print "                       FILES=<relppath1> <relpath2> #' ' separated list of files relative paths\n";
  print "                       FILES=<relpath3>\n";
  print "                       SKIP_FILES=<regexp1> <regexp2> #Optional: ' ' separated list regular expressions\n";
  print "                       SKIP_FILES=<regexp3>\n";
  print "                       SKIP_INCLUDES=<regexp1_InFile>:<regexp2_IncludedFile>  <regexp3_InFile>:<regexp4_IncludeFile>\n";
  print "                         #Optional: ' ' separated list regular expressions\n";
  print "                       SKIP_INCLUDES=<regexp5_InFile>:<regexp5_IncludeFile>\n";
  print "                       SKIP_INCLUDE_INDIRECT_ADD=<regexp_InFile>:<regexp_IncludeFile>\n";
  print "                       LOCAL_HEADERS=<regexp_InFile>:<regexp_IncludeFile> <regexp_InFile>:<regexp_IncludeFile>\n";
  print "                       BASE_DIR=<dir1> <dir2> #Path where all the FILES exists.\n";
  print "                       BASE_DIR=<dir3>\n";
  print "                         #One can use it multiple times to provide many base directories.\n";
  print "                       OWNHEADER=<regexp1>:\"<regexp2>\"\n";
  print "                         #by default file1.h is assumed to be aheader file of file1.C|Cc|Cxx etc.\n";
  print "                         #But if your source files and header files do not exist in same directory then\n";
  print "                         #you can provide your own filters. e.g.\n";
  print "                         #OWNHEADER=^src\\/(.+?)\\.C:\"interface/\$1.h\"\n";
  print "                         #means that a own header for file src/file1.C exists in interface/file1.h\n";
  print "                     One can redefine COMPILER_FLAGS so the rest of the FILES\n";
  print "                     will use these new flags.\n";
  print "  --tmpdir <path>    Tmp directory where the tmp files will be generated.\n";
  print "                     Default is /tmp/delete_me_includechecker_<pid>.\n";
  print "  --filter <regexp>  Process only those files which matches the filter <regexp>.\n";
  print "  --redo <regexp>    Recheck all the files which matches the filter <regexp>.\n";
  print "  --redoerr          Recheck all the files had errors durring last run.\n";
  print "  --keep             Do not delete the tmp files.\n";
  print "  --recursive        Recursively check all the included headers if\n";
  print "                     they exist in one of the BASE_DIR.\n";
  print "  --unique           If a header is already included in one of your included headers then remove it.\n";
  print "                     e.g. if B.h includes A.h and C.h include both A.h and B.h then remove A.h for\n";
  print "                     C.h as its alreay included by B.h. If this option is not used then script will\n";
  print "                     check if a included header is really used or not by compiling it couple of time.\n";
  print "  --includeall       Add all the headers from an included header file (which are not already included in file to check).\n";
  print "  --sysheader        Do not skip checking for system headers.\n";
  print "  --local            Do not skip checking for local headers(for fast processing local headers of a package are not checked).\n";
  print "  --detail           To get more detailed output.\n";
  print "  --help             To see this help message.\n";
  exit 0;
}
