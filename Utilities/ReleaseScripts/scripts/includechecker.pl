#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use Getopt::Long;
use SCRAMGenUtils;
$|=1;

my $INSTALL_PATH = dirname($0);
my $curdir=`/bin/pwd`; chomp $curdir;
my $tmp_inc="includechecker/src";
my $dummy_inc="dummy_include";
my $config_cache={};
my $cache_file="config_cache";
my $max_pack_depth=2;
my %extra_dummy_file;
$extra_dummy_file{string}{"bits/stringfwd.h"}=1;
$config_cache->{COMPILER}="c++";
$config_cache->{COMPILER_FLAGS}[0]="";
$config_cache->{HEADER_EXT}="\\.(h|hpp)";
$config_cache->{SOURCE_EXT}="\\.(cc|CC|cpp|C|c|CPP|cxx|CXX)";
$config_cache->{INCLUDE_FILTER}=".+";
$config_cache->{SKIP_INCLUDES}{".+?:.+?\\.icc"}=1;
$config_cache->{OWNHEADER}{"^(.*?)\\.(cc|CC|cpp|C|c|CPP|cxx|CXX)"}="\"\${1}.h\"";
$config_cache->{FWDHEADER}{"^.*?(\\/|)[^\\/]*[Ff][Ww][Dd].h"}=1;

if(&GetOptions(
               "--config=s",\$config,
	       "--tmpdir=s",\$tmp_dir,
	       "--keep",\$keep,
               "--detail",\$detail,
               "--recursive",\$recursive,
	       "--includeall",\$includeall,
               "--unique",\$unique,
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

if(defined $includeall){$includeall=1;}
else{$includeall=0;}

if(defined $detail){$detail=1;}
else{$detail=0;}

if(defined $recursive){$recursive=1;}
else{$recursive=0;}

if(!defined $keep){$keep=0;}
else{$keep=1;}

if ((!defined $tmp_dir) || ($tmp_dir=~/^\s*$/)) {$tmp_dir="/tmp/delete_me_includechecker_$$";}
else
{
  if($tmp_dir=~/^[^\/]/){$tmp_dir=&SCRAMGenUtils::fixPath("${curdir}/${tmp_dir}");}
  $keep=1;
}

system("mkdir -p $tmp_dir");
print "TMP directory:$tmp_dir\n";
&init ($config);
chdir($tmp_dir);

foreach my $f (sort(keys %{$config_cache->{FILES}}))
{&check_file($f);}

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
  my $origrel_dir=dirname($origfile);
  my $orig_dir="${base_dir}/${origrel_dir}";
  my $filter=$config_cache->{INCLUDE_FILTER};
  
  &read_file ("${base_dir}/${origfile}", $cache);
  
  my $total_inc=scalar(@{$cache->{includes}});
  my $inc_added=0;
  my $actual_inc_added=0;
  my $inc_removed=0;
  my $inc_type="ALL_INCLUDES_REMOVED";
  if($includeall){$inc_type="ALL_INCLUDES";}
  my $otype ="${inc_type}_ORDERED";
  
  for(my $i=0; $i<$total_inc; $i++)
  {$config_cache->{FILES}{$origfile}{INCLUDES}{$cache->{includes}[$i]}=$cache->{includes_line}[$i];}
  
  my $skip=&is_skipped($origfile);
  my $deperror=0;
  my %pincs=();
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
	$pincs{$inc_file}=1;
	print "MSG: Private header $pinc in $origfile file.($inc_file : $inc_line)\n";
      }
    }
    else{$b=&find_file ($inc_file);}
    
    if((exists $config_cache->{FILES}{$inc_file}) && ($config_cache->{FILES}{$inc_file}{DONE}) && (exists $config_cache->{FILES}{$inc_file}{ERROR}))
    {if($config_cache->{FILES}{$inc_file}{ERROR} > 0){$deperror++;next;}}

    my $inc_skip = &is_skipped($inc_file);
    if(!$inc_skip && $includeall && (!exists $config_cache->{FILES}{$origfile}{ALL_INCLUDES}{$inc_file}))
    {
      push @{$config_cache->{FILES}{$origfile}{ALL_INCLUDES_ORDERED}}, $inc_file;
      $config_cache->{FILES}{$origfile}{ALL_INCLUDES}{$inc_file}=$inc_line;
    }
    
    if (!$recursive && !exists $config_cache->{FILES}{$inc_file}){next;}
    if ($inc_file!~/$filter/){next;}
    if ($b ne "")
    {
      if (!exists $config_cache->{FILES}{$inc_file})
      {
	my $incdir1=$inc_file;
	my $dirdepth=0;
	while((!exists $config_cache->{FILES}{$inc_file}) && ($dirdepth<$max_pack_depth))
	{
	  $dirdepth++;
	  $incdir1=dirname($incdir1);
	  if (($incdir1 eq "/") || ($incdir1 eq ".") || ($incdir1 eq "")){last;}
	  foreach my $tmpfile (keys %{$config_cache->{FILES}})
	  {
	    if ($config_cache->{FILES}{$tmpfile}{BASE_DIR} eq $b)
	    {
	      my $incdir2=$tmpfile;
	      for(my $k=0;$k<$dirdepth;$k++){$incdir2=dirname($incdir2);}
	      if ($incdir1 eq $incdir2)
	      {$config_cache->{FILES}{$inc_file}{COMPILER_FLAGS_INDEX}=$config_cache->{FILES}{$tmpfile}{COMPILER_FLAGS_INDEX};last;}
	    }
	  }
	}
	$config_cache->{FILES}{$inc_file}{BASE_DIR}=$b;
	if (!exists $config_cache->{FILES}{$inc_file}{COMPILER_FLAGS_INDEX})
	{$config_cache->{FILES}{$inc_file}{COMPILER_FLAGS_INDEX}=scalar(@{$config_cache->{COMPILER_FLAGS}})-1;}
      }
      &check_file($inc_file);
      if(exists $config_cache->{FILES}{$inc_file}{ERROR})
      {if($config_cache->{FILES}{$inc_file}{ERROR} > 0){$deperror++;next;}}
      
      my $num=$cache->{includes_line_number}[$i];
      my $cur_total = scalar(@{$cache->{includes}});
      if($includeall)
      {
        foreach my $inc (@{$config_cache->{FILES}{$inc_file}{ALL_INCLUDES_ORDERED}})
        {
	  my $l=$config_cache->{FILES}{$inc_file}{ALL_INCLUDES}{$inc};
	  if(!exists $config_cache->{FILES}{$origfile}{ALL_INCLUDES}{$inc})
	  {
	    push @{$config_cache->{FILES}{$origfile}{ALL_INCLUDES_ORDERED}}, $inc;
	    $config_cache->{FILES}{$origfile}{ALL_INCLUDES}{$inc}=$l;
	  }
        }
        if($inc_skip && (!exists $config_cache->{FILES}{$origfile}{ALL_INCLUDES}{$inc_file}))
        {
          push @{$config_cache->{FILES}{$origfile}{ALL_INCLUDES_ORDERED}}, $inc_file;
          $config_cache->{FILES}{$origfile}{ALL_INCLUDES}{$inc_file}=$inc_line;
        }
      }
      if($skip)
      {
        foreach my $inc (@{$config_cache->{FILES}{$inc_file}{ALL_INCLUDES_REMOVED_ORDERED}})
	{
	  my $l=$config_cache->{FILES}{$inc_file}{ALL_INCLUDES_REMOVED}{$inc};
	  push @{$config_cache->{FILES}{$origfile}{ALL_INCLUDES_REMOVED_ORDERED}}, $inc;
	  $config_cache->{FILES}{$origfile}{ALL_INCLUDES_REMOVED}{$inc}=$l;
	}
	next;
      }
      foreach my $inc (@{$config_cache->{FILES}{$inc_file}{$otype}})
      {
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
	if($detail)
	{print "Added \"$inc\" as it was removed from or included in \"$inc_file\" file.\n";}
	$num++;
	$cur_total++;
	$config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}{$inc}=1;
	$config_cache->{FILES}{$origfile}{INCLUDES}{$inc}=$l;
	$inc_added++;
      }
    }
  }
  print "#################################################\n";
  print "File: $origfile\n";
  print "  Total lines            : ".$cache->{total_lines}."\n";
  print "  Code lines             : ".$cache->{code_lines}."\n";
  print "  Commented lines        : ".$cache->{comments_lines}."\n";
  print "  Empty lines            : ".$cache->{empty_lines}."\n";
  print "  Number of includes     : ".$total_inc."\n";
  print "  Additional Includes    : ".$inc_added."\n\n";
  
  $config_cache->{FILES}{$origfile}{ERROR}=$deperror;
  if($skip || $deperror)
  {
    if($detail)
    {
      my $msg="Skipping \"$origfile\" due to ";
      if($skip && $deperror)
      {$msg.="SKIP_FILES flag in the config file and errors ($deperror) in files included by it.";}
      elsif($deperror){$msg.="errors ($deperror) in files included by it.";}
      else{$msg.="SKIP_FILES flag in the config file.";}
      print "$msg\n";
    }
    return;
  }
  
  my $stime=time;
  my $oflags=$config_cache->{COMPILER_FLAGS}[$config_cache->{FILES}{$origfile}{COMPILER_FLAGS_INDEX}];
  my $xflags="-I-";  
  #my $xflags="";
  $oflags=~s/\B$xflags\B//;
  my $flags="-shared -fsyntax-only -I $xflags -I${tmp_dir}/${tmp_inc}/${origrel_dir} -I${tmp_dir}/${tmp_inc}  -I${orig_dir} $oflags";
  my $compiler=$config_cache->{COMPILER};
  my $error=0;
  my @origwarns=`$compiler $flags -o ${srcfile}.o $srcfile 2>&1`;
  if ($? != 0)
  {$error=1;}
  
  my $origwarns_count=scalar(@origwarns);
  foreach my $warn (@origwarns)
  {chomp $warn;$warn=~s/$srcfile/${base_dir}\/${origfile}/;}

  $total_inc=scalar(@{$cache->{includes}});
  if($detail && (exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}))
  {
    if($includeall)
    {print "Following files are added (because those were removed or indirectly added from included headers)\n";}
    else{print "Following files are added (because those were removed from included headers)\n"}
    print "inorder to make the compilation work.\n";
    foreach my $f (keys %{$config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}})
    {print "  $f\n";}
    print "\n";
  }
  if ($error)
  {
    print "File should be compiled without errors.\n";
    print "Compilation errors are:\n";
    foreach my $w (@origwarns)
    {
      $w=~s/$srcfile/${base_dir}\/${origfile}/;
      print "$w\n";
    }
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
    print "\nCompiler flags used are:\n$flags\n";
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
    foreach my $fil (keys %{$config_cache->{OWNHEADER}})
    {
      my $h=$origfile;
      my $fil2=$config_cache->{OWNHEADER}{$fil};
      if($h=~/$fil/)
      {$h=eval $fil2;}
      for(my $i=0; $i < $total_inc; $i++)
      {
        my $f=$cache->{includes}[$i];
	if ($f=~/$h/)
	{$own_header=$f;last;}
      }
    }
  }
  if($detail && $origwarns_count>0)
  {
    print "-----------------------------------\n";
    print "ORIGINAL WARNINGS\n";
    foreach my $w (@origwarns)
    {print "$w\n";}
    print "-----------------------------------\n";
  }
  
  my $num = -1;
  my $fwdcheck=0; 
  $flags=~s/\B$xflags\B//;
  while(1){
    my $i=&find_next_index ($cache, $num, $fwdcheck);
    if ($i==-1)
    {
      if($fwdcheck==1){last;}
      else{$fwdcheck=1;$num=-1;next;}
    }
    $num = $cache->{includes_line_number}[$i];
    my $inc_file=$cache->{includes}[$i];
    my $inc_line=$cache->{includes_line}[$i];
    
    my $skip_inc="$origfile:$inc_file";
    my $sinc_exp="";
    foreach my $sinc (keys %{$config_cache->{SKIP_INCLUDES}})
    {if($skip_inc=~/$sinc$/){$skip_inc=""; $sinc_exp=$sinc;last;}}
    if ($skip_inc eq "")
    {
      if($detail){print "  Skipping checking of \"$inc_file\" in \"$origfile\" due to \"$sinc_exp\" SKIP_INCLUDES flag in the config file\n";}
      next;
    }
    if ($inc_file eq $own_header)
    { 
      if($detail){print "  Skipping the checking of $inc_file (Assumption: .cc always needs its own .h)\n";}
      next;
    }
    &comment_line ($srcfile, $num);
    if(!$includeall)
    {if(exists $extra_dummy_file{$inc_file}){&createdummyfile ($inc_file,1);}}
    my $inc_req=0;
    my $exists_in_own_header=0;
    if (($own_header ne "") && 
        (exists $config_cache->{FILES}{$own_header}) &&
	(exists $config_cache->{FILES}{$own_header}{INCLUDES}{$inc_file}))
    {$exists_in_own_header=1;}
    my $loop=2;
    if($unique){$loop=1;};
    my @warns=();
    for(my $x=0;$x<$loop;$x++)
    {
      @warns=();
      @warns=`$compiler -I $xflags -I${tmp_dir}/${dummy_inc} $flags -o ${srcfile}.o $srcfile 2>&1`;
      my $ret_val=$?;
      foreach my $w (@warns)
      {chomp $w;$w=~s/$srcfile/${base_dir}\/${origfile}/;}
      if ($ret_val != 0)
      {
	if($x==0){$inc_req=1;}
	else
	{
	  my @warns1=();
	  my $sf="${base_dir}/${origfile}";
	  foreach my $w (@warns)
	  {
	    if($w=~/^\s*$sf:\d+:\s*/)
	    {$inc_req=1;push @warns1, $w;}
	    elsif(($inc_req==0) && ($w=~/^.+?:\d+:\s+confused by earlier errors/))
	    {
	      $inc_req=1;
	      foreach my $w (@warns)
	      {push @warns1, $w;}
	      last;
	    }
	  }
	  @warns=@warns1;
	}
      }
      elsif ($origwarns_count == scalar(@warns))
      {
        for(my $j=0; $j<$origwarns_count; $j++)
        {
          my $warn=$warns[$j];
          if ("$warn" ne "$origwarns[$j]")
          {$inc_req=1;last;}
        }
      }
      else{$inc_req=1;}
      if(($x==0) && (!$inc_req) && (!$unique) && (!$exists_in_own_header))
      {
        if($includeall || (!exists $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT}{$inc_file})){&createdummyfile ($inc_file);}
	else{last;}
      }
      else{last;}
    }
    &cleanupdummydir();
    if($inc_req==0)
    {
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
      
      if(($detail || $extra) && scalar(@warns) > 0)
      {
	print "-----------------------------------\n";
	print "Difference in warnings/errors after removing \"$inc_file\"\n";
	print "NEW WARNINGS\n";
	foreach my $w (@warns)
	{print "$w\n";}
        print "-----------------------------------\n";
      }
    }
    system("rm -f ${srcfile}.o");
  }
  system("grep -v \"//INCLUDECHECKER: Removed this line:.*//INCLUDECHECKER: Added this line\" $srcfile > ${srcfile}.new");
  my $diff=`diff ${srcfile}.new ${base_dir}/${origfile}`; chomp $diff;
  if ($diff ne "")
  {
    my $name=basename($origfile);
    my $dir="${tmp_dir}/${tmp_inc}/${origrel_dir}";
    system("mkdir -p $dir; rm -f ${dir}/${name}; mv ${srcfile}.new ${dir}/${name}");
    foreach my $pinc (keys %pincs)
    {
      my $file="${tmp_dir}/${tmp_inc}/${pinc}";
      my $b=$config_cache->{FILES}{$pinc}{BASE_DIR};
      print "MSG: Private Include Copy: ${b}/${pinc} => $file\n";
      if(!-f $file)
      {
        $dir=dirname($file);
        system("mkdir -p $dir; cp ${b}/${pinc} $file");
      }
    }
  }
  print "\n";
  print "  Include added          : ".$inc_added."\n";
  print "  Include removed        : ".$inc_removed."\n";
  print "  Actual include added   : ".$actual_inc_added."\n";
  print "  Actual include removed : ",$inc_removed-$inc_added+$actual_inc_added,"\n";
  my $dtime=time-$stime;
  print "  Processing time(sec)   : ".$dtime."\n";
  delete $config_cache->{FILES}{$origfile}{INCLUDE_ADDED_INDIRECT};
}

sub createdummyfile ()
{
  my $file=shift;
  my $extra_only=shift || 0;;
  my $dir="${tmp_dir}/${dummy_inc}/".dirname($file);
  my $file1=basename($file);
  if(!$extra_only)
  {system("mkdir -p $dir; touch ${dir}/${file1}");}
  if(exists $extra_dummy_file{$file})
  {
    foreach my $f (keys %{$extra_dummy_file{$file}})
    {&createdummyfile ($f);}
  }
}

sub cleanupdummydir()
{
  system("rm -rf ${tmp_dir}/${dummy_inc}/* 2>&1 > /dev/null");
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

sub check_file ()
{
  my $file=shift;
  my $depth = scalar(@{$config_cache->{INC_LIST}});
  if ($depth > 0)
  {
    for(my $i=0; $i<$depth; $i++)
    {
      if($config_cache->{INC_LIST}[$i] eq $file)
      {
        print "WARNING: Cyclic includes:\n";
	for(my $j=$i; $j<$depth;$j++)
	{print $config_cache->{INC_LIST}[$j]." -> ";}
	print "$file\n";
      }
    }
  }
  if (exists $config_cache->{FILES}{$file}{DONE})
  {
    if($detail){print "Already done: $file\n";}
    return;
  }
  $depth++;
  $config_cache->{FILES}{$file}{DONE}=1;
  delete $config_cache->{FILES}{$file}{ERROR};
  delete $config_cache->{FILES}{$file}{FINAL_DONE};
  push @{$config_cache->{INC_LIST}}, $file;
  
  my $base=$config_cache->{FILES}{$file}{BASE_DIR};
  my $check=1;
  my $tmpfile="";
  my %cache=();

  my $dir="${tmp_dir}/tmp_${depth}";
  system("mkdir -p $dir");
  
  my $header_ext=$config_cache->{HEADER_EXT};
  my $src_ext=$config_cache->{SOURCE_EXT};
  
  if($file=~/$header_ext$/)
  {
    $tmpfile=rearrangePath ("${dir}/".basename($file).".cc");
    $cache{$tmpfile}{isheader}=1;
  }
  elsif($file=~/$src_ext$/)
  {
    $tmpfile=rearrangePath ("${dir}/".basename($file));
    $cache{$tmpfile}{isheader}=0;
  }
  else
  {
    print "$file does not match the either source($src_ext) or header($header_ext) extensions.\n";
    $config_cache->{FILES}{$file}{FINAL_DONE}=1;
  }
  
  if ($tmpfile ne "")
  {
     system("cp -pf ${base}/${file} $tmpfile; chmod u+w $tmpfile");
     $cache{$tmpfile}{original}=$file;
     &check_includes ($tmpfile, $cache{$tmpfile});
     if($config_cache->{FILES}{$file}{ERROR} == 0)
     {$config_cache->{FILES}{$file}{FINAL_DONE}=1;}
     if($keep){&SCRAMGenUtils::writeHashCache($config_cache->{FILES}{$file}, "${tmp_dir}/cache/files/$file");}
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

  my $total_lines=scalar(@{$cache->{lines}});
  my $first_ifndef=0;
  my $extern=0;
  for(my $i=0;$i<$total_lines;$i++)
  {
    my $line=$cache->{lines}[$i];
    my $num=$cache->{line_numbers}[$i];
    if ($cache->{isheader} && !$first_ifndef && ($line=~/^\s*#\s*ifndef\s+/)){$first_ifndef=1; next;}
    if($line=~/^\s*#\s*if(n|\s+|)def(ined|\s+|)/)
    {$i=&SCRAMGenUtils::skipIfDirectiveCXX ($cache->{lines}, $i+1, $total_lines);next;}
    
    while($line=~/\\\//){$line=~s/\\\//\//;}
    if($line=~/^\s*extern\s+\"C\"\s+\{\s*$/)
    {$extern=1;next;}
    if($extern && $line=~/^\s*\}.*/){$extern=0; next;}
    if ($line=~/^\s*#\s*include\s*([\"<](.+?)[\">])\s*/)
    {
      if($extern){next;}
      my $inc_file=$2;
      push @{$cache->{includes}}, $inc_file;
      push @{$cache->{includes_line_number}}, $num;
      push @{$cache->{includes_line}}, $line;
    }
  }
}

sub init ()
{
  my $config=shift;
  if ("$tmp_dir" ne "/")
  {
    system("mkdir -p ${tmp_dir}/${tmp_inc} ${tmp_dir}/${dummy_inc} ${tmp_dir}/cache");
    system("/bin/rm -rf ${tmp_dir}/tmp_* 2>&1");
    system("/bin/rm -rf ${tmp_dir}/${dummy_inc}/* 2>&1");
    if (-f "${tmp_dir}/cache/${cache_file}")
    {
      $config_cache=&SCRAMGenUtils::readHashCache("${tmp_dir}/cache/${cache_file}");
      foreach my $f (keys %{$config_cache->{FILES}})
      {
	if((!exists $config_cache->{FILES}{$f}{DONE}) && (-f "${tmp_dir}/cache/files/$f"))
	{
	  $config_cache->{FILES}{$f}={};
	  $config_cache->{FILES}{$f}=&SCRAMGenUtils::readHashCache("${tmp_dir}/cache/files/$f");
	}
	if ($config_cache->{FILES}{$f}{ERROR})
	{
	  delete $config_cache->{FILES}{$f}{DONE};
	  delete $config_cache->{FILES}{$f}{ERROR};
	  if($detail){print "REDO due to errors in previous run: $f\n";}
	}
	elsif (exists $config_cache->{FILES}{$f}{FINAL_DONE}){$config_cache->{FILES}{$f}{DONE}=1;print "ALREADY DONE:$f\n";}
	else{delete $config_cache->{FILES}{$f}{DONE};}
        foreach my $regexp (keys %{$config_cache->{SKIP_FILES}})
        {
          if ($f=~/$regexp/)
          {
            delete $config_cache->{FILES}{$f};
	    print "SKIPPED:$f\n";
	    last;
          }
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
      delete $config_cache->{INC_LIST};
      &SCRAMGenUtils::writeHashCache($config_cache, "${tmp_dir}/cache/${cache_file}");
      return;
    }
  }
  &read_config ($config);
  $config_cache->{INCLUDEALL}=$includeall;
  delete $config_cache->{INC_LIST};
  if($keep){&SCRAMGenUtils::writeHashCache($config_cache, "${tmp_dir}/cache/${cache_file}");}
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
      $line=$1;
      $config_cache->{COMPILER}=$line;
      #if($detail){print "COMPILER=$line\n";}
    }
    elsif($line=~/^\s*COMPILER_FLAGS\s*=\s*(.+)$/)
    {
      $line="$1";
      push @{$config_cache->{COMPILER_FLAGS}}, $line;
      $flag_index++;
      #if($detail){print "COMPILER_FLAGS=$line\n";}
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
          push @{$config_cache->{BASE_DIR_ORDERED}}, $dir;
          #if($detail){print "BASE_DIR=$dir\n";}
        }
      }
    }
    elsif($line=~/^\s*HEADER_EXT\s*=\s*(.+)$/)
    {
      $line=$1;
      $config_cache->{HEADER_EXT}=$line;
    }
    elsif($line=~/^\s*SOURCE_EXT\s*=\s*(.+)$/)
    {
      $line=$1;
      $config_cache->{SOURCE_EXT}=$line;
    }
    elsif($line=~/^\s*INCLUDE_FILTER\s*=\s*(.+)$/)
    {
      $line=$1;
      $config_cache->{INCLUDE_FILTER}=$line;
    }
    elsif($line=~/^\s*SKIP_FILES\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $file (split /\s+/,$line)
      {
        $config_cache->{SKIP_FILES}{$file}=1;
      }
    }
    elsif($line=~/^\s*SKIP_INCLUDES\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $exp (split /\s+/,$line)
      {
        $config_cache->{SKIP_INCLUDES}{$exp}=1;
      }
    }
    elsif($line=~/^\s*FILES\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $file (split /\s+/,$line)
      {$config_cache->{FILES}{$file}{COMPILER_FLAGS_INDEX}=$flag_index;}
    }
    elsif($line=~/^\s*OWNHEADER\s*=\s*(.+)$/)
    {
      $line=$1;
      foreach my $x (split /\s+/,$line)
      {
	my ($v, $w)=split /:/, $x;
	if(($v ne "") && ($w ne ""))
	{$config_cache->{OWNHEADER}{$v}=$w;}
      }
    }
  }
  if (!exists $config_cache->{OWNHEADER})
  {$config_cache->{OWNHEADER}{"^(.+?)\\.[^\\.].+"}="\$1.h";}
  if(!exists $config_cache->{FWDHEADER})
  {$config_cache->{FWDHEADER}{"^.*?(\\/|)[^\\/]*fwd.h"}=1;}
  foreach my $f (sort keys %{$config_cache->{FILES}})
  {
    foreach my $regexp (keys %{$config_cache->{SKIP_FILES}})
    {
      if ($f=~/$regexp/)
      {
        delete $config_cache->{FILES}{$f};
    	$f="";
    	last;
      }
    }
    if($f)
    {
      my $b=&find_file($f);
      if($b){$config_cache->{FILES}{$f}{BASE_DIR}=$b;}
      else{delete $config_cache->{FILES}{$f}; print "$f does not exist in any base directory.\n";}
    }
  }
}

sub is_skipped()
{
  my $file=shift;
  my $skip=0;
  foreach my $exp (keys %{$config_cache->{SKIP_FILES}})
  {
    if ($file=~/${exp}/){return 1;}
  }
  foreach my $fwd (keys %{$config_cache->{FWDHEADER}})
  {
    if($file=~/$fwd/){return 1;}
  }
  return 0;
}

sub usage_msg()
{
  print "Usage: \n$0 \\\n\t[--config <file>] [--tmpdir <path>]\\\n";
  print "   [--keep] [--detail] [--recursive] [--unique] [--help]\n\n";
  print "  --config <file>    File which contains the list of files to check.\n";
  print "                     File format is:\n";
  print "                       COMPILER=<compiler path> #Optional: Default is \"".$config_cache->{COMPILER}."\".\n";
  print "                       COMPILER_FLAGS=<flags>   #Optional: Default are \"".$config_cache->{COMPILER_FLAGS}[0]."\".\n";
  print "                       HEADER_EXT=<regexp>      #Optional: Default is \"".$config_cache->{HEADER_EXT}."\".\n";
  print "                       SOURCE_EXT=<regexp>      #Optional: Default is \"".$config_cache->{SOURCE_EXT}."\".\n";
  print "                       INCLUDE_FILTER=<regexp>  #Optional: Default is \"".$config_cache->{INCLUDE_FILTER}."\"\n";
  print "                         #This filter is use to find the included files.\n";
  print "                       FILES=<relppath1> <relpath2> #' ' separated list of files relative paths\n";
  print "                       FILES=<relpath3>\n";
  print "                       SKIP_FILES=<regexp1> <regexp2> #Optional: ' ' separated list regular expressions\n";
  print "                       SKIP_FILES=<regexp3>\n";
  print "                       SKIP_INCLUDES=<regexp1_InFile>:<regexp2_IncludedFile>  <regexp3_InFile>:<regexp4_IncludeFile>\n";
  print "                         #Optional: ' ' separated list regular expressions\n";
  print "                       SKIP_INCLUDES=<regexp5_InFile>:<regexp5_IncludeFile>\n";
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
  print "  --keep             Do not delete the tmp files.\n";
  print "  --recursive        Recursively check all the included headers if\n";
  print "                     they exist in one of the BASE_DIR.\n";
  print "  --unique           If a header is already included in one of your included headers then remove it.\n";
  print "                     e.g. if B.h includes A.h and C.h include both A.h and B.h then remove A.h for\n";
  print "                     C.h as its alreay included by B.h. If this option is not used then script will\n";
  print "                     check if a included header is really used or not by compiling it couple of time.\n";
  print "  --includeall       Add all the headers from an included header file (which are not already included in file to check).\n";
  print "  --detail           To get more detailed output.\n";
  print "  --help             To see this help message.\n";
  exit 0;
}
