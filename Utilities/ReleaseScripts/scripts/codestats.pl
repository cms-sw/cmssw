#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use Getopt::Long;
use SCRAMGenUtils;
$|=1;

my $INSTALL_PATH = dirname($0);
my $curdir=`/bin/pwd`; chomp $curdir;
my $precentage_prec=3;
my $value_length=10;
my $html=0;
if(&GetOptions(
               "--log=s",\$log,
	       "--tmpdir=s",\$dir,
               "--help",\$help,
              ) eq ""){print "ERROR: Wrong arguments.\n"; &usage_msg();}

if (defined $help){&usage_msg();}
if ((!defined $log) || ($log eq "") || (!-f $log))
{print "Log file missing.\n";&usage_msg();}
if ((defined $dir) && ($dir!~/^\s*$/) || (-d $dir))
{
  print "<html><head></head><body><pre>\n";
  $html=1;
}

if(!open(LOGFILE, $log)){die "Can not open \"$log\" file for reading.";}
my $file="";
my $cache={};
my %urls=();
if($html){$urls{src}=1;}
while(my $line=<LOGFILE>)
{
  chomp $line;
  if ($line=~/^File:\s+([^\s]+?)\s*$/)
  {
    $file=$1;
    foreach my $key ("Total Lines", "Code Lines", "Commented Lines", "Empty Lines", "Include Statements", "Include Added", "Include Removed")
    {&addValue ($cache, $file, $key);}
    &addValue ($cache,$file,"Files",1);
    if($html && -f "${dir}/includechecker/src/${file}")
    {
      my $f=$file;
      while($f ne "."){$urls{"src/${f}"}=1;$f=dirname($f);}
    }
  }
  elsif ($line=~/^\s+Total\s+lines\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "Total Lines", $1);}
  elsif ($line=~/^\s+Code\s+lines\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "Code Lines", $1);}
  elsif ($line=~/^\s+Commented\s+lines\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "Commented Lines", $1);}
  elsif ($line=~/^\s+Empty\s+lines\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "Empty Lines", $1);}
  elsif ($line=~/^\s+Number\s+of\s+includes\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "Include Statements", $1);}
  elsif ($line=~/^\s+Actual\s+include\s+added\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "Include Added", $1);}
  elsif ($line=~/^\s+Actual\s+include\s+removed\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "Include Removed", $1);}
}
close(LOGFILE);
&process ($cache);
if($html){print "</pre></body></html>\n";}

sub process ()
{
  my $cache = shift;
  my $base=shift || "";
  my $totals = shift || {};
  if (!defined $cache){return;}
  foreach my $key (sort keys %$cache)
  {
    if($key eq "_DATA"){next;}
    my $ctype = "Total Lines";
    my $total = $cache->{$key}{_DATA}{$ctype};
    my $num   = scalar(@{$totals->{$ctype}});
    my $empty = $cache->{$key}{_DATA}{"Empty Lines"};
    my $code = $cache->{$key}{_DATA}{"Code Lines"};
    my $comment = $cache->{$key}{_DATA}{"Commented Lines"};
    my $lines=($comment - ($total - $empty - $code))/2;
    if($lines != int($lines)){$code++;$lines=int($lines)+1;}
    $code = $code - $lines;
    $comment = $comment - $lines;
    $cache->{$key}{_DATA}{"Code Lines"} = $code;
    $cache->{$key}{_DATA}{"Commented Lines"} = $comment;
    my $url="";
    if($html)
    {
      if(exists $urls{"${base}${key}"}){$url="${base}${key}";}
      #else{next;}
    }
    print "###########################################################################\n";
    if($url){print "For <A href=\"$url\">${base}${key}</a>\n";}
    else{print "For ${base}${key}\n";}
    foreach my $skey ("Total Lines", "Code Lines", "Commented Lines", "Empty Lines")
    {
      my $value=$cache->{$key}{_DATA}{$skey};
      my $precentage = "-";
      if ($total>0)
      {$precentage=&SCRAMGenUtils::setPrecision(($value * 100)/$total, $precentage_prec);}
      print &SCRAMGenUtils::leftAdjust($skey, 20).": ".&SCRAMGenUtils::leftAdjust($value, $value_length).&SCRAMGenUtils::rightAdjust("$precentage%",$precentage_prec+5)."  ";
      for(my $i=$num-1; $i>=0; $i--)
      {
        my $t=$totals->{$ctype}[$i];
	if ($t>0)
	{$precentage=&SCRAMGenUtils::setPrecision(($value * 100)/$t, $precentage_prec);}
	print &SCRAMGenUtils::rightAdjust("$precentage%",$precentage_prec+5)."  ";
      }
      print "\n";
    }
    $ctype = "Files";
    $total=$cache->{$key}{_DATA}{$ctype};
    $num = scalar(@{$totals->{$ctype}});
    foreach my $skey ("Files")
    {
      my $value=$cache->{$key}{_DATA}{$skey};
      if($value<=1){next;}
      my $precentage = "-";
      if ($total>0)
      {$precentage=&SCRAMGenUtils::setPrecision(($value * 100)/$total, $precentage_prec);}
      print &SCRAMGenUtils::leftAdjust($skey, 20).": ".&SCRAMGenUtils::leftAdjust($value, $value_length).&SCRAMGenUtils::rightAdjust("$precentage%",$precentage_prec+5)."  ";
      for(my $i=$num-1; $i>=0; $i--)
      {
        my $t=$totals->{$ctype}[$i];
	if ($t>0)
	{$precentage=&SCRAMGenUtils::setPrecision(($value * 100)/$t, $precentage_prec);}
	print &SCRAMGenUtils::rightAdjust("$precentage%",$precentage_prec+5)."  ";
      }
      print "\n";
    }
    $ctype = "Include Statements";
    $total=$cache->{$key}{_DATA}{$ctype};
    $num = scalar(@{$totals->{$ctype}});
    foreach my $skey ("Include Statements", "Include Added", "Include Removed")
    {
      my $value=$cache->{$key}{_DATA}{$skey};
      if($value==0){next;}
      my $precentage = "-";
      if ($total>0)
      {$precentage=&SCRAMGenUtils::setPrecision(($value * 100)/$total, $precentage_prec);}
      print &SCRAMGenUtils::leftAdjust($skey, 20).": ".&SCRAMGenUtils::leftAdjust($value, $value_length).&SCRAMGenUtils::rightAdjust("$precentage%",$precentage_prec+5)."  ";
      for(my $i=$num-1; $i>=0; $i--)
      {
        my $t=$totals->{$ctype}[$i];
	if ($t>0)
	{$precentage=&SCRAMGenUtils::setPrecision(($value * 100)/$t, $precentage_prec);}
	print &SCRAMGenUtils::rightAdjust("$precentage%",$precentage_prec+5)."  ";
      }
      print "\n";
    }
  }
  foreach my $key (sort keys %$cache)
  {
    if($key eq "_DATA"){next;}
    foreach my $ctype ("Total Lines", "Files", "Include Statements")
    {push @{$totals->{$ctype}}, $cache->{$key}{_DATA}{$ctype};}
    &process($cache->{$key}, "${base}${key}/",$totals);
    foreach my $ctype ("Total Lines", "Files", "Include Statements")
    {pop @{$totals->{$ctype}};}
  }
}

sub addValue ()
{
  my $cache=shift || return;
  my $file=shift || return;
  my $type=shift || return;
  my $value=shift || 0;
  my $str="src";
  $cache->{$str} ||= {};
  $cache=$cache->{$str};
  my $oldvalue=$cache->{_DATA}{$type} || 0;
  $cache->{_DATA}{$type}=$oldvalue+$value;
  foreach $str (split /\/+/, $file)
  {
    $cache->{$str} ||= {};
    $cache=$cache->{$str};
    $oldvalue=$cache->{_DATA}{$type} || 0;
    $cache->{_DATA}{$type}=$oldvalue+$value;
  }
}

sub usage_msg()
{
  print "Usage: \n$0 \\\n\t[--log <file>] [--help]\n\n";
  print "  --log    <file>    Log file whcih contains the output of includechecker\n";
  print "  --tmpdir <dir>     Path of directory tmp directory where the newly generated files are available.\n";       
  print "  --help             To see this help message.\n";
  exit 0;
}
