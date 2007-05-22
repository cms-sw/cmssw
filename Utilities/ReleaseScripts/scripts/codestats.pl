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
if(&GetOptions(
               "--log=s",\$log,
               "--help",\$help,
              ) eq ""){print "ERROR: Wrong arguments.\n"; &usage_msg();}

if (defined $help){&usage_msg();}
if ((!defined $log) || ($log eq "") || (!-f $log))
{print "Log file missing.\n";&usage_msg();}

if(!open(LOGFILE, $log)){die "Can not open \"$log\" file for reading.";}
my $file="";
my $cache={};
while(my $line=<LOGFILE>)
{
  chomp $line;
  if ($line=~/^File:\s+([^\s]+?)\s*$/)
  {
    $file=$1;
    foreach my $key ("TotalLines", "CodeLines", "CommentedLines", "EmptyLines", "IncludeStatements", "IncludeAdded", "IncludeRemoved", "ActualAdded")
    {&addValue ($cache, $file, $key);}
  }
  elsif ($line=~/^\s+Total\s+lines\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "TotalLines", $1);}
  elsif ($line=~/^\s+Code\s+lines\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "CodeLines", $1);}
  elsif ($line=~/^\s+Commented\s+lines\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "CommentedLines", $1);}
  elsif ($line=~/^\s+Empty\s+lines\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "EmptyLines", $1);}
  elsif ($line=~/^\s+Number\s+of\s+includes\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "IncludeStatements", $1);}
  elsif ($line=~/^\s+Include\s+added\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "IncludeAdded", $1);}
  elsif ($line=~/^\s+Include\s+removed\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "IncludeRemoved", $1);}
  elsif ($line=~/^\s+Actual\s+include\s+added\s+:\s+(\d+)\s*$/)
  {&addValue ($cache, $file, "ActualAdded", $1);}
}
close(LOGFILE);
&process ($cache);

sub process ()
{
  my $cache = shift;
  my $base=shift || "";
  my $totals = shift || [];
  my $num=scalar(@$totals);
  if (!defined $cache){return;}
  foreach my $key (sort keys %$cache)
  {
    if($key eq "_DATA"){next;}
    my $total = $cache->{$key}{_DATA}{TotalLines};
    my $empty = $cache->{$key}{_DATA}{EmptyLines};
    my $code = $cache->{$key}{_DATA}{CodeLines};
    my $comment = $cache->{$key}{_DATA}{CommentedLines};
    my $lines=($comment - ($total - $empty - $code))/2;
    $code = $code - $lines;
    $comment = $comment - $lines;
    $cache->{$key}{_DATA}{CodeLines} = $code;
    $cache->{$key}{_DATA}{CommentedLines} = $comment;
    print "################################\n";
    print "For ${base}${key}\n";
    foreach my $skey ("TotalLines", "CodeLines", "CommentedLines", "EmptyLines")
    {
      my $value=$cache->{$key}{_DATA}{$skey};
      my $precentage = "-";
      if ($total>0)
      {$precentage=&SCRAMGenUtils::setPrecision(($value * 100)/$total, $precentage_prec);}
      print &SCRAMGenUtils::leftAdjust($skey, 20).": ".&SCRAMGenUtils::leftAdjust($value, $value_length).&SCRAMGenUtils::rightAdjust("$precentage%",$precentage_prec+5)."  ";
      for(my $i=$num-1; $i>=0; $i--)
      {
        my $t=$totals->[$i];
	if ($t>0)
	{$precentage=&SCRAMGenUtils::setPrecision(($value * 100)/$t, $precentage_prec);}
	print &SCRAMGenUtils::rightAdjust("$precentage%",$precentage_prec+5)."  ";
      }
      print "\n";
    }
    my $diff=$cache->{$key}{_DATA}{IncludeStatements}-$cache->{$key}{_DATA}{IncludeRemoved};
    print &SCRAMGenUtils::leftAdjust("Includes", 20).": $diff\n";
    print &SCRAMGenUtils::leftAdjust("Includes added", 20).": ".$cache->{$key}{_DATA}{ActualAdded}."\n";
    $diff = $cache->{$key}{_DATA}{IncludeRemoved} - ($cache->{$key}{_DATA}{IncludeAdded} - $cache->{$key}{_DATA}{ActualAdded});
    print &SCRAMGenUtils::leftAdjust("Includes removed", 20).": $diff\n";
  }
  foreach my $key (sort keys %$cache)
  {
    if($key eq "_DATA"){next;}
    push @$totals, $cache->{$key}{_DATA}{TotalLines};
    &process($cache->{$key}, "${base}${key}/", $totals);
    pop @$totals;
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
  print "  --log <file>       Log file whcih contains the output of includechecker\n";
  print "  --help             To see this help message.\n";
  exit 0;
}
