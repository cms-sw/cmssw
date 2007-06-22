#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use SCRAMGenUtils;

$|=1;

my $bf1=shift;
my $bf2=shift;
my @output=();

if(!defined $bf1){die "Usage: $0 <buildfile1> <buildfile2>\n";}
if(!defined $bf2){die "Usage: $0 <buildfile1> <buildfile2>\n";}

if(!-f $bf1){die "No such file \"$bf1\".";}
if(!-f $bf2){die "No such file \"$bf2\".";}

my $refbf1=&SCRAMGenUtils::readBuildFile($bf1);
my $refbf2=&SCRAMGenUtils::readBuildFile($bf2);
&cleanup($refbf1);
&cleanup($refbf2);

my $output1={};
my $output2={};
my $res1=&compare ($refbf1,$refbf2,"",$output1);
my $res2=&compare ($refbf2,$refbf1,"",$output2);
if($res1 && $res2){print "OK SAME Files.\n"; exit 0;}
if(!$res1)
{
  print "Missing tags/values in $bf2 w.r.t $bf1\n";
  &printOut($output1);
}
if(!$res2)
{
  print "Missing tags/values in $bf1 w.r.t $bf2\n";
  &printOut($output2);
}
exit 1;

sub printOut ()
{
  my $out=shift || return;
  my $base=shift || "";
  foreach my $k (keys %$out)
  {
    my $v=$out->{$k};
    if($k eq "COMPARE_BUILDFILE_ERROR_MSG"){print "$base=> $v\n";}
    else
    {
      my $base1=$base;
      if(($k ne "deps") || ($base!~/^(bin|library)\:/))
      {
        if($base1){$base1.=":$k";}
        else{$base1="$k";}
      }
      &printOut($v,$base1);
    }
  }
}

sub cleanup ()
{ 
  my $bf=shift;
  my $data=shift || {};
  my $f=&SCRAMGenUtils::findBuildFileTag($data,$bf,"flags");
  foreach my $a (keys %$f)
  {
    foreach my $c (@{$f->{$a}})
    {
      foreach my $f1 (keys %{$c->{flags}})
      {
        my $v=$c->{flags}{$f1};
        if(ref($v) eq "ARRAY"){foreach my $x (@$v){if(exists $x->{q}){delete $x->{q};}}}
      }
    }
  }
  if(exists $data->{prodtype}){return;}
  foreach my $type ("bin", "library")
  {
    if(exists $bf->{$type})
    {
      foreach my $name (keys %{$bf->{$type}})
      {
        $data->{prodname}=$name;
	$data->{prodtype}=$type;
	&cleanup($bf,$data);
      }
    }
  }
}

sub compare ()
{
  my $data1=shift || ();
  my $data2=shift || ();
  my $tab=shift || "";
  my $seq=shift || {};
  
  my $ref1=ref($data1);
  my $ref2=ref($data2);
  my $res=1;
  if ($ref1 ne $ref2){$res=0;$seq->{COMPARE_BUILDFILE_ERROR_MSG}="MISMATCH REF:\"$ref1\"!=\"$ref2\"";}
  elsif(($ref1 eq "SCALAR") || ($ref1 eq ""))
  {if ($data1 ne $data2){$res=0;$seq->{COMPARE_BUILDFILE_ERROR_MSG}="MISMATCH VAL:\"$data1\"!=\"$data2\"";}}
  elsif ($ref1 eq "ARRAY")
  {
    my $count = scalar(@$data1);
    for(my $i=0;$i<$count;$i++)
    {
      $seq->{$i}={};
      my $x=$data1->[$i];
      my $rx=ref($x);
      if(($rx eq "SCALAR") || ($rx eq ""))
      {
        if(!&existsInArray($x,$data2)){$res=0;$seq->{$i}{COMPARE_BUILDFILE_ERROR_MSG}="MISSING VAL:\"$x\"";}
	else{delete $seq->{$i};}
      }
      elsif(!&compare($data1->[$i],$data2->[$i],"$tab  ",$seq->{$i})){$res=0;}
      else{delete $seq->{$i};}
    }
  }
  else
  {
    foreach my $k (keys %{$data1})
    {
      $seq->{$k}={};
      if(exists $data2->{$k})
      {
        push @output,"$tab  Checking:\"$k\"\n";
        if (!&compare($data1->{$k},$data2->{$k},"$tab  ",$seq->{$k})){$res=0;}
	else{delete $seq->{$k};}
      }
      else{$res=0;$seq->{$k}{COMPARE_BUILDFILE_ERROR_MSG}="Missing Key";}
    }
  }
  return $res;
}

sub existsInArray ()
{
  my $val=shift;
  my $array=shift;
  foreach my $x (@$array){if($x eq $val){return 1;}}
  return 0;
}
