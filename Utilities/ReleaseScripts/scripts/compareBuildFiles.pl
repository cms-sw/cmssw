#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use SCRAMGenUtils;

$|=1;

my $bf1=shift;
my $bf2=shift;
my $html=shift || 0;
my @output=();

if(!defined $bf1){die "Usage: $0 <buildfile1> <buildfile2>\n";}
if(!defined $bf2){die "Usage: $0 <buildfile1> <buildfile2>\n";}

if(!-f $bf1){die "No such file \"$bf1\".";}
if(!-f $bf2){die "No such file \"$bf2\".";}

my $rel=`/bin/pwd`; chomp $rel; $rel=&SCRAMGenUtils::fixPath($rel);
if ($bf1!~/^\//){$bf1=&SCRAMGenUtils::fixPath("${rel}/${bf1}");}
if ($bf2!~/^\//){$bf2=&SCRAMGenUtils::fixPath("${rel}/${bf2}");}

my $buildfile=$bf2; $buildfile=~s/^$rel\/src\///;

$rel=&SCRAMGenUtils::scramReleaseTop($rel);
&SCRAMGenUtils::init($rel);
my $refbf1=&SCRAMGenUtils::readBuildFile($bf1);
my $refbf2=&SCRAMGenUtils::readBuildFile($bf2);
&findDiff($refbf1,$refbf2);
exit 0;

sub FixToolName()
{
  my $name=shift;
  my $tool=lc($name);
  if (-e "${rel}/.SCRAM/$ENV{SCRAM_ARCH}/timestamps/${tool}"){return $tool;}
  return $name;
}

sub printHead()
{
  if ($html)
  {
    print "<html><body><center><h2>Difference for $buildfile</h2></center><pre>\n";
  }
}

sub printTail()
{
  if ($html){print "</pre></body></html>\n";}
}

sub printLine ()
{
  my $line=shift;
  if ($html)
  {
    my $tag=shift || "";
    if ($tag){print "<$tag>";}
    print "$line\n";
    if ($tag){print "</$tag>";}
  }
  else{print "$line\n";}
}

sub findDiff ()
{
  my $bf1=shift;
  my $bf2=shift;
  my $u1={}; my $u2={};
  my $hasprod=0;
  for(my $i=1;$i<=2;$i++)
  {
    my $u={};
    my $bf=$bf1;
    if ($i==2){$bf=$bf2;}
    foreach my $type ("","bin", "library")
    {
      my $data={};    
      if ($type eq ""){&getAllUse($bf,$data,$u);}
      elsif(exists $bf->{$type})
      {
	foreach my $name (keys %{$bf->{$type}})
        {
          $hasprod=1;
	  $data->{prodname}=$name;
	  $data->{prodtype}=$type;
	  &getAllUse($bf,$data,$u);
        }
      }
    }
    if ($i==1){$u1=$u;}
    else{$u2=$u;}
  }
  my $diff={};
  foreach my $t (keys %$u1)
  {
    foreach my $p (keys %{$u1->{$t}})
    {
      foreach my $u (keys %{$u1->{$t}{$p}})
      {
	if((exists $u2->{$t}) && (exists $u2->{$t}{$p}) && (exists $u2->{$t}{$p}{$u})){delete $u1->{$t}{$p}{$u}; delete $u2->{$t}{$p}{$u};}
	else{$diff->{$t}{$p}{"+"}{$u}=1;}
      }
    }
  }
  foreach my $t (keys %$u2)
  {
    foreach my $p (keys %{$u2->{$t}})
    {
      foreach my $u (keys %{$u2->{$t}{$p}}){$diff->{$t}{$p}{"-"}{$u}=1;}
    }
  }
  &printHead();
  foreach my $t (sort keys %$diff)
  {
    foreach my $p (sort keys %{$diff->{$t}})
    {
      if ($hasprod)
      {
        if ($t eq "common"){&printLine("* Common Non-Export Section:","b");}
        else{&printLine("* $p:","b");}
      }
      else{&printLine("* Non-Export Section:","b");}
      foreach my $a (sort keys %{$diff->{$t}{$p}})
      {
	foreach my $u (sort keys %{$diff->{$t}{$p}{$a}})
	{
	  print "  $a $u\n";
	}
      }
      print "\n\n";
    }
  }
  &printTail();
}

sub getAllUse ()
{
  my $bf=shift;
  my $data=shift;
  my $use=shift;
  my $f=&findTag($data,$bf,"use");
  my $type="common"; my $name="all";
  if (exists $data->{prodname}){$type="prods"; $name=$data->{prodname};}
  $use{$type}{$name}={};
  foreach my $c (@{$f})
  {
    foreach my $u (keys %{$c->{use}}){$use->{$type}{$name}{&FixToolName($u)}=1;}
  }
}

sub findTag ()
{
  my $data=shift;
  my $bf=shift;
  my $tag=shift;
  my $d=shift || [];
  my $pt=$data->{prodtype};
  my $pn=$data->{prodname};
  if ($pt eq ""){if(exists $bf->{$tag}){push @{$d},$bf;}}
  elsif((exists $bf->{$pt}) && (exists $bf->{$pt}{$pn}) && (exists $bf->{$pt}{$pn}{deps}))
  {&findTag({},$bf->{$pt}{$pn}{deps},$tag,$d);}
  return $d;
}
