#!/usr/bin/env perl
use File::Basename;
use lib dirname($0);
use Getopt::Long;
use SCRAMGenUtils;

$|=1;
if(&GetOptions(
	       "--dir=s",\$dir,
	       "--common",\$common,
	       "--help",\$help,
	      ) eq ""){print STDERR "#Wrong arguments.\n"; &usage_msg();}
if(defined $help){&usage_msg();}
if((!defined $dir) || ($dir=~/^\s*$/) || (!-d $dir)){print STDERR "Missing directory path where the newly auto generated BuildFiles exist.\n"; &usage_msg();}
if(defined $common){$common=1;}
else{$common=0;}
&process($dir);
exit 0;

sub process ()
{
  my $dir=shift || return;
  if(!-d $dir){return;}
  my %bfs=();
  foreach my $file (&SCRAMGenUtils::readDir("$dir",0))
  {
    my $fpath="${dir}/${file}";
    if(-d $fpath){&process($fpath);}
    elsif($file=~/^(.+?)BuildFile\.auto$/){$bfs{$file}=0;}
  }
  if(scalar(keys %bfs)==0){return;}
  print "Working on $dir\n";
  my %commontools=();
  if($common)
  {
    my $flag=0;
    foreach my $file (keys %bfs)
    {
      if(($flag) && (scalar(keys %commontools)==0)){last;}
      my $bf=&SCRAMGenUtils::readBuildFile("${dir}/${file}");
      foreach my $type ("bin", "library")
      {
        if(exists $bf->{$type})
        {
          foreach my $prod (keys %{$bf->{$type}})
	  {
	    my %local=();
	    if(exists $bf->{$type}{$prod}{deps}{use})
	    {foreach my $u (keys %{$bf->{$type}{$prod}{deps}{use}}){$local{$u}=1;}}
	    if($flag==0)
	    {
	      foreach my $u (keys %local){$commontools{$u}=1;}
	      $flag=1;
	    }
	    else{foreach my $u (keys %commontools){if(!exists $local{$u}){delete $commontools{$u};}}}
	  }
        }
      }
    }
  }
  my $mbf="${dir}/BuildFile.auto";
  my $outfile;
  open($outfile,">$mbf") || die "Can not open \"$mbf\" for writing.";
  if($common){foreach my $u (keys %commontools){print $outfile "<use name=$u>\n";}}
  my $c=scalar(keys %bfs);
  print "Files:$c\n";
  foreach my $file (keys %bfs)
  {
    my $infile;
    open($infile,"${dir}/${file}") || die "Can not open \"${dir}/${file}\" for reading.";
    my $line="";
    while($line=$line || <$infile>)
    {
      chomp $line;
      if($common)
      {
        foreach my $u (keys %commontools)
        {if($line=~/^\s*<\s*use\s+name\s*=\s*$u\s*>(.*)$/){$line=$1;}}
      }
      if($line!~/^\s*$/){print $outfile "$line\n";$line="";}
    }
    close($infile);
  }
  close($outfile);
}
	      
sub usage_msg() 
{
  my $script=basename($0);
  print "Usage: $script --dir <dir> [--common]\n\n";
  print "  --dir    <dir>  Path of the directory where the newly generated BuildFile.auto files exist.\n";
  print "  --common        To Search for common tools used by different products (library/bin) in a\n";
  print "                  test/bin area and add those tools once in the BuildFile.\n";
  exit 0;
}
