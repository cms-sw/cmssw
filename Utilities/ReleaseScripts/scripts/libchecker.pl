#!/usr/bin/env perl 
#
#  Script to identify unused direct dependencies which are NOT unused direct
#  dependencies of the direct dependencies. (Somewhat kludgey and filled
#  with hardcoded stuff, but at least it is a start....)
#
#  Peter Elmer, Princeton University                         11 July, 2009
#
#############################################################################

  use Getopt::Std;
  use File::Basename;
  use lib dirname($0);
  use SCRAMGenUtils;
  getopts("rv");

  $libname = shift; chomp($libname);
  $tmpcache = $ENV{CMSSW_BASE}."/tmp/".$ENV{SCRAM_ARCH}."/libcheck";
  system("mkdir -p $tmpcache");
  # Should check that library actually exists

  # Get the list of direct dependencies
  
  my $LddData=&getLddData($libname,$tmpcache);
  foreach my $data (@$LddData) {
    my $AAA=$data->[0]; my $CCC=$data->[1];
    $ldddepmap{$AAA} = $CCC;
    $lddrevdepmap{$CCC} = $AAA;
    $lddneedmap{$AAA} = 0;
    $lddrevneedmap{$CCC} = 0;
  }

  for my $bkey ( keys %lddrevdepmap ) {
    my $LddData=&getLddData($bkey,$tmpcache);
    foreach my $data (@$LddData) {
      my $AAA=$data->[0]; my $CCC=$data->[1];
      $lddrevneedmap{$CCC} += 1;  # Mark this dep as needed by another dep
    }
  }

  # Get the list of unneeded libraries (includes some header lines)
  @UNUSEDLIST = `ldd -u -r $libname`;
 
  # Loop and print only unused libraries which are not deps of other deps
  # (i.e. those that must have been added as direct deps of the library 
  # in question, not things that come in indirectly via its deps)
  foreach $dkey (@UNUSEDLIST) {
    chomp($dkey);
    $dkey =~ s/(^\s+|\s+$)//g; # Drop the whitespace
    if (exists($lddrevdepmap{$dkey})) {
      if ($opt_v) {print "Found unneeded library => $dkey\n"};
      if ($lddrevneedmap{$dkey}==0) {
        print "Unnecessary direct dependence ".$lddrevdepmap{$dkey}."\n";
      }
    } else {
      if ($opt_v) {print "Reject header line => $dkey\n"};
      if ($opt_v) {print "  With => ".$lddrevdepmap{$dkey}."\n"};
    }
  }
  
  sub getLddData() {
    my ($lib,$cache)=@_;
    my $cfile=$cache."/".basename($lib);
    my $data=[];
    my $mtime=(stat($lib))[9];
    if ((-f $cfile) && ($mtime==(stat($cfile))[9])){$data=&SCRAMGenUtils::readHashCache($cfile);}
    else{
      my @LDDLIST = `ldd $lib`;
      foreach my $akey (@LDDLIST) {
        chomp($akey);
        $akey=~s/\s+\(.+\)\s*$//;
        if ($akey=~/^\s*\/.+/){next;}
        my ($AAA,$BBB,$CCC) = split(' ',$akey);
        if (($BBB eq "=>") && ($CCC=~/^\//) && ($AAA!~/(linux-gate\.so|linux-vdso\.so|libgcc_s.so)/)) {
	  push @$data,[$AAA,$CCC];
        }
      }
      &SCRAMGenUtils::writeHashCache($data,$cfile);
      utime($mtime, $mtime, $cfile);
    }
    return $data;
  }
