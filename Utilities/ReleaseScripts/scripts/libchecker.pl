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
  getopts("rv");

  $libname = shift; chomp($libname);
  # Should check that library actually exists


  # Get the list of direct dependencies
  @LDDLIST = `ldd $libname`;

  foreach $akey (@LDDLIST) {
    chomp($akey);
    ($AAA,$BBB,$CCC) = split(' ',$akey);
    # Remove most extraneous stuff
    if (($BBB eq "=>") && !($AAA =~ /linux-gate.so/) && !($AAA =~ /libgcc_s.so/)) { 
      $ldddepmap{$AAA} = $CCC;
      $lddrevdepmap{$CCC} = $AAA;

      $lddneedmap{$AAA} = 0;
      $lddrevneedmap{$CCC} = 0;
    } 
  }

  # Loop over the list of direct dependencies and look at their dependencies
  for my $bkey ( keys %lddrevdepmap ) {
    #my $value = $lddrevdepmap{$bkey};
    #print "$bkey => $value\n";
    @LDDLIST2 = `ldd $bkey`;

    foreach $ckey (@LDDLIST2) {
      chomp($ckey);
      ($AAA,$BBB,$CCC) = split(' ',$ckey);
      # Remove most extraneous stuff
      if (($BBB eq "=>") && !($AAA =~ /linux-gate.so/) && !($AAA =~ /libgcc_s.so/)) { 
        $lddrevneedmap{$CCC} += 1;  # Mark this dep as needed by another dep
      } 
    }

  }

  # Get the list of unneeded libraries (includes some header lines)
  @UNUSEDLIST = `ldd -u -r $libname`;
 
  # Loop and print only unused libraries which are not deps of other deps
  # (i.e. those that must have been added as direct deps of the library 
  #  in question, not things that come in indirectly via its deps)
  foreach $dkey (@UNUSEDLIST) {
    chomp($dkey);
    $dkey =~ s/(^\s+|\s+$)//g; # Drop the whitespace
    if (exists($lddrevdepmap{$dkey})) {
      if ($opt_v) {print "Found unneeded library => $dkey\n"};
      #print "XXX library $dkey used ".$lddrevneedmap{$dkey}." times\n";
      if ($lddrevneedmap{$dkey}==0) {
        print "Unnecessary direct dependence ".$lddrevdepmap{$dkey}."\n";
      } else {
        #print "Used Dependence $dkey ".$lddrevneedmap{$dkey}." times\n";
      }
    } else {
      if ($opt_v) {print "Reject header line => $dkey\n"};
      if ($opt_v) {print "  With => ".$lddrevdepmap{$dkey}."\n"};
    }
  }
