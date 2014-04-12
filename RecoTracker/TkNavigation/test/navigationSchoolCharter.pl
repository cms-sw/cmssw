#!/usr/bin/env perl

## Navigation school plotter:
##   Takes the output of the NavigationSchoolAnalyzer, and produces a '.dot' file
##   to be visualized with graphviz (http://www.graphviz.org/)
##
## Usage:
##   - edit the NavigationSchoolAnalyzer cfg.py to select which navigation school to print
##   - run the cfg.py for the NavigationSchoolAnalyzer
##   - perl -w navigationSchoolCharter.pl detailedInfo.log > yourSchool.dot
##   - graphviz -Tpng -o yourSchool.png < yourSchool.dot
##
## Output:
##   - red lines are 'inside-out' only (red = hot = bang = proton-proton collisions = in-out)
##   - blue lines are 'outside-in' only (blue = sky = cosmics = out-in)
##   - green lines are two-way links (with the arrow set inside-out)

use Data::Dumper;

## Parse the set of lines describing one layer
sub parseLayer {
    my $txt = shift;
    my ($be, $name) = ($txt =~ m/^(barrel|endcap) subDetector: (\w+)/m) or die "Can't parse layer block\n" . $txt . "\n";
    my ($l) = ($txt =~ m/^(?:wheel|layer): (\d+)/m) or die "Can't parse layer block\n" . $txt . "\n";
    my ($s) = ($txt =~ m/^side: (\w+)/m);
    $s = '' if $be ne "endcap";
    return sprintf('%s%s_%d', $name,$s,$l);
}

## Parse the set of lines describing all links starting from one layer
my %layers;
sub parseNavi {
   my $txt = shift;
   my $start = parseLayer($txt);
   my @outIn = (); 
   my @inOut = ();
   if ($txt =~ m/^\*\*\* INsideOUT CONNECTED TO \*\*\*\n((([a-z0-9]+.*\n)+-+\n)+)/m) {
        my $list = $1;
        #die "VVVVVVVVVVVVVVV\n$list\n^^^^^^^^^^^^^^^^^^^^\n";
        foreach (split(/---+/, $list)) { m/subDetector/ and push @inOut, parseLayer($_); }
   }
   if ($txt =~ m/^\*\*\* OUTsideIN CONNECTED TO \*\*\*\n((([a-z0-9]+.*\n)+-+\n)+)/m) {
        my $list = $1;
        #die "VVVVVVVVVVVVVVV\n$list\n^^^^^^^^^^^^^^^^^^^^\n";
        foreach (split(/---+/, $list)) { m/subDetector/ and push @outIn, parseLayer($_); }
   }
   $layers{$start} = { 'inOut' => [ @inOut ], 'outIn' => [ @outIn ] };
}

## Read input and parse it layer by layer
my $text = join('', <>);
while ($text =~ m/^(####+\nLayer.*\n([^#].*\n)+)/gm) {
    parseNavi($1);
}

#print Dumper(\%layers);

## Write the output in a graphviz syntax
print "digraph G {\n";
foreach my $k (sort(keys(%layers))) {
    print "$k\n";
    foreach my $l1 ( @{$layers{$k}->{'inOut'}} ) {
        my $color = 'red';
        if (grep($_ eq $k, @{$layers{$l1}->{'outIn'}})) {
            $color = 'darkgreen';
        }
        print "\t$k -> $l1 [color=$color]\n";
    }
    foreach my $l2 ( @{$layers{$k}->{'outIn'}} ) {
        next if (grep($_ eq $k, @{$layers{$l2}->{'inOut'}})) ;
        print "\t$k -> $l2 [color=blue]\n";
    }
}
print "}\n";
