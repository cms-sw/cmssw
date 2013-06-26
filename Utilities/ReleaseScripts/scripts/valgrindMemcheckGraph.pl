#!/usr/bin/env perl
# $Id: valgrindMemcheckGraph.pl,v 1.5 2013/03/15 18:02:18 muzaffar Exp $
# Created: June 2007
# Author: Gioivanni Petrucciani, INFN Pisa
#
use strict;
use warnings;
use Data::Dumper;
use Date::Format;
use Getopt::Long;

my $mstart = qr/^==\d+== (\S.*? bytes) in \S+ blocks are (.*?) in loss (record \S+ of \S+)/;
my $mtrace = qr/^==\d+== \s+(?:at|by)\s.*?:\s+(.*?)\s\((.*)\)/;
my $version = undef; #"CMSSW_1_5_0_pre3";
my @showstoppers = qq(libFWCoreFramework);

my %presets = (
    'trash' => [ '__static_initialization_and_destruction_0', 'G__exec_statement', 'dlopen\@\@GLIBC_2', '_dl_lookup_symbol_x' ],
    'fwk' => [ qw(EventSetup  ESProd  castor  ROOT  Pool  Reflex  PluginManager  RFIO  xerces  G_) ],
    'tom'  =>  [ qw(EventSetup ESProd castor ROOT Pool Reflex PluginManager RFIO xerces G_ libGraf createES),
                 qw(Streamer python static MessageLogger ServiceRegistry) ],
    'prod' => [ '::(produce|filter)\(\s*edm::Event\s*&' , '::analyze\(\s*(?:const\s+)?edm::Event(?:\s+const)?\s*&' ],
    'prod1' => [ '::produce\(\s*\w+(?:\s+const)?\s*&\w*\s*\)' ],
    'prod1+' => [ '::produce\(\s*\w+(?:\s+const)?\s*&\w*\s*\)', 'edm::eventsetup::DataProxyTemplate<'  ],
);
my $preset_names = join(', ', sort(keys(%presets)));

my @trace = (); my @libs = (); my @presets = (); my @dump_presets = ();
my $help = '';  my $all = ''; my $onecolumn = ''; my $outdir = $ENV{'HOME'} . "/public_html/leaks";

GetOptions(
        'rel|release|r=s' => \$version,
        'libs|l=s' => \@libs,
        'trace|t=s' => \@trace,
        'stopper|showstopper'=> \@showstoppers,
        'onecolumn|1' => \$onecolumn,
        'all|a' => \$all,
        'preset=s'   => \@presets,
        'dump-preset=s'   => \@dump_presets,
        'out=s' => \$outdir,
        'help|h|?' => \$help);

if ($help) {
        print <<_END;
   Usage: valgrindMemcheckParser.pl [ --rel RELEASE ] 
                 [ --libs lib1,lib2,-lib3 ]
                 [ --trace match1,match2,-match3 ]
                 [ --stopper lib1,lib2 ]
                 [ --preset name,name,-name,+name,... ]
                 [ --all ]
                 [ --onecolumn ]
                 [ --out dir ]
                 logfile [ logfile2 logfile3 ... ]
        
  It will output a set of files in a single folder, specified through '--out' option

  If no input file is specified, reads from standard input.

  It needs a graphviz dot program with PNG support, you can get mine from AFS with:
    export LD_LIBRARY_PATH=/afs/cern.ch/user/g/gpetrucc/scratch0/graphviz/lib:\${LD_LIBRARY_PATH}
    export PATH=/afs/cern.ch/user/g/gpetrucc/scratch0/graphviz/bin:\${PATH}

  FILTERS
    --libs: coma-separated list of libs to require in the library stack trace 
            (or to exclude, if prefixed by a "-"). 
            Can be used multiple times. 
            Abbreviation is "-l" 
    --trace: coma-separated list of regexps to match in the stack trace
             (or to exclude, if prefixed by a "-"). 
             Can be used multiple times.
             Abbreviation is "-t" 
    --stopper: coma-separated list of libs to cut the stack trace at;
               libFWCoreFramework.so is in by default.
               set it to "none" to never break stack trace.
               use full library name.
    --preset: use a specified preset filter for exclusion or inclusion.
        filter names are $preset_names
        --preset name : require at least one of the regexps in "name" to match
                        in the stack trace
        --preset +name: requires all the regexp to match the in each stack trace 
                        (not all on the same stack trace element, of course)
        --preset -name: exclude the event if at least one regexp in name matches
                        in the stack trace
        to get the contents of a preset use "--dump-preset name" 

    --all: show all leaks, skipping any filter
             Abbreviation is "-a" 

    Note: you can use PERL regexps in "libs", "trace" 

  HTML & LINKING OPTIONS
    --onecolunm: output things in one column, avoiding the column with the library name,
                 for easier cut-n-paste in savannah
                 an alias is "-1"
    --rel: CMSSW_*, or "nightly" (default: $version) to set LXR links
           aliases are "--release" and "-r"
    --link-files: if set to true (default is false), links to Uppercase identifiers are
                  made using filename search instead of identifier search)
      [NOT IMPLEMENTED]

  HELP
    --help : prints this stuff (also -h, -?)
    --dump-preset name: dumps the content of a preset and exit
    --out  : output path (defaults to  ~/public_html/leaks/)
_END
    exit;  
}
if (@dump_presets) {
    foreach my $ps (@dump_presets) {
        print "Preset $ps: \n";
        print map("\t * '$_'\n", @{$presets{$ps}});
        print "\n";
    }
    exit;
}

#if ($version eq 'nightly') { $version = time2str('%Y-%m-%d',time()); }
@libs = split(/,/, join(',',@libs));
@trace = split(/,/, join(',',@trace));
@presets = split(/,/, join(',',@presets));
@showstoppers= split(/,/, join(',',@showstoppers));
if (grep($_ eq 'none', @showstoppers)) { @showstoppers = (); }
my @trace_in  = map (qr($_), grep ( $_ !~ m/^-/, @trace ));
my @trace_out = map (qr($_), grep ( s/^-//g, @trace ));
my @libs_in   = map (qr($_), grep ( $_ !~ m/^-/, @libs ));
my @libs_out  = map (qr($_), grep ( s/^-//g, @libs ));
my %stopmap = (); foreach (@showstoppers) { $stopmap{$_} = 1; }
my %presets_c = ();
foreach my $ps (keys(%presets)) { $presets_c{$ps} = [ map(qr($_), @{$presets{$ps}}) ] ; }
my @leaks = ();

sub cfilter {   
    my @trace = @{$_->{'trace'}};
    my $rx; 
    foreach $rx (@trace_in ) { return 0 unless ( grep( $_->[0] =~ $rx, @trace) ); }
    foreach $rx (@trace_out) { return 0 if     ( grep( $_->[0] =~ $rx, @trace) ); }
    foreach $rx (@libs_in )  { return 0 unless ( grep( $_->[1] =~ $rx, @trace) ); }
    foreach $rx (@libs_out)  { return 0 if     ( grep( $_->[1] =~ $rx, @trace) ); }
    foreach my $ps (@presets) {
        my ($op, $name) = ($ps =~ m/^([+\-]?)(\S+)/);
        if ($op eq '') {
            my $ok = 0;
            foreach $rx (@{$presets_c{$name}}) {
                if ( grep( $_->[0] =~ $rx, @trace) ) { $ok = 1; last; }
            }
            return 0 unless $ok;
        } elsif ($op eq '-') {
            foreach $rx (@{$presets_c{$name}}) {
                return 0 if     ( grep( $_->[0] =~ $rx, @trace) );
            }
        } elsif ($op eq '+') {
            foreach $rx (@{$presets_c{$name}}) {
                return 0 unless ( grep( $_->[0] =~ $rx, @trace) );
            }
        }
    }
    return 1;
}

sub realsize {
        my ($num) = ($_[0] =~ m/^([0-9,]+)/) or return 0;
        $num =~ s/,//g;
        return eval($num);
}
sub fformat {
        my $vstring = (defined($version) ? "v=$version;" : "");
        my $func = &escapeHTML($_[0]);
        $func =~ s!(\b[A-Z]\w\w\w\w+)!<a class='obj' href='http://cmssdt.cern.ch/SDT/lxr/ident?${vstring}i=$1'>$1</a>!g;
        $func =~ s!::(\w+)\(!::<a class='func' href='http://cmssdt.cern.ch/SDT/lxr/ident?${vstring}i=$1'>$1</a>(!g;
        return $func;
}
sub escapeHTML {
        my $data=$_[0];
        $data =~ s!&!&amp;!g;
        $data =~ s!<!&lt;!g;
        $data =~ s!>!&gt;!g;
        $data =~ s!"!&quot;!g;
        return $data;
}

BEGIN {
    my $id = 0;
    sub toId { return sprintf("ID\%04d", $id++);  }
}

my %legend = ();
sub pretty {
        my ($func, $id, $count, $size) = @_;
        my ($nm) = ($func =~ m/^\s*([^\s\(']+)/);
        $nm = "Unknown" unless $nm;
        $nm .= sprintf('\n(%d leaks/ %.0f bytes)', $count, $size);
        $legend{$id} = $func;
        return $nm;
}

while (<>) {
  if (/$mstart/) {
        my ($size, $status, $record) = ($1, $2, $3);
        #print STDERR "\nLoss size=$size, status=$status\n" if $#leaks < 20;

        my %libs = (); my @trace = ();
        while (<>) {
                my ($func, $lib) = /$mtrace/ or last;
                #$lib =~ s/^in \S+\/((?:lib|plugin)\w+)\.so/$1/ or next;
                $lib =~ s/^in \S+\/((?:lib|plugin)\w+)\.so/$1/; # or $lib = "";
                last if $stopmap{$lib};
                $libs{$lib} = 1; push @trace, [$func, $lib];
                die "I'm not defined" unless (defined($func) and defined($lib));
                #print STDERR "   lib=$lib, func=$func\n"  if $#leaks < 20;
        }

        push @leaks, { 'size'=>$size, 'realsize' => realsize($size), 'status'=>$status, 'record'=>$record, 'libs'=>[keys(%libs)], 'trace'=>\@trace};
  }
}


#print STDERR Dumper(\@leaks);
my @gleaks = ($all ? @leaks : grep ( cfilter($_), @leaks));
my @sleaks = sort {$b->{'realsize'} <=> $a->{'realsize'}} @gleaks ;
my $count = scalar(@sleaks); 
print STDERR "Selected $count leaks of " , scalar(@leaks) , ".\n";

my %leak_map = ();

my $idx = 0;
foreach my $l (@sleaks) {
        my %L = %{$l}; $idx++;
        my $top = undef;
        foreach my $sf (reverse(@{$L{'trace'}})) {
                my $func = $sf->[0];
                next if $func =~ /operator\s+new/;

                unless (defined($leak_map{$func})) {
                        $leak_map{$func} = { 'count'=>0, 'size'=> 0, 'links'=>{}, 'depcount'=>0, 'id'=>toId($func), 'items'=>{} };
                }

                $leak_map{$func}->{'count'}++;
                $leak_map{$func}->{''}++;
                $leak_map{$func}->{'size'} += $L{'realsize'};

                $leak_map{$func}->{'items'}->{$idx} = 1;
                
                if (defined($top)) {
                    $leak_map{$func}->{'links'}->{$top} = 1;
                    $leak_map{$top}->{'depcount'}++;
                }
                $top = $func;
        }
}

mkdir $outdir unless (-d $outdir);

open DOT, "> $outdir/leak.dot";
print DOT "digraph G { \n";
#print DOT "\trankdir=LR\n";
foreach my $func (keys(%leak_map)) {
        if (!defined($leak_map{$func})) { die "BOH ? $func " . Dumper(\%leak_map); }
        my %L = %{$leak_map{$func}};
        my $nm = pretty($func,$L{'id'},$L{'count'},$L{'size'});
        my $col = ($leak_map{$func}->{'count'} > $leak_map{$func}->{'depcount'} ? 'orange' : 'green');
        my $url = "#" .  $L{'id'};
        print DOT "\t", sprintf('%s [ shape=rect label="%s" style=filled fillcolor=%s URL="%s"] ', $L{'id'}, $nm, $col, $url), "\n";
        foreach my $link (keys(%{$L{'links'}})) {
                print DOT "\t\t",  $L{'id'}, " -> ", $leak_map{$link}->{'id'}, "\n";
        }
}
print DOT "}\n";
close DOT;

open CSS, "> $outdir/valgrindMemcheckParser.css";
print CSS <<EOF;
th.header { font-size: large; color: red; text-align: left; padding-top: 1em;}
td.libs   { font-size: normal; color: ; padding-left: 2em; }
tr.trace  { font-size: small; }
td.func   { font-family: "Courier New", "courier", monospace; text-indent: -2em; padding-left: 2.5em; }
td.lib    { font-family: "Courier New", "courier", monospace; color: navy;}
a         { text-decoration: none; }
a.obj     { color: #007700; }
a.func    { color: #000077; }
a:hover  { text-decoration: underline; }
EOF
close CSS;

open HTM, "> $outdir/index.html";

my $imgmap = join('', qx(dot -Tcmapx < $outdir/leak.dot));
system("dot -Tpng -o $outdir/leak.png < $outdir/leak.dot");

my $footer = "Done at " . scalar(localtime());

print HTM <<EOF;
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" 
   "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en">
<head>
        <title>Valgrind MemCheck Graph </title>
        <link rel='stylesheet' type='text/css' href='valgrindMemcheckParser.css' />
</head>
<body>
        <h1>Valgrind MemCheck Graph ($count leaks)</h1>

<h3>Overview</h3>
<ul>
        <li><a href="#plot">Leak graph</a></li>
        <li><a href="#legend">Graph legend</a></li>
        <li><a href="#detail">Leak details</a></li>
</ul>
<h1><a name="plot" id="plot">Plot</a> (<a href="leak.png">PNG</a>)</h1>
<p style="text-align: center">
      <img src="leak.png" alt="Leak map" usemap="#G" />
      $imgmap
</p>
<h1><a name="legend" id="legend">Legend</a></h1>
<dl>
EOF
foreach my $id (sort(keys(%legend))) {
        print HTM "\t<dt class='id'>Frame <a name=\"$id\" id=\"$id\">$id</a></dt>\n";
        print HTM "\t<dd class='func'>Function: <tt>" , fformat($legend{$id}), "</tt></dd>\n";
        print HTM "\t<dd class='refs'>Leaks: ", 
                join(', ', map(sprintf('<a href="#LK%04d" class="leak">#%d</a>', $_, $_), 
                                sort(keys(%{$leak_map{$legend{$id}}->{'items'}}))
                           )), "</dd>\n";
}
print HTM <<EOF; 
</dl>

<h1><a name="detail" id="detail">Detailed leak list</a></h1>
<table width="100%">
EOF
$idx = 0;
foreach my $l (@sleaks) {
        my %L = %{$l}; $idx++;
        my $colspan = ($onecolumn ? 1 : 2);
        my $id = sprintf("LK%04d", $idx);
        print HTM "<tr class='header'><th class='header' colspan='$colspan'><a name=\"$id\" id=\"$id\">Leak $idx</a>: $L{size} $L{status} ($L{record}) <a href=\"#$id\">[href]</a></th></tr>\n";
        foreach my $sf (@{$L{'trace'}}) {
                print HTM "<tr class='trace'><td class='func'>"  . fformat($sf->[0]) . "</td>";
                print HTM "<td class='lib'>" . $sf->[1]. "</td>" unless $onecolumn;
                print HTM "</tr>\n";
        }
}
print HTM <<EOF;
</table>

<p class='footer'>$footer</p>
</body>
</html>
EOF


