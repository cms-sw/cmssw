Here's a quick but not very brief summary of how to invoke the tools:

ignominy
--------

This is the scanner.  The options it accepts are:
  --root=DIR        Define DIR as the root of the area
                    to analyse

  --build-root=DIR  Define DIR as relative directory
                    from the --root DIR as the build
                    area root

  --src-root=DIR    Like --build-root but for sources.

  --bindir=DIR      Define DIR as the absolute path
                    of the --root DIR's binaries dir.
  --libdir=DIR      Like --bindir but for libraries
  --incdir=DIR      Like --bindir but for generated
                    header files (if not in package
                    specific directories).

  --verbose         Be incredibly verbose (to stderr)

  --skip=makedeps   Skip .d file analysis
  --skip=source     Skip #include scanning
  --skip=binaries   Skip binaries analysis
                    (= ldd/nm/symbol resolution)

  --conf=FILE       Read configuration file FILE

  --mode=flat       Assume a flat structure:
                      --bindir => --root + /bin
                      --libdir => --root + /lib
                      --incdir => --root + /include
                    Otherwise assumes SCRAM structure
                    for options not specified.

The configuration files can use the following:
  scan libs DIR-OR-FILE [DIR-OR-FILE...]
     Add DIR-OR-FILE to the library analysis.  By default
     Ignominy analyses --bindir and --libdir contents so
     those don't need to be specified.

  scan bins DIR-OR-FILE [DIR-OR-FILE...]
     Like `scan libs' but for programs.  Doesn't really
     matter since they get treated the same anyway.

  scan src DIR [DIR...]
     Scan DIR (recursively) for #include analysis.  By
     default Ignominy scans the --src-root area, so this
     doesn't need to be specified.  Basically meant for
     scanning "external" software (e.g. Anaphe) to get
     more accurate dependencies.  Within --src-root
     automatic package matches will translate paths to
     packages.  For external software you'll probably
     need `match' directives to map DIR to a package
     name.

  scan reject REGEXP [REGEXP...]
     While scanning sources for #includes, reject file
     names (and directory names) that match REGEXP.  Use
     this to disable the scanning of test directories etc.
     Note that REGEXP is matched against the full path so
     you'll have to be careful not to match too much.

  symbol prefer REGEXP
     For multiply defined symbols, prefer a binary that
     matches REGEXP.  Note that when multiple `symbol
     prefer's are given, they all count; there is no
     ordering.

  symbol ignore TYPE-REGEXP NAME-REGEXP
     Ignore symbols that match TYPE-REGEXP for type (as
     reported by nm: a single uppercase letter) and
     NAME-REGEXP for name.  If you want a specific symbol,
     be careful to anchor the regexp with '^' and '$' to
     ensure you are not matching in random place in some
     long mangled C++ name.

  match REGEXP:PERL-EXPRESSION
     If REGEXP matches the path (of source file or binary),
     use the eval'ed value of PERL-EXPRESSION as the package.
     PERL-EXPRESSION can use the $1, $2, etc. from the match.

  output ignore package REGEXP
     On analysis output ignore packages that match REGEXP.
     Normally you want to use this very little, and exclude
     packages in the diagram generation.

  output ignore binary REGEXP
     Like `output ignore package' but restricts which
     binaries get reported for a package.

  alias package ALIAS:NAME
     When outputting package dependencies, pretend that
     ALIAS is actually called NAME.  Can be used to merge
     several package names into one logical package in the
     reporting.

  reject include REGEXP
     Don't try to look for owner package for #includes that
     match REGEXP.

  search include PACKAGE-REGEXP:FILE-REGEXP:PERL-EXPRESSION
     If a #include can't be found from transitive dependencies
     (*.d files), look for it with this rule.  The package in
     which #include is must match PACKAGE-REGEXP.  The file
     #included must match FILE-REGEXP.  The file to look for
     is given by eval'ling PERL-EXPRESSION, which may use the
     match results ($1, $2, ...) as well as the following:
       $name        full package name (matched PACKAGE-REGEXP)
       $root        --root value
       $build_root  --build-root value
       $src_root    --src-root value
       $incdir      --incdir value
       $bindir      --bindir value
       $libdir      --libdir value

     If evaluating the expression results in a relative path,
     it is prefixed with the source directory of the file
     which has the #include.

     Each `search include' is considered in the order they are
     given in the configuration files and only the first match
     is considered.  Hence the order is important.  Sometimes
     it is a good idea to restrict the package that can match
     the search rule.  Note that improper use of this directive
     can easily result in completely false dependencies being
     generated.

  option match pre-gen
  option match post-gen
     Put subsequent `match' expression before/after the
     automatically generated match rules (the automatic ones
     are generated for each package found in the area source
     tree).

  option flat
     Same as giving --mode=flat on the command line.

  option define package PERL-EXPRESSION
     Define the rule that determines what in the source
     tree is a package.  PERL-EXPRESSION should evaluate
     to true if (and only if) $dir/$file is a package.
     $relative can also be used; it will be true if $dir
     contains more than one level inside the source tree.

  option define package-type PERL-EXPRESSION
     Define the rule that determines the type of the package.
     The outcome should be "subsystem" or "leaf" (these perl
     strings).  The full name of the package is $path; the
     expression may also use the hash %packages, where the
     keys of the hash are the full names of packages.

  option define package-match-rule PERL-EXPRESSION
     Eval PERL-EXPRESSION for each package and add the
     result as the path `match' for that package.  If more
     than one `package-match-rule' is given, they are all
     evaluated in the order given.  The order in which the
     `package-match-rule's are evaluated for the packages
     is undefined.  The variables that can be used in the
     PERL-EXPRESSION are like for `search include' except
     that $name is $fullname (the full name of the package
     and at the same time relative path from the area's
     source root).

  option define package-build-dir PERL-EXPRESSION
     Eval PERL-EXPRESSION to get the build directory for
     the package whose full name is $fullname (= relative
     path from the area's source root).  If not defined,
     assumes it is $build_root/$fullname.  Used to locate
     the .d files for the package.

  option define reject-binary PERL-EXPRESSION
     If eval'ling PERL-EXPRESSION results in true, then
     the binary $name in directory $dir should not be
     scanned for symbol information.

  import FILE
     Import another configuration file FILE (at that point
     in processing).


Ignominy produces on standard output a dependency database of the
format:
  #### (# x 80)
  # SECTION NAME
  PACKAGE:
  <SPACES> DEPENDS-ON-PACKAGE

  #### (# x 80)
  # SECTION NAME
  PACKAGE (BINARY):
  <SPACES> DEPENDS-ON-PACKAGE

Logical dependencies
--------------------

The following three tools also accept logical dependency files in
addition to the ignominy output.  The idea is to put into the logical
dependency files the dependencies that you know to exist in the software
but which are not detected by ignominy.  The format of the logical
dependency files are thus:
  # COMMENT
  REASON: [-> TARGET-PACKAGE [,TARGET-PACKAGE ...]]
  <SPACES> SOURCE-PACKAGE [-> TARGET-PACKAGE [,TARGET-PACKAGE...]]

The interpretation is: there is a dependency from SOURCE-PACKAGE to
TARGET-PACKAGE for REASON (that is, source uses target).  Each REASON
has its own dependencies; the TARGETs on the REASON line gives the
default target packages.  The TARGETs on the SOURCE line takes
precedence over the TARGETs on the REASON line.

Where logical dependencies are accepted, you will need to give a regular
expression that matches on REASONs.  Usually one would just request all
REASONs.

deps2dot
--------

This program takes the dependency database created by ignominy and turns
it into a DOT file that can be fed to `dot' from AT&T's graphviz
package.  If used with the `--help' option, prints out a brief usage.
The meaning of the options is:

  --group=FILE   Define package groups via FILE.  It has the
                 format `group NAME REGEXP [REGEXP...] -- H S V'.
                 Here all packages matching REGEXP(s) will be
                 in group NAME and coloured H S V (the text will
                 be white if the Y in YIQ coordinates is < .5,
                 black otherwise).

  --rename=FILE  Define package renaming (applied after --group).
                 This can be used to mangle the labels for package
                 nodes on the diagram (e.g. to shorten them). The
                 file has format `REGEXP:PERL-EXPRESSION'; if
                 REGEXP matches the package name, PERL-EXPRESSION
                 is eval'led to get the new name.

  --url=FILE     Define package node URL mapping.  The file has
                 format `REGEXP:PERL-EXPRESSION'; if REGEXP matches
                 the package name (before --rename), then the
                 expression is eval'led to get the URL for the
                 node.  The URL is available in the DOT PS output.
                 (See also eps2gif and ps2map to get the values to
                 GIF/HTML maps.)

  --ratio=RATIO  Define page ratio RATIO into the DOT file.
                 Default is to let DOT choose the page geometry
                 according to the graph.

  --concentrate=yes
  --concentrate=no
                 Define whether edges will be concentrated
                 (merged).  Default is yes.

  --shape=SHAPE  Define the shape to be used for the nodes.
                 It must be shape known by DOT.

The other arguments are:
  REPORT-FILE REPORT-PART DOT-TITLE INCLUSIVE? [RE...]
  [! RESTRICT-INCLUSIVE? RESTRICT-RE...]
  [-- EXTRA EXTRA-INCLUSIVE? EXTRA-RE...]

REPORT-FILE      is the ignominy dependency report.
REPORT-PART      defines a regular expression to match the
                  "SECTION NAME" from the report (see above
                   on the report syntax).
DOT-TITLE        is the diagram title
INCLUSIVE?       should be `yes' if the REs should be taken
                  to be inclusive (to define which packages
                  to select) or `no' for the opposite (to
                  define which package not to select)
RE               regexps to match on the package names; if
                 none are given, '.*' is assumed

RESTRICT-INCLUSIVE?
                 if restrictions are given (after `!'), this
                  defines whether the restriction cut regexps
                  are inclusive (`yes') or not (`no').  If
                  inclusive, only edges that point to packages
                  matched by the regexps are considered.  If
                  not inclusive, only edges not pointing to
                  packages matched by the regexps are taken.
RESTRICT-RE      restriction cut regexps; default is '.*'

EXTRA            if extra logical dependencies are given (after
                 '--'), this defines the file containing them.
EXTRA-INCLUSIVE? defines whether the reason regexps are inclusive
EXTRA-RE         regular expressions to be matched on the reason
                  names; default is `.*'

deps2metrics
------------

This program takes the dependency database created by ignominy and turns
it into a metrics as defined by Lakos' book.  The output is produced to
standard output in the the same format as in Lakos' tools.

Doesn't take any options other than `--help'; arguments are like with
`deps2dot'.

xmerge
------

Munges and outputs extra dependency statistics from #include files and
symbols used.  It can produce the output in three formats: text, html or
side-by-side PostScript plot; in each case it produces a different view
of the data.  The HTML output has comprehensive cross-referencing
tables.  The text output includes only the summary table (last in the
HTML output).  The side-by-side shows two columns, with rows being the
packages; the left column show the outgoing edges of the package, and
the right column shows the incoming edges, and the density of the lines
in between shows something about the package relation density.  Logical
dependencies are shown in all three; in the latter one as gray lines
(hard dependencies are shown in black).

The tool takes the --group (for side-by-side, defines the colouring and
ordering of the groups), and --rename (all forms) options as defined for
`deps2dot', as well as the standard `--help' option.  The arguments
taken are:
  INCS LIBS INCLUSIVE? [RE...]
  [! RESTRICT-INCLUSIVE? RESTRICT-RE...]
  [-- EXTRA EXTRA-INCLUSIVE? EXTRA-RE...]

INCS is the include dependency statistics, LIBS is similar for symbols
(how many #includes/symbols used from a package to another).  The other
arguments are as for `deps2dot' and `deps2metrics'.


Others
------

For the other utilities, feel free to peruse the source to understand
how they function.  They are mostly generic tools you might well find
use for elsewhere (I do).  A brief summary of each follows:
  eps2gif        Converts EPS or PS files to GIF files;
                 allows size limitation, and is able to
                 cache away how the image was mutated
                 (cropped and scaled) in the conversion

  rescale        Rescale PS files to some maximum size;
                 also may rotate the file if fits into
                 the given size better that way.  Also
                 adjusts ADSC PS comments.

  ps2map         Convert PDF hyperlinks in PS file into
                 a HTML map.  Uses information in the
                 format eps2gif stashed it away.

  map2html       Convert `dot'-generated map files into
                 HTML maps (not used).

  graysearch     Search for colours in the Y-axis of the
                 YIQ colour space.  Reports HSV values of
                 each colour found in .1 steps on the Y
                 axis.  YIQ colour space has all grayscale
                 information on the Y axis, hence this
                 utility can be used to search for HSV
                 values that print to separable grays on
                 black-and-white printers.

  hsv2rgb        Convert HSV colour values to RGB values.
