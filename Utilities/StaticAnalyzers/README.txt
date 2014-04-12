
=== CLANG CMS ===

This are the CMS clang static analyzer checkers to crawl C/C++ code for constructs which my become problematic when running multi-threaded code or do violate CMS Framework Rules.

by Thomas Hauth - Thomas.Hauth@cern.ch

This is still in early development/evaluation phase. You are welcome to contribute your commens and source code changes.

== Available Checks == 

* Non-const local statics
  
  int foo()
  {
    static int myEvilLocalState;
  }

* Non-const global statics

  static g_myGlobalStatic;
  
* use of the mutable keyword ( breaks const-corrcetness )

  struct Foo{
    mutable int myEvilValue;
  };
  
* use of const_cast to remove const-ness

  std::string s = "23";
  std::string const& r_const = s;
  std::string & r = const_cast< std::string & >( r_const );

* every explicit cast statement that removes const-ness

  std::string s = "23";
  std::string const& r_const = s;
  std::string & r = (std::string &) ( r_const );  

Dedicated Checkers exist for each of this code constructs.

== Compile LLVM / clang with this extensions ==

Follow the directions to obtain and compile LLVM/clang here:

http://clang.llvm.org/get_started.html#build

Stick to the directory structure suggested by this website, but run configure with the option --enable-optimized which will speed-up llvm/clang by some factors. Compile LLVM/clang and see if this is working. The root path of the LLVM subversion folder will in the following be aliased by . The folder where you built llvm is aliased

Now, checkout the repository which contains the CMS extensions into the same folder as <llvm_src> resides (CERN account needed):

svn co https://svn.cern.ch/reps/cmscoperformance/projects/clang_cms 

and run

cmake .
make

inside the clang_cms folder. If you encounter problems with missing files or directories, you may need to edit the file CMakeLists.txt to adapt it to your specific build configuration.

The CMS specific checkers have now been compiled into an external library in clang_cms/lib. 

== Test on a small example (non-CMSSW) ==

Export the path to the new clang binary ( Bash example ):

export PATH=<llvm_bin>/Release+Asserts/bin/:$PATH


To see a listing of all available checkers, also the CMS-specific ones, you can run the scan-build command:

<llvm_src>/tools/clang/tools/scan-build/scan-build -load-plugin lib/ClangCms.so


Test out the newly compiled and modified clang, cd into the clang_cms/test folder and run:

<llvm_src>/tools/clang/tools/scan-build/scan-build -load-plugin ../lib/ClangCms.so -enable-checker threadsafety.ConstCast -enable-checker threadsafety.ConstCastAway -enable-checker threadsafety.GlobalStatic -enable-checker threadsafety.MutableMember -enable-checker threadsafety.StaticLocal make -B

This wil produce a clang static analyzer html your can open in your favorite browser. You can find the location in the output line, something along the lines:

scan-build: 6 bugs found.
scan-build: Run 'scan-view /tmp/scan-build-2012-04-26-13' to examine bug reports.


You then call:
firefox /tmp/scan-build-2012-04-26-13/index.html

== Run within a SCRAM-based build ==

Create a project area with arch slc5_amd64_gcc470

export SCRAM_ARCH=slc5_amd64_gcc470
source cmsset_default.sh
scram pro CMSSW CMSSW_6_0_0_pre3

In the project area edit

config/toolbox/slc5_amd64_gcc470/tools/selected/cxxcompiler

and change these lines

#      <environment name="CXX" value="$GCCBINDIR/c++"/>
      <environment name="CXX" value="(llvm src path)/tools/clang/tools/scan-build/c++-analyzer"/>

Then setup the project area with the new cxxcompiler settings

scram setup cxxcompiler

Now you can run a build as usual expect that you will run scan-build to collect the results and then run scan-view to view them. The output by default goes to /tmp/scan-view-date-1

cd src/(your package dir)
scan-build scram b -v -k -j 4
scan-view /tmp/scan-view-(date)-(##)

You will need to include the paths to clang, scan-build and scan-view in your path

export PATH=$PATH\:(llvm install path)/bin/\:(llvm src path)/tools/clang/tools/scan-build/\:(llvm src path)/tools/clang/tools/scan-view/

If you also want to generate the reports for thread-safety, you also need to add the additional parameters to scan-build.

 
