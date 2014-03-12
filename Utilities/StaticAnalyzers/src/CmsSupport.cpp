//===--- CmsSupport.cpp - Provides support functions ------------*- C++ -*-===//
//
// by Shahzad Malik MUZAFFAR [ Shahzad.Malik.MUZAFFAR@cern.ch ]
//
//===----------------------------------------------------------------------===//
#include <clang/Basic/FileManager.h>
#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include <llvm/ADT/SmallString.h>
#include "llvm/Support/raw_ostream.h"
#include "CmsSupport.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>


using namespace clangcms;
using namespace clang;
using namespace llvm;
bool support::isCmsLocalFile(const char* file)
{
  static char* LocalDir=0;
  static int DirLen=-1;
  if (DirLen==-1)
  {
    DirLen=0;
    LocalDir = getenv ("LOCALRT");
    if (LocalDir!=NULL) DirLen=strlen(LocalDir);
  }
  if ((DirLen==0) || (strncmp(file,LocalDir,DirLen)!=0) || (strncmp(&file[DirLen],"/src/",5)!=0)) return false;
  return true;
}
// This is a wrapper around NamedDecl::getQualifiedNameAsString.
// It produces more qualified output to distinguish several cases
// which would otherwise be ambiguous.
std::string support::getQualifiedName(const clang::NamedDecl &d) {
  std::string ret;
  const DeclContext *ctx = d.getDeclContext();
  if (ctx->isFunctionOrMethod() && isa<NamedDecl>(ctx))
  {
    // This is a local variable.
    // d.getQualifiedNameAsString() will return the unqualifed name for this
    // but we want an actual qualified name so we can distinguish variables
    // with the same name but that are in different functions.
    ret = getQualifiedName(*cast<NamedDecl>(ctx)) + "::" + d.getNameAsString();
  }
  else
  {
    ret = d.getQualifiedNameAsString();
  }

  if (const FunctionDecl *fd = dyn_cast<FunctionDecl>(&d))
  {
    // This is a function. getQualifiedNameAsString will return a string
    // like "ANamespace::AFunction". To this we append the list of parameters
    // so that we can distinguish correctly between
    // void ANamespace::AFunction(int);
    // and
    // void ANamespace::AFunction(float);
    ret += "(";
    const FunctionType *ft = fd->getType()->castAs<FunctionType>();
    if (const FunctionProtoType *fpt = dyn_cast<FunctionProtoType>(ft))
    {
      unsigned num_params = fd->getNumParams();
      for (unsigned i = 0; i < num_params; ++i) {
        if (i)
          ret += ", ";
        ret += fd->getParamDecl(i)->getType().getAsString();
      }

      if (fpt->isVariadic()) {
        if (num_params > 0)
          ret += ", ";
        ret += "...";
      }
    }
    ret += ")";
    if (ft->isConst())
      ret += " const";
  }

  return ret;
}

bool support::isSafeClassName(const std::string &name) {
  std::string atomic = "std::atomic";
  std::string uatomic = "std::__atomic_";
  std::string mutex = "std::mutex";
  std::string rmutex = "std::recursive_mutex";
  std::string btsp = "boost::thread_specific_ptr";
  std::string catomic = "class std::atomic";
  std::string cuatomic = "class std::__atomic_";
  std::string cmutex = "class std::mutex";
  std::string crmutex = "class std::recursive_mutex";
  std::string cbtsp = "class boost::thread_specific_ptr";
  std::string ctbb = "class tbb::";
  std::string eap = "edm::AtomicPtrCache";
  std::string ceap = "class edm::AtomicPtrCache";
  
  if ( name.substr(0,atomic.length()) == atomic || name.substr(0,catomic.length()) == catomic
	|| name.substr(0,uatomic.length()) == uatomic  || name.substr(0,cuatomic.length()) == cuatomic
	|| name.substr(0,mutex.length()) == mutex || name.substr(0,cmutex.length()) == cmutex 
	|| name.substr(0,rmutex.length()) == rmutex || name.substr(0,crmutex.length()) == rmutex 
	|| name.substr(0,btsp.length()) == btsp || name.substr(0,cbtsp.length()) == cbtsp 
	|| name.substr(0,ctbb.length()) == ctbb 
	|| name.substr(0,eap.length()) == eap || name.substr(0,ceap.length()) == ceap
	) 
	return true;	
  return false;
}

bool support::isDataClass(const std::string & name) {
	std::string buf;
	llvm::raw_string_ostream os(buf);
	clang::FileSystemOptions FSO;
	clang::FileManager FM(FSO);
	const char * lPath = std::getenv("LOCALRT");
	const char * rPath = std::getenv("CMSSW_RELEASE_BASE");
	std::string lname(""); 
	std::string rname(""); 
	std::string iname(""); 
	if ( lPath != NULL && rPath != NULL ) {
		lname = std::string(lPath);
		rname = std::string(rPath);
	}
		
	std::string tname("/tmp/classes.txt");
	std::string sname("/src/Utilities/StaticAnalyzers/scripts/classes.txt");
	std::string fname1 = lname + tname;
	std::string fname2 = rname + sname;
	if (!FM.getFile(fname1) && !FM.getFile(fname2) ) {
		llvm::errs()<<"\n\nChecker cannot find classes.txt. Run \"USER_LLVM_CHECKERS='-enable-checker optional.ClassDumperCT -enable-checker optional.ClassDumperFT scram b checker to create $LOCALRT/tmp/classes.txt.\n\n\n";
		exit(1);
		}
	if ( FM.getFile(fname1) ) 
		iname = fname1;
	else 
		iname = fname2;	
	os <<"class '"<< name <<"'\n";
	std::ifstream ifile;
	ifile.open(iname.c_str(),std::ifstream::in);
	std::string ifilecontents((std::istreambuf_iterator<char>(ifile)),std::istreambuf_iterator<char>() );
	if (ifilecontents.find(os.str()) != std::string::npos ) return true;
	return false;
}
