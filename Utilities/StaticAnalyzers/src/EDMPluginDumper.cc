#include "EDMPluginDumper.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm>

using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {

  void EDMPluginDumper::checkASTDecl(const clang::ClassTemplateDecl *TD,
                                     clang::ento::AnalysisManager &mgr,
                                     clang::ento::BugReporter &BR) const {
    std::string tname = TD->getTemplatedDecl()->getQualifiedNameAsString();
    if (tname == "edm::WorkerMaker") {
      for (auto I = TD->spec_begin(), E = TD->spec_end(); I != E; ++I) {
        for (unsigned J = 0, F = I->getTemplateArgs().size(); J != F; ++J) {
          llvm::SmallString<100> buf;
          llvm::raw_svector_ostream os(buf);
#if LLVM_VERSION_MAJOR >= 13
          I->getTemplateArgs().get(J).print(mgr.getASTContext().getPrintingPolicy(), os, false);
#else
          I->getTemplateArgs().get(J).print(mgr.getASTContext().getPrintingPolicy(), os);
#endif
          std::string rname = os.str().str();
          std::string fname("plugins.txt.unsorted");
          std::string ostring = rname + "\n";
          support::writeLog(ostring, fname);
        }
      }
    }

  }  //end class

}  // namespace clangcms
