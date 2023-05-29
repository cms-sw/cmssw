//== GlobalStaticChecker.cpp - Checks for non-const global statics --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#include "GlobalStaticChecker.h"

#include <clang/AST/Attr.h>
#include "CmsSupport.h"
using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

  void GlobalStaticChecker::checkASTDecl(const clang::VarDecl *D,
                                         clang::ento::AnalysisManager &Mgr,
                                         clang::ento::BugReporter &BR) const {
    if (D->hasAttr<CMSThreadGuardAttr>() || D->hasAttr<CMSThreadSafeAttr>() || D->hasAttr<CMSSaAllowAttr>())
      return;
    if (D->getTSCSpec() == clang::ThreadStorageClassSpecifier::TSCS_thread_local)
      return;
    clang::QualType t = D->getType();
    if (D->hasGlobalStorage() && !D->isStaticDataMember() && !D->isStaticLocal() && !support::isConst(t)) {
      clang::ento::PathDiagnosticLocation DLoc =
          clang::ento::PathDiagnosticLocation::createBegin(D, BR.getSourceManager());

      if (!m_exception.reportGlobalStaticForType(t, DLoc, BR))
        return;
      if (support::isSafeClassName(t.getCanonicalType().getAsString()) || support::isSafeClassName(t.getAsString()))
        return;

      std::string buf;
      llvm::raw_string_ostream os(buf);
      os << "Non-const variable '" << t.getAsString() << " " << *D << "' is static and might be thread-unsafe";

      BR.EmitBasicReport(D, this, "non-const global static variable", "ThreadSafety", os.str(), DLoc);
      return;
    }
  }

}  // namespace clangcms
