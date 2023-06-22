//== StaticLocalChecker.cpp - Checks for non-const static locals --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#include "StaticLocalChecker.h"

#include "CmsSupport.h"
#include <iostream>
#include <clang/AST/Attr.h>
using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

  void StaticLocalChecker::checkASTDecl(const clang::VarDecl *D,
                                        clang::ento::AnalysisManager &Mgr,
                                        clang::ento::BugReporter &BR) const {
    clang::QualType t = D->getType();
    if (D->hasAttr<CMSThreadGuardAttr>() || D->hasAttr<CMSThreadSafeAttr>() || D->hasAttr<CMSSaAllowAttr>())
      return;
    if (((D->isStaticLocal() || D->isStaticDataMember()) &&
         D->getTSCSpec() != clang::ThreadStorageClassSpecifier::TSCS_thread_local) &&
        !support::isConst(t)) {
      clang::ento::PathDiagnosticLocation DLoc =
          clang::ento::PathDiagnosticLocation::createBegin(D, BR.getSourceManager());

      if (!m_exception.reportGlobalStaticForType(t, DLoc, BR))
        return;
      if (support::isSafeClassName(t.getCanonicalType().getAsString()))
        return;

      std::string buf;
      llvm::raw_string_ostream os(buf);
      os << "Non-const variable '" << t.getAsString() << " " << D->getQualifiedNameAsString()
         << "' is static local or static member data and might be thread-unsafe";
      if (!BT)
        BT = std::make_unique<clang::ento::BugType>(this, "non-const static variable", "ThreadSafety");
      std::unique_ptr<clang::ento::BasicBugReport> R =
          std::make_unique<clang::ento::BasicBugReport>(*BT, llvm::StringRef(os.str()), DLoc);
      R->setDeclWithIssue(D);
      R->addRange(D->getSourceRange());
      BR.emitReport(std::move(R));
      return;
    }
  }

}  // namespace clangcms
