//== MutableMemberChecker.h - Checks for mutable members --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#ifndef Utilities_StaticAnalyzers_MutableMemberChecker_h
#define Utilities_StaticAnalyzers_MutableMemberChecker_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "CmsException.h"
#include "CmsSupport.h"

namespace clangcms {
  class MutableMemberChecker : public clang::ento::Checker<clang::ento::check::ASTDecl<clang::FieldDecl> > {
    CMS_SA_ALLOW mutable std::unique_ptr<clang::ento::BuiltinBug> BT;

  public:
    void checkASTDecl(const clang::FieldDecl *D, clang::ento::AnalysisManager &Mgr, clang::ento::BugReporter &BR) const;

  private:
    CmsException m_exception;
  };
}  // namespace clangcms

#endif
