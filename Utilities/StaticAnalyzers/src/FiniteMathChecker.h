#ifndef Utilities_StaticAnalyzers_FastMathChecker_h
#define Utilities_StaticAnalyzers_FastMathChecker_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "CmsException.h"

namespace clangcms {
  class FiniteMathChecker : public clang::ento::Checker<clang::ento::check::ASTDecl<clang::CXXRecordDecl> > {
  public:
    void checkASTDecl(const clang::CXXRecordDecl *CRD,
                      clang::ento::AnalysisManager &mgr,
                      clang::ento::BugReporter &BR) const;
  };
}  // namespace clangcms

#endif
