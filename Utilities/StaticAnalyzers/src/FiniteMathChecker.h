#ifndef Utilities_StaticAnalyzers_FastMathChecker_h
#define Utilities_StaticAnalyzers_FastMathChecker_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "CmsException.h"

namespace clangcms {
class FiniteMathChecker : public clang::ento::Checker<clang::ento::check::PreStmt<clang::CallExpr> > {
  CMS_THREAD_SAFE mutable std::unique_ptr<clang::ento::BugType> BT;
public:
  void checkPreStmt(const clang::CallExpr *ref, clang::ento::CheckerContext &C) const;
};  
}

#endif
