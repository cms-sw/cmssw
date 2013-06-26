#ifndef Utilities_StaticAnalyzers_FastMathChecker_h
#define Utilities_StaticAnalyzers_FastMathChecker_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>

#include "CmsException.h"

namespace clangcms {
class FiniteMathChecker : public clang::ento::Checker<clang::ento::check::PreStmt<clang::CallExpr> > {
  mutable llvm::OwningPtr<clang::ento::BugType> BT;
public:
  void checkPreStmt(const clang::CallExpr *ref, clang::ento::CheckerContext &C) const;
};  
}

#endif
