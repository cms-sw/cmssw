#ifndef Utilities_StaticAnalyzers_ArgSizeChecker_h
#define Utilities_StaticAnalyzers_ArgSizeChecker_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>

#include "CmsException.h"

namespace clangcms {
class ArgSizeChecker : public clang::ento::Checker<clang::ento::check::PreStmt<clang::CXXMemberCallExpr> > {
  mutable llvm::OwningPtr<clang::ento::BugType> BT;
public:
  void checkPreStmt(const clang::CXXMemberCallExpr *ref, clang::ento::CheckerContext &C) const;
};  
}

#endif
