#ifndef Utilities_StaticAnalyzers_ArgSizeChecker_h
#define Utilities_StaticAnalyzers_ArgSizeChecker_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>

#include "CmsException.h"

namespace clangcms {
class ArgSizeChecker : public clang::ento::Checker<clang::ento::check::PreStmt<clang::CXXConstructExpr>, 
						clang::ento::check::ASTDecl<clang::CXXMethodDecl>	> {
  mutable llvm::OwningPtr<clang::ento::BugType> BT;
public:
  void checkPreStmt(const clang::CXXConstructExpr *ref, clang::ento::CheckerContext &C) const;
  void checkASTDecl(const clang::CXXMethodDecl *CMD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const;
};  
}

#endif
