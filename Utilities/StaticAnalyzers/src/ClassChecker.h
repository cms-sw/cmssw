#ifndef Utilities_StaticAnalyzers_MemberChecker_h
#define Utilities_StaticAnalyzers_MemberChecker_h
#include <clang/AST/DeclCXX.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>

#include "CmsException.h"

namespace clangcms {
class ClassChecker : public clang::ento::Checker< clang::ento::check::ASTDecl< clang::CXXRecordDecl> > {
  mutable clang::OwningPtr< clang::ento::BuiltinBug> BT;

public:
  void checkASTDecl(const clang::CXXRecordDecl *D,
                      clang::ento::AnalysisManager &Mgr,
                      clang::ento::BugReporter &BR) const;
private:
  CmsException m_exception;
};  
}

#endif
