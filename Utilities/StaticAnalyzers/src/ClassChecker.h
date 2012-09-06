#ifndef Utilities_StaticAnalyzers_MemberChecker_h
#define Utilities_StaticAnalyzers_MemberChecker_h
#include <clang/AST/DeclCXX.h>
#include <clang/AST/StmtVisitor.h>
#include <llvm/Support/SaveAndRestore.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h>
#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugReporter.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include <llvm/ADT/SmallString.h>

#include "CmsException.h"

namespace clangcms {
class ClassCheckerMDecl : public clang::ento::Checker< clang::ento::check::ASTDecl< clang::CXXMethodDecl> > {
  mutable clang::OwningPtr< clang::ento::BuiltinBug> BT;

public:
  void checkASTDecl(const clang::CXXMethodDecl *D,
                      clang::ento::AnalysisManager &Mgr,
                      clang::ento::BugReporter &BR) const;

private:
  CmsException m_exception;
};  

class ClassCheckerMCall : public clang::ento::Checker< clang::ento::check::PostStmt< clang::CXXMemberCallExpr> > {
  mutable clang::OwningPtr<clang::ento::BugType> BT;

public:
  void checkPostStmt(const clang::CXXMemberCallExpr *CE,
		clang::ento::CheckerContext &C) const;


private:
  CmsException m_exception;
};

class ClassCheckerRDecl : public clang::ento::Checker<clang::ento::check::ASTDecl<clang::CXXRecordDecl> > {
  mutable clang::OwningPtr< clang::ento::BugType> BT;


public:
  void checkASTDecl(const clang::CXXRecordDecl *CRD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const ;

private:
  CmsException m_exception;
};

}
#endif
