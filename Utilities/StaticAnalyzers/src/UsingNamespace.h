//==-- UsingNamespace.h - Checks for using namespace and using std:: in headers --------------*- C++ -*--==//
//
// by Shahzad Malik MUZAFFAR [ Shahzad.Malik.MUZAFFAR@cern.ch ]
//
//===----------------------------------------------------------------------===//
#ifndef Utilities_StaticAnalyzers_UsingNamespace_h
#define Utilities_StaticAnalyzers_UsingNamespace_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/AST/DeclCXX.h>

namespace clangcms {
  class UsingNamespace : public clang::ento::Checker< clang::ento::check::ASTDecl <clang::UsingDirectiveDecl>,clang::ento::check::ASTDecl <clang::UsingDecl> >
  {
  public:
    void checkASTDecl (const clang::UsingDirectiveDecl *D, clang::ento::AnalysisManager &Mgr, clang::ento::BugReporter &BR) const;
    void checkASTDecl (const clang::UsingDecl *D, clang::ento::AnalysisManager &Mgr, clang::ento::BugReporter &BR) const;
  private:
    bool isDeclOK(const clang::NamedDecl *D, clang::ento::BugReporter &BR) const;
    void reportBug(const char* bug, const clang::Decl *D, clang::ento::BugReporter &BR) const;
  };
} 
#endif
