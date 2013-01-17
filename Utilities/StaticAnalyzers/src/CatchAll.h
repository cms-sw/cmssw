//==-- CatchAll.h - Checks for catch(...) in source files --------------*- C++ -*--==//
//
// by Shahzad Malik MUZAFFAR [ Shahzad.Malik.MUZAFFAR@cern.ch ]
//
//===----------------------------------------------------------------------===//
#ifndef Utilities_StaticAnalyzers_CatchAll_h
#define Utilities_StaticAnalyzers_CatchAll_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/AST/StmtCXX.h>

namespace clangcms {
  class CatchAll : public clang::ento::Checker< clang::ento::check::ASTCodeBody >
  {
  public:
    void checkASTCodeBody(const clang::Decl*& D, clang::ento::AnalysisManager&, clang::ento::BugReporter& BR) const;
  private:
    const clang::Stmt* process(const clang::Stmt* S) const;
    inline bool checkCatchAll(const clang::CXXCatchStmt* S) const {return S->getCaughtType().isNull();}
  };
} 
#endif
