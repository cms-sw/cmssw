#ifndef Utilities_StaticAnalyzers_getParamDumper_h
#define Utilities_StaticAnalyzers_getParamDumper_h
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"

namespace clangcms {

  class getParamDumper
      : public clang::ento::Checker<clang::ento::check::ASTDecl<clang::CXXRecordDecl>, clang::ento::eval::Call> {
    void analyzerEval(const clang::CallExpr *CE, clang::ento::CheckerContext &C) const;

    typedef void (getParamDumper::*FnCheck)(const clang::CallExpr *, clang::ento::CheckerContext &C) const;

  public:
    bool evalCall(const clang::ento::CallEvent &Call, clang::ento::CheckerContext &C) const;

    void checkASTDecl(const clang::CXXRecordDecl *CRD,
                      clang::ento::AnalysisManager &mgr,
                      clang::ento::BugReporter &BR) const;
  };

}  // namespace clangcms
#endif
