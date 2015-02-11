#ifndef Utilities_StaticAnalyzers_getParamDumper_h
#define Utilities_StaticAnalyzers_getParamDumper_h
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>


namespace clangcms {

class getParamDumper : public clang::ento::Checker< clang::ento::eval::Call> 
{

 void analyzerEval(const clang::CallExpr *CE, clang::ento::CheckerContext &C) const;

 typedef void (getParamDumper::*FnCheck)(const clang::CallExpr *, clang::ento::CheckerContext &C) const;

 public:

  bool evalCall(const clang::CallExpr *CE, clang::ento::CheckerContext &C) const;

};

}
  #endif
