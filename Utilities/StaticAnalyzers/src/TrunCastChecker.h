#ifndef Utilities_StaticAnalyzers_CastSizeChecker_h
#define Utilities_StaticAnalyzers_CastSizeChecker_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include "CmsException.h" 

namespace clangcms {

class TrunCastChecker: public clang::ento::Checker< clang::ento::check::ASTDecl<clang::TranslationUnitDecl> > {
public:
     mutable std::unique_ptr<clang::ento::BugType> BT;
     void checkASTDecl(const clang::TranslationUnitDecl *D, clang::ento::AnalysisManager& Mgr, clang::ento::BugReporter &BR) const; 

private:
  CmsException m_exception;

};

} 

#endif
