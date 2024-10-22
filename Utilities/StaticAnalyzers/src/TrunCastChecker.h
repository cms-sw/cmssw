#ifndef Utilities_StaticAnalyzers_CastSizeChecker_h
#define Utilities_StaticAnalyzers_CastSizeChecker_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include "CmsException.h"

namespace clangcms {

  class TrunCastChecker : public clang::ento::Checker<clang::ento::check::ASTDecl<clang::CXXRecordDecl> > {
  public:
    void checkASTDecl(const clang::CXXRecordDecl *D,
                      clang::ento::AnalysisManager &Mgr,
                      clang::ento::BugReporter &BR) const;

  private:
    CmsException m_exception;
  };

}  // namespace clangcms

#endif
