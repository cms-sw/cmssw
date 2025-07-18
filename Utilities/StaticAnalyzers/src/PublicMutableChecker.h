//== PublicMutableChecker.h - Checks for public mutable members --------------*- C++ -*--==//
//
// Check for public mutable members
//
// by Ivan Razumov
//
//===----------------------------------------------------------------------===//
#ifndef Utilities_StaticAnalyzers_PublicMutableChecker_h
#define Utilities_StaticAnalyzers_PublicMutableChecker_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "CmsException.h"

namespace clangcms {

  class PublicMutableChecker : public clang::ento::Checker<clang::ento::check::ASTDecl<clang::FieldDecl>> {
  public:
    CMS_SA_ALLOW mutable std::unique_ptr<clang::ento::BugType> BT;
    void checkASTDecl(const clang::FieldDecl *D, clang::ento::AnalysisManager &Mgr, clang::ento::BugReporter &BR) const;

  private:
    CmsException m_exception;
  };

}  // namespace clangcms

#endif
