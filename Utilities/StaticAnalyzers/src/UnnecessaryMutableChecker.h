#ifndef Utilities_StaticAnalyzers_UnnecessaryMutableChecker_h
#define Utilities_StaticAnalyzers_UnnecessaryMutableChecker_h

#include <clang/AST/ASTContext.h>
#include <clang/AST/DeclCXX.h>
#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/CheckerManager.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugReporter.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h>
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "CmsException.h"
#include "CmsSupport.h"

namespace clangcms {

  class UnnecessaryMutableChecker : public clang::ento::Checker<clang::ento::check::ASTDecl<clang::CXXRecordDecl>> {
  public:
    void checkASTDecl(const clang::CXXRecordDecl *RD,
                      clang::ento::AnalysisManager &Mgr,
                      clang::ento::BugReporter &BR) const;

  private:
    bool isMutableMemberModified(const clang::FieldDecl *Field, const clang::CXXRecordDecl *RD) const;
    bool analyzeStmt(const clang::Stmt *S, const clang::FieldDecl *Field) const;
    CMS_SA_ALLOW mutable std::unique_ptr<clang::ento::BugType> BT;
    CmsException m_exception;
  };

}  // namespace clangcms

#endif /* Utilities_StaticAnalyzers_UnnecessaryMutableChecker_h */
