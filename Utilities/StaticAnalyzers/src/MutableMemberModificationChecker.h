//== MutableMemberChecker.h - Checks for mutable members --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#ifndef Utilities_StaticAnalyzers_MutableMemberChecker_h
#define Utilities_StaticAnalyzers_MutableMemberChecker_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "CmsException.h"
#include "CmsSupport.h"

namespace clangcms {
  class MutableMemberModificationChecker : public clang::ento::Checker<clang::ento::check::PreStmt<clang::MemberExpr>> {
  public:
    CMS_SA_ALLOW mutable std::unique_ptr<clang::ento::BugType> BT;
    void checkPreStmt(const clang::MemberExpr *ME, clang::ento::CheckerContext &C) const;

  private:
    CmsException m_exception;
    bool checkAssignToMutable(const clang::MemberExpr *ME, clang::ento::CheckerContext &C) const;
    bool checkCallNonConstOfMutable(const clang::MemberExpr *ME, clang::ento::CheckerContext &C) const;
  };
}  // namespace clangcms

#endif
