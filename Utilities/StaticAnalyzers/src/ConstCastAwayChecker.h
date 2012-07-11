//== ConstCastAwayChecker.h - Checks for removed const qualfiers --------------*- C++ -*--==//
//
// Check in a generic way if an explicit cast removes a const qualifier.
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//
#ifndef Utilities_StaticAnalyzers_ConstCastAwayChecker_h
#define Utilities_StaticAnalyzers_ConstCastAwayChecker_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include "CmsException.h" 

namespace clangcms {

class ConstCastAwayChecker: public clang::ento::Checker< clang::ento::check::PreStmt< clang::ExplicitCastExpr> > {
public:
	mutable clang::OwningPtr<clang::ento::BugType> BT;
	void checkPreStmt(const clang::ExplicitCastExpr *CE, clang::ento::CheckerContext &C) const;

private:
  CmsException m_exception;

};

} 

#endif
