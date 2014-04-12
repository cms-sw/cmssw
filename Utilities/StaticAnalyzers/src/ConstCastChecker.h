//== ConstCastChecker.h - Checks for const_cast<> --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#ifndef Utilities_StaticAnalyzers_ConstCastChecker_h
#define Utilities_StaticAnalyzers_ConstCastChecker_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include "CmsException.h"

namespace clangcms {

class ConstCastChecker: public clang::ento::Checker< clang::ento::check::PreStmt< clang::CXXConstCastExpr> > {
public:
	mutable clang::OwningPtr<clang::ento::BugType> BT;
	void checkPreStmt(const clang::CXXConstCastExpr *CE, clang::ento::CheckerContext &C) const;

private:
  CmsException m_exception;
};
} 

#endif
