//== ConstCastChecker.cpp - Checks for const_cast<> --------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#pragma once

#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"


using namespace clang;
using namespace ento;

namespace clangcms {

class ConstCastChecker: public Checker<check::PreStmt<CXXConstCastExpr> > {
public:
	mutable OwningPtr<BugType> BT;
	void checkPreStmt(const CXXConstCastExpr *CE, CheckerContext &C) const;
};
} 


