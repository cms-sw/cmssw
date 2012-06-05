//== ConstCastAwayChecker.cpp - Checks for removed const qualfiers --------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Check in a generic way if an explicit cast removes a const qualifier.
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//
#pragma once

#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"

#include "ClangCheckerPluginDef.h"

using namespace clang;
using namespace ento;

namespace clangcms {


class ConstCastAwayChecker: public Checker<check::PreStmt<ExplicitCastExpr> > {
public:
	mutable OwningPtr<BugType> BT;
	void checkPreStmt(const ExplicitCastExpr *CE, CheckerContext &C) const;
};

} 



