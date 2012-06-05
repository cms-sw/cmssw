//== StaticLocalChecker.cpp - Checks for non-const static locals --------------*- C++ -*--==//
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

#include "CmsException.h"


using namespace clang;
using namespace ento;

namespace clangcms {
class StaticLocalChecker : public Checker< check::ASTDecl<VarDecl> > {
  mutable OwningPtr<BuiltinBug> BT;

public:
  void checkASTDecl(const VarDecl *D,
                      AnalysisManager &Mgr,
                      BugReporter &BR) const;
private:
  CmsException m_exception;
};  
}




