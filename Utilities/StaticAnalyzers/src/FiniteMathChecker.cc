#include "FiniteMathChecker.h"
#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/DeclGroup.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Expr.h>


#include "CmsSupport.h"
#include <iostream>

namespace clangcms {

void FiniteMathChecker::checkPreStmt(const clang::CallExpr *CE, clang::ento::CheckerContext &ctx) const
{
  const clang::ento::ProgramStateRef state = ctx.getState();
  const clang::LocationContext *LC = ctx.getLocationContext();
  const clang::Expr *Callee = CE->getCallee();
  const clang::FunctionDecl *FD = state->getSVal(Callee, LC).getAsFunctionDecl();

  if (!FD)
    return;

  // Get the name of the callee.
  clang::IdentifierInfo *II = FD->getIdentifier();
  if (!II)   // if no identifier, not a simple C function
    return;

  if (!II->isStr("isnan") && !II->isStr("isinf")) 
    return;

  clang::ento::ExplodedNode *N = ctx.generateSink();
  if (!N)
    return;

  if (!BT)
    BT.reset(new clang::ento::BugType("std::isnan / std::isinf does not work when fast-math is used. Please use edm::isNotFinite from 'FWCore/Utilities/interface/isNotFinite.h'", "fastmath plugin"));

  clang::ento::BugReport *report = new clang::ento::BugReport(*BT, BT->getName(), N);
  report->addRange(Callee->getSourceRange());
  ctx.emitReport(report);
}
}

