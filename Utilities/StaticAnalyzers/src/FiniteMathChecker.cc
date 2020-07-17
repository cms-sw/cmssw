#include "FiniteMathChecker.h"
#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/DeclGroup.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Expr.h>

#include "CmsSupport.h"
#include <iostream>
#include <memory>

#include <utility>

namespace clangcms {

  void FiniteMathChecker::checkPreStmt(const clang::CallExpr *CE, clang::ento::CheckerContext &ctx) const {
    const clang::ento::ProgramStateRef state = ctx.getState();
    const clang::LocationContext *LC = ctx.getLocationContext();
    const clang::Expr *Callee = CE->getCallee();
    const clang::FunctionDecl *FD = state->getSVal(Callee, LC).getAsFunctionDecl();

    if (!FD)
      return;

    // Get the name of the callee.
    clang::IdentifierInfo *II = FD->getIdentifier();
    if (!II)  // if no identifier, not a simple C function
      return;

    if (!II->isStr("isnan") && !II->isStr("isinf"))
      return;

    clang::ento::ExplodedNode *N = ctx.generateErrorNode();
    if (!N)
      return;

    if (!BT)
      BT = std::make_unique<clang::ento::BugType>(
          this,
          "std::isnan / std::isinf does not work when fast-math is used. Please use "
          "edm::isNotFinite from 'FWCore/Utilities/interface/isFinite.h'",
          "fastmath plugin");
    std::unique_ptr<clang::ento::PathSensitiveBugReport> PSBR =
        std::make_unique<clang::ento::PathSensitiveBugReport>(*BT, BT->getCheckerName(), N);
    std::unique_ptr<clang::ento::BasicBugReport> report =
        std::make_unique<clang::ento::BasicBugReport>(*BT, BT->getCheckerName(), PSBR->getLocation());
    report->addRange(Callee->getSourceRange());
    ctx.emitReport(std::move(report));
  }
}  // namespace clangcms
