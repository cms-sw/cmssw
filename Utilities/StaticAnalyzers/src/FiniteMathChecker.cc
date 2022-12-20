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

using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {

  class FMWalkAST : public clang::StmtVisitor<FMWalkAST> {
    const CheckerBase *Checker;
    clang::ento::BugReporter &BR;
    clang::AnalysisDeclContext *AC;
    const NamedDecl *ND;

  public:
    FMWalkAST(const CheckerBase *checker,
              clang::ento::BugReporter &br,
              clang::AnalysisDeclContext *ac,
              const NamedDecl *nd)
        : Checker(checker), BR(br), AC(ac), ND(nd) {}

    // Stmt visitor methods.
    void VisitChildren(clang::Stmt *S);
    void VisitStmt(clang::Stmt *S) { VisitChildren(S); }
    void VisitCallExpr(clang::CallExpr *CE);
  };

  void FMWalkAST::VisitChildren(clang::Stmt *S) {
    for (clang::Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I)
      if (clang::Stmt *child = *I) {
        Visit(child);
      }
  }

  void FMWalkAST::VisitCallExpr(clang::CallExpr *CE) {
    const clang::Expr *Callee = CE->getCallee();
    const FunctionDecl *FD = CE->getDirectCallee();
    if (!FD)
      return;

    const char *sfile = BR.getSourceManager().getPresumedLoc(CE->getExprLoc()).getFilename();
    std::string sname(sfile);
    if (!support::isInterestingLocation(sname))
      return;

    // Get the name of the callee.
    clang::IdentifierInfo *II = FD->getIdentifier();
    if (!II)  // if no identifier, not a simple C function
      return;

    if (!II->isStr("isnan") && !II->isStr("isinf"))
      return;

    clang::ento::PathDiagnosticLocation CELoc =
        clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(), AC);
    BugType *BT = new clang::ento::BugType(Checker,
                                           "std::isnan / std::isinf does not work when fast-math is used. Please use "
                                           "edm::isNotFinite from 'FWCore/Utilities/interface/isFinite.h'",
                                           "fastmath plugin");
    std::unique_ptr<clang::ento::BasicBugReport> report =
        std::make_unique<clang::ento::BasicBugReport>(*BT, BT->getCheckerName(), CELoc);
    BR.emitReport(std::move(report));
  }

  void FiniteMathChecker::checkASTDecl(const clang::CXXRecordDecl *RD,
                                       clang::ento::AnalysisManager &mgr,
                                       clang::ento::BugReporter &BR) const {
    const clang::SourceManager &SM = BR.getSourceManager();
    const char *sfile = SM.getPresumedLoc(RD->getLocation()).getFilename();
    if (!support::isCmsLocalFile(sfile))
      return;

    for (clang::CXXRecordDecl::method_iterator I = RD->method_begin(), E = RD->method_end(); I != E; ++I) {
      clang::CXXMethodDecl *MD = llvm::cast<clang::CXXMethodDecl>((*I)->getMostRecentDecl());
      clang::Stmt *Body = MD->getBody();
      if (Body) {
        FMWalkAST walker(this, BR, mgr.getAnalysisDeclContext(MD), MD);
        walker.Visit(Body);
      }
    }
  }
}  // namespace clangcms
