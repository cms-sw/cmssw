//== ConstCastAwayChecker.cpp - Checks for removed const qualfiers --------------*- C++ -*--==//
//
// Check in a generic way if an explicit cast removes a const qualifier.
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//
#include <clang/AST/ExprCXX.h>
#include <clang/AST/Attr.h>
#include <clang/AST/ParentMap.h>
#include <clang/AST/Stmt.h>

#include <memory>

#include "ConstCastAwayChecker.h"
#include "CmsSupport.h"

using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {

  void ConstCastAwayChecker::checkPreStmt(const clang::ExplicitCastExpr *CE, clang::ento::CheckerContext &C) const {
    if (!(clang::CStyleCastExpr::classof(CE) || clang::CXXConstCastExpr::classof(CE)))
      return;
    auto P = C.getCurrentAnalysisDeclContext()->getParentMap().getParent(CE);
    while (P && !(isa<AttributedStmt>(P) || isa<DeclStmt>(P)) &&
           C.getCurrentAnalysisDeclContext()->getParentMap().hasParent(P)) {
      P = C.getCurrentAnalysisDeclContext()->getParentMap().getParent(P);
    }
    if (P && isa<AttributedStmt>(P)) {
      const AttributedStmt *AS = dyn_cast_or_null<AttributedStmt>(P);
      if (AS && (hasSpecificAttr<CMSSaAllowAttr>(AS->getAttrs()) || hasSpecificAttr<CMSThreadSafeAttr>(AS->getAttrs())))
        return;
    }
    if (P && isa<DeclStmt>(P)) {
      const DeclStmt *DS = dyn_cast_or_null<DeclStmt>(P);
      if (DS) {
        for (auto D : DS->decls()) {
          if (hasSpecificAttr<CMSSaAllowAttr>(D->getAttrs()) || hasSpecificAttr<CMSThreadSafeAttr>(D->getAttrs())) {
            return;
          }
        }
      }
    }

    const Expr *SE = CE->getSubExpr();
    const CXXRecordDecl *CRD = nullptr;
    std::string cname;
    if (SE->getType()->isPointerType())
      CRD = SE->getType()->getPointeeCXXRecordDecl();
    else
      CRD = SE->getType()->getAsCXXRecordDecl();

    if (CRD)
      cname = CRD->getQualifiedNameAsString();

    clang::ASTContext &Ctx = C.getASTContext();
    clang::QualType OrigTy = Ctx.getCanonicalType(SE->getType());
    clang::QualType ToTy = Ctx.getCanonicalType(CE->getType());

    if (support::isConst(OrigTy) && !support::isConst(ToTy)) {
      if (clang::ento::ExplodedNode *errorNode = C.generateErrorNode()) {
        if (!BT)
          BT = std::make_unique<clang::ento::BugType>(this, "const cast away", "ConstThreadSafety");
        std::string buf;
        llvm::raw_string_ostream os(buf);
        os << "const qualifier was removed via a cast, this may result in thread-unsafe code.";
        std::unique_ptr<clang::ento::PathSensitiveBugReport> PSBR =
            std::make_unique<clang::ento::PathSensitiveBugReport>(*BT, llvm::StringRef(os.str()), errorNode);
        PSBR->addRange(CE->getSourceRange());
        if (!m_exception.reportConstCastAway(*PSBR, C))
          return;
        C.emitReport(std::move(PSBR));
        if (cname.empty())
          return;
        std::string tname = "constcastaway-checker.txt.unsorted";
        std::string tolog = "flagged class '" + cname + "' const qualifier cast away";
        support::writeLog(tolog, tname);
      }
    }
  }

}  // namespace clangcms
