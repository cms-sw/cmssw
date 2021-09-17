//== ConstCastAwayChecker.cpp - Checks for removed const qualfiers --------------*- C++ -*--==//
//
// Check in a generic way if an explicit cast removes a const qualifier.
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//
#include <clang/AST/ExprCXX.h>
#include <clang/AST/Attr.h>

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
        std::unique_ptr<clang::ento::BasicBugReport> R =
            std::make_unique<clang::ento::BasicBugReport>(*BT, llvm::StringRef(os.str()), PSBR->getLocation());
        R->addRange(CE->getSourceRange());
        if (!m_exception.reportConstCastAway(*R, C))
          return;
        C.emitReport(std::move(R));
        if (cname.empty())
          return;
        std::string tname = "constcastaway-checker.txt.unsorted";
        std::string tolog = "flagged class '" + cname + "' const qualifier cast away";
        support::writeLog(tolog, tname);
      }
    }
  }

}  // namespace clangcms
