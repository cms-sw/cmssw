//== ConstCastChecker.cpp - Checks for const_cast<> --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#include <clang/AST/Attr.h>
#include <clang/AST/ExprCXX.h>

#include <memory>

#include "ConstCastChecker.h"
#include "CmsSupport.h"

using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {

  void ConstCastChecker::checkPreStmt(const clang::CXXConstCastExpr *CE, clang::ento::CheckerContext &C) const {
    const Expr *SE = CE->getSubExprAsWritten();
    const CXXRecordDecl *CRD = nullptr;
    std::string cname;
    if (SE->getType()->isPointerType())
      CRD = SE->getType()->getPointeeCXXRecordDecl();
    else
      CRD = SE->getType()->getAsCXXRecordDecl();
    if (CRD)
      cname = CRD->getQualifiedNameAsString();
    if (clang::ento::ExplodedNode *errorNode = C.generateErrorNode()) {
      if (!BT)
        BT = std::make_unique<clang::ento::BugType>(this, "const_cast used on pointer to class", "ConstThreadSafety");
      std::string buf;
      llvm::raw_string_ostream os(buf);
      os << "const_cast was used, this may result in thread-unsafe code.";
      std::unique_ptr<clang::ento::PathSensitiveBugReport> PSBR =
          std::make_unique<clang::ento::PathSensitiveBugReport>(*BT, llvm::StringRef(os.str()), errorNode);
      std::unique_ptr<clang::ento::BasicBugReport> R =
          std::make_unique<clang::ento::BasicBugReport>(*BT, llvm::StringRef(os.str()), PSBR->getLocation());
      R->addRange(CE->getSourceRange());
      if (!m_exception.reportConstCast(*R, C))
        return;
      C.emitReport(std::move(R));
      if (cname.empty())
        return;
      std::string tname = "constcast-checker.txt.unsorted";
      std::string tolog = "flagged class '" + cname + "' const_cast used ";
      support::writeLog(tolog, tname);
    }
  }

}  // namespace clangcms
