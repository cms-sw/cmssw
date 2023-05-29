#include "ArgSizeChecker.h"
#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/DeclGroup.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Expr.h>
#include <clang/AST/CharUnits.h>
#include <llvm/ADT/SmallString.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugReporter.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include <clang/AST/ParentMap.h>

#include "CmsSupport.h"
#include <iostream>
using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

  void ArgSizeChecker::checkPreStmt(const CXXConstructExpr *E, CheckerContext &ctx) const {
    CmsException m_exception;
    clang::LangOptions LangOpts;
    LangOpts.CPlusPlus = true;
    clang::PrintingPolicy Policy(LangOpts);

    const clang::ento::PathDiagnosticLocation ELoc =
        clang::ento::PathDiagnosticLocation::createBegin(E, ctx.getSourceManager(), ctx.getLocationContext());

    if (!m_exception.reportGeneral(ELoc, ctx.getBugReporter()))
      return;
    llvm::SmallString<100> buf;
    llvm::raw_svector_ostream os(buf);

    for (clang::Stmt::const_child_iterator I = E->child_begin(), F = E->child_end(); I != F; ++I) {
      const Expr *child = llvm::dyn_cast_or_null<Expr>(*I);
      if (!child)
        continue;
      if (llvm::isa<DeclRefExpr>(child->IgnoreImpCasts())) {
        const NamedDecl *ND = llvm::cast<DeclRefExpr>(child->IgnoreImpCasts())->getFoundDecl();
        if (llvm::isa<ParmVarDecl>(ND)) {
          const ParmVarDecl *PVD = llvm::cast<ParmVarDecl>(ND);
          QualType QT = PVD->getOriginalType();
          if (QT->isIncompleteType() || QT->isDependentType())
            continue;
          clang::QualType PQT = QT.getCanonicalType();
          PQT.removeLocalConst();
          if (PQT->isReferenceType() || PQT->isPointerType() || PQT->isMemberFunctionPointerType() ||
              PQT->isArrayType() || PQT->isBuiltinType() || PQT->isUnionType() || PQT->isVectorType())
            continue;
          uint64_t size_param = ctx.getASTContext().getTypeSize(PQT);
          uint64_t max_bits = 128;
          if (size_param <= max_bits)
            continue;
          std::string qname = QT.getAsString();
          std::string pname = PQT.getAsString();
          const CXXMethodDecl *MD =
              llvm::dyn_cast_or_null<CXXMethodDecl>(ctx.getCurrentAnalysisDeclContext()->getDecl());
          os << "Function parameter copied by value with size '" << size_param << "' bits > max size '" << max_bits
             << "' bits parameter type '" << pname << "' function '";

          if (MD) {
            std::string fname = MD->getNameAsString();
            os << fname << "' class '" << MD->getParent()->getNameAsString();
          }
          os << "'\n";

          std::string oname = "operator";

          const clang::ento::PathDiagnosticLocation DLoc =
              clang::ento::PathDiagnosticLocation::createBegin(PVD, ctx.getSourceManager());

          BugType *BT = new BugType(this, "Function parameter copied by value with size > max", "ArgSize");
          std::unique_ptr<BasicBugReport> report =
              std::make_unique<BasicBugReport>(*BT, llvm::StringRef(os.str()), DLoc);
          report->addRange(PVD->getSourceRange());
          ctx.emitReport(std::move(report));
        }
      }
    }
  }

  void ArgSizeChecker::checkASTDecl(const CXXMethodDecl *MD, AnalysisManager &mgr, BugReporter &BR) const {
    const SourceManager &SM = BR.getSourceManager();
    CmsException m_exception;
    PathDiagnosticLocation DLoc = PathDiagnosticLocation::createBegin(MD, SM);

    if (!m_exception.reportGeneral(DLoc, BR))
      return;

    for (CXXMethodDecl::param_const_iterator I = MD->param_begin(), E = MD->param_end(); I != E; I++) {
      llvm::SmallString<100> buf;
      llvm::raw_svector_ostream os(buf);
      QualType QT = (*I)->getOriginalType();
      if (QT->isIncompleteType() || QT->isDependentType())
        continue;
      clang::QualType PQT = QT.getCanonicalType();
      PQT.removeLocalConst();
      if (PQT->isReferenceType() || PQT->isPointerType() || PQT->isMemberFunctionPointerType() || PQT->isArrayType() ||
          PQT->isBuiltinType() || PQT->isUnionType() || PQT->isVectorType())
        continue;
      uint64_t size_param = mgr.getASTContext().getTypeSize(PQT);
      uint64_t max_bits = 128;
      if (size_param <= max_bits)
        continue;
      std::string qname = QT.getAsString();
      std::string pname = PQT.getAsString();
      std::string fname = MD->getNameAsString();
      os << "Function parameter passed by value with size of parameter '" << size_param << "' bits > max size '"
         << max_bits << "' bits parameter type '" << pname << "' function '" << fname << "' class '"
         << MD->getParent()->getNameAsString() << "'\n";
      std::string oname = "operator";

      BugType *BT = new BugType(this, "Function parameter with size > max", "ArgSize");
      std::unique_ptr<BasicBugReport> report = std::make_unique<BasicBugReport>(*BT, llvm::StringRef(os.str()), DLoc);
      BR.emitReport(std::move(report));
    }
  }

}  // namespace clangcms
