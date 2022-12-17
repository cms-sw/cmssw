#include <clang/AST/DeclCXX.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/StmtCXX.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <llvm/ADT/StringSwitch.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Attr.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/ParentMap.h>
#include <clang/Analysis/CFGStmtMap.h>
#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugReporter.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h>
#include <llvm/Support/SaveAndRestore.h>
#include <llvm/ADT/SmallString.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm>

#include "CmsException.h"
#include "CmsSupport.h"
#include "getParamDumper.h"

using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

  void getParamDumper::analyzerEval(const clang::CallExpr *CE, clang::ento::CheckerContext &C) const {
    if (!C.getSourceManager().isInMainFile(CE->getExprLoc()))
      return;

    const FunctionDecl *FD = CE->getDirectCallee();

    if (!FD)
      return;

    std::string mname = support::getQualifiedName(*FD);
    const char *sfile = C.getSourceManager().getPresumedLoc(CE->getExprLoc()).getFilename();
    std::string sname(sfile);
    if (!support::isInterestingLocation(sname))
      return;
    std::string mdname;
    const FunctionDecl *MD = C.getCurrentAnalysisDeclContext()->getDecl()->getAsFunction();
    if (!MD)
      return;
    mdname = MD->getQualifiedNameAsString();
    for (unsigned I = 0, E = MD->getNumParams(); I != E; ++I) {
      std::string ps = "const class edm::ParameterSet ";
      std::string ups = "const class edm::UntrackedParameterSet ";
      std::string pname = MD->getParamDecl(I)->getQualifiedNameAsString();
      std::string qname = MD->getParamDecl(I)->getType().getCanonicalType().getAsString();
      //             if (qname.substr(0,ps.length()) == ps || qname.substr(0,ups.length()) == ups) {
      std::string buf;
      llvm::raw_string_ostream os(buf);
      os << "in function decl '" << mdname << "' with parameter '" << qname << " " << pname << "'\n";
      //                 }
    }
    const CXXMemberCallExpr *CXE = llvm::dyn_cast_or_null<CXXMemberCallExpr>(CE);
    if (!CXE)
      return;
    const Expr *IOA = CXE->getImplicitObjectArgument();
    std::string tname = "getparam-dumper.txt.unsorted";
    std::string gp = "edm::ParameterSet::getParameter";
    std::string gup = "edm::ParameterSet::getUntrackedParameter";
    if (mname.substr(0, gp.length()) == gp || mname.substr(0, gup.length()) == gup) {
      std::string buf;
      llvm::raw_string_ostream os(buf);
      os << "in function decl '" << mdname << "' member function call '";
      clang::LangOptions LangOpts;
      LangOpts.CPlusPlus = true;
      clang::PrintingPolicy Policy(LangOpts);
      os << support::getQualifiedName(*(CXE->getMethodDecl()));
      os << "' with args '";
      for (unsigned I = 0, E = CE->getNumArgs(); I != E; ++I) {
        if (I)
          os << ", ";
        CE->getArg(I)->printPretty(os, nullptr, Policy);
      }
      os << "' with implicit object '";
      const Expr *E = IOA->IgnoreParenNoopCasts(C.getASTContext());
      QualType QE = E->getType().getCanonicalType();
      os << QE.getAsString() << " ";
      switch (E->getStmtClass()) {
        case Stmt::MemberExprClass:
          os << dyn_cast<MemberExpr>(E)->getMemberDecl()->getQualifiedNameAsString();
          break;
        case Stmt::DeclRefExprClass:
          os << dyn_cast<DeclRefExpr>(E)->getDecl()->getQualifiedNameAsString();
          break;
        case Stmt::CXXOperatorCallExprClass:
          dyn_cast<CXXOperatorCallExpr>(E)->printPretty(os, nullptr, Policy);
          break;
        case Stmt::CXXBindTemporaryExprClass:
          dyn_cast<CXXBindTemporaryExpr>(E)->printPretty(os, nullptr, Policy);
          break;
        case Stmt::CXXMemberCallExprClass:
          dyn_cast<CXXMemberCallExpr>(E)->printPretty(os, nullptr, Policy);
          break;
        case Stmt::UnaryOperatorClass:
          dyn_cast<UnaryOperator>(E)->printPretty(os, nullptr, Policy);
          break;
        default:
          E->printPretty(os, nullptr, Policy);
          os << " unhandled expr class " << E->getStmtClassName();
      }
      os << "'\n";

      support::writeLog(os.str(), tname);
    }
    return;
  }

  bool getParamDumper::evalCall(const CallEvent &Call, CheckerContext &C) const {
    const auto *CE = llvm::dyn_cast_or_null<clang::CallExpr>(Call.getOriginExpr());
    if (!CE)
      return false;

    FnCheck Handler = llvm::StringSwitch<FnCheck>(C.getCalleeName(CE))
                          .Case("getParameter", &getParamDumper::analyzerEval)
                          .Case("getUntrackedParameter", &getParamDumper::analyzerEval)
                          .Default(nullptr);

    if (!Handler)
      return false;

    (this->*Handler)(CE, C);

    return true;
  }

  class gpWalkAST : public clang::StmtVisitor<gpWalkAST> {
    const CheckerBase *Checker;
    clang::ento::BugReporter &BR;
    clang::AnalysisDeclContext *AC;
    const NamedDecl *ND;

  public:
    gpWalkAST(const CheckerBase *checker,
              clang::ento::BugReporter &br,
              clang::AnalysisDeclContext *ac,
              const NamedDecl *nd)
        : Checker(checker), BR(br), AC(ac), ND(nd) {}

    // Stmt visitor methods.
    void VisitChildren(clang::Stmt *S);
    void VisitStmt(clang::Stmt *S) { VisitChildren(S); }
    void VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *CE);
  };

  void gpWalkAST::VisitChildren(clang::Stmt *S) {
    for (clang::Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I)
      if (clang::Stmt *child = *I) {
        Visit(child);
      }
  }

  void gpWalkAST::VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *CE) {
    const FunctionDecl *FD = CE->getMethodDecl();
    if (!FD)
      return;

    std::string mname = FD->getQualifiedNameAsString();
    const char *sfile = BR.getSourceManager().getPresumedLoc(CE->getExprLoc()).getFilename();
    std::string sname(sfile);
    if (!support::isInterestingLocation(sname))
      return;
    std::string mdname = ND->getQualifiedNameAsString();
    const Expr *IOA = CE->getImplicitObjectArgument();
    std::string tname = "getparam-dumper.txt.unsorted";
    std::string ps = "const class edm::ParameterSet ";
    std::string ups = "const class edm::UntrackedParameterSet ";
    std::string gp = "edm::ParameterSet::getParameter";
    std::string gup = "edm::ParameterSet::getUntrackedParameter";
    if (mname.substr(0, gp.length()) == gp || mname.substr(0, gup.length()) == gup) {
      std::string buf;
      llvm::raw_string_ostream os(buf);
      const NamedDecl *nd = llvm::dyn_cast<NamedDecl>(AC->getDecl());
      if (FunctionDecl::classof(ND)) {
        os << "function decl '" << nd->getQualifiedNameAsString();
        os << "' this '" << mdname;
      } else {
        os << "constructor decl '" << nd->getQualifiedNameAsString();
        os << "' initializer for member decl '" << mdname;
      }
      clang::LangOptions LangOpts;
      LangOpts.CPlusPlus = true;
      clang::PrintingPolicy Policy(LangOpts);
      os << "' with call args '";
      for (unsigned I = 0, E = CE->getNumArgs(); I != E; ++I) {
        if (I)
          os << ", ";
        os << CE->getType().getCanonicalType().getAsString() << " ";
        CE->getArg(I)->printPretty(os, nullptr, Policy);
      }
      os << "' with implicit object '";
      const Expr *E = IOA->IgnoreParenCasts();
      QualType QE = E->getType().getCanonicalType();
      os << QE.getAsString() << " ";
      switch (E->getStmtClass()) {
        case Stmt::MemberExprClass:
          os << dyn_cast<MemberExpr>(E)->getMemberDecl()->getQualifiedNameAsString();
          break;
        case Stmt::DeclRefExprClass:
          os << dyn_cast<DeclRefExpr>(E)->getDecl()->getQualifiedNameAsString();
          break;
        case Stmt::CXXOperatorCallExprClass:
          dyn_cast<CXXOperatorCallExpr>(E)->printPretty(os, nullptr, Policy);
          break;
        case Stmt::CXXBindTemporaryExprClass:
          dyn_cast<CXXBindTemporaryExpr>(E)->printPretty(os, nullptr, Policy);
          break;
        case Stmt::CXXMemberCallExprClass:
          dyn_cast<CXXMemberCallExpr>(E)->printPretty(os, nullptr, Policy);
          break;
        case Stmt::UnaryOperatorClass:
          dyn_cast<UnaryOperator>(E)->printPretty(os, nullptr, Policy);
          break;
        default:
          E->printPretty(os, nullptr, Policy);
          os << " unhandled expr class " << E->getStmtClassName();
      }
      os << "'\n";

      support::writeLog(os.str(), tname);
    }
    return;
  }

  void getParamDumper::checkASTDecl(const clang::CXXRecordDecl *RD,
                                    clang::ento::AnalysisManager &mgr,
                                    clang::ento::BugReporter &BR) const {
    const clang::SourceManager &SM = BR.getSourceManager();
    const char *sfile = SM.getPresumedLoc(RD->getLocation()).getFilename();
    if (!support::isCmsLocalFile(sfile))
      return;

    std::string tname = "getparam-dumper.txt.unsorted";
    std::string ps = "const class edm::ParameterSet ";
    std::string ups = "const class edm::UntrackedParameterSet ";

    for (clang::CXXRecordDecl::ctor_iterator I = RD->ctor_begin(), E = RD->ctor_end(); I != E; ++I) {
      clang::CXXConstructorDecl *CD = llvm::dyn_cast<clang::CXXConstructorDecl>((*I)->getMostRecentDecl());
      for (unsigned I = 0, E = CD->getNumParams(); I != E; ++I) {
        std::string pname = CD->getParamDecl(I)->getQualifiedNameAsString();
        std::string qname = CD->getParamDecl(I)->getType().getCanonicalType().getAsString();
        if (qname.substr(0, ps.length()) == ps || qname.substr(0, ups.length()) == ups) {
          std::string buf;
          llvm::raw_string_ostream os(buf);
          os << "constructor decl '" << CD->getQualifiedNameAsString() << "' with parameter '" << qname << " " << pname
             << "'\n";
          support::writeLog(os.str(), tname);
          for (CXXConstructorDecl::init_iterator J = CD->init_begin(), E = CD->init_end(); J != E; ++J) {
            if (FieldDecl *fd = (*J)->getAnyMember()) {
              std::string fname = fd->getQualifiedNameAsString();
              std::string fqname = fd->getType().getCanonicalType().getAsString();
              os << "constructor decl '" << CD->getQualifiedNameAsString() << "' initializer for member decl '" << fname
                 << "' of type '" << fqname << "'\n";
              Expr *e = (*J)->getInit();
              if (e) {
                gpWalkAST walker(this, BR, mgr.getAnalysisDeclContext(CD), fd);
                walker.Visit(e);
              }
            }
          }
          support::writeLog(os.str(), tname);
        }
      }
    }

    for (clang::CXXRecordDecl::method_iterator I = RD->method_begin(), E = RD->method_end(); I != E; ++I) {
      clang::CXXMethodDecl *MD = llvm::cast<clang::CXXMethodDecl>((*I)->getMostRecentDecl());
      for (unsigned I = 0, E = MD->getNumParams(); I != E; ++I) {
        std::string pname = MD->getParamDecl(I)->getQualifiedNameAsString();
        std::string qname = MD->getParamDecl(I)->getType().getCanonicalType().getAsString();
        if (qname.substr(0, ps.length()) == ps || qname.substr(0, ups.length()) == ups) {
          std::string buf;
          llvm::raw_string_ostream os(buf);
          clang::Stmt *Body = MD->getBody();
          if (Body) {
            os << "function decl '" << MD->getQualifiedNameAsString() << "' with parameter '" << qname << " " << pname
               << "'\n";
            gpWalkAST walker(this, BR, mgr.getAnalysisDeclContext(MD), MD);
            walker.Visit(Body);
          }
          support::writeLog(os.str(), tname);
        }
      }
    }

    return;
  }

}  // namespace clangcms
