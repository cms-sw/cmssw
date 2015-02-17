#include <clang/AST/DeclCXX.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/StmtCXX.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <llvm/ADT/StringSwitch.h>
#include "CmsException.h"
#include "CmsSupport.h"
#include "getParamDumper.h"

using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

void getParamDumper::analyzerEval(const clang::CallExpr *CE, clang::ento::CheckerContext &C) const {

  if ( ! C.getSourceManager().isInMainFile(CE->getExprLoc()) ) return;

  const FunctionDecl * FD = CE->getDirectCallee();

  if (!FD) return;

    std::string mname = support::getQualifiedName(*FD);
    const char *sfile=C.getSourceManager().getPresumedLoc(CE->getExprLoc()).getFilename();
    std::string sname(sfile);
    if ( ! support::isInterestingLocation(sname) ) return;
    std::string mdname;
    const FunctionDecl * MD = C.getCurrentAnalysisDeclContext()->getDecl()->getAsFunction();
    if (MD) mdname = support::getQualifiedName(*MD);
    const CXXMemberCallExpr * CXE = llvm::dyn_cast_or_null<CXXMemberCallExpr>(CE);
    if (!CXE) return;
    const Expr * IOA = CXE->getImplicitObjectArgument();
    std::string tname = "getparam-dumper.txt.unsorted";
    std::string gp = "edm::ParameterSet::getParameter";
    std::string gup = "edm::ParameterSet::getUntrackedParameter";
    if (mname.substr(0,gp.length()) == gp || mname.substr(0,gup.length()) == gup ) {
         std::string buf;
         llvm::raw_string_ostream os(buf);
         os << "in function decl '" << mdname << "' member function call '";
         clang::LangOptions LangOpts;
         LangOpts.CPlusPlus = true;
         clang::PrintingPolicy Policy(LangOpts);
         os << support::getQualifiedName(*(CXE->getMethodDecl()));
         os << "' with args '";
         for ( unsigned I=0, E=CE->getNumArgs(); I != E; ++I) {
              if (I) os <<", ";
              CE->getArg(I)->printPretty(os,0,Policy);
         }
         os << "' with implicit object '";
         const Expr * E = IOA->IgnoreParenNoopCasts(C.getASTContext());
         switch( E->getStmtClass() ) {
             case Stmt::MemberExprClass: 
                 os << support::getQualifiedName(*(dyn_cast<MemberExpr>(E)->getMemberDecl()));
                 break;
             case Stmt::DeclRefExprClass:
                 os << support::getQualifiedName(*(dyn_cast<DeclRefExpr>(E)->getDecl()));
                 break;
             case Stmt::CXXOperatorCallExprClass:  
                 dyn_cast<CXXOperatorCallExpr>(E)->getArg(0)->printPretty(os,0,Policy);
                 break;
             case Stmt::CXXBindTemporaryExprClass:
                 { 
                     const Expr * SE = dyn_cast<CXXBindTemporaryExpr>(E)->getSubExpr();
                     const Expr * SOA = dyn_cast<CXXMemberCallExpr>(SE)->getImplicitObjectArgument();
                     SOA->printPretty(os,0,Policy);
                 }
                 break;
             case Stmt::CXXMemberCallExprClass:
                 dyn_cast<CXXMemberCallExpr>(E)->getImplicitObjectArgument()->printPretty(os,0,Policy);
                 break;
             case Stmt::UnaryOperatorClass:
                 dyn_cast<UnaryOperator>(E)->getSubExpr()->printPretty(os,0,Policy);
                 break;
             default:
                 E->printPretty(os,0,Policy);
                 os << " unhandled expr class " <<E->getStmtClassName();
             }


         os<<"'\n";

         support::writeLog(os.str(),tname);
  }
  return ;
}


bool getParamDumper::evalCall(const CallExpr *CE, CheckerContext &C) const {

  FnCheck Handler = llvm::StringSwitch<FnCheck>(C.getCalleeName(CE))
    .Case("getParameter",&getParamDumper::analyzerEval)
    .Case("getUntrackedParameter",&getParamDumper::analyzerEval)
    .Default(nullptr);

  if (!Handler) return false;

  (this->*Handler)(CE,C);

  return true;
}



}

