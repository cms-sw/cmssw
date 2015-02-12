#include <clang/AST/DeclCXX.h>
#include <clang/AST/ExprCXX.h>
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

const std::string getArgumentValueString(const CallExpr *CE, CheckerContext &C, unsigned &I) {
     if (CE->getNumArgs() == 0)
       return "Missing arguments";
   
     ExplodedNode *N = C.getPredecessor();
     const LocationContext *LC = N->getLocationContext();
     ProgramStateRef State = N->getState();
   
     const Expr *arg  = CE->getArg(I);
     SVal argVal = State->getSVal(arg, LC);
     
     if (argVal.isUndef()) return "UNDEFINED";
   
     ProgramStateRef StTrue, StFalse;
     std::tie(StTrue, StFalse) = State->assume(argVal.castAs<DefinedOrUnknownSVal>());
   
     if (StTrue) {
       std::string buf;
       llvm::raw_string_ostream os(buf);
       argVal.dumpToStream(os);
       return os.str();
     }
}



void getParamDumper::analyzerEval(const clang::CallExpr *CE, clang::ento::CheckerContext &C) const {

  const FunctionDecl * FD = CE->getDirectCallee();

  if (!FD) return;

    std::string mname = support::getQualifiedName(*FD);
    const char *sfile=C.getSourceManager().getPresumedLoc(CE->getExprLoc()).getFilename();
    std::string sname(sfile);
//    if ( ! support::isInterestingLocation(sname) ) return;
    std::string mdname;
    const FunctionDecl * MD = C.getCurrentAnalysisDeclContext()->getDecl()->getAsFunction();
    if (MD) mdname = support::getQualifiedName(*MD);
    std::string tname = "getparam-dumper.txt.unsorted";
    std::string ostring = "function '"+mdname+ "' calls function'" + mname + "' with args '";
    std::string gp = "edm::ParameterSet::getParameter";
    std::string gup = "edm::ParameterSet::getUntrackedParameter";
    if (mname.substr(0,gp.length()) == gp || mname.substr(0,gup.length()) == gup ) {
         for ( unsigned I=0, E=CE->getNumArgs(); I != E; ++I) {
             const std::string qt = getArgumentValueString(CE, C, I);
             ostring = ostring + qt + " ";
         }
    ostring += "'\n";
    support::writeLog(ostring,tname);
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

