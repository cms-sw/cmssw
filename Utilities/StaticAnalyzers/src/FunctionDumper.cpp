#include <clang/AST/DeclCXX.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/ParentMap.h>
#include <clang/Analysis/CFGStmtMap.h>
#include <clang/Analysis/CallGraph.h>
#include <llvm/Support/SaveAndRestore.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h>
#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugReporter.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include <llvm/ADT/SmallString.h>
#include <clang/AST/DeclTemplate.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm> 
#include "CmsException.h"
#include "CmsSupport.h"
#include "FunctionDumper.h"


using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

class FDumper : public clang::StmtVisitor<FDumper> {
  clang::ento::BugReporter &BR;
  clang::AnalysisDeclContext *AC;
  const FunctionDecl *AD;

  enum Kind { NotVisited,
              Visited };

  /// A DenseMap that records visited states of CallExpr.
  llvm::DenseMap<const clang::Expr *, Kind> VisitedExpr;

public:
  FDumper(clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac, const FunctionDecl * fd )
    : BR(br),
      AC(ac),
      AD(fd) {}


  /// This method adds a CallExpr to the worklist 
  void setVisited(Expr * E) {
      Kind &K = VisitedExpr[E];
      if ( K = NotVisited ) {
       VisitedExpr[E] = Visited;
       return;
      }
  }

  bool wasVisited(Expr * E) {
      Kind &K = VisitedExpr[E];
      if ( K = Visited ) return true;
      return false;
  }

  const clang::Stmt * ParentStmt(const Stmt *S) {
  const Stmt * P = AC->getParentMap().getParentIgnoreParens(S);
  if (!P) return 0;
  return P;
  }

  void VisitChildren(clang::Stmt *S );
  void VisitStmt( clang::Stmt *S) { VisitChildren(S); }
  void VisitCallExpr( CallExpr *CE ); 
  void VisitCXXMemberCallExpr( CXXMemberCallExpr *CXE ); 
  void VisitCXXConstructExpr( CXXConstructExpr *CCE ); 
 
};

void FDumper::VisitChildren( clang::Stmt *S) {
      for (clang::Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
        if (clang::Stmt *child = *I) {
          Visit(child);
      }
}

void FDumper::VisitCXXConstructExpr( CXXConstructExpr *CCE ) {
     std::string buf;
     llvm::raw_string_ostream os(buf);
     LangOptions LangOpts;
     LangOpts.CPlusPlus = true;
     PrintingPolicy Policy(LangOpts);
     std::string mdname = support::getQualifiedName(*AD);
     CXXConstructorDecl * CCD = CCE->getConstructor();
     if (!CCD) return;
     const char *sfile=BR.getSourceManager().getPresumedLoc(CCE->getExprLoc()).getFilename();
     std::string sname(sfile);
     if ( ! support::isInterestingLocation(sname) ) return;
     std::string mname;
     mname = support::getQualifiedName(*CCD);
     std::string tname = "function-dumper.txt.unsorted";
     std::string ostring = "function '"+ mdname +  "' " + "calls function '" + mname + "'\n";
     support::writeLog(ostring,tname);
 
     VisitChildren(CCE);
}

void FDumper::VisitCXXMemberCallExpr( CXXMemberCallExpr *CXE ) {
     std::string buf;
     llvm::raw_string_ostream os(buf);
     LangOptions LangOpts;
     LangOpts.CPlusPlus = true;
     PrintingPolicy Policy(LangOpts);
     std::string mdname = support::getQualifiedName(*AD);
     CXXMethodDecl * MD = CXE->getMethodDecl();
     if (!MD) return;
     const char *sfile=BR.getSourceManager().getPresumedLoc(CXE->getExprLoc()).getFilename();
     std::string sname(sfile);
     if ( ! support::isInterestingLocation(sname) ) return;
     std::string mname;
     mname = support::getQualifiedName(*MD);
     std::string tname = "function-dumper.txt.unsorted";
     std::string ostring;
     if ( MD->isVirtual()) ostring = "function '"+ mdname +  "' " + "calls function '" + mname + " virtual'\n";
     else ostring = "function '"+ mdname +  "' " + "calls function '" + mname + "'\n"; 
     support::writeLog(ostring,tname);

     VisitChildren(CXE);
}

void FDumper::VisitCallExpr( CallExpr *CE ) {
     std::string buf;
     llvm::raw_string_ostream os(buf);
     LangOptions LangOpts;
     LangOpts.CPlusPlus = true;
     PrintingPolicy Policy(LangOpts);
     std::string mdname = support::getQualifiedName(*AD);
     FunctionDecl * FD = CE->getDirectCallee();
     if (!FD) return;
     const char *sfile=BR.getSourceManager().getPresumedLoc(CE->getExprLoc()).getFilename();
     std::string sname(sfile);
     if ( ! support::isInterestingLocation(sname) ) return;
     std::string mname;
     mname = support::getQualifiedName(*FD);
     std::string tname = "function-dumper.txt.unsorted";
     std::string ostring;
     if (FD->isVirtualAsWritten() || FD->isPure())
         ostring = "function '"+ mdname +  "' " + "calls function '" + mname + " virtual'\n";
     else ostring = "function '"+ mdname +  "' " + "calls function '" + mname + "'\n"; 
     support::writeLog(ostring,tname);
   
     VisitChildren(CE);
}

void FunctionDumper::checkASTDecl(const CXXMethodDecl *MD, AnalysisManager& mgr,
                    BugReporter &BR) const {
      if (MD->getLocation().isInvalid()) return;
      const char *sfile=BR.getSourceManager().getPresumedLoc(MD->getLocation()).getFilename();
      std::string sname(sfile);
      if ( ! support::isInterestingLocation(sname) ) return;
      if ( ! support::isCmsLocalFile(sfile) ) return;
      if (!MD->doesThisDeclarationHaveABody()) return;
      FDumper walker(BR, mgr.getAnalysisDeclContext(MD), MD);
      walker.Visit(MD->getBody());
      std::string mname = support::getQualifiedName(*MD);
      std::string tname = "function-dumper.txt.unsorted";
      for (auto I = MD->begin_overridden_methods(), E = MD->end_overridden_methods(); I!=E; ++I) {
          std::string oname = support::getQualifiedName(*(*I));
          std::string ostring = "function '" +  mname + "' " + "overrides function '" + oname + " virtual'\n";
          support::writeLog(ostring,tname);
      }
             return;
} 

void FunctionDumper::checkASTDecl(const FunctionDecl *MD, AnalysisManager& mgr,
                    BugReporter &BR) const {
      if (MD->getLocation().isInvalid()) return;
      const char *sfile=BR.getSourceManager().getPresumedLoc(MD->getLocation()).getFilename();
      std::string sname(sfile);
      if ( ! support::isInterestingLocation(sname) ) return;
      if ( ! support::isCmsLocalFile(sfile) ) return;
      if (!MD->doesThisDeclarationHaveABody()) return;
      FDumper walker(BR, mgr.getAnalysisDeclContext(MD), MD);
      walker.Visit(MD->getBody());
             return;
} 



void FunctionDumper::checkASTDecl(const FunctionTemplateDecl *TD, AnalysisManager& mgr,
                    BugReporter &BR) const {
     if (TD->getLocation().isInvalid()) return;
     const char *sfile=BR.getSourceManager().getPresumedLoc(TD->getLocation ()).getFilename();
     std::string sname(sfile);
     if ( ! support::isInterestingLocation(sname) ) return;
     if ( ! support::isCmsLocalFile(sfile) ) return;
     for (FunctionTemplateDecl::spec_iterator I = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_begin(), 
               E = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_end(); I != E; ++I) 
          {
               if (I->doesThisDeclarationHaveABody()) {
                   FDumper walker(BR, mgr.getAnalysisDeclContext(*I), (*I));
                   walker.Visit(I->getBody());
                   }
          }
     return;
}



}
