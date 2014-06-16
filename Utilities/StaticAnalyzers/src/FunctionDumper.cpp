#include "FunctionDumper.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm> 

using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

class FDumper : public clang::StmtVisitor<FDumper> {
  clang::ento::BugReporter &BR;
  clang::AnalysisDeclContext *AC;

  enum Kind { NotVisited,
              Visited };

  /// A DenseMap that records visited states of CallExpr.
  llvm::DenseMap<const clang::Expr *, Kind> VisitedExpr;

public:
  FDumper(clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac )
    : BR(br),
      AC(ac) {}


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
  void VisitCXXConstructExpr( CXXConstructExpr *CCE ); 
 
};

void FDumper::VisitChildren( clang::Stmt *S) {
  for (clang::Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (clang::Stmt *child = *I) {
      Visit(child);
    }
}

void FDumper::VisitCXXConstructExpr( CXXConstructExpr *CCE ) {

	LangOptions LangOpts;
	LangOpts.CPlusPlus = true;
	PrintingPolicy Policy(LangOpts);
	const Decl * D = AC->getDecl();
	std::string mdname =""; 
	if (const NamedDecl * ND = llvm::dyn_cast<NamedDecl>(D)) mdname = support::getQualifiedName(*ND);
	CXXConstructorDecl * CCD = CCE->getConstructor();
	if (!CCD) return;
	const char *sfile=BR.getSourceManager().getPresumedLoc(CCE->getExprLoc()).getFilename();
	std::string sname(sfile);
	if ( sname.find("/test/") != std::string::npos) return;
	std::string mname = support::getQualifiedName(*CCD);
	const char * pPath = std::getenv("LOCALRT");
	std::string tname = ""; 
	if ( pPath != NULL ) tname += std::string(pPath);
	tname+="/tmp/function-dumper.txt.unsorted";
	std::string ostring = "function '"+ mdname +  "' " + "calls function '" + mname + "'\n"; 
	std::ofstream file(tname.c_str(),std::ios::app);
	file<<ostring;	
	VisitChildren(CCE);
}


void FDumper::VisitCallExpr( CallExpr *CE ) {
	LangOptions LangOpts;
	LangOpts.CPlusPlus = true;
	PrintingPolicy Policy(LangOpts);
	const Decl * D = AC->getDecl();
	std::string mdname =""; 
	if (const NamedDecl * ND = llvm::dyn_cast<NamedDecl>(D)) mdname = support::getQualifiedName(*ND);
	FunctionDecl * FD = CE->getDirectCallee();
	if (!FD) return;
 	const char *sfile=BR.getSourceManager().getPresumedLoc(CE->getExprLoc()).getFilename();
	std::string sname(sfile);
	if ( sname.find("/test/") != std::string::npos) return;
 	std::string mname = support::getQualifiedName(*FD);
	const char * pPath = std::getenv("LOCALRT");
	std::string tname = ""; 
	if ( pPath != NULL ) tname += std::string(pPath);
	tname+="/tmp/function-dumper.txt.unsorted";
	std::string ostring;
	CXXMemberCallExpr * CXE = llvm::dyn_cast<CXXMemberCallExpr>(CE);
	if (CXE) {
		const CXXMethodDecl * CD = CXE->getMethodDecl();
		const CXXRecordDecl * RD = CXE->getRecordDecl();
		const Expr * IOA = CXE->getImplicitObjectArgument();
		const CXXMethodDecl * AMD = llvm::dyn_cast<CXXMethodDecl>(D);
		if ( AMD && CD && RD && CD->isVirtual() && RD == AMD->getParent() ) ostring = "function '"+ mdname +  "' " + "calls function '" + mname + " virtual'\n";
		else ostring = "function '"+ mdname +  "' " + "calls function '" + mname + "'\n"; 
	} else {
		if (FD->isVirtualAsWritten() || FD->isPure()) ostring = "function '"+ mdname +  "' " + "calls function '" + mname + " virtual'\n"; 
		else ostring = "function '"+ mdname +  "' " + "calls function '" + mname + "'\n"; 
	}
	std::ofstream file(tname.c_str(),std::ios::app);
	file<<ostring;
	VisitChildren(CE);
}

void FunctionDumper::checkASTDecl(const CXXMethodDecl *MD, AnalysisManager& mgr,
                    BugReporter &BR) const {

 	const char *sfile=BR.getSourceManager().getPresumedLoc(MD->getLocation()).getFilename();
	std::string sname(sfile);
	if ( sname.find("/test/") != std::string::npos) return;
	if (!MD->doesThisDeclarationHaveABody()) return;
	FDumper walker(BR, mgr.getAnalysisDeclContext(MD));
	walker.Visit(MD->getBody());
        std::string mname = support::getQualifiedName(*MD);
	const char * pPath = std::getenv("LOCALRT");
	std::string tname=""; 
	if ( pPath != NULL ) tname += std::string(pPath);
	tname += "/tmp/function-dumper.txt.unsorted";
	for (auto I = MD->begin_overridden_methods(), E = MD->end_overridden_methods(); I!=E; ++I) {
		std::string oname = support::getQualifiedName(*(*I));
		std::string ostring = "function '" +  mname + "' " + "overrides function '" + oname + " virtual'\n";
		std::ofstream file(tname.c_str(),std::ios::app);
		file<<ostring;
	}
       	return;
} 

void FunctionDumper::checkASTDecl(const FunctionTemplateDecl *TD, AnalysisManager& mgr,
                    BugReporter &BR) const {

 	const char *sfile=BR.getSourceManager().getPresumedLoc(TD->getLocation ()).getFilename();
	std::string sname(sfile);
	if ( sname.find("/test/") != std::string::npos) return;
  
	for (FunctionTemplateDecl::spec_iterator I = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_begin(), 
			E = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_end(); I != E; ++I) 
		{
			if (I->doesThisDeclarationHaveABody()) {
				FDumper walker(BR, mgr.getAnalysisDeclContext(*I));
				walker.Visit(I->getBody());
				}
		}	
	return;
}



}
