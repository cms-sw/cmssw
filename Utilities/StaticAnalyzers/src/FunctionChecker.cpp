#include <clang/AST/DeclCXX.h>
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

#include "FunctionChecker.h"

using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

class FWalker : public clang::StmtVisitor<FWalker> {
  clang::ento::BugReporter &BR;
  clang::AnalysisDeclContext *AC;

public:
  FWalker(clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac )
    : BR(br),
      AC(ac) {}

  const clang::Stmt * ParentStmt(const Stmt *S) {
  	const Stmt * P = AC->getParentMap().getParentIgnoreParens(S);
	if (!P) return 0;
	return P;
  }


  void VisitChildren(clang::Stmt *S );
  void VisitStmt( clang::Stmt *S) { VisitChildren(S); }
  void VisitDeclRefExpr( clang::DeclRefExpr * DRE);
  void ReportDeclRef( const clang::DeclRefExpr * DRE);
 
};

void FWalker::VisitChildren( clang::Stmt *S) {
  for (clang::Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (clang::Stmt *child = *I) {
      Visit(child);
    }
}


void FWalker::VisitDeclRefExpr( clang::DeclRefExpr * DRE) {
  if (clang::VarDecl * D = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl()) ) {
	if ( support::isSafeClassName(D->getCanonicalDecl()->getQualifiedNameAsString() ) ) return;
	ReportDeclRef(DRE);
  	}
}

void FWalker::ReportDeclRef ( const clang::DeclRefExpr * DRE) {
  

  if (const clang::VarDecl * D = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl())) {

	if ( support::isSafeClassName( D->getCanonicalDecl()->getQualifiedNameAsString() ) ) return;

 	const char *sfile=BR.getSourceManager().getPresumedLoc(D->getLocation()).getFilename();
	std::string fname(sfile);
	if ( fname.find("stdio.h") != std::string::npos 
		|| fname.find("iostream") != std::string::npos 
		|| fname.find("placeholders.hpp") != std::string::npos) return;	

	clang::QualType t =  D->getType();  
	const Decl * PD = AC->getDecl();
	std::string dname =""; 
	std::string sdname =""; 
	if (const NamedDecl * ND = llvm::dyn_cast<NamedDecl>(PD)) {
		sdname = support::getQualifiedName(*ND);
		dname = ND->getQualifiedNameAsString();
	}

 	clang::ento::PathDiagnosticLocation DLoc = clang::ento::PathDiagnosticLocation::createBegin(D, BR.getSourceManager());

	if ( (D->isStaticLocal() && D->getTSCSpec() != clang::ThreadStorageClassSpecifier::TSCS_thread_local ) && ! clangcms::support::isConst( t ) )
	{
	    if ( support::isSafeClassName( t.getAsString() ) ) return;
		std::string buf;
	    	llvm::raw_string_ostream os(buf);
	   	os << "function '"<<dname << "' accesses or modifies non-const static local variable '" << D->getNameAsString() << "'.\n";
		BR.EmitBasicReport(D, "FunctionChecker : non-const static local variable accessed or modified","ThreadSafety",os.str(), DLoc);
 		llvm::errs() <<  "function '"<<sdname << "' static variable '" << support::getQualifiedName(*D) << "'.\n\n";
		return;
	}

	if ( (D->isStaticDataMember() && D->getTSCSpec() != clang::ThreadStorageClassSpecifier::TSCS_thread_local ) && ! clangcms::support::isConst( t ) )
	{
	    	std::string buf;
	    	llvm::raw_string_ostream os(buf);
	    	os << "function '"<<dname<< "' accesses or modifies non-const static member data variable '" << D->getNameAsString() << "'.\n";
		BR.EmitBasicReport(D, "FunctionChecker : non-const static local variable accessed or modified","ThreadSafety",os.str(), DLoc);
 		llvm::errs() <<  "function '"<<sdname << "' static variable '" << support::getQualifiedName(*D) << "'.\n\n";
	    return;
	}

	
	if ( D->hasGlobalStorage() &&
			  !D->isStaticDataMember() &&
			  !D->isStaticLocal() &&
			  !clangcms::support::isConst( t ) )
	{
	    	std::string buf;
	    	llvm::raw_string_ostream os(buf);
	    	os << "function '"<<dname << "' accesses or modifies non-const global static variable '" << D->getNameAsString() << "'.\n";
		BR.EmitBasicReport(D, "FunctionChecker : non-const static local variable accessed or modified","ThreadSafety",os.str(), DLoc);
 		llvm::errs() <<  "function '"<<sdname << "' static variable '" << support::getQualifiedName(*D) << "'.\n\n";
	    return;
	
	}

  }


}


void FunctionChecker::checkASTDecl(const CXXMethodDecl *MD, AnalysisManager& mgr,
                    BugReporter &BR) const {

 	const char *sfile=BR.getSourceManager().getPresumedLoc(MD->getLocation()).getFilename();
 	if (!support::isCmsLocalFile(sfile)) return;
	std::string fname(sfile);
	if ( fname.find("/test/") != std::string::npos) return;
  
      	if (!MD->doesThisDeclarationHaveABody()) return;
	FWalker walker(BR, mgr.getAnalysisDeclContext(MD));
	walker.Visit(MD->getBody());
       	return;
} 

void FunctionChecker::checkASTDecl(const FunctionTemplateDecl *TD, AnalysisManager& mgr,
                    BugReporter &BR) const {

 	const char *sfile=BR.getSourceManager().getPresumedLoc(TD->getLocation ()).getFilename();
   	if (!support::isCmsLocalFile(sfile)) return;
	std::string fname(sfile);
	if ( fname.find("/test/") != std::string::npos) return;
  
	for (FunctionTemplateDecl::spec_iterator I = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_begin(), 
			E = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_end(); I != E; ++I) 
		{
			if (I->doesThisDeclarationHaveABody()) {
				FWalker walker(BR, mgr.getAnalysisDeclContext(*I));
				walker.Visit(I->getBody());
				}
		}	
	return;
}



}
