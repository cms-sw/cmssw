#include <clang/AST/DeclCXX.h>
#include <clang/AST/Attr.h>
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
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm> 

#include "FunctionChecker.h"

using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

class FWalker : public clang::StmtVisitor<FWalker> {
  const CheckerBase *Checker;
  clang::ento::BugReporter &BR;
  clang::AnalysisDeclContext *AC;

public:
  FWalker(const CheckerBase * checker, clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac )
    : Checker(checker),
      BR(br),
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
  if (const clang::VarDecl * D = llvm::dyn_cast_or_null<clang::VarDecl>(DRE->getDecl()) ) {
	if ( D && support::isSafeClassName(D->getCanonicalDecl()->getQualifiedNameAsString() ) ) return;
	ReportDeclRef(DRE);
  }
}

void FWalker::ReportDeclRef ( const clang::DeclRefExpr * DRE) {
  
        const clang::VarDecl * D = llvm::dyn_cast_or_null<clang::VarDecl>(DRE->getDecl());
	if ( D && ( D->hasAttr<CMSThreadGuardAttr>() || D->hasAttr<CMSThreadSafeAttr>())) return;
	if ( support::isSafeClassName( D->getCanonicalDecl()->getQualifiedNameAsString() ) ) return;

 	const char *sfile=BR.getSourceManager().getPresumedLoc(D->getLocation()).getFilename();
	std::string fname(sfile);
	if (!support::isInterestingLocation(fname)) return;
	clang::QualType t =  D->getType();  
	if ( support::isSafeClassName( t.getCanonicalType().getAsString() ) ) return;
	const Decl * PD = AC->getDecl();
	std::string dname =""; 
	std::string sdname =""; 
	if (const NamedDecl * ND = llvm::dyn_cast_or_null<NamedDecl>(PD)) {
		sdname = support::getQualifiedName(*ND);
		dname = ND->getQualifiedNameAsString();
	}
	clang::ento::PathDiagnosticLocation DLoc;
	if (support::isCmsLocalFile(sfile)) {
		if (D->getLocation().isMacroID()) 
			DLoc = clang::ento::PathDiagnosticLocation(D->getLocation(),BR.getSourceManager());
		else 
			DLoc = clang::ento::PathDiagnosticLocation::createBegin(D, BR.getSourceManager());
	} else 
		DLoc = clang::ento::PathDiagnosticLocation::createBegin(DRE, BR.getSourceManager(), AC);

	std::string tname ="function-checker.txt.unsorted";

	std::string vname = support::getQualifiedName(*D);
	std::string svname = D->getNameAsString();
	if ( D->getTSCSpec() == clang::ThreadStorageClassSpecifier::TSCS_thread_local ) return;
	if ( D->isStaticLocal()  && ! clangcms::support::isConst( t ) )
	{
		std::string buf;
	    	llvm::raw_string_ostream os(buf);
		os << "function '"<<dname << "' accesses or modifies non-const static local variable '" << svname<< "'.\n";
//		BR.EmitBasicReport(D, Checker, "non-const static local variable accessed or modified","ThreadSafety",os.str(), DLoc);
		std::string ostring =  "function '"+ sdname + "' static variable '" + vname + "'.\n";
		support::writeLog(ostring,tname);
		return;
	}

	if ( D->isStaticDataMember() && ! clangcms::support::isConst( t ) )
	{
	    	std::string buf;
	    	llvm::raw_string_ostream os(buf);
		os << "function '"<<dname<< "' accesses or modifies non-const static member data variable '" << svname << "'.\n";
//		BR.EmitBasicReport(D, Checker, "non-const static local variable accessed or modified","ThreadSafety",os.str(), DLoc);
		std::string ostring =  "function '" + sdname + "' static variable '" + vname + "'.\n";
		support::writeLog(ostring,tname);
	    return;
	}

	
	if ( D->hasGlobalStorage() &&
			  !D->isStaticDataMember() &&
			  !D->isStaticLocal() &&
			  !clangcms::support::isConst( t ) )
	{
	    	std::string buf;
	    	llvm::raw_string_ostream os(buf);
		os << "function '"<<dname << "' accesses or modifies non-const global static variable '" << svname << "'.\n";
//		BR.EmitBasicReport(D, Checker, "non-const static local variable accessed or modified","ThreadSafety",os.str(), DLoc);
		std::string ostring =  "function '" + sdname + "' static variable '" + vname + "'.\n";
		support::writeLog(ostring,tname);
	    return;
	
	}


}


void FunctionChecker::checkASTDecl(const CXXMethodDecl *MD, AnalysisManager& mgr,
                    BugReporter &BR) const {
	if ( MD->hasAttr<CMSThreadSafeAttr>()) return;
 	const char *sfile=BR.getSourceManager().getPresumedLoc(MD->getLocation()).getFilename();
 	if (!support::isCmsLocalFile(sfile)) return;
	std::string fname(sfile);
	if ( !support::isInterestingLocation(fname) ) return;
      	if (!MD->doesThisDeclarationHaveABody()) return;
	FWalker walker(this, BR, mgr.getAnalysisDeclContext(MD));
	walker.Visit(MD->getBody());
       	return;
} 

void FunctionChecker::checkASTDecl(const FunctionDecl *FD, AnalysisManager& mgr,
                    BugReporter &BR) const {
	if ( FD->hasAttr<CMSThreadSafeAttr>()) return;
        if (FD-> isInExternCContext()) {
                std::string buf;
                std::string dname = FD->getQualifiedNameAsString();
                if ( dname.compare(dname.size()-1,1,"_") == 0 ) {
                llvm::raw_string_ostream os(buf);
                os << "function '"<< dname << "' is in an extern \"C\" context and most likely accesses or modifies fortran variables in a 'COMMONBLOCK'.\n";
                clang::ento::PathDiagnosticLocation::createBegin(FD, BR.getSourceManager());
//		BR.EmitBasicReport(FD, "COMMONBLOCK variable accessed or modified","ThreadSafety",os.str(), FDLoc);
                std::string ostring =  "function '" + dname + "' static variable 'COMMONBLOCK'.\n";
		std::string tname = "function-checker.txt.unsorted";
		support::writeLog(ostring,tname);
		}
        }

 	const char *sfile=BR.getSourceManager().getPresumedLoc(FD->getLocation ()).getFilename();
   	if (!support::isCmsLocalFile(sfile)) return;
	std::string fname(sfile);
	if ( !support::isInterestingLocation(fname) ) return;
	if (FD->doesThisDeclarationHaveABody()) {
		FWalker walker(this, BR, mgr.getAnalysisDeclContext(FD));
		walker.Visit(FD->getBody());
		}
}

void FunctionChecker::checkASTDecl(const FunctionTemplateDecl *TD, AnalysisManager& mgr,
                    BugReporter &BR) const {

	if ( TD->hasAttr<CMSThreadSafeAttr>()) return;
 	const char *sfile=BR.getSourceManager().getPresumedLoc(TD->getLocation ()).getFilename();
   	if (!support::isCmsLocalFile(sfile)) return;
	std::string fname(sfile);
	if ( !support::isInterestingLocation(fname) ) return;
	for (FunctionTemplateDecl::spec_iterator I = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_begin(), 
			E = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_end(); I != E; ++I) 
		{
			if (I->doesThisDeclarationHaveABody()) {
				FWalker walker(this,BR, mgr.getAnalysisDeclContext(*I));
				walker.Visit(I->getBody());
				}
		}
	return;
}



}
