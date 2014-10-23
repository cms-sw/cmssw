#include <clang/AST/Attr.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/Expr.h>
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
#include "TrunCastChecker.h"
#include "CmsSupport.h" 

using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms
{

class ICEVisitor: public clang::RecursiveASTVisitor<ICEVisitor> {
	const clang::ento::CheckerBase *Checker;
	clang::ento::BugReporter &BR;
	clang::AnalysisDeclContext* AC;

 public:
   ICEVisitor(const clang::ento::CheckerBase *checker, clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac)
       : Checker(checker), BR(br), AC(ac) {}
 
bool VisitImplicitCastExpr( ImplicitCastExpr *CE);

};

bool ICEVisitor::VisitImplicitCastExpr( ImplicitCastExpr *CE )  
{
 	const char *sfile=BR.getSourceManager().getPresumedLoc(CE->getExprLoc()).getFilename();
	std::string sname(sfile);
	if ( ! support::isInterestingLocation(sname) || ! support::isCmsLocalFile(sfile) ) return true;	
	BR.getContext();
  	llvm::SmallString<100> buf;
  	llvm::raw_svector_ostream os(buf);
	const Expr * SE = CE->getSubExprAsWritten();
	std::string sename = SE->getType().getAsString();
	std::string evn = "EventNumber_t";
	std::string lname;
	std::string rname;
	const clang::Expr *E = CE->getSubExpr();
	if ( BinaryOperator::classof(SE) ) {
		const BinaryOperator * BO = dyn_cast<BinaryOperator>(SE);
		lname = BO->getLHS()->getType().getAsString();
		rname = BO->getRHS()->getType().getAsString();
	}
	if (!(sename==evn || lname==evn || rname==evn )) return true;	
	clang::QualType OrigTy = BR.getContext().getCanonicalType(E->getType());
	clang::QualType ToTy = BR.getContext().getCanonicalType(CE->getType());
	if (!(ToTy->isIntegerType()||ToTy->isFloatingType()) ) return true;
	if ( ToTy->isBooleanType() ) return true;
	CharUnits size_otype = BR.getContext().getTypeSizeInChars(OrigTy);
	CharUnits size_ttype = BR.getContext().getTypeSizeInChars(ToTy);
	std::string oname = OrigTy.getAsString();
	std::string tname = ToTy.getAsString();
	if ( ToTy->isFloatingType() ) {
		os <<"Cast-to type, "<<tname<<". Cast-from type, "<<oname; 
		clang::ento::PathDiagnosticLocation CELoc = clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
 		BugType * BT = new BugType(CheckName(), "implicit cast of int type to float type","CMS code rules");
		BugReport * R = new BugReport(*BT,os.str(),CELoc);
		R->addRange(CE->getSourceRange());
		BR.emitReport(R);
		return true;
		} 
	else {
		if (  (size_otype > size_ttype) || ( size_otype == size_ttype && ToTy->getTypeClass() != OrigTy->getTypeClass() ) ) {
			os <<"Cast-to type, "<<tname<<". Cast-from type, "<<oname<<". Cast may result in truncation."; 
			clang::ento::PathDiagnosticLocation CELoc = clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
 			BugType * BT = new BugType(CheckName(), "implicit cast of int type to different int type could truncate","CMS code rules");
			BugReport * R = new BugReport(*BT,os.str(),CELoc);
			R->addRange(CE->getSourceRange());
			BR.emitReport(R);
			return true;
		} 
	}
	return true;
}

void TrunCastChecker::checkASTDecl(const TranslationUnitDecl *D, AnalysisManager& Mgr,BugReporter &BR) const  {
   	ICEVisitor icevisitor(this, BR, Mgr.getAnalysisDeclContext(D));
     	icevisitor.TraverseDecl(const_cast<TranslationUnitDecl *>(D));
}


}
