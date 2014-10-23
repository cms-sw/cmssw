#include <clang/AST/Attr.h>
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
	BR.getContext();
  	llvm::SmallString<100> buf;
  	llvm::raw_svector_ostream os(buf);
	const Expr * SE = CE->getSubExprAsWritten();	
	std::string sname = SE->getType().getAsString();
	if (!(sname=="EventNumber_t")) return true;
	const clang::Expr *E = CE->getSubExpr();
	clang::QualType OrigTy = BR.getContext().getCanonicalType(E->getType());
	clang::QualType ToTy = BR.getContext().getCanonicalType(CE->getType());
	if (!OrigTy->isIntegerType()) return true;
	if (!(ToTy->isIntegerType()||ToTy->isFloatingType())) return true;
	uint64_t size_otype = BR.getContext().getTypeSize(OrigTy);
	uint64_t size_ttype = BR.getContext().getTypeSize(ToTy);
	std::string oname = OrigTy.getAsString();
	std::string tname = ToTy.getAsString();

	if ( size_ttype < size_otype || ToTy->isFloatingType() ) {
		os <<"Size of cast-to type, "<<tname<<", is smaller than cast-from type, "<<oname<<", and may result in truncation"; 
		clang::ento::PathDiagnosticLocation CELoc = clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
 		BugType * BT = new BugType(CheckName(), "implicit cast size truncates","CMS code rules");
		BugReport * R = new BugReport(*BT,os.str(),CELoc);
		R->addRange(CE->getSourceRange());
		BR.emitReport(R);
		return true;
	}
	return true;
}

void TrunCastChecker::checkASTDecl(const TranslationUnitDecl *D, AnalysisManager& Mgr,BugReporter &BR) const  {
   	ICEVisitor icevisitor(this, BR, Mgr.getAnalysisDeclContext(D));
     	icevisitor.TraverseDecl(const_cast<TranslationUnitDecl *>(D));
}


}
