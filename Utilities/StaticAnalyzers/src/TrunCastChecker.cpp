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

class ICEVisitor: public clang::StmtVisitor<ICEVisitor> {
	const clang::ento::CheckerBase *Checker;
	clang::ento::BugReporter &BR;
	clang::AnalysisDeclContext* AC;

 public:
   ICEVisitor(const clang::ento::CheckerBase *checker, clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac)
       : Checker(checker), BR(br), AC(ac) {}
 
void VisitImplicitCastExpr( ImplicitCastExpr *CE );
void VisitBinaryOperator( BinaryOperator *BO );
void VisitChildren( clang::Stmt *S );
void VisitStmt( clang::Stmt *S ) { VisitChildren(S); }
};

void ICEVisitor::VisitChildren( clang::Stmt *S ) {
  for ( auto I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (clang::Stmt *child = *I) {
      Visit(child);
    }
}


void ICEVisitor::VisitBinaryOperator( BinaryOperator *BO )
{
	const NamedDecl * ACD = dyn_cast_or_null<NamedDecl>(AC->getDecl());
	VisitChildren(BO);
	std::string ename = "EventNumber_t";
	clang::Expr * LHS = BO->getLHS();
	clang::Expr * RHS = BO->getRHS();
	if (!LHS || !RHS) return;
	std::string lname = LHS->getType().getAsString();
	std::string rname = RHS->getType().getAsString();
	if (IntegerLiteral::classof(LHS->IgnoreCasts()) || IntegerLiteral::classof(RHS->IgnoreCasts())) return;
	if (!(lname == ename || rname == ename)) return;
	if (  lname == ename && rname == ename ) return;
	clang::QualType OTy;
	clang::QualType TTy;
	if (lname == ename && ImplicitCastExpr::classof(RHS) ) {
		ImplicitCastExpr * ICE = dyn_cast_or_null<ImplicitCastExpr>(RHS);
		TTy = BR.getContext().getCanonicalType(LHS->getType());
		OTy = BR.getContext().getCanonicalType(ICE->getSubExprAsWritten()->getType());
	}
	if (rname == ename && ImplicitCastExpr::classof(LHS) ) {
		ImplicitCastExpr * ICE = dyn_cast_or_null<ImplicitCastExpr>(LHS);
		TTy = BR.getContext().getCanonicalType(RHS->getType());
		OTy = BR.getContext().getCanonicalType(ICE->getSubExprAsWritten()->getType());
	}
	if ( TTy.isNull() || OTy.isNull() ) return;
	QualType ToTy = TTy.getUnqualifiedType();
	QualType OrigTy = OTy.getUnqualifiedType();
	if (!(ToTy->isIntegerType()||ToTy->isFloatingType()) ) return;
	if ( ToTy->isBooleanType() ) return;
	CharUnits size_otype = BR.getContext().getTypeSizeInChars(OrigTy);
	CharUnits size_ttype = BR.getContext().getTypeSizeInChars(ToTy);
	std::string oname = OrigTy.getAsString();
	std::string tname = ToTy.getAsString();
	if ( ToTy->isFloatingType() ) {
  		llvm::SmallString<100> buf;
  		llvm::raw_svector_ostream os(buf);
		os <<"Cast-to type, "<<tname<<". Cast-from type, "<<oname<<" . "<<support::getQualifiedName(*(ACD)); 
		clang::ento::PathDiagnosticLocation CELoc = clang::ento::PathDiagnosticLocation::createBegin(BO, BR.getSourceManager(),AC);
		BR.EmitBasicReport(ACD,CheckName(),"implicit cast of int type to float type","CMS code rules", os.str(),CELoc, BO->getSourceRange());
		} 
	if (  (size_otype > size_ttype) ) {
  		llvm::SmallString<100> buf;
  		llvm::raw_svector_ostream os(buf);
		os <<"Cast-to type, "<<tname<<". Cast-from type, "<<oname<<". Cast may result in truncation. "<<support::getQualifiedName(*(ACD)); 
		clang::ento::PathDiagnosticLocation CELoc = clang::ento::PathDiagnosticLocation::createBegin(BO, BR.getSourceManager(),AC);
		BR.EmitBasicReport(ACD,CheckName(),"implicit cast of int type to smaller int type could truncate","CMS code rules", os.str(),CELoc, BO->getSourceRange());
		}
	if ( ( size_otype == size_ttype ) && (ToTy->hasSignedIntegerRepresentation() &&  OrigTy->hasUnsignedIntegerRepresentation() || 
		ToTy->hasUnsignedIntegerRepresentation() &&  OrigTy->hasSignedIntegerRepresentation() ) ) {
  		llvm::SmallString<100> buf;
  		llvm::raw_svector_ostream os(buf);
		os <<"Cast-to type, "<<tname<<". Cast-from type, "<<oname<<". Changes int sign type. "<<support::getQualifiedName(*(ACD)); 
		clang::ento::PathDiagnosticLocation CELoc = clang::ento::PathDiagnosticLocation::createBegin(BO, BR.getSourceManager(),AC);
		BR.EmitBasicReport(ACD,CheckName(),"implicit cast ins sign type","CMS code rules", os.str(),CELoc, BO->getSourceRange());
	}
	return;
return;
}

void ICEVisitor::VisitImplicitCastExpr( ImplicitCastExpr *CE )  
{
	const NamedDecl * ACD = dyn_cast<NamedDecl>(AC->getDecl());
	VisitChildren(CE);
	const Expr * SE = CE->getSubExprAsWritten();
	std::string sename = SE->getType().getAsString();
	const clang::Expr *E = CE->getSubExpr();
	if (!(sename=="EventNumber_t")) return;	
	QualType OTy = BR.getContext().getCanonicalType(E->getType());
	QualType TTy = BR.getContext().getCanonicalType(CE->getType());
	QualType ToTy = TTy.getUnqualifiedType();
	QualType OrigTy = OTy.getUnqualifiedType();
	if (!(ToTy->isIntegerType()||ToTy->isFloatingType()) ) return;
	if ( ToTy->isBooleanType() ) return;
	CharUnits size_otype = BR.getContext().getTypeSizeInChars(OrigTy);
	CharUnits size_ttype = BR.getContext().getTypeSizeInChars(ToTy);
	std::string oname = OrigTy.getAsString();
	std::string tname = ToTy.getAsString();
	if ( ToTy->isFloatingType() ) {
  		llvm::SmallString<100> buf;
  		llvm::raw_svector_ostream os(buf);
		os <<"Cast-to type, "<<tname<<". Cast-from type, "<<oname<<" . "<<support::getQualifiedName(*(ACD)); 
		clang::ento::PathDiagnosticLocation CELoc = clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
		BR.EmitBasicReport(ACD,CheckName(),"implicit cast of int type to float type","CMS code rules", os.str(),CELoc, CE->getSourceRange());
		} 
	if (  (size_otype > size_ttype) ) {
  		llvm::SmallString<100> buf;
  		llvm::raw_svector_ostream os(buf);
		os <<"Cast-to type, "<<tname<<". Cast-from type, "<<oname<<". Cast may result in truncation. "<<support::getQualifiedName(*(ACD)); 
		clang::ento::PathDiagnosticLocation CELoc = clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
		BR.EmitBasicReport(ACD,CheckName(),"implicit cast of int type to smaller int type could truncate","CMS code rules", os.str(),CELoc, CE->getSourceRange());
		}
	if ( ToTy->hasSignedIntegerRepresentation() &&  OrigTy->hasUnsignedIntegerRepresentation() || 
		ToTy->hasUnsignedIntegerRepresentation() &&  OrigTy->hasSignedIntegerRepresentation() ) {	
  		llvm::SmallString<100> buf;
 	 	llvm::raw_svector_ostream os(buf);
		os <<"Cast-to type, "<<tname<<". Cast-from type, "<<oname<<". Changes int sign type. "<<support::getQualifiedName(*(ACD)); 
		clang::ento::PathDiagnosticLocation CELoc = clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
		BR.EmitBasicReport(ACD,CheckName(),"implicit cast changes int sign type","CMS code rules", os.str(),CELoc, CE->getSourceRange());
	}
	return;
}

void TrunCastChecker::checkASTDecl(const CXXRecordDecl *D, AnalysisManager& Mgr,BugReporter &BR) const  {
	for ( auto I = D->method_begin(), E = D->method_end(); I != E; ++I)  {
		if ( !llvm::isa<clang::CXXMethodDecl>((*I)) ) continue;
		clang::CXXMethodDecl * MD = llvm::cast<clang::CXXMethodDecl>((*I)->getMostRecentDecl());
		if ( ! MD->hasBody() ) continue;
		clang::Stmt *Body = MD->getBody();
   		ICEVisitor icevisitor(this, BR, Mgr.getAnalysisDeclContext(MD));
     		icevisitor.Visit(Body);
	}
}

}
