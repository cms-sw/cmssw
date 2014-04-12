// ClassChecker.cpp by Patrick Gartung (gartung@fnal.gov)
//
// Objectives of this checker
//
// For each special function of a class (produce, beginrun, endrun, beginlumi, endlumi)
//
//	1) identify member data being modified
//		built-in types reseting values
//		calling non-const member function object if member data is an object
//	2) for each non-const member functions of self called
//		do 1) above
//	3) for each non member function (external) passed in a member object
//		complain if arguement passed in by non-const ref
//		pass by value OK
//		pass by const ref & pointer OK
//
//
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

#include "ClassChecker.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm> 

using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {


void writeLog(std::string ostring) {
	const char * pPath = std::getenv("LOCALRT");
	std::string tname = ""; 
	if ( pPath != NULL ) tname += std::string(pPath);
	tname+="/tmp/class-checker.txt.unsorted";
	std::fstream file;
	file.open(tname.c_str(),std::ios::in|std::ios::out|std::ios::app);
	file<<ostring<<"\n";
	file.flush();
	file.close();
	return;
}



class WalkAST : public clang::StmtVisitor<WalkAST> {
  clang::ento::BugReporter &BR;
  clang::AnalysisDeclContext *AC;

  typedef const clang::CXXMemberCallExpr * WorkListUnit;
  typedef clang::SmallVector<WorkListUnit, 50> DFSWorkList;


  /// A vector representing the worklist which has a chain of CallExprs.
  DFSWorkList WList;
  
  // PreVisited : A CallExpr to this FunctionDecl is in the worklist, but the
  // body has not been visited yet.
  // PostVisited : A CallExpr to this FunctionDecl is in the worklist, and the
  // body has been visited.
  enum Kind { NotVisited,
              Visiting,  /**< A CallExpr to this FunctionDecl is in the 
                                worklist, but the body has not yet been
                                visited. */
              Visited  /**< A CallExpr to this FunctionDecl is in the
                                worklist, and the body has been visited. */
  };

  /// A DenseMap that records visited states of FunctionDecls.
  llvm::DenseMap<const clang::CXXMemberCallExpr *, Kind> VisitedFunctions;

  /// The CallExpr whose body is currently being visited.  This is used for
  /// generating bug reports.  This is null while visiting the body of a
  /// constructor or destructor.
  const clang::CXXMemberCallExpr *visitingCallExpr;

public:
  WalkAST(clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac)
    : BR(br),
      AC(ac),
      visitingCallExpr(0) {}

 

  bool hasWork() const { return !WList.empty(); }

  /// This method adds a CallExpr to the worklist 
  void Enqueue(WorkListUnit WLUnit) {
    Kind &K = VisitedFunctions[WLUnit];
    if (K = Visiting) {
	return;
	}
    K = Visiting;
    WList.push_back(WLUnit);
  }

  /// This method returns an item from the worklist without removing it.
  WorkListUnit Dequeue() {
    assert(!WList.empty());
    return WList.back();    
  }
  
  void Execute() {
      if (WList.empty()) return;
      WorkListUnit WLUnit = Dequeue();
      const clang::CXXMethodDecl *FD = WLUnit->getMethodDecl();
      if (!FD) return;
      llvm::SaveAndRestore<const clang::CXXMemberCallExpr *> SaveCall(visitingCallExpr, WLUnit);
      if (FD && FD->hasBody()) Visit(FD->getBody());
      VisitedFunctions[WLUnit] = Visited;
      WList.pop_back();
  }

  const clang::Stmt * ParentStmt(const Stmt *S) {
  	const Stmt * P = AC->getParentMap().getParentIgnoreParens(S);
	if (!P) return 0;
	return P;
  }

  void WListDump(llvm::raw_ostream & os) {
  	clang::LangOptions LangOpts;
  	LangOpts.CPlusPlus = true;
  	clang::PrintingPolicy Policy(LangOpts);
	if (!WList.empty()) {
		for (llvm::SmallVectorImpl<const clang::CXXMemberCallExpr *>::iterator 
			I = WList.begin(), E = WList.end(); I != E; I++) {
			(*I)->printPretty(os, 0 , Policy);
			os <<" ";
		}
	}       
  }

  // Stmt visitor methods.
  void VisitChildren(clang::Stmt *S);
  void VisitStmt( clang::Stmt *S) { VisitChildren(S); }
  void VisitMemberExpr(clang::MemberExpr *E);
  void VisitCXXMemberCallExpr( clang::CXXMemberCallExpr *CE);
  void VisitDeclRefExpr(clang::DeclRefExpr * DRE);
  void VisitCXXConstCastExpr(clang::CXXConstCastExpr *CCE);
  void ReportDeclRef( const clang::DeclRefExpr * DRE);
  void CheckCXXOperatorCallExpr(const clang::CXXOperatorCallExpr *CE,const clang::MemberExpr *E);
  void CheckBinaryOperator(const clang::BinaryOperator * BO,const clang::MemberExpr *E);
  void CheckUnaryOperator(const clang::UnaryOperator * UO,const clang::MemberExpr *E);
  void CheckExplicitCastExpr(const clang::ExplicitCastExpr * CE,const clang::MemberExpr *E);
  void CheckReturnStmt(const clang::ReturnStmt * RS, const clang::MemberExpr *E);
  void ReportCast(const clang::ExplicitCastExpr *CE);
  void ReportCall(const clang::CXXMemberCallExpr *CE);
  void ReportMember(const clang::MemberExpr *ME);
  void ReportCallReturn(const clang::ReturnStmt * RS);
  void ReportCallArg(const clang::CXXMemberCallExpr *CE, const int i);
};

//===----------------------------------------------------------------------===//
// AST walking.
//===----------------------------------------------------------------------===//




void WalkAST::VisitChildren( clang::Stmt *S) {
  for (clang::Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (clang::Stmt *child = *I) {
      Visit(child);
    }
}


void WalkAST::CheckBinaryOperator(const clang::BinaryOperator * BO,const clang::MemberExpr *IME) {
  if (BO->isAssignmentOp()) {

	if (clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(BO->getLHS())){
			if (ME->isImplicitAccess()) ReportMember(ME);
		}
  } else  {
	if (clang::UnaryOperator * UO = llvm::dyn_cast<clang::UnaryOperator>(BO->getLHS()->IgnoreParenImpCasts()) ) {
		if (UO->getOpcode() == clang::UnaryOperatorKind::UO_Deref) {
			if (clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(UO->getSubExpr()->IgnoreParenImpCasts())){
				if (ME->isImplicitAccess()) ReportMember(ME);
				}
			if (clang::DeclRefExpr * DRE =dyn_cast<clang::DeclRefExpr>(UO->getSubExpr()->IgnoreParenImpCasts())){
				if (const clang::VarDecl * D = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl())) {
					clang::QualType t =  D->getType();
					const clang::Expr * E = llvm::dyn_cast<clang::Expr>(D->getInit());
					if (E && t->isPointerType() ) {
						const clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(E->IgnoreParenImpCasts());
						if (ME && ME->isImplicitAccess()) ReportMember(ME);
						}	
						
					} 		
				}
			}
		}
  }
}

void WalkAST::CheckUnaryOperator(const clang::UnaryOperator * UO,const clang::MemberExpr *E) {
  if (UO->isIncrementDecrementOp()) {
		if (clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(UO->getSubExpr()->IgnoreParenImpCasts())) ReportMember(ME);
	}
}


void WalkAST::CheckCXXOperatorCallExpr(const clang::CXXOperatorCallExpr *OCE,const clang::MemberExpr *E) {
  switch ( OCE->getOperator() ) {

	case OO_Equal:	
	case OO_PlusEqual:
	case OO_MinusEqual:
	case OO_StarEqual:
	case OO_SlashEqual:
	case OO_PercentEqual:
	case OO_AmpEqual:
	case OO_PipeEqual:
	case OO_LessLessEqual:
	case OO_GreaterGreaterEqual:
	if (const clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(OCE->arg_begin()->IgnoreParenImpCasts())){
		if (ME->isImplicitAccess())
			ReportMember(ME);
	} 

	case OO_PlusPlus:
	case OO_MinusMinus:
	if (const clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(OCE->getCallee()->IgnoreParenCasts())) {
		if (ME->isImplicitAccess())
			ReportMember(ME);
	} 
	
  }

}


void WalkAST::CheckExplicitCastExpr(const clang::ExplicitCastExpr * CE,const clang::MemberExpr *ME){

	const clang::Expr *E = CE->getSubExpr();
	clang::ASTContext &Ctx = AC->getASTContext();
	clang::QualType OrigTy = Ctx.getCanonicalType(E->getType());
	clang::QualType ToTy = Ctx.getCanonicalType(CE->getType());

	if ( support::isConst( OrigTy ) && ! support::isConst(ToTy) )
		ReportCast(CE);

}
 

void WalkAST::CheckReturnStmt(const clang::ReturnStmt * RS, const clang::MemberExpr * E){
	if (const clang::Expr * RE = RS->getRetValue()) {
		clang::QualType QT = RE->getType();
		clang::ASTContext &Ctx = AC->getASTContext();
		clang::QualType Ty = Ctx.getCanonicalType(QT);
		const clang::CXXMethodDecl * MD;
		if (visitingCallExpr) 
			MD = visitingCallExpr->getMethodDecl();
		else 
			MD = llvm::dyn_cast<clang::CXXMethodDecl>(AC->getDecl());
		if ( llvm::isa<clang::CXXNewExpr>(RE) ) return; 
		clang::QualType RQT = MD->getResultType();
		clang::QualType RTy = Ctx.getCanonicalType(RQT);
		clang::QualType CQT = MD->getCallResultType();
		clang::QualType CTy = Ctx.getCanonicalType(CQT);
		if ( (RTy->isPointerType() || RTy->isReferenceType() ) ) {
		if( !support::isConst(RTy) ) {
			ReportCallReturn(RS);
			}
		}

	}
}


void WalkAST::VisitCXXConstCastExpr(clang::CXXConstCastExpr *CCE) {
	
	CmsException m_exception;
	clang::ento::PathDiagnosticLocation CELoc = clang::ento::PathDiagnosticLocation::createBegin(CCE, BR.getSourceManager(),AC);
	if (!m_exception.reportClass( CELoc, BR ) ) return;
	const clang::CXXMethodDecl * MD = llvm::cast<clang::CXXMethodDecl>(AC->getDecl());
	std::string buf;
	llvm::raw_string_ostream os(buf);
	os <<"const_cast used\n";
	std::string tolog = "data class '"+MD->getParent()->getNameAsString()+"' const function '" + support::getQualifiedName(*MD)+ "' Warning: "+os.str()+".";
	writeLog(tolog);
	BugType * BT = new BugType("ClassChecker : const_cast used in const function ","ThreadSafety");
	BugReport * R = new BugReport(*BT,tolog,CELoc);
	BR.emitReport(R);
	return;
}

void WalkAST::VisitDeclRefExpr( clang::DeclRefExpr * DRE) {
  if (clang::VarDecl * D = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl()) ) { 
  	clang::SourceLocation SL = DRE->getLocStart();
  	if (BR.getSourceManager().isInSystemHeader(SL) || BR.getSourceManager().isInExternCSystemHeader(SL)) return;
	if ( support::isSafeClassName( D->getCanonicalDecl()->getQualifiedNameAsString() ) ) return;
	ReportDeclRef( DRE );

  }
}

void WalkAST::ReportDeclRef( const clang::DeclRefExpr * DRE) {
 
 if (const clang::VarDecl * D = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl())) {
	clang::QualType t =  D->getType();
	const clang::Stmt * PS = ParentStmt(DRE);
 	CmsException m_exception;
  	clang::LangOptions LangOpts;
  	LangOpts.CPlusPlus = true;
  	clang::PrintingPolicy Policy(LangOpts);


  	clang::ento::PathDiagnosticLocation CELoc = clang::ento::PathDiagnosticLocation::createBegin(DRE, BR.getSourceManager(),AC);
	if (!m_exception.reportClass( CELoc, BR ) ) return;
	if ( support::isSafeClassName( t.getAsString() ) ) return;
        if ( D->hasAttr<CMSThreadGuardAttr>() || D->hasAttr<CMSThreadSafeAttr>()) return;
	if ( D->isStaticLocal() && D->getTSCSpec() != clang::ThreadStorageClassSpecifier::TSCS_thread_local && ! support::isConst( t ) )
	{
		std::string buf;
	    	llvm::raw_string_ostream os(buf);
	   	os << "Non-const variable '" << D->getNameAsString() << "' is static local and accessed in statement '";
	    	PS->printPretty(os,0,Policy);
		os << "'.\n";
		const clang::CXXMethodDecl * MD = llvm::cast<clang::CXXMethodDecl>(AC->getDecl());
		std::string tolog = "data class '"+MD->getParent()->getNameAsString()+"' const function '" + support::getQualifiedName(*MD) + "' Warning: "+os.str();
		writeLog(tolog);
	    	BugType * BT = new BugType("ClassChecker : non-const static local variable accessed","ThreadSafety");
		BugReport * R = new BugReport(*BT,os.str(),CELoc);
		BR.emitReport(R);
		return;
	}

	if ( D->isStaticDataMember() &&  D->getTSCSpec() != clang::ThreadStorageClassSpecifier::TSCS_thread_local && ! support::isConst( t ) )
	{
	    	std::string buf;
	    	llvm::raw_string_ostream os(buf);
	    	os << "Non-const variable '" << D->getNameAsString() << "' is static member data and accessed in statement '";
	    	PS->printPretty(os,0,Policy);
		os << "'.\n";
		const clang::CXXMethodDecl * MD = llvm::cast<clang::CXXMethodDecl>(AC->getDecl());
		std::string tolog = "data class '"+MD->getParent()->getNameAsString()+"' const function '" + support::getQualifiedName(*MD) + "' Warning: "+os.str();
		writeLog(tolog);
	    	BugType * BT = new BugType("ClassChecker : non-const static member variable accessed","ThreadSafety");
		BugReport * R = new BugReport(*BT,os.str(),CELoc);
		BR.emitReport(R);
	    return;
	}


	if ( (D->getStorageClass() == clang::SC_Static) &&
			  !D->isStaticDataMember() &&
			  !D->isStaticLocal() &&
			  !support::isConst( t ) )
	{

	    	std::string buf;
	    	llvm::raw_string_ostream os(buf);
	    	os << "Non-const variable '" << D->getNameAsString() << "' is global static and accessed in statement '";
	    	PS->printPretty(os,0,Policy);
		os << "'.\n";
		const clang::CXXMethodDecl * MD = llvm::cast<clang::CXXMethodDecl>(AC->getDecl());
		std::string tolog = "data class '"+MD->getParent()->getNameAsString()+"' const function '" + support::getQualifiedName(*MD) + "' Warning: "+os.str();
		writeLog(tolog);
	    	BugType * BT = new BugType("ClassChecker : non-const global static variable accessed","ThreadSafety");
		BugReport * R = new BugReport(*BT,os.str(),CELoc);
		BR.emitReport(R);
	    return;
	
	}

  }
}


void WalkAST::VisitMemberExpr( clang::MemberExpr *ME) {

  clang::SourceLocation SL = ME->getExprLoc();
  if (BR.getSourceManager().isInSystemHeader(SL) || BR.getSourceManager().isInExternCSystemHeader(SL)) return;


  if (!(ME->isImplicitAccess())) return;
  Stmt * P = AC->getParentMap().getParent(ME);
	while (AC->getParentMap().hasParent(P)) {
		if (const clang::UnaryOperator * UO = llvm::dyn_cast<clang::UnaryOperator>(P)) 
			{ WalkAST::CheckUnaryOperator(UO,ME);}
		if (const clang::BinaryOperator * BO = llvm::dyn_cast<clang::BinaryOperator>(P)) 
			{ WalkAST::CheckBinaryOperator(BO,ME);}
		if (const clang::CXXOperatorCallExpr *OCE = llvm::dyn_cast<clang::CXXOperatorCallExpr>(P)) 
			{ WalkAST::CheckCXXOperatorCallExpr(OCE,ME);}
		if (const clang::ExplicitCastExpr * CE = llvm::dyn_cast<clang::ExplicitCastExpr>(P))
			{ WalkAST::CheckExplicitCastExpr(CE,ME);}
		if (const clang::ReturnStmt * RS = llvm::dyn_cast<clang::ReturnStmt>(P)) 
			{ WalkAST::CheckReturnStmt(RS,ME); }
		if (const clang::CXXConstCastExpr * CCE = llvm::dyn_cast<clang::CXXConstCastExpr>(P))
			{ WalkAST::ReportCast(CCE);}
		if (const clang::CXXNewExpr * NE = llvm::dyn_cast<clang::CXXNewExpr>(P)) break;
		P = AC->getParentMap().getParent(P);
	}
}




void WalkAST::VisitCXXMemberCallExpr( clang::CXXMemberCallExpr *CE) {

  if (BR.getSourceManager().isInSystemHeader(CE->getExprLoc()) || BR.getSourceManager().isInExternCSystemHeader(CE->getExprLoc())) return;

  clang::CXXMethodDecl * MD = CE->getMethodDecl();

  if (! MD)  return;                                                                                                      

  Enqueue(CE);
  Execute();
  Visit(CE->getImplicitObjectArgument()->IgnoreParenCasts());

  const Expr * IOA = CE->getImplicitObjectArgument()->IgnoreParenCasts();
  if ( llvm::isa<clang::MemberExpr>(IOA) ) {
	const MemberExpr * ME = dyn_cast<MemberExpr>(IOA);
	if ( !MD->isConst() && ME->isImplicitAccess() ) ReportCall(CE);
  }

  for(int i=0, j=CE->getNumArgs(); i<j; i++) {
    if (CE->getArg(i)) {
	if ( const clang::Expr *E = llvm::dyn_cast<clang::Expr>(CE->getArg(i)))  {
		clang::QualType qual_arg = E->getType();
		if (const clang::MemberExpr *ME=llvm::dyn_cast<clang::MemberExpr>(E)) {
			if (ME->isImplicitAccess()) {
				clang::ParmVarDecl *PVD=llvm::dyn_cast<clang::ParmVarDecl>(MD->getParamDecl(i));
				clang::QualType QT = PVD->getOriginalType();
				const clang::Type * T = QT.getTypePtr();
				if (!support::isConst(QT) && T->isReferenceType()) ReportCallArg(CE,i);
			}
		}
	}
    }
  }

}

void WalkAST::ReportMember(const clang::MemberExpr *ME) {
 if ( visitingCallExpr ) {
	clang::Expr * IOA = visitingCallExpr->getImplicitObjectArgument();
	if (!( IOA->isImplicitCXXThis() || llvm::dyn_cast<CXXThisExpr>(IOA->IgnoreParenCasts()))) return;
	}
  std::string buf;
  llvm::raw_string_ostream os(buf);
  CmsException m_exception;
  clang::ento::PathDiagnosticLocation CELoc;
  clang::SourceRange R;
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

  CELoc = clang::ento::PathDiagnosticLocation::createBegin(ME, BR.getSourceManager(),AC);
  R = ME->getSourceRange();

  os << "Member data '";
  ME->printPretty(os,0,Policy);
  os << "' is directly or indirectly modified in const function\n";
  const clang::CXXMethodDecl * MD = llvm::cast<clang::CXXMethodDecl>(AC->getDecl());
  std::string tolog = "data class '"+MD->getParent()->getNameAsString()+"' const function '" + support::getQualifiedName(*MD) + "' Warning: " + os.str();
  if (!m_exception.reportClass( CELoc, BR ) ) return;
  writeLog(tolog);
  BR.EmitBasicReport(AC->getDecl(),"Class Checker : Member data modified in const function","ThreadSafety",os.str(),CELoc,R);
}

void WalkAST::ReportCall(const clang::CXXMemberCallExpr *CE) {

  const clang::CXXRecordDecl * RD = CE->getRecordDecl();
  const clang::CXXMethodDecl * MD = CE->getMethodDecl();
  if ( !RD || support::isSafeClassName( RD->getQualifiedNameAsString() ) ) return; 
  std::string buf;
  llvm::raw_string_ostream os(buf);

  CmsException m_exception;
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);
 
  os << "call expr '";
  CE->printPretty(os,0,Policy);
  os << "' with implicit object argument '";
  CE->getImplicitObjectArgument()->IgnoreParenCasts()->printPretty(os,0,Policy);  
  os << "'";
  os<<"' is a non-const member function '"<<MD->getQualifiedNameAsString();
  os<<"' that could modify member data object of type '"<<RD->getQualifiedNameAsString()<<"'\n";
  const clang::CXXMethodDecl * ACMD = llvm::cast<clang::CXXMethodDecl>(AC->getDecl()); 
  std::string tolog = "data class '"+ACMD->getParent()->getNameAsString()+"' const function '" + support::getQualifiedName(*ACMD) + "' Warning: "+os.str();
  clang::ento::PathDiagnosticLocation CELoc =
    clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
  if ( support::isSafeClassName(MD->getQualifiedNameAsString()) ) return;
  if (!m_exception.reportClass( CELoc, BR ) ) return;
  writeLog(tolog);
  BugType * BT = new BugType("Class Checker : Non-const member function could modify member data object","ThreadSafety");
  BugReport * R = new BugReport(*BT,os.str(),CELoc);
  BR.emitReport(R);
  
	 

}


void WalkAST::ReportCast(const clang::ExplicitCastExpr *CE) {
  std::string buf;
  llvm::raw_string_ostream os(buf);

  CmsException m_exception;
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

 
  os << "Const qualifier of member data object";
  os <<" was removed via cast expression '";
  const clang::CXXMethodDecl * MD = llvm::cast<clang::CXXMethodDecl>(AC->getDecl());
  std::string tolog = "data class '"+MD->getParent()->getNameAsString()+"' const function '" + support::getQualifiedName(*MD) + "' Warning: "+os.str();
  clang::ento::PathDiagnosticLocation CELoc =
    clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);

  if (!m_exception.reportClass( CELoc, BR ) ) return;
  writeLog(tolog);
  BugType * BT = new BugType("Class Checker : Const cast away from member data in const function","ThreadSafety");
  BugReport * R = new BugReport(*BT,os.str(),CELoc);
  BR.emitReport(R);
	 

}

void WalkAST::ReportCallArg(const clang::CXXMemberCallExpr *CE,const int i) {

  std::string buf;
  llvm::raw_string_ostream os(buf);
  CmsException m_exception;
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

  clang::CXXMethodDecl * CMD = llvm::dyn_cast<clang::CXXMemberCallExpr>(CE)->getMethodDecl();
  const clang::MemberExpr *E = llvm::dyn_cast<clang::MemberExpr>(CE->getArg(i));
  clang::ParmVarDecl *PVD=llvm::dyn_cast<clang::ParmVarDecl>(CMD->getParamDecl(i));
  clang::ValueDecl * VD = llvm::dyn_cast<clang::ValueDecl>(E->getMemberDecl());
  os << "Member data '" << VD->getQualifiedNameAsString();
  os << "' is passed to a non-const reference parameter";
  os <<" of CXX method '" << CMD->getQualifiedNameAsString() << "' in const function";
  os << "\n";
  const clang::CXXMethodDecl * MD = llvm::cast<clang::CXXMethodDecl>(AC->getDecl());
  std::string tolog = "data class '"+CMD->getParent()->getNameAsString()+"' const function '" + support::getQualifiedName(*MD) + "' Warning: "+os.str();

  clang::ento::PathDiagnosticLocation ELoc =
   clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
  clang::SourceLocation L = E->getExprLoc();

  if (!m_exception.reportClass( ELoc, BR ) ) return;
  writeLog(tolog);
  BR.EmitBasicReport(CE->getCalleeDecl(),"Class Checker :  Member data passed to non-const reference","ThreadSafety",os.str(),ELoc,L);

}

void WalkAST::ReportCallReturn(const clang::ReturnStmt * RS) {
  std::string buf;
  llvm::raw_string_ostream os(buf);

  CmsException m_exception;
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

  os << "Returns a pointer or reference to a non-const member data object ";
  os << "in const function in statement '";
  RS->printPretty(os,0,Policy);
  os << "\n";
  const clang::CXXMethodDecl * MD = llvm::cast<clang::CXXMethodDecl>(AC->getDecl());
  std::string tolog = "data class '"+MD->getParent()->getNameAsString()+"' const function '" + support::getQualifiedName(*MD) + "' Warning: "+os.str();
  clang::ento::PathDiagnosticLocation CELoc =
    clang::ento::PathDiagnosticLocation::createBegin(RS, BR.getSourceManager(),AC);
  

  if (!m_exception.reportClass( CELoc, BR ) ) return;
  writeLog(tolog);
  BugType * BT = new BugType("Class Checker : Const function returns pointer or reference to non-const member data object","ThreadSafety");
  BugReport * R = new BugReport(*BT,os.str(),CELoc);
  BR.emitReport(R);
	 
}


void ClassChecker::checkASTDecl(const clang::CXXRecordDecl *RD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const {

	const clang::SourceManager &SM = BR.getSourceManager();
	
  	std::string buf;
  	llvm::raw_string_ostream os(buf);
	std::string name = RD->getNameAsString();
	if ( ! support::isDataClass(name) ) return;
	clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( RD, SM );
	if (  !m_exception.reportClass( DLoc, BR ) ) return;

// Check the class methods (member methods).
	for (clang::CXXRecordDecl::method_iterator
		I = RD->method_begin(), E = RD->method_end(); I != E; ++I)  {
		clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( (*I) , SM );
		if (  !m_exception.reportClass( DLoc, BR ) ) continue;
		if ( !llvm::isa<clang::CXXMethodDecl>((*I)) ) continue;
		if (!(*I)->isConst()) continue;
		clang::CXXMethodDecl * MD = llvm::cast<clang::CXXMethodDecl>((*I)->getMostRecentDecl());
		if ( MD->hasAttr<CMSThreadGuardAttr>() || MD->hasAttr<CMSThreadSafeAttr>()) continue;
				if ( MD->hasBody() ) {
					clang::Stmt *Body = MD->getBody();
					clangcms::WalkAST walker(BR, mgr.getAnalysisDeclContext(MD));
	       				walker.Visit(Body);
					clang::QualType CQT = MD->getCallResultType();
					clang::QualType RQT = MD->getResultType();
					clang::ASTContext &Ctx = BR.getContext();
					clang::QualType RTy = Ctx.getCanonicalType(RQT);
					clang::QualType CTy = Ctx.getCanonicalType(CQT);
					clang::ento::PathDiagnosticLocation ELoc =clang::ento::PathDiagnosticLocation::createBegin( MD , SM );
					if ( (RTy->isPointerType() || RTy->isReferenceType() ) 
					&&(!support::isConst(RTy) ) 
					&& ( MD->getNameAsString().find("clone")==std::string::npos ) )  
					{
						std::string buf;
						llvm::raw_string_ostream os(buf);
						os << MD->getQualifiedNameAsString() << " is a const member function that returns a pointer or reference to a non-const object \n";
						std::string tolog = "data class '"+MD->getParent()->getNameAsString()+"' const function '" + MD->getNameAsString() + "' Warning: "+os.str();
						writeLog(tolog);
						clang::SourceRange SR = MD->getSourceRange();
						BR.EmitBasicReport(MD, "Class Checker : Const function returns pointer or reference to non-const object.","ThreadSafety",os.str(),ELoc);
					}
				}
   	}	/* end of methods loop */


} //end of class

} //end namespace
