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


#include "ClassChecker.h"

using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {

class WalkAST : public clang::StmtVisitor<WalkAST> {
  clang::ento::BugReporter &BR;
  clang::AnalysisDeclContext *AC;

  typedef const clang::CXXMemberCallExpr * WorkListUnit;
  typedef clang::SmallVector<WorkListUnit, 50> DFSWorkList;
  typedef const clang::Stmt * StmtListUnit;
  typedef clang::SmallVector<StmtListUnit, 50> StmtList;


  /// A vector representing the worklist which has a chain of CallExprs.
  DFSWorkList WList;
  StmtList SList;
  
  // PreVisited : A CallExpr to this FunctionDecl is in the worklist, but the
  // body has not been visited yet.
  // PostVisited : A CallExpr to this FunctionDecl is in the worklist, and the
  // body has been visited.
  enum Kind { NotVisited,
              PreVisited,  /**< A CallExpr to this FunctionDecl is in the 
                                worklist, but the body has not yet been
                                visited. */
              PostVisited  /**< A CallExpr to this FunctionDecl is in the
                                worklist, and the body has been visited. */
  };

  /// A DenseMap that records visited states of FunctionDecls.
  llvm::DenseMap<const clang::FunctionDecl *, Kind> VisitedFunctions;

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
    WList.push_back(WLUnit);
  }

  /// This method returns an item from the worklist without removing it.
  WorkListUnit Dequeue() {
    assert(!WList.empty());
    return WList.back();    
  }
  
  void Execute() {
      WorkListUnit WLUnit = Dequeue();
      if (visitingCallExpr && WLUnit->getMethodDecl() == visitingCallExpr->getMethodDecl()) {
//		llvm::errs()<<"\nRecursive call to ";
//		WLUnit->getDirectCallee()->printName(llvm::errs());
//		llvm::errs()<<" , ";
//		WLUnit->dumpPretty(AC->getASTContext());
//		llvm::errs()<<"\n";
   		WList.pop_back();
		return;
		}
      const clang::CXXMethodDecl *FD = clang::dyn_cast<clang::CXXMethodDecl>(WLUnit->getMethodDecl());
      llvm::SaveAndRestore<const clang::CXXMemberCallExpr *> SaveCall(visitingCallExpr, WLUnit);
      if (FD && FD->hasBody()) Visit(FD->getBody());
      WList.pop_back();
  }

const clang::Stmt * ParentStmt() {
	if (!SList.empty()) {
		if ( visitingCallExpr ){
 	  		for (llvm::SmallVectorImpl<const clang::Stmt *>::iterator 
				I = SList.begin(), E = SList.end(); I != E; I++) {
				if ((*I)==visitingCallExpr ) {
					return (*(I+1));
				}
			}
		}
		else {
			return ( *(SList.begin()));
		}
	}
	return 0;
  }


  void SListDump(llvm::raw_ostream & os) {
  	clang::LangOptions LangOpts;
  	LangOpts.CPlusPlus = true;
  	clang::PrintingPolicy Policy(LangOpts);
  	ParentStmt()->printPretty(os,0,Policy);
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

bool SListwalkback() {
	for (llvm::SmallVectorImpl<const clang::Stmt *>::reverse_iterator I = SList.rbegin(), E = SList.rend(); I != E; ++I) {
		if (!(*I)) continue;
		if (visitingCallExpr && (*I)==visitingCallExpr) return true;
		if (const clang::CXXNewExpr * NE = llvm::dyn_cast<clang::CXXNewExpr>(*I)) return true;
		if (const clang::UnaryOperator * UO = llvm::dyn_cast<clang::UnaryOperator>(*I)) 
			{ WalkAST::UnaryOperator(UO);}
		if (const clang::BinaryOperator * BO = llvm::dyn_cast<clang::BinaryOperator>(*I)) 
			{ WalkAST::BinaryOperator(BO);}
		if (const clang::CXXOperatorCallExpr *OCE = llvm::dyn_cast<clang::CXXOperatorCallExpr>(*I)) 
			{ WalkAST::CXXOperatorCallExpr(OCE);}
		if (const clang::ExplicitCastExpr * CE = llvm::dyn_cast<clang::ExplicitCastExpr>(*I))
			{ WalkAST::ExplicitCastExpr(CE);} 
		if (const clang::ReturnStmt * RS = llvm::dyn_cast<clang::ReturnStmt>(*I)) 
			{ WalkAST::ReturnStmt(RS); }
	}
	return true;
}

  // Stmt visitor methods.
  void VisitChildren(clang::Stmt *S);
  void VisitStmt( clang::Stmt *S) { VisitChildren(S); }
  void VisitMemberExpr(clang::MemberExpr *E);
  void VisitCXXMemberCallExpr( clang::CXXMemberCallExpr *CE);
  void VisitDeclRefExpr(clang::DeclRefExpr * DRE);
  void ReportDeclRef( const clang::DeclRefExpr * DRE);
  void CXXOperatorCallExpr(const clang::CXXOperatorCallExpr *CE);
  void BinaryOperator(const clang::BinaryOperator * BO);
  void UnaryOperator(const clang::UnaryOperator * UO);
  void ExplicitCastExpr(const clang::ExplicitCastExpr * CE);
  void ReturnStmt(const clang::ReturnStmt * RS);
  void ReportCast(const clang::ExplicitCastExpr *CE,const clang::Expr *E);
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
      SList.push_back(child);
      Visit(child);
      SList.pop_back();
    }
}


void WalkAST::BinaryOperator(const clang::BinaryOperator * BO) {
//	BO->dump(); 
//	llvm::errs()<<"\n";
if (BO->isAssignmentOp()) {
	if ( dyn_cast<clang::DeclRefExpr>(*SList.rbegin())) 
	if (clang::DeclRefExpr * DRE =dyn_cast<clang::DeclRefExpr>(BO->getLHS())){
			ReportDeclRef(DRE);
		} else
	if ( dyn_cast<clang::MemberExpr>(*SList.rbegin())) 
	if (clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(BO->getLHS())){
			if (ME->isImplicitAccess()) ReportMember(ME);
		}
	}
}
void WalkAST::UnaryOperator(const clang::UnaryOperator * UO) {
//	UO->dump(); 
//	llvm::errs()<<"\n";
  if (UO->isIncrementDecrementOp())  {
	if ( dyn_cast<clang::DeclRefExpr>(*SList.rbegin())) 
	if (clang::DeclRefExpr * DRE =dyn_cast<clang::DeclRefExpr>(UO->getSubExpr()->IgnoreParenImpCasts())) {
		ReportDeclRef(DRE);
	} else 
	if ( dyn_cast<clang::MemberExpr>(*SList.rbegin())) 
	if (clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(UO->getSubExpr()->IgnoreParenImpCasts())) {
		ReportMember(ME);
	}
  }
}

void WalkAST::CXXOperatorCallExpr(const clang::CXXOperatorCallExpr *OCE) {
// OCE->dump(); 
//  llvm::errs()<<"\n";
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
	if ( dyn_cast<clang::DeclRefExpr>(*SList.rbegin())) 
	if (const clang::DeclRefExpr * DRE =dyn_cast<clang::DeclRefExpr>(OCE->getCallee()->IgnoreParenImpCasts())) {
		ReportDeclRef(DRE);
	} else
	if ( dyn_cast<clang::MemberExpr>(*SList.rbegin())) 
	if (const clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(OCE->getCallee()->IgnoreParenImpCasts())){
		if (ME->isImplicitAccess())
			ReportMember(ME);
	} 

	case OO_PlusPlus:
	case OO_MinusMinus:
	if ( dyn_cast<clang::DeclRefExpr>(*SList.rbegin())) 
	if (const clang::DeclRefExpr * DRE =dyn_cast<clang::DeclRefExpr>(OCE->arg_begin()->IgnoreParenCasts())) {
		ReportDeclRef(DRE);
	} else
	if ( dyn_cast<clang::MemberExpr>(*SList.rbegin())) 
	if (const clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(OCE->getCallee()->IgnoreParenCasts())) {
		if (ME->isImplicitAccess())
			ReportMember(ME);
	} 
	
}

}


void WalkAST::ExplicitCastExpr(const clang::ExplicitCastExpr * CE){
//	CE->dump();
//	llvm::errs()<<"\n";

	const clang::Expr *E = CE->getSubExpr();
	clang::ASTContext &Ctx = AC->getASTContext();
	clang::QualType OrigTy = Ctx.getCanonicalType(E->getType());
	clang::QualType ToTy = Ctx.getCanonicalType(CE->getType());

	if (const clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(*SList.rbegin())) {
	if ( support::isConst( OrigTy ) && ! support::isConst(ToTy) )
		ReportCast(CE,ME);
	}

	if (const clang::DeclRefExpr * DRE = dyn_cast<clang::DeclRefExpr>(*SList.rbegin())) {
	if ( support::isConst( OrigTy ) && ! support::isConst(ToTy) )
		ReportCast(CE,DRE);
	}
}
 

void WalkAST::ReturnStmt(const clang::ReturnStmt * RS){
	if (const clang::Expr * RE = RS->getRetValue()) {
		clang::QualType RQT = RE->getType();
		if ( llvm::isa<clang::CXXNewExpr>(RE) ) return; 
		if ( RQT.getTypePtr()->isPointerType() && !(RQT.getTypePtr()->getPointeeType().isConstQualified()) ) {
//		llvm::errs()<<"\nReturn Expression\n";
//		RS->dump();
//		llvm::errs()<<"\n";
//		RQT->dump();
//		llvm::errs()<<"\n";
		if ( const clang::MemberExpr *ME = dyn_cast<clang::MemberExpr>(*SList.rbegin()))
			if (ME->isImplicitAccess()) 
			{
			ReportCallReturn(RS);
			return;
			} else
		if (const clang::DeclRefExpr * DRE = dyn_cast<clang::DeclRefExpr>(*SList.rbegin()))
			{
			return;
			}
		}
	}
	
}

void WalkAST::VisitDeclRefExpr( clang::DeclRefExpr * DRE) {
  if (clang::VarDecl * D = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl()) ) { 
  	clang::SourceLocation SL = DRE->getLocStart();
  	if (BR.getSourceManager().isInSystemHeader(SL) || BR.getSourceManager().isInExternCSystemHeader(SL)) return;

		if (SListwalkback()) return;

//	llvm::errs()<<"Declaration Ref Expr\t";
//	dyn_cast<Stmt>(DRE)->dumpPretty(AC->getASTContext());
//	DRE->dump();
//	llvm::errs()<<"\n";
//	SListDump(llvm::errs());
//	llvm::errs()<<"\n";
  	}
}

void WalkAST::ReportDeclRef( const clang::DeclRefExpr * DRE) {

 if (const clang::VarDecl * D = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl())) {
	clang::QualType t =  D->getType();  
	const clang::Stmt * PS = ParentStmt();
 	CmsException m_exception;
  	clang::LangOptions LangOpts;
  	LangOpts.CPlusPlus = true;
  	clang::PrintingPolicy Policy(LangOpts);


  	clang::ento::PathDiagnosticLocation CELoc = clang::ento::PathDiagnosticLocation::createBegin(DRE, BR.getSourceManager(),AC);
	if (!m_exception.reportClass( CELoc, BR ) ) return;

	if ( D->isStaticLocal() && ! support::isConst( t ) )
	{
		std::string buf;
	    	llvm::raw_string_ostream os(buf);
	   	os << "Non-const variable '" << D->getNameAsString() << "' is static local and modified in statement \n";
	    	PS->printPretty(os,0,Policy);
	    	BugType * BT = new BugType("ClassChecker : non-const static local variable","ThreadSafety");
		BugReport * R = new BugReport(*BT,os.str(),CELoc);
		BR.emitReport(R);
		return;
	}

	if ( D->isStaticDataMember() && ! support::isConst( t ) )
	{
	    	std::string buf;
	    	llvm::raw_string_ostream os(buf);
	    	os << "Non-const variable '" << D->getNameAsString() << "' is static member data and modified in statement  \n";
	    	PS->printPretty(os,0,Policy);
	    	BugType * BT = new BugType("ClassChecker : non-const static member variable","ThreadSafety");
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
	    	os << "Non-const variable '" << D->getNameAsString() << "' is global static and modified in statement \n";
	    	PS->printPretty(os,0,Policy);
	    	BugType * BT = new BugType("ClassChecker : non-const global static variable","ThreadSafety");
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
	
	if (SListwalkback()) return;
//  	clang::Expr * E = ME->getBase();
//  	clang::QualType qual_base = E->getType();
//  	clang::ValueDecl * VD = ME->getMemberDecl();
//  	const clang::Stmt * PS = (*(SList.rbegin()+1));

  
	ReportMember(ME);
	llvm::errs()<<"Member Expr '";
	dyn_cast<Stmt>(ME)->dumpPretty(AC->getASTContext());
	llvm::errs()<<"' not handled by Class Checker in statement\n\n";
	SListDump(llvm::errs());
	llvm::errs()<<"\n";
	WListDump(llvm::errs());
	llvm::errs()<<"\n";
}




void WalkAST::VisitCXXMemberCallExpr( clang::CXXMemberCallExpr *CE) {

  clang::SourceLocation SL = CE->getLocStart();
  if (BR.getSourceManager().isInSystemHeader(SL) || BR.getSourceManager().isInExternCSystemHeader(SL)) return;
  if (!(CE->getMethodDecl())) return;                                                                                                      
//  if (!( CE->getImplicitObjectArgument()->isImplicitCXXThis() || llvm::dyn_cast<CXXThisExpr>(CE->getImplicitObjectArgument()->IgnoreParenCasts()))) return;
  	
  Enqueue(CE);
  Execute();
  Visit(CE->getImplicitObjectArgument());

  clang::CXXMethodDecl * MD = dyn_cast<clang::CXXMethodDecl>(CE->getMethodDecl());
  if (llvm::isa<clang::MemberExpr>(CE->getImplicitObjectArgument()->IgnoreParenCasts() ))
	if ( ! MD->isConst()  )
	ReportCall(CE);

  for(int i=0, j=CE->getNumArgs(); i<j; i++) {
	if (CE->getArg(i))
	if ( const clang::Expr *E = llvm::dyn_cast<clang::Expr>(CE->getArg(i)))  {
		clang::QualType qual_arg = E->getType();
		if (const clang::MemberExpr *ME=llvm::dyn_cast<clang::MemberExpr>(E))		
		if (ME->isImplicitAccess()) {
//			clang::ValueDecl * VD = llvm::dyn_cast<clang::ValueDecl>(ME->getMemberDecl());
//			clang::QualType qual_decl = llvm::dyn_cast<clang::ValueDecl>(ME->getMemberDecl())->getType();
			clang::ParmVarDecl *PVD=llvm::dyn_cast<clang::ParmVarDecl>(MD->getParamDecl(i));
			clang::QualType QT = PVD->getOriginalType();
			const clang::Type * T = QT.getTypePtr();
			if (!support::isConst(QT))
			if (T->isReferenceType())
				{
				ReportCallArg(CE,i);
				}
			}
	}
  }

//	llvm::errs()<<"\n--------------------------------------------------------------\n";
//	llvm::errs()<<"\n------CXXMemberCallExpression---------------------------------\n";
//	llvm::errs()<<"\n--------------------------------------------------------------\n";
//	CE->dump();
//	llvm::errs()<<"\n--------------------------------------------------------------\n";
//	return;


}

void WalkAST::ReportMember(const clang::MemberExpr *ME) {
  llvm::SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);
  CmsException m_exception;
  clang::ento::PathDiagnosticLocation CELoc;
  clang::SourceRange R;
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

  CELoc = clang::ento::PathDiagnosticLocation::createBegin(ME, BR.getSourceManager(),AC);
  R = ME->getSourceRange();


  os << " Member data ";
  os << ME->getMemberDecl()->getQualifiedNameAsString();
//  ME->printPretty(os,0,Policy);
  os << " is directly or indirectly modified in const function ";
//  os << llvm::dyn_cast<clang::CXXMethodDecl>(AC->getDecl())->getQualifiedNameAsString();
//  if (hasWork()) {
//  	os << " in call stack ";
//  	WListDump(os);
//  }
//  os << " \n";

//  ME->printPretty(llvm::errs(),0,Policy);
//  llvm::errs()<<"\n";
//  ME->dump();
//  llvm::errs()<<"\n";
//  SListDump(llvm::errs());
//  llvm::errs()<<"\n";
//  WListDump(llvm::errs());
//  llvm::errs()<<"\n";
//  if (visitingCallExpr) visitingCallExpr->getImplicitObjectArgument()->dump();
//  llvm::errs()<<"\n";



  if (!m_exception.reportClass( CELoc, BR ) ) return;
  BR.EmitBasicReport(AC->getDecl(),"Class Checker : Member data modified in const function","ThreadSafety",os.str(),CELoc,R);
}

void WalkAST::ReportCall(const clang::CXXMemberCallExpr *CE) {
  llvm::SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);

  CmsException m_exception;
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

 
  CE->printPretty(os,0,Policy);
  os<<" is a non-const member function could modify member data object ";
  CE->getImplicitObjectArgument()->printPretty(os,0,Policy);
  os << "\n";
  clang::ento::PathDiagnosticLocation CELoc =
    clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
  

  if (!m_exception.reportClass( CELoc, BR ) ) return;
  BugType * BT = new BugType("Class Checker : Non-const member function could modify member data object","ThreadSafety");
  BugReport * R = new BugReport(*BT,os.str(),CELoc);
  BR.emitReport(R);
	 

}


void WalkAST::ReportCast(const clang::ExplicitCastExpr *CE,const clang::Expr *E) {
  llvm::SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);

  CmsException m_exception;
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

 
  os << "Const qualifier of data object ";
  E->printPretty(os,0,Policy);
  os <<" was removed via cast expression ";
  CE->printPretty(os,0,Policy);
//  llvm::errs() << llvm::dyn_cast<clang::CXXMethodDecl>(AC->getDecl())->getQualifiedNameAsString();
//  if (visitingCallExpr) {
//	llvm::errs() << " in call stack ";
//	WListDump(llvm::errs());
//  }
//  llvm::errs()<<"\n";


  clang::ento::PathDiagnosticLocation CELoc =
    clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
  

  if (!m_exception.reportClass( CELoc, BR ) ) return;
  BugType * BT = new BugType("Class Checker : Const cast away from member data in const function","ThreadSafety");
  BugReport * R = new BugReport(*BT,os.str(),CELoc);
  BR.emitReport(R);
	 

}

void WalkAST::ReportCallArg(const clang::CXXMemberCallExpr *CE,const int i) {

  llvm::SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);
  CmsException m_exception;
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

  clang::CXXMethodDecl * MD = llvm::dyn_cast<clang::CXXMemberCallExpr>(CE)->getMethodDecl();
  const clang::MemberExpr *E = llvm::dyn_cast<clang::MemberExpr>(CE->getArg(i));
  clang::ParmVarDecl *PVD=llvm::dyn_cast<clang::ParmVarDecl>(MD->getParamDecl(i));
  clang::ValueDecl * VD = llvm::dyn_cast<clang::ValueDecl>(E->getMemberDecl());
  os << " Member data " << VD->getQualifiedNameAsString();
  os<< " is passed to a non-const reference parameter";
  os <<" of CXX method " << MD->getQualifiedNameAsString()<<" in const function ";
//  llvm::errs() << llvm::dyn_cast<clang::CXXMethodDecl>(AC->getDecl())->getQualifiedNameAsString();
//  if (visitingCallExpr) {
//	llvm::errs() << " in call stack ";
//	WListDump(llvm::errs());
//  }
//  llvm::errs()<<"\n";


  clang::ento::PathDiagnosticLocation ELoc =
   clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
  clang::SourceLocation L = E->getExprLoc();

  if (!m_exception.reportClass( ELoc, BR ) ) return;
  BR.EmitBasicReport(CE->getCalleeDecl(),"Class Checker :  Member data passed to non-const reference","ThreadSafety",os.str(),ELoc,L);

}

void WalkAST::ReportCallReturn(const clang::ReturnStmt * RS) {
  llvm::SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);

  CmsException m_exception;
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

  os << "Returns a pointer to a non-const member data object ";
  os << "in const function.\n ";
  RS->printPretty(os,0,Policy);
//  llvm::errs() << llvm::dyn_cast<clang::CXXMethodDecl>(AC->getDecl())->getQualifiedNameAsString();
//  if (visitingCallExpr) {
//	llvm::errs() << " in call stack ";
//	WListDump(llvm::errs());
//  }
//  llvm::errs()<<"\n";


  clang::ento::PathDiagnosticLocation CELoc =
    clang::ento::PathDiagnosticLocation::createBegin(RS, BR.getSourceManager(),AC);
  

  if (!m_exception.reportClass( CELoc, BR ) ) return;
  BugType * BT = new BugType("Class Checker : Const function returns pointer to non-const member data object","ThreadSafety");
  BugReport * R = new BugReport(*BT,os.str(),CELoc);
  BR.emitReport(R);
	 
}


void ClassChecker::checkASTDecl(const clang::CXXRecordDecl *RD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const {

	const clang::SourceManager &SM = BR.getSourceManager();
	
  	llvm::SmallString<100> buf;
  	llvm::raw_svector_ostream os(buf);
	clang::FileSystemOptions FSO;
	clang::FileManager FM(FSO);
	if (!FM.getFile("/tmp/classes.txt") ) {
		llvm::errs()<<"\n\nChecker optional.ClassChecker cannot find /tmp/classes.txt. Run 'scram b checker' with USER_LLVM_CHECKERS='-enable-checker optional.ClassDumperCT -enable-checker optional.ClassDumperFT' to create /tmp/classes.txt.\n\n\n";
		exit(1);
		}
	llvm::MemoryBuffer * buffer = FM.getBufferForFile(FM.getFile("/tmp/classes.txt"));
		os <<"class "<<RD->getQualifiedNameAsString()<<"\n";
		llvm::StringRef Rname(os.str());
		if (buffer->getBuffer().find(Rname) == llvm::StringRef::npos ) {return;}
	clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( RD, SM );
	if (  !m_exception.reportClass( DLoc, BR ) ) return;
//	clangcms::WalkAST walker(BR, mgr.getAnalysisDeclContext(RD));
//	clang::LangOptions LangOpts;
//	LangOpts.CPlusPlus = true;
//	clang::PrintingPolicy Policy(LangOpts);
//	RD->print(llvm::errs(),Policy,0,0);
//	llvm::errs() << s.str(); 
//	llvm::errs()<<"\n\n\n\n";
//	RD->dump();
//	llvm::errs()<<"\n\n\n\n";

 
// Check the constructors.
//			for (clang::CXXRecordDecl::ctor_iterator I = RD->ctor_begin(), E = RD->ctor_end();
//         			I != E; ++I) {
//        			if (clang::Stmt *Body = I->getBody()) {
//				llvm::errs()<<"Visited Constructors for\n";
//				llvm::errs()<<RD->getNameAsString();
//          				walker.Visit(Body);
//          				walker.Execute();
//        				}
//    				}
	


// Check the class methods (member methods).
	for (clang::CXXRecordDecl::method_iterator
		I = RD->method_begin(), E = RD->method_end(); I != E; ++I)  
		{
		clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( (*I) , SM );
		if (  !m_exception.reportClass( DLoc, BR ) ) continue;
		if ( !((*I)->hasBody()) ) continue;      
		if ( !llvm::isa<clang::CXXMethodDecl>((*I)) ) continue;
		if (!(*I)->isConst()) continue;
		if ((*I)->isVirtualAsWritten()) continue;
		clang::CXXMethodDecl * MD = llvm::cast<clang::CXXMethodDecl>((*I)->getMostRecentDecl());
//        	llvm::errs() << "\n\nmethod "<<MD->getQualifiedNameAsString()<<"\n\n";
//		for (clang::CXXMethodDecl::method_iterator J = MD->begin_overridden_methods(), F = MD->end_overridden_methods(); J != F; ++J) {
//			llvm::errs()<<"\n\n overwritten method "<<(*J)->getQualifiedNameAsString()<<"\n\n";
//			}


//				llvm::errs()<<"\n*****************************************************\n";
//				llvm::errs()<<"\nVisited CXXMethodDecl\n";
//				llvm::errs()<<RD->getNameAsString();
//				llvm::errs()<<"::";
//				llvm::errs()<<MD->getNameAsString();
//				llvm::errs()<<"\n*****************************************************\n";
				if ( MD->hasBody() ) {
					clang::Stmt *Body = MD->getBody();
//					clang::LangOptions LangOpts;
//					LangOpts.CPlusPlus = true;
//					clang::PrintingPolicy Policy(LangOpts);
//					std::string TypeS;
//	       				llvm::raw_string_ostream s(TypeS);
//	       				llvm::errs() << "\n\n+++++++++++++++++++++++++++++++++++++\n\n";
//	      				llvm::errs() << "\n\nPretty Print\n\n";
//	       				Body->printPretty(s, 0, Policy);
//        				llvm::errs() << s.str();
//					Body->dumpAll();
//	       				llvm::errs() << "\n\n+++++++++++++++++++++++++++++++++++++\n\n";
					clangcms::WalkAST walker(BR, mgr.getAnalysisDeclContext(MD));
	       				walker.Visit(Body);
					clang::QualType RQT = MD->getCallResultType();
					clang::QualType CQT = MD->getResultType();
					if (RQT.getTypePtr()->isPointerType() && !RQT.getTypePtr()->getPointeeType().isConstQualified() 
							&& MD->getName().lower().find("clone")==std::string::npos )  {
						llvm::SmallString<100> buf;
						llvm::raw_svector_ostream os(buf);
						os << MD->getQualifiedNameAsString() << " is a const member function that returns a pointer to a non-const object";
						os << "\n";
						clang::ento::PathDiagnosticLocation ELoc =clang::ento::PathDiagnosticLocation::createBegin( MD , SM );
						clang::SourceRange SR = MD->getSourceRange();
						BR.EmitBasicReport(MD, "Class Checker : Const function returns pointer to non-const object.","ThreadSafety",os.str(),ELoc);
					}
				}
   	}	/* end of methods loop */


} //end of class


} //end namespace
