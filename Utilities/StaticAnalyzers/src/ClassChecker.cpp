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
//	llvm::errs()<<"\nRecursive call to ";
//	WLUnit->getDirectCallee()->printName(llvm::errs());
//	llvm::errs()<<" , ";
//	WLUnit->dumpPretty(AC->getASTContext());
//	llvm::errs()<<"\n";
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
//      if (visitingCallExpr && WLUnit->getMethodDecl() == visitingCallExpr->getMethodDecl()) {
//		llvm::errs()<<"\nRecursive call to ";
//		WLUnit->getDirectCallee()->printName(llvm::errs());
//		llvm::errs()<<" , ";
//		WLUnit->dumpPretty(AC->getASTContext());
//		llvm::errs()<<"\n";
//   		WList.pop_back();
//		return;
//		}
      
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
  void ReportDeclRef( const clang::DeclRefExpr * DRE);
  void CheckCXXOperatorCallExpr(const clang::CXXOperatorCallExpr *CE,const clang::Expr *E);
  void CheckBinaryOperator(const clang::BinaryOperator * BO,const clang::Expr *E);
  void CheckUnaryOperator(const clang::UnaryOperator * UO,const clang::Expr *E);
  void CheckExplicitCastExpr(const clang::ExplicitCastExpr * CE,const clang::Expr *E);
  void CheckReturnStmt(const clang::ReturnStmt * RS, const clang::Expr *E);
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
      Visit(child);
    }
}


void WalkAST::CheckBinaryOperator(const clang::BinaryOperator * BO,const clang::Expr *E) {
//	BO->dump(); 
//	llvm::errs()<<"\n";
if (BO->isAssignmentOp()) {
	if (clang::DeclRefExpr * DRE =dyn_cast<clang::DeclRefExpr>(BO->getLHS())){
			ReportDeclRef(DRE);
		} else
	if (clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(BO->getLHS())){
			if (ME->isImplicitAccess()) ReportMember(ME);
		}
	}
}
void WalkAST::CheckUnaryOperator(const clang::UnaryOperator * UO,const clang::Expr *E) {
//	UO->dump(); 
//	llvm::errs()<<"\n";
  if (UO->isIncrementDecrementOp())  {
	if ( dyn_cast<clang::DeclRefExpr>(E)) 
	if (clang::DeclRefExpr * DRE =dyn_cast<clang::DeclRefExpr>(UO->getSubExpr()->IgnoreParenImpCasts())) {
		ReportDeclRef(DRE);
	} else 
	if ( dyn_cast<clang::MemberExpr>(E)) 
	if (clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(UO->getSubExpr()->IgnoreParenImpCasts())) {
		ReportMember(ME);
	}
  }
}

void WalkAST::CheckCXXOperatorCallExpr(const clang::CXXOperatorCallExpr *OCE,const clang::Expr *E) {
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
	if (const clang::DeclRefExpr * DRE =dyn_cast<clang::DeclRefExpr>(OCE->arg_begin()->IgnoreParenImpCasts())) {
		ReportDeclRef(DRE);
	} else
	if (const clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(OCE->arg_begin()->IgnoreParenImpCasts())){
		if (ME->isImplicitAccess())
			ReportMember(ME);
	} 

	case OO_PlusPlus:
	case OO_MinusMinus:
	if ( dyn_cast<clang::DeclRefExpr>(E)) 
	if (const clang::DeclRefExpr * DRE =dyn_cast<clang::DeclRefExpr>(OCE->arg_begin()->IgnoreParenCasts())) {
		ReportDeclRef(DRE);
	} else
	if ( dyn_cast<clang::MemberExpr>(E)) 
	if (const clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(OCE->getCallee()->IgnoreParenCasts())) {
		if (ME->isImplicitAccess())
			ReportMember(ME);
	} 
	
}

}


void WalkAST::CheckExplicitCastExpr(const clang::ExplicitCastExpr * CE,const clang::Expr *expr){
//	CE->dump();
//	llvm::errs()<<"\n";

	const clang::Expr *E = CE->getSubExpr();
	clang::ASTContext &Ctx = AC->getASTContext();
	clang::QualType OrigTy = Ctx.getCanonicalType(E->getType());
	clang::QualType ToTy = Ctx.getCanonicalType(CE->getType());

	if (const clang::MemberExpr * ME = dyn_cast<clang::MemberExpr>(expr)) {
	if ( support::isConst( OrigTy ) && ! support::isConst(ToTy) )
		ReportCast(CE,ME);
	}

	if (const clang::DeclRefExpr * DRE = dyn_cast<clang::DeclRefExpr>(expr)) {
	if ( support::isConst( OrigTy ) && ! support::isConst(ToTy) )
		ReportCast(CE,DRE);
	}
}
 

void WalkAST::CheckReturnStmt(const clang::ReturnStmt * RS, const Expr * E){
//		llvm::errs()<<"\nReturn Expression\n";
//		RE->dump();
//		llvm::errs()<<"\n";
//		RQT->dump();
//		llvm::errs()<<"\n";
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
		if ( (RTy->isPointerType() || RTy->isReferenceType() ) )
		if( !support::isConst(RTy) ) {
		if ( const clang::MemberExpr *ME = dyn_cast<clang::MemberExpr>(E))
			if (ME->isImplicitAccess()) 
			{
			ReportCallReturn(RS);
			return;
			}
		}
	}
	
}

void WalkAST::VisitDeclRefExpr( clang::DeclRefExpr * DRE) {
  if (clang::VarDecl * D = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl()) ) { 
  	clang::SourceLocation SL = DRE->getLocStart();
  	if (BR.getSourceManager().isInSystemHeader(SL) || BR.getSourceManager().isInExternCSystemHeader(SL)) return;
	Stmt * P = AC->getParentMap().getParent(DRE);
	while (AC->getParentMap().hasParent(P)) {
		if (const clang::UnaryOperator * UO = llvm::dyn_cast<clang::UnaryOperator>(P)) 
			{ WalkAST::CheckUnaryOperator(UO,DRE);}
		if (const clang::BinaryOperator * BO = llvm::dyn_cast<clang::BinaryOperator>(P)) 
			{ WalkAST::CheckBinaryOperator(BO,DRE);}
		if (const clang::CXXOperatorCallExpr *OCE = llvm::dyn_cast<clang::CXXOperatorCallExpr>(P)) 
			{ WalkAST::CheckCXXOperatorCallExpr(OCE,DRE);}
		if (const clang::ExplicitCastExpr * CE = llvm::dyn_cast<clang::ExplicitCastExpr>(P))
			{ WalkAST::CheckExplicitCastExpr(CE,DRE);}
		if (const clang::CXXNewExpr * NE = llvm::dyn_cast<clang::CXXNewExpr>(P)) break; 
		P = AC->getParentMap().getParent(P);
	}


//	llvm::errs()<<"Declaration Ref Expr\t";
//	dyn_cast<Stmt>(DRE)->dumpPretty(AC->getASTContext());
//	DRE->dump();
//	llvm::errs()<<"\n";
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

	if ( D->isStaticLocal() && ! support::isConst( t ) )
	{
		std::string buf;
	    	llvm::raw_string_ostream os(buf);
	   	os << "Non-const variable '" << D->getNameAsString() << "' is static local and modified in statement '";
	    	PS->printPretty(os,0,Policy);
		os << "'.\n";
	    	BugType * BT = new BugType("ClassChecker : non-const static local variable modified","ThreadSafety");
		BugReport * R = new BugReport(*BT,os.str(),CELoc);
		BR.emitReport(R);
		return;
	}

	if ( D->isStaticDataMember() && ! support::isConst( t ) )
	{
	    	std::string buf;
	    	llvm::raw_string_ostream os(buf);
	    	os << "Non-const variable '" << D->getNameAsString() << "' is static member data and modified in statement '";
	    	PS->printPretty(os,0,Policy);
		os << "'.\n";
	    	BugType * BT = new BugType("ClassChecker : non-const static member variable modified","ThreadSafety");
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
	    	os << "Non-const variable '" << D->getNameAsString() << "' is global static and modified in statement '";
	    	PS->printPretty(os,0,Policy);
		os << "'.\n";
	    	BugType * BT = new BugType("ClassChecker : non-const global static variable modified","ThreadSafety");
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
  Visit(CE->getImplicitObjectArgument());

  std::string name = MD->getNameAsString();
  if (name == "end" || name == "begin" || name == "find") return;
  if (llvm::isa<clang::MemberExpr>(CE->getImplicitObjectArgument()->IgnoreParenCasts() ) ) 
  if ( !MD->isConst() ) ReportCall(CE);

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
 if ( visitingCallExpr ) {
	clang::Expr * IOA = visitingCallExpr->getImplicitObjectArgument();
	if (!( IOA->isImplicitCXXThis() || llvm::dyn_cast<CXXThisExpr>(IOA->IgnoreParenCasts()))) return;
	}
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


  os << " Member data '";
//  os << ME->getMemberDecl()->getQualifiedNameAsString();
   ME->printPretty(os,0,Policy);
  os << "' is directly or indirectly modified in const function '";
  llvm::errs() << os.str();
  llvm::errs() << llvm::dyn_cast<clang::CXXMethodDecl>(AC->getDecl())->getQualifiedNameAsString();
  if (visitingCallExpr) {
   llvm::errs() << "' in function call '";
    visitingCallExpr->printPretty(os,0,Policy);
  }
  if (hasWork()) {
  	llvm::errs() << "' in call stack '";
  	WListDump(llvm::errs());
  }
  llvm::errs() << "'.\n";

//  ME->printPretty(llvm::errs(),0,Policy);
//  llvm::errs()<<"\n";
//  ME->dump();
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
  os<<"' is a non-const member function that could modify member data object '";
  CE->getImplicitObjectArgument()->printPretty(os,0,Policy);
  os << llvm::dyn_cast<clang::CXXMethodDecl>(AC->getDecl())->getQualifiedNameAsString();
  os << "'.\n";
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

 
  os << "Const qualifier of data object '";
  E->printPretty(os,0,Policy);
  os <<"' was removed via cast expression '";
  CE->printPretty(os,0,Policy);
  os <<  "'.\n";
  llvm::errs()<<os.str();
  llvm::errs() << llvm::dyn_cast<clang::CXXMethodDecl>(AC->getDecl())->getQualifiedNameAsString();
  if (visitingCallExpr) {
	llvm::errs() << "' in call stack '";
	WListDump(llvm::errs());
  }
  llvm::errs()<<"'.\n";


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
  os <<" of CXX method '" << MD->getQualifiedNameAsString()<<"' in const function '";
  os<< llvm::dyn_cast<clang::CXXMethodDecl>(AC->getDecl())->getQualifiedNameAsString();
  if (visitingCallExpr) {
	os << "' in call stack '";
	WListDump(os);
  }
  os<<"'.\n";


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

  os << "Returns a pointer or reference to a non-const member data object ";
  os << "in const function in statement '";
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
  BugType * BT = new BugType("Class Checker : Const function returns pointer or reference to non-const member data object","ThreadSafety");
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
		if ( !llvm::isa<clang::CXXMethodDecl>((*I)) ) continue;
		if (!(*I)->isConst()) continue;
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
//				llvm::errs()<<"\n";
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
					clang::QualType CQT = MD->getCallResultType();
					clang::QualType RQT = MD->getResultType();
					clang::ASTContext &Ctx = BR.getContext();
					clang::QualType RTy = Ctx.getCanonicalType(RQT);
					clang::QualType CTy = Ctx.getCanonicalType(CQT);
//					llvm::errs()<<"\n"<<MD->getQualifiedNameAsString()<<"\n\n";
//					llvm::errs()<<"Call Result Type\n";
//					CTy->dump();
//					llvm::errs()<<"\n";
//					llvm::errs()<<"Result Type\n";
//					RTy->dump();
//					llvm::errs()<<"\n";

					clang::ento::PathDiagnosticLocation ELoc =clang::ento::PathDiagnosticLocation::createBegin( MD , SM );
					if ((RTy->isPointerType() || RTy->isReferenceType() ))
					if (!support::isConst(RTy) ) 
					if ( MD->getNameAsString().find("clone")==std::string::npos )  
					{
						llvm::SmallString<100> buf;
						llvm::raw_svector_ostream os(buf);
						os << MD->getQualifiedNameAsString() << " is a const member function that returns a pointer or reference to a non-const object";
						os << "\n";
						llvm::errs()<<os.str();
						clang::SourceRange SR = MD->getSourceRange();
						BR.EmitBasicReport(MD, "Class Checker : Const function returns pointer or reference to non-const object.","ThreadSafety",os.str(),ELoc);
					}
				}
   	}	/* end of methods loop */


} //end of class


} //end namespace
