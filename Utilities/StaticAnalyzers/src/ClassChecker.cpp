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
  const clang::CXXRecordDecl &RUT;

  typedef const clang::CallExpr * WorkListUnit;
  typedef clang::SmallVector<WorkListUnit, 50> DFSWorkList;

  typedef const clang::Expr * ExprListUnit;
  typedef clang::SmallVector<ExprListUnit, 50> ExprList;


  /// A vector representing the worklist which has a chain of CallExprs.
  DFSWorkList WList;
  
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
  const clang::CallExpr *visitingCallExpr;
  const clang::Expr *visitingExpr;
  
public:
  WalkAST(clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac,const clang::CXXRecordDecl &rut)
    : BR(br),
      AC(ac),
      RUT(rut),
      visitingCallExpr(0),
      visitingExpr(0) {}

  
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
      if (WLUnit == visitingCallExpr) {
		llvm::errs()<<"\nRecursive call to ";
		WLUnit->getDirectCallee()->printName(llvm::errs());
		llvm::errs()<<" , ";
		WLUnit->dumpPretty(AC->getASTContext());
		llvm::errs()<<"\n";
		return;
		}
      const clang::FunctionDecl *FD = WLUnit->getDirectCallee();
      llvm::SaveAndRestore<const clang::CallExpr *> SaveCall(visitingCallExpr, WLUnit);
      if (FD && FD->getBody()) Visit(FD->getBody());
      WList.pop_back();
  }

  // Stmt visitor methods.
  void VisitMemberExpr(clang::MemberExpr *E);
  void VisitCallExpr(clang::CallExpr *CE);
  void VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *CE);
  void VisitStmt(clang::Stmt *S) { VisitChildren(S); }
  void VisitChildren(clang::Stmt *S);
  
  void ReportCall(const clang::CallExpr *CE);
  void ReportMember(const clang::MemberExpr *ME);
  void ReportCallArg(const clang::CallExpr *CE, const int i);
  void ReportCallParam(const clang::CXXMethodDecl * MD, const clang::ParmVarDecl *PVD); 
};

//===----------------------------------------------------------------------===//
// AST walking.
//===----------------------------------------------------------------------===//

void WalkAST::VisitChildren(clang::Stmt *S) {
  for (clang::Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (clang::Stmt *child = *I)
      Visit(child);
}

void WalkAST::VisitMemberExpr(clang::MemberExpr *ME) {

const clang::CXXMethodDecl * ACD = llvm::dyn_cast<clang::CXXMethodDecl>(AC->getDecl());
clang::Expr * E = ME->getBase();
clang::QualType qual_base = E->getType();
clang::ValueDecl * VD = ME->getMemberDecl();
clang::QualType qual_vd = VD->getType();
clang::QualType qual_md = ACD->getThisType(AC->getASTContext());
if (!(ME->isBoundMemberFunction(AC->getASTContext())))
	if (visitingCallExpr)
	if (qual_base == qual_md) 
	{
	ReportMember(ME);
//	llvm::errs()<<"\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
//	llvm::errs()<<"\n CXXMemberExpr \n";
//	llvm::errs()<<"\n___________\n";           
//	ME->dump();
//	llvm::errs()<<"\n___________\n";           
//	qual_md->dump();
//	llvm::errs()<<"\n___________\n";           
//	qual_base->dump();
//	llvm::errs()<<"\n___________\n";
//		ACD->printName(llvm::errs());
//		llvm::errs() << "\n";           
//	 	for (llvm::SmallVectorImpl<const clang::CallExpr *>::iterator I = WList.begin(),
//	      	E = WList.end(); I != E; ++I) {
//		    	const clang::FunctionDecl *FD = (*(I))->getDirectCallee();
//		    	assert(FD);
//		    	if (VisitedFunctions[FD] == PostVisited) {
//			FD->printName(llvm::errs());
//		      	llvm::errs() << "\n" ;
//		    }
//		}
//	llvm::errs()<<"\n___________\n";
//	visitingCallExpr->getDirectCallee()->printName(llvm::errs());
//	llvm::errs()<<"\n";
//	llvm::errs()<<"\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
	}
}

void WalkAST::VisitCallExpr(clang::CallExpr *CE) {
  Enqueue(CE);
  Execute();
  VisitChildren(CE);
}


void WalkAST::VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *CE) {


  Enqueue(CE);
  Execute();
  VisitChildren(CE);

//clang::Expr * IOA = CE->getImplicitObjectArgument();
//clang::QualType qual_ioa = llvm::dyn_cast<clang::Expr>(IOA)->getType();
//clang::MemberExpr * ME = clang::dyn_cast<clang::MemberExpr>(CE->getCallee());
//clang::QualType qual_exp = llvm::dyn_cast<clang::Expr>(ME)->getType();

//clang::CXXRecordDecl * RD = llvm::dyn_cast<clang::CXXRecordDecl>(CE->getRecordDecl());
clang::CXXMethodDecl * MD = CE->getMethodDecl();
//clang::CXXRecordDecl * MRD = llvm::dyn_cast<clang::CXXRecordDecl>(MD->getParent());
clang::QualType MQT = MD->getThisType(AC->getASTContext());
//const clang::CXXRecordDecl * ACD = llvm::dyn_cast<clang::CXXRecordDecl>(AC->getDecl());
                                                                                                               


for(int i=0, j=CE->getNumArgs(); i<j; i++) {
	if ( const clang::Expr *E = llvm::dyn_cast<clang::Expr>(CE->getArg(i)))
		{
		clang::QualType qual_arg = E->getType();
		if (const clang::MemberExpr *ME=llvm::dyn_cast<clang::MemberExpr>(E))		
		if (ME->isImplicitAccess())
			{
//			clang::ValueDecl * VD = llvm::dyn_cast<clang::ValueDecl>(ME->getMemberDecl());
//			clang::QualType qual_decl = llvm::dyn_cast<clang::ValueDecl>(ME->getMemberDecl())->getType();
			clang::ParmVarDecl *PVD=llvm::dyn_cast<clang::ParmVarDecl>(MD->getParamDecl(i));
			clang::QualType QT = PVD->getOriginalType();
			const clang::Type * T = QT.getTypePtr();
			if (!support::isConst(QT))
			if (T->isReferenceType())
				{
//				ME->dump();
//				llvm::errs()<<"\n";
//				VD->dump();
//				llvm::errs()<<"\n";
//				qual_decl.dump();
//				llvm::errs()<<"\n";
//				PVD->dump();
//				llvm::errs()<<"\n";
//				QT->dump();
//				llvm::errs()<<"\n---------------------------------------\n";
				ReportCallArg(CE,i);
				}
			}
		}
}

//if (ME->isImplicitAccess())
//if (!support::isConst(qual_ioa))
//if (MD->getAccess()==clang::AccessSpecifier::AS_public)
//for(int i=0, j=MD->getNumParams();i<j;i++) {
//	if (clang::ParmVarDecl *PVD=llvm::dyn_cast<clang::ParmVarDecl>(MD->getParamDecl(i)))
//	{
//			clang::QualType QT = PVD->getOriginalType();
//			const clang::Type * T = QT.getTypePtr();
//			if (!support::isConst(QT))
//			if (T->isReferenceType())
//				{
//				ReportCallParam(MD,PVD);
//				}
//	}
//}


//	llvm::errs()<<"\n--------------------------------------------------------------\n";
//	llvm::errs()<<"\n------CXXMemberCallExpression---------------------------------\n";
//	llvm::errs()<<"\n--------------------------------------------------------------\n";
//	llvm::errs()<<"\n";
//	CE->dump();
//	llvm::errs()<<"\n";
//	if (MD->hasBody()) {
//		MD->getBody()->dump();
//		Visit(MD->getBody());
//		}
//	llvm::errs()<<"\n";
//	llvm::errs()<<"\n--------------------------------------------------------------\n";
//	return;
}

void WalkAST::ReportMember(const clang::MemberExpr *ME) {
  llvm::SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);

  CmsException m_exception;
  clang::ValueDecl * VD = ME->getMemberDecl();
	  os << " Member function is indirectly accessing member data ";
	  VD->printName(os);
	  os << " in  call stack\n";
	  llvm::dyn_cast<clang::NamedDecl>(AC->getDecl())->printName(os);
 	  for (llvm::SmallVectorImpl<const clang::CallExpr *>::iterator I = WList.begin(),
	      	E = WList.end(); I != E; I++) {
		    	const clang::FunctionDecl *FD = (*(I))->getDirectCallee();
		      	os << "-->" ;
		    	assert(FD);
			FD->printName(os);
			}
	
  clang::ento::PathDiagnosticLocation CELoc =
    clang::ento::PathDiagnosticLocation::createBegin(*(WList.begin()), BR.getSourceManager(),AC);
  clang::SourceRange R = ME->getSourceRange();

  if (!m_exception.reportClass( CELoc, BR ) ) return;
  BR.EmitBasicReport(AC->getDecl(),"Class Checker : Member data indirect modify check","ThreadSafety",os.str(),CELoc,R);
}

void WalkAST::ReportCall(const clang::CallExpr *CE) {
  llvm::SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);

  CmsException m_exception;
  os << "Call Expression ";
  // Name of current visiting CallExpr.
  os << *CE->getDirectCallee();
// Name of the CallExpr whose body is current walking.
  if (visitingCallExpr) 
    os << " <-- " << *visitingCallExpr->getDirectCallee();
// Names of FunctionDecls in worklist with state PostVisited.
  for (llvm::SmallVectorImpl<const clang::CallExpr *>::iterator I = WList.end(),
         E = WList.begin(); I != E; --I) {
	    const clang::FunctionDecl *FD = (*(I-1))->getDirectCallee();
	    assert(FD);
	    if (VisitedFunctions[FD] == PostVisited)
	      os << " <-- " << *FD;
  }
     os << " is a function.\n";

// Names of args  
    clang::LangOptions LangOpts;
    LangOpts.CPlusPlus = true;
    clang::PrintingPolicy Policy(LangOpts);
    for(int i=0, j=CE->getNumArgs(); i<j; i++)
	{
	std::string TypeS;
        llvm::raw_string_ostream s(TypeS);
        CE->getArg(i)->printPretty(s, 0, Policy);
        os << "arg: " << s.str() << " ";
	}	

  os << "\n";

  clang::ento::PathDiagnosticLocation CELoc =
    clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
  clang::SourceRange R = CE->getCallee()->getSourceRange();
  

// llvm::errs()<<os.str();
  if (!m_exception.reportClass( CELoc, BR ) ) return;
  BR.EmitBasicReport(CE->getCalleeDecl(),"Class Checker : Method Call Expr check","ThreadSafety",os.str(),CELoc,R);
	 
}

void WalkAST::ReportCallArg(const clang::CallExpr *CE,const int i) {

  llvm::SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);
  CmsException m_exception;
  clang::CXXMethodDecl * MD = llvm::dyn_cast<clang::CXXMemberCallExpr>(CE)->getMethodDecl();
  const clang::MemberExpr *E = llvm::dyn_cast<clang::MemberExpr>(CE->getArg(i));
  clang::ParmVarDecl *PVD=llvm::dyn_cast<clang::ParmVarDecl>(MD->getParamDecl(i));
  const clang::MemberExpr * ME = clang::dyn_cast<clang::MemberExpr>(CE->getCallee());
  clang::ValueDecl * VD = llvm::dyn_cast<clang::ValueDecl>(E->getMemberDecl());
  os << *CE->getDirectCallee();
  if (ME->isImplicitAccess()) 
  	os << " is a member function acting on member data.\n";
  else 
  	os << " is a non-member function acting on member data.\n";
  os << " Member data ";
  VD->printName(os);
  os<< " is passed to a non-const reference parameter ";
  PVD->printName(os);

  clang::ento::PathDiagnosticLocation ELoc =
   clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
  clang::SourceLocation L = E->getExprLoc();

  if (!m_exception.reportClass( ELoc, BR ) ) return;
  BR.EmitBasicReport(CE->getCalleeDecl(),"Class Checker :  Method Call Expr Arg check","ThreadSafety",os.str(),ELoc,L);

}

void WalkAST::ReportCallParam(const clang::CXXMethodDecl * MD,const clang::ParmVarDecl *PVD) {
  llvm::SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);
  CmsException m_exception;
  os << "method declaration ";
  MD->printName(os);
  os << " with parameter decl ";
  PVD->printName(os);
  os << " , a non const reference\n";
  clang::ento::PathDiagnosticLocation ELoc =
   clang::ento::PathDiagnosticLocation::createBegin(MD, BR.getSourceManager());
  clang::SourceRange R = PVD->getSourceRange();

  if (!m_exception.reportClass( ELoc, BR ) ) return;
  BR.EmitBasicReport(MD,"Class Checker :  Method Decl Param Decl check","ThreadSafety",os.str(),ELoc,R);

}

void ClassCheckerRDecl::checkASTDecl(const clang::CXXRecordDecl *CRD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const {

	const clang::CXXRecordDecl *RD=CRD;
//	clangcms::WalkAST walker(BR, mgr.getAnalysisDeclContext(RD));
	const clang::SourceManager &SM = BR.getSourceManager();
	clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( RD, SM );
	if (  !m_exception.reportClass( DLoc, BR ) ) return;
//	clang::LangOptions LangOpts;
//	LangOpts.CPlusPlus = true;
//	clang::PrintingPolicy Policy(LangOpts);
//	std::string TypeS;
//	llvm::raw_string_ostream s(TypeS);
//	RD->print(s,Policy,0,0);
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

			if ( I->getNameAsString() == "produce" 
				|| I->getNameAsString() == "beginRun" 
				|| I->getNameAsString() == "endRun" 
				|| I->getNameAsString() == "beginLuminosityBlock" 
				|| I->getNameAsString() == "endLuminosityBlock" )
			{
				const clang::CXXMethodDecl *  MD = &llvm::cast<const clang::CXXMethodDecl>(*I->getMostRecentDecl());
				if (MD->isVirtualAsWritten()) continue;
				clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( MD , SM );
				clang::SourceRange R = MD->getSourceRange();
				clangcms::WalkAST walker(BR, mgr.getAnalysisDeclContext(MD),*RD);
//				llvm::errs()<<"\n*****************************************************\n";
//				llvm::errs()<<"\nVisited CXXMethodDecl\n";
//				llvm::errs()<<RD->getNameAsString();
//				llvm::errs()<<"::";
//				llvm::errs()<<I->getNameAsString();
//				llvm::errs()<<"\n*****************************************************\n";
				if (  !m_exception.reportClass( DLoc, BR ) ) continue;
				if ( I->hasBody() ){
					clang::Stmt *Body = I->getBody();
//					clang::LangOptions LangOpts;
//					LangOpts.CPlusPlus = true;
//					clang::PrintingPolicy Policy(LangOpts);
//					std::string TypeS;
//	       				llvm::raw_string_ostream s(TypeS);
//	       				llvm::errs() << "\n\n+++++++++++++++++++++++++++++++++++++\n\n";
//	      				llvm::errs() << "\n\nPretty Print\n\n";
//	       				Body->printPretty(s, 0, Policy);
//        				llvm::errs() << s.str();
//        				llvm::errs() << "\n\nDump\n\n";
//					Body->dumpAll();
//	       				llvm::errs() << "\n\n+++++++++++++++++++++++++++++++++++++\n\n";
	       				walker.Visit(Body);
//	       				walker.Execute();
       				}
			} /* end of Name check */
   	}	/* end of methods loop */


} //end of class


}


