// ClassChecker.cpp by Patrick Gartung (gartung@fnal.gov)
//
// Objectives of this checker
//
// For each special function of a class (produce, beginrun, endrun, beginlumi, endlumi)
//
//	1) indentify member data being modified
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

namespace clangcms {

void ClassCheckerMDecl::checkASTDecl(const clang::CXXMethodDecl *D,
                    clang::ento::AnalysisManager &Mgr,
                    clang::ento::BugReporter &BR) const
{
//	if ( D->getNameAsString() == "produce" 
//		|| D->getNameAsString() == "beginRun" 
//		|| D->getNameAsString() == "endRun" 
//		|| D->getNameAsString() == "beginLuminosityBlock" 
//		|| D->getNameAsString() == "endLuminosityBlock" )
	{	    

	    clang::ento::PathDiagnosticLocation PLoc =
  	    clang::ento::PathDiagnosticLocation::createBegin(D, BR.getSourceManager());
	    if ( ! m_exception.reportGeneral( PLoc, BR ) )
		return; 
	    const clang::CXXRecordDecl* P= D->getParent(); 
	    std::string buf;
	    llvm::raw_string_ostream os(buf);
	    os << "Declaration of Method  ";
	    D->printName(os);
	    os <<" in Class " << *P <<" .\n";
	    BR.EmitBasicReport(D, "Class Checker CXXMethodDecl","ThreadSafety",os.str(), PLoc);
		if (D->hasBody()){
			clang::Stmt* S = D->getBody();
			for (clang::Stmt::const_child_iterator
				c = S->child_begin(), e=S->child_end();c !=e; ++c) {
				if ( llvm::isa<clang::CXXMemberCallExpr>(*c) ) {
					const clang::CXXMemberCallExpr * ce = llvm::cast<clang::CXXMemberCallExpr>(*c);
					clang::CXXMethodDecl *D = ce->getMethodDecl();
					const clang::CXXRecordDecl* P = ce->getRecordDecl();
					clang::ento::PathDiagnosticLocation DLoc = 
						clang::ento::PathDiagnosticLocation::createBegin(ce->getDirectCallee(), BR.getSourceManager());
    					std::string buf;
	    				llvm::raw_string_ostream os(buf);
					os<< "CXXMemberCallExpr "<< *ce->getDirectCallee();
					os<<" MethodDecl "<<*D <<" RecordDecl " << *P << " .\n";
					if ( m_exception.reportClass( DLoc, BR ) )  { 
						BR.EmitBasicReport(ce->getCalleeDecl(), "Class Checker CXXMemberCallExpr in CXXMethodDecl","ThreadSafety",os.str(), DLoc);
						}
					}
				}
			} 

	}
}


void ClassCheckerRDecl::checkASTDecl(const clang::CXXRecordDecl *D,
                    clang::ento::AnalysisManager &Mgr,
                   clang::ento::BugReporter &BR) const
{
//		const clang::SourceManager &SM = BR.getSourceManager();
//	    	clang::ento::PathDiagnosticLocation DLoc = clang::ento::PathDiagnosticLocation::createBegin(D, BR.getSourceManager());
//	    	if ( ! m_exception.reportGeneral( DLoc, BR ) ) return;
//  		for (clang::CXXRecordDecl::method_iterator
//		I = D->method_begin(), E = D->method_end(); I != E; ++I)  
//		{
//			if ( I->getNameAsString() == "produce" 
//				|| I->getNameAsString() == "beginRun" 
//				|| I->getNameAsString() == "endRun" 
//				|| I->getNameAsString() == "beginLuminosityBlock" 
//				|| I->getNameAsString() == "endLuminosityBlock" )
//			{
//				const clang::CXXMethodDecl *  MD = &(*I);
//				MD->dump();
//				clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( MD, SM );
//				std::string buf;
//	    			llvm::raw_string_ostream os(buf);
//  				os << "Method  " << (*I) << " Parent Class "<< *D<<". \n";
// 				if ( ! m_exception.reportClass( DLoc, BR ) )
//					return; 
//				BR.EmitBasicReport( MD, "Class Checker CXXMethodDecl in CXXRecordDecl","ThreadSafety",os.str(), DLoc);
//
//			for (clang::CXXMethodDecl::method_iterator
//				J=I->begin_overridden_methods(), F=I->end_overridden_methods(); J !=F; J++)
//				{
//				
//				const clang::CXXMethodDecl * MD = (*J);
//				MD->dump();
//				clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( MD, SM );
//				std::string buf;
//	    			llvm::raw_string_ostream os(buf);
//    				os << "Overridden Method  " << (**J) << " Method "<< (*I)<<". \n";
//	    			llvm::outs()<<os.str();
// 				if ( ! m_exception.reportClass( DLoc, BR ) )
//					return; 
//				DLoc.asLocation().dump();
//				BR.EmitBasicReport( *J, "Class Checker Overridden CXXMethodDecl in CXXRecordDecl","ThreadSafety",os.str(), DLoc);
//				}				   				
//			}			
//		} /* end method loop */
//
//
//  		for (clang::CXXRecordDecl::field_iterator
//		       	I = D->field_begin(), E = D->field_end(); I != E; ++I)  {
//			const clang::FieldDecl * RD = &(*I);
//	    		std::string buf;
//	    		llvm::raw_string_ostream os(buf);
//  			os << "Declaration of Field  " << *RD << " in Class "<<*D<<" .\n";
//			clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( RD, SM );
//			BR.EmitBasicReport(D, "Class Checker FieldDecl in CXXRecordDecl","ThreadSafety",os.str(), DLoc);
// 		}  /* end of field loop */

} /* end class */



void ClassCheckerMCall::checkPostStmt(const clang::CXXMemberCallExpr *CE,
		clang::ento::CheckerContext &C) const 
{	
	
	clang::CXXMethodDecl *D = CE->getMethodDecl();
	const clang::CXXRecordDecl *P = CE->getRecordDecl();
	if ( CE->getNumArgs() == 0) return;	
	
	if (clang::ento::ExplodedNode *errorNode = C.addTransition()) {
		if (!BT)
			BT.reset(new clang::ento::BugType("Class Checker CXXMemberCallExpr", "ThreadSafety"));
	    	std::string buf;
	    	llvm::raw_string_ostream os(buf);
	    	os << "CXXMemberCallExpression"<< CE->getCalleeDecl()<<" MethodDecl "<< *D << " CXXRecordDecl " << *P <<".\n";
		for ( clang::CallExpr::const_arg_iterator I=CE->arg_begin(), E=CE->arg_end(); I != E;++I)
		{
			os <<" Arg "<< *I <<" ";
		} 
		clang::ento::BugReport *R = new clang::ento::BugReport(*BT, os.str(), errorNode);
		R->addRange(CE->getSourceRange());
	   	if ( ! m_exception.reportConstCast( *R, C ) ) C.EmitReport(R);
	}

}



class WalkAST : public clang::StmtVisitor<WalkAST> {
  clang::ento::BugReporter &BR;
  clang::AnalysisDeclContext *AC;

  typedef const clang::CallExpr * WorkListUnit;
  typedef clang::SmallVector<WorkListUnit, 20> DFSWorkList;

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
  
public:
  WalkAST(clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac)
    : BR(br),
      AC(ac),
      visitingCallExpr(0) {}
  
  bool hasWork() const { return !WList.empty(); }

  /// This method adds a CallExpr to the worklist and marks the callee as
  /// being PreVisited.
  void Enqueue(WorkListUnit WLUnit) {
    const clang::FunctionDecl *FD = WLUnit->getDirectCallee();
    if (!FD || !FD->getBody())
      return;    
    Kind &K = VisitedFunctions[FD];
    if (K != NotVisited)
      return;
    K = PreVisited;
    WList.push_back(WLUnit);
  }

  /// This method returns an item from the worklist without removing it.
  WorkListUnit Dequeue() {
    assert(!WList.empty());
    return WList.back();    
  }
  
  void Execute() {
    while (hasWork()) {
      WorkListUnit WLUnit = Dequeue();
      const clang::FunctionDecl *FD = WLUnit->getDirectCallee();
      assert(FD && FD->getBody());

      if (VisitedFunctions[FD] == PreVisited) {
        // If the callee is PreVisited, walk its body.
        // Visit the body.
        llvm::SaveAndRestore<const clang::CallExpr *> SaveCall(visitingCallExpr, WLUnit);
        Visit(FD->getBody());
        
        // Mark the function as being PostVisited to indicate we have
        // scanned the body.
        VisitedFunctions[FD] = PostVisited;
        continue;
      }

      // Otherwise, the callee is PostVisited.
      // Remove it from the worklist.
      assert(VisitedFunctions[FD] == PostVisited);
      WList.pop_back();
    }
  }

  // Stmt visitor methods.
  void VisitCallExpr(clang::CallExpr *CE);
  void VisitCXXMemberCallExpr(clang::CallExpr *CE);
  void VisitStmt(clang::Stmt *S) { VisitChildren(S); }
  void VisitChildren(clang::Stmt *S);
  
  void ReportCall(const clang::CallExpr *CE, bool isPure);

};

//===----------------------------------------------------------------------===//
// AST walking.
//===----------------------------------------------------------------------===//

void WalkAST::VisitChildren(clang::Stmt *S) {
  for (clang::Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (clang::Stmt *child = *I)
      Visit(child);
}

void WalkAST::VisitCallExpr(clang::CallExpr *CE) {
  VisitChildren(CE);
  Enqueue(CE);
}

void WalkAST::VisitCXXMemberCallExpr(clang::CallExpr *CE) {
  VisitChildren(CE);
  bool callIsNonVirtual = false;
  
  // Several situations to elide for checking.
  if (clang::MemberExpr *CME = llvm::dyn_cast<clang::MemberExpr>(CE->getCallee())) {
    // If the member access is fully qualified (i.e., X::F), then treat
    // this as a non-virtual call and do not warn.
    if (CME->getQualifier())
      callIsNonVirtual = true;

    // Elide analyzing the call entirely if the base pointer is not 'this'.
    if (clang::Expr *base = CME->getBase()->IgnoreImpCasts())
      if (!clang::isa<clang::CXXThisExpr>(base))
        return;
  }

  // Get the callee.
  const clang::CXXMethodDecl *MD = llvm::dyn_cast<clang::CXXMethodDecl>(CE->getDirectCallee());
//  if (MD && MD->isVirtual() && !callIsNonVirtual)
    ReportCall(CE, MD->isPure());

  Enqueue(CE);
}

void WalkAST::ReportCall(const clang::CallExpr *CE, bool isPure) {
  llvm::SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);

  CmsException m_exception;
  os << "Call Path : ";
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
   os <<"/n";
  // Names of args  
  for ( clang::CallExpr::const_arg_iterator I=CE->arg_begin(), E=CE->arg_end(); I != E;++I){
	os<<"Args "<<*I<<" ";
  }	


  os << "\n";

  clang::ento::PathDiagnosticLocation CELoc =
    clang::ento::PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(), AC);
  clang::SourceRange R = CE->getCallee()->getSourceRange();


  
  if (m_exception.reportClass( CELoc, BR ) ) {
  BR.EmitBasicReport(AC->getDecl(),
                       "CMSSW Class Method CallExpr ",
                      "ThreadSafety",
                       os.str(), CELoc, &R,1);
    return;
  }
  else {
//    os << "\n Class Methods Checker";
//    BR.EmitBasicReport(AC->getDecl(),
//                       "External Class Method CallExpr",
//                       "Thread Safety",
//                       os.str(), CELoc, &R,1);
    return;

  }
}

void ClassChecker::checkASTDecl(const clang::CXXRecordDecl *RD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const {

	clangcms::WalkAST walker(BR, mgr.getAnalysisDeclContext(RD));

	const clang::SourceManager &SM = BR.getSourceManager();
	clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( RD, SM );
//	if (  m_exception.reportClass( DLoc, BR ) ) RD->dump();

// Check the class methods (member methods).
	for (clang::CXXRecordDecl::method_iterator
		I = RD->method_begin(), E = RD->method_end(); I != E; ++I)  
	{      

			if ( I->hasBody() ){
				clang::Stmt *Body = I->getBody();
        			walker.Visit(Body);
        			walker.Execute();
        			}

//			if ( I->getNameAsString() == "produce" 
//				|| I->getNameAsString() == "beginRun" 
//				|| I->getNameAsString() == "endRun" 
//				|| I->getNameAsString() == "beginLuminosityBlock" 
//				|| I->getNameAsString() == "endLuminosityBlock" )
			{
				const clang::CXXMethodDecl *  MD = &(*I);
//				clang::CXXMethodDecl *  CD = (*I).getCorrespondingMethodInClass( RD ,true );
				clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( MD , SM );
//				if (  m_exception.reportClass( DLoc, BR ) ) MD->dump();
				std::string buf;
	    			llvm::raw_string_ostream os(buf);
  				os << "Method  " << (*I) << " Parent Class "<< *RD<<". \n";
				if (  m_exception.reportClass( DLoc, BR ) ) 
					BR.EmitBasicReport( MD , "Class Checker MethodDecl","ThreadSafety",os.str(), DLoc);
				for (clang::CXXMethodDecl::method_iterator
					J=I->begin_overridden_methods(), F=I->end_overridden_methods(); J !=F; J++)
					{
					const clang::CXXMethodDecl * MD = (*J);
					clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( MD, SM );
//					if ( m_exception.reportClass( DLoc, BR ) ) (*J)->dump();
					std::string buf;
	    				llvm::raw_string_ostream os(buf);
   					os << "Overridden Method  " << (**J) << " Method "<< (*I)<<". \n";
 					if ( m_exception.reportClass( DLoc, BR ) ) 
						BR.EmitBasicReport( *J, "Class Checker Overridden MethodDecl","ThreadSafety",os.str(), DLoc);
					}				   				
			}
    	}	/* end of methods loop */

  		for (clang::CXXRecordDecl::field_iterator
		       	I = RD->field_begin(), E = RD->field_end(); I != E; ++I)  {
			if ( I->hasBody() ){
				clang::Stmt *Body = I->getBody();
				Body->dump();
        			walker.Visit(Body);
        			walker.Execute();
        			}
			const clang::FieldDecl * FD = &(*I);
	    		std::string buf;
	    		llvm::raw_string_ostream os(buf);
  			os << "Declaration of Field  " << *FD << " in Class "<<*RD<<" .\n";
			clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( FD, SM );
//			if ( m_exception.reportClass( DLoc, BR ) ) 
//				BR.EmitBasicReport(FD, "Class Checker FieldDecl","ThreadSafety",os.str(), DLoc);
 		}  /* end of field loop */

	

};

}


