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

using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {

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
  //void VisitMemberExpr(clang::Expr *E);
  void VisitCallExpr(clang::CallExpr *CE);
  void VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *CE);
  void VisitStmt(clang::Stmt *S) { VisitChildren(S); }
  void VisitChildren(clang::Stmt *S);
  
  void ReportCall(const clang::CallExpr *CE);
  void ReportCallArg(const clang::CallExpr *CE, const clang::Expr *E);

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

//void WalkAST::VisitMemberExpr(clang::Expr *E) {

//const clang::CXXRecordDecl * ACD = llvm::dyn_cast<clang::CXXRecordDecl>(AC->getDecl());
//clang::QualType qual = llvm::dyn_cast<clang::Expr>(E)->getType();
//clang::MemberExpr * ME = llvm::dyn_cast<clang::MemberExpr>(E);
//clang::ValueDecl * VD = ME->getMemberDecl();
//if (!(ME->isBoundMemberFunction(AC->getASTContext()))&&
//	ME->isImplicitAccess() &&
//	!(support::isConst(qual)) )
//{

//clang::FieldDecl * MD = llvm::dyn_cast<clang::FieldDecl>(VD);
//clang::CXXRecordDecl * MRD = llvm::dyn_cast<clang::CXXRecordDecl>(MD->getParent());
//if (MRD==ACD) 
//	{
//	llvm::errs()<<"\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
//	llvm::errs()<<"\n CXXMemberExpr \n";
//	ME->dumpPretty(AC->getASTContext());
//	llvm::errs()<<"\n\n";           
//	qual.dump();
//	llvm::errs()<<"\n\n";
//	VD->dump();
//	llvm::errs()<<"\n\n";
//	VD->getType().dump();
//	llvm::errs()<<"\n\n";
//	MD->dump();
//	llvm::errs()<<"\n\n";
//	llvm::errs()<<"\n Record Decl name\n";
//	MRD->printName(llvm::errs());
//	llvm::errs()<<"\n";
//	ACD->printName(llvm::errs());
//	llvm::errs()<<"\n";
//
//	llvm::errs()<<"\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
//	}
//}
//VisitChildren(E);  
//}

void WalkAST::VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *CE) {



// check for non const reference or pointer passed as args
clang::Expr * IOA = CE->getImplicitObjectArgument();
clang::QualType qual_ioa = llvm::dyn_cast<clang::Expr>(IOA)->getType();
if (!support::isConst(qual_ioa) )
if ( 	!qual_ioa.getTypePtr()->isScalarType())
if (	!qual_ioa.getTypePtr()->isObjectType() )
{
ReportCall(CE);
}


//clang::MemberExpr * ME = clang::dyn_cast<clang::MemberExpr>(CE->getCallee());
//clang::CXXRecordDecl * RD = llvm::dyn_cast<clang::CXXRecordDecl>(CE->getRecordDecl());
//clang::CXXMethodDecl * MD = CE->getMethodDecl();
//clang::CXXRecordDecl * MRD = llvm::dyn_cast<clang::CXXRecordDecl>(MD->getParent());
//clang::QualType MQT = MD->getThisType(AC->getASTContext());
//const clang::CXXRecordDecl * ACD = llvm::dyn_cast<clang::CXXRecordDecl>(AC->getDecl());
//clang::QualType qual_exp = llvm::dyn_cast<clang::Expr>(ME)->getType();

//llvm::errs()<<"\n-------------------------------------------------------------------------------------\n";
//		llvm::errs()<<"\n CXXMemberCallExpr \n";
//		CE->dumpPretty(AC->getASTContext());
//		llvm::errs()<<"\n";
//		llvm::errs()<<"\n Qual Type CallExpr\n";
//		qual_exp->dump();
//		llvm::errs()<<"\n";
//		llvm::errs()<<"\n Record Decl name\n";
//		RD->printName(llvm::errs());
//		llvm::errs()<<"\n";
//		ACD->printName(llvm::errs());
//		llvm::errs()<<"\n";
//		llvm::errs()<<"\n Method Decl name\n";
//		MD->printName(llvm::errs());
//		llvm::errs()<<"\n";
//		llvm::errs()<<"\n Qual Type MD This\n";
//		MQT.dump();
//		llvm::errs()<<"\n";
//		llvm::errs()<<"\n MD Parent Decl name\n";
//		MRD->printName(llvm::errs())e
//		llvm::errs()<<"\n";
//		llvm::errs()<<"\n CE Parent Decl name\n";
//		RD->printName(llvm::errs());
//		llvm::errs()<<"\n";
//		llvm::errs()<<"\n CE Implicit Object Argument\n";
//		IOA->dumpPretty(AC->getASTContext());
//		llvm::errs()<<"\n";
//		llvm::errs()<<"\n Qual Type CallExpr IOA\n";
//		qual_ioa->dump();
//		llvm::errs()<<"\n";
//		llvm::errs()<<"\n Member Expr \n";
//		ME->dumpPretty(AC->getASTContext());
//		llvm::errs()<<"\n";
		


// 	for (clang::Stmt::child_range SubStmts = CE->children(); SubStmts; ++SubStmts){
//		if (const clang::Stmt *S = *SubStmts) 
//			{ 
//					clang::QualType qual_sub = 
//						llvm::dyn_cast<clang::Expr>(S)->getType();
//					if (!(support::isConst(qual_sub))&&
//						!(llvm::dyn_cast<clang::Expr>(S)->isRValue()))
//						if(qual_arg.getTypePtr()->isPointerType()||
//							qual_arg.getTypePtr()->isReferenceType())
//						if(!(qual_sub.getTypePtr()->isScalarType())&&
//							!(qual_sub.getTypePtr()->isObjectType()))
//						{					
//							llvm::errs()<<"\n CXXMemberCallExpr \n";
//							CE->dumpPretty(AC->getASTContext());
//							llvm::errs()<<"\n";
//							llvm::errs()<<"\n////////////////////\n";
//							llvm::errs()<<"\n Sub  Expr \n";
//							S->dumpPretty(AC->getASTContext());
//							llvm::errs()<<"\n";
//							llvm::errs()<<"\nQual Type Sub\n";
//							qual_sub->dump();
//							llvm::errs()<<"\n";
//							llvm::errs()<<"\n Record Decl name\n";
//							RD->printName(llvm::errs());
//							llvm::errs()<<"\n";
//							ACD->printName(llvm::errs());
//							llvm::errs()<<"\n";
//							llvm::errs()<<"\n////////////////////\n";
//						}
//			}
//	}
	




// 	for(int i=0, j=CE->getNumArgs(); i<j; i++) {
//		if ( const clang::Expr *E = CE->getArg(i))
//			{
//			clang::QualType qual_arg = llvm::dyn_cast<clang::Expr>(E)->getType();
//
//			if (!(support::isConst(qual_arg)) &&
//				!(llvm::dyn_cast<clang::Expr>(E)->isRValue()))
//				if(qual_arg.getTypePtr()->isPointerType()||qual_arg.getTypePtr()->isReferenceType())
//				if(!(qual_arg.getTypePtr()->isScalarType())&&
//					!(qual_arg.getTypePtr()->isObjectType()))
//				{
//					llvm::errs()<<"\n CXXMemberCallExpr \n";
//					CE->dumpPretty(AC->getASTContext());
//					llvm::errs()<<"\n";
//					llvm::errs()<<"\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n";
//					llvm::errs()<<"\n Arg Expr \n";
//					CE->getArg(i)->dumpPretty(AC->getASTContext());
//					llvm::errs()<<"\n";
//					llvm::errs()<<"\nQual Type Arg\n";
//					qual_arg->dump();
//					llvm::errs()<<"\n";
//					llvm::errs()<<"\n Record Decl name\n";
//					RD->printName(llvm::errs());
//					llvm::errs()<<"\n";
//					ACD->printName(llvm::errs());
//					llvm::errs()<<"\n";
//					llvm::errs()<<"\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n";
//					ReportCallArg(CE,E);
//				}
//				 
//			}
//	} 
//llvm::errs()<<"\n-------------------------------------------------------------------------------------\n";

}

  VisitChildren(CE);
  Enqueue(CE);

}


void WalkAST::ReportCall(const clang::CallExpr *CE) {
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
     os << "\n";

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
  BR.EmitBasicReport(CE->getCalleeDecl(),"Class Checker : CallExpr Report","ThreadSafety",os.str(),CELoc,R);
	 
}

void WalkAST::ReportCallArg(const clang::CallExpr *CE,const clang::Expr *E) {
  llvm::SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);
  CmsException m_exception;

  os << "CXXMethod Call Expression : ";
  os << *CE->getDirectCallee();
  os << "\n";
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);
  std::string TypeS;
  llvm::raw_string_ostream s(TypeS);
  E->printPretty(s, 0, Policy);
  os << "arg: " << s.str() << " ";
  os << "is not const qualified and not RValue.";

  clang::ento::PathDiagnosticLocation ELoc =
    clang::ento::PathDiagnosticLocation::createBegin(E, BR.getSourceManager(),AC);
  clang::SourceLocation L = E->getExprLoc();

  if (!m_exception.reportClass( ELoc, BR ) ) return;
  BR.EmitBasicReport(CE->getCalleeDecl(),"Class Checker :  CallExpr Arg check","ThreadSafety",os.str(),ELoc,L);

}


void ClassCheckerRDecl::checkASTDecl(const clang::CXXRecordDecl *CRD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const {

	const clang::CXXRecordDecl *RD=CRD;
	clangcms::WalkAST walker(BR, mgr.getAnalysisDeclContext(RD));
	const clang::SourceManager &SM = BR.getSourceManager();
	clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( RD, SM );
	if (  !m_exception.reportClass( DLoc, BR ) ) return;
//	clang::LangOptions LangOpts;
//	LangOpts.CPlusPlus = true;
//	clang::PrintingPolicy Policy(LangOpts);
//	std::string TypeS;
//	llvm::raw_string_ostream s(TypeS);
//	RD->print(s,Policy,0,0);
//	llvm::errs() << s.str() <<"\n\n\n\n";
//	RD->dump();
 
bool first = true;
// Check the class methods (member methods).
	for (clang::CXXRecordDecl::method_iterator
		I = RD->method_begin(), E = RD->method_end(); I != E; ++I)  
	{      

//			if ( I->getNameAsString() == "produce" 
//				|| I->getNameAsString() == "beginRun" 
//				|| I->getNameAsString() == "endRun" 
//				|| I->getNameAsString() == "beginLuminosityBlock" 
//				|| I->getNameAsString() == "endLuminosityBlock" )
			{
				const clang::CXXMethodDecl *  MD = &llvm::cast<const clang::CXXMethodDecl>(*I->getMostRecentDecl());
				if (MD->isVirtualAsWritten()) continue;
				clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( MD , SM );
				clang::SourceRange R = MD->getSourceRange();
//				llvm::errs()<<"\n*****************************************************\n";
//				llvm::errs()<<"Visited CXXMethodDecl\n";
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
//       				llvm::raw_string_ostream s(TypeS);
//        				llvm::errs() << "\n\n+++++++++++++++++++++++++++++++++++++\n\n";
//	      				llvm::errs() << "\n\nPretty Print\n\n";
//	       				Body->printPretty(s, 0, Policy);
//        				llvm::errs() << s.str();
//        				llvm::errs() << "\n\nDump\n\n";
//					Body->dump();
//        				llvm::errs() << "\n\n+++++++++++++++++++++++++++++++++++++\n\n";
	       				walker.Visit(Body);
	       				walker.Execute();
       				}
// Check the constructors.
				   
			    if (first) {
				for (clang::CXXRecordDecl::ctor_iterator I = RD->ctor_begin(), E = RD->ctor_end();
         			I != E; ++I) {
        			if (clang::Stmt *Body = I->getBody()) {
//				llvm::errs()<<"Visited Constructors for\n";
//				llvm::errs()<<RD->getNameAsString();
          				walker.Visit(Body);
          				walker.Execute();
        				}
    				}
				first=false;
			    }	
			} /* end of Name check */
   	}	/* end of methods loop */


} //end of class


}


