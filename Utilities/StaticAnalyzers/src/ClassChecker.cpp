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
	if ( D->getNameAsString() == "produce" 
		|| D->getNameAsString() == "beginRun" 
		|| D->getNameAsString() == "endRun" 
		|| D->getNameAsString() == "beginLuminosityBlock" 
		|| D->getNameAsString() == "endLuminosityBlock" )
	{	    

	    clang::ento::PathDiagnosticLocation PLoc =
    	    clang::ento::PathDiagnosticLocation::createBegin(D, BR.getSourceManager());
	    if ( ! m_exception.reportGeneral( PLoc, BR ) )
		return; 
	    const clang::CXXRecordDecl* P= D->getParent(); 
	    if (D->hasInlineBody()) return;	
	    std::string buf;
	    llvm::raw_string_ostream os(buf);
	    os << "Declaration of Method  ";
	    D->printName(os);
	    os <<" in Class " << *P <<" .\n";
//    	    llvm::outs()<<os.str();
	    BR.EmitBasicReport(D, "Class Checker CXXMethodDecl","ThreadSafety",os.str(), PLoc);
		if (D->hasBody()){
			clang::Stmt* S = D->getBody();
//			S->dump();
			for (clang::Stmt::const_child_iterator
				c = S->child_begin(), e=S->child_end();c !=e; ++c) {
				if ( llvm::isa<clang::CXXMemberCallExpr>(*c) ) {
//					c->dump();
					const clang::CXXMemberCallExpr * ce = llvm::cast<clang::CXXMemberCallExpr>(*c);
//					ce->dump();
					clang::CXXMethodDecl *D = ce->getMethodDecl();
					const clang::CXXRecordDecl* P = ce->getRecordDecl();
					clang::ento::PathDiagnosticLocation DLoc = clang::ento::PathDiagnosticLocation::createBegin(ce->getDirectCallee(), BR.getSourceManager());
					DLoc.asLocation().dump();
     					std::string buf;
	    				llvm::raw_string_ostream os(buf);
					os<< "CXXMemberCallExpr\n";
					os<<"MethodDecl "<<*D <<" RecordDecl " << *P << " .\n";
//					llvm::outs()<<os.str();
					if ( ! m_exception.reportClass( DLoc, BR ) )
					return; 
					BR.EmitBasicReport(ce->getCalleeDecl(), "Class Checker CXXMemberCallExpr in CXXMethodDecl","ThreadSafety",os.str(), DLoc);
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
//		D->dump();
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
//				DLoc.asLocation().dump();
//				std::string buf;
//	    			llvm::raw_string_ostream os(buf);
//    				os << "Method  " << (*I) << " Parent Class "<< *D<<". \n";
//	    			llvm::outs()<<os.str();
// 				if ( ! m_exception.reportClass( DLoc, BR ) )
//					return; 
//				BR.EmitBasicReport( MD, "Class Checker CXXMethodDecl in CXXRecordDecl","ThreadSafety",os.str(), DLoc);

//			for (clang::CXXMethodDecl::method_iterator
//				J=I->begin_overridden_methods(), F=I->end_overridden_methods(); J !=F; J++)
//				{
//				
//				const clang::CXXMethodDecl * MD = (*J);
//				MD->dump();
//				clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( MD, SM );
//				std::string buf;
//	    			llvm::raw_string_ostream os(buf);
//     				os << "Overridden Method  " << (**J) << " Method "<< (*I)<<". \n";
//	    			llvm::outs()<<os.str();
// 				if ( ! m_exception.reportClass( DLoc, BR ) )
//					return; 
//				DLoc.asLocation().dump();
//				BR.EmitBasicReport( *J, "Class Checker Overridden CXXMethodDecl in CXXRecordDecl","ThreadSafety",os.str(), DLoc);
//				}				


    				
//			}			
//		} /* end method loop */


//  		for (clang::CXXRecordDecl::field_iterator
//       	I = D->field_begin(), E = D->field_end(); I != E; ++I)  {
//	    		std::string buf;
//	    		llvm::raw_string_ostream os(buf);
//     			os << "Declaration of Field  " << *I << " in Class "<<*D<<" .\n";
//	    		llvm::outs()<<os.str();
//			BR.EmitBasicReport(D, "Class Field Checker","ThreadSafety",os.str(), DLoc);
//  		}  /* end of field loop */

} /* end class */



void ClassCheckerCall::checkPostStmt(const clang::CXXMemberCallExpr *CE,
		clang::ento::CheckerContext &C) const 
{	

//	clang::CXXMethodDecl *D = CE->getMethodDecl();
//	const clang::CXXRecordDecl* P = CE->getRecordDecl();
//	if ( CE->getNumArgs() == 0) return;	
//	if (clang::ento::ExplodedNode *errorNode = C.addTransition()) {
//		if (!BT)
//			BT.reset(new clang::ento::BugType("Class Call Checker", "ThreadSafety"));
//	    	std::string buf;
//	    	llvm::raw_string_ostream os(buf);
//	    	os << "CXXMemberCallExpression "<<*P<<" ";
//	    	D->printName(os);
//		os<<".\n";
//		clang::ento::BugReport *R = new clang::ento::BugReport(*BT, os.str(), errorNode);
//		R->addRange(CE->getSourceRange());
//	   	if ( ! m_exception.reportConstCast( *R, C ) )
//			return;
///	    	llvm::outs()<<os.str();
//		C.EmitReport(R);
//	}

}



} /* end namespace */

