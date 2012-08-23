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

void ClassCheckerDecl::checkASTDecl(const clang::CXXRecordDecl *D,
                    clang::ento::AnalysisManager &Mgr,
                    clang::ento::BugReporter &BR) const
{
	return; /* Don't do anything since this is the wrong approach */
	if ( D->hasDefinition() )
	{	    

	    clang::CXXRecordDecl *CD = D->getDefinition();
	    clang::ento::PathDiagnosticLocation PLoc =
    	    clang::ento::PathDiagnosticLocation::createBegin(CD, BR.getSourceManager());

	    if ( ! m_exception.reportGeneral( PLoc, BR ) )
		return; 
	    std::string buf;
	    llvm::raw_string_ostream os(buf);
	    os << "Declaration of Class  " << *CD << " .";
//	    BR.EmitBasicReport(CD, "Class Method Checker","ThreadSafety",os.str(), DLoc);

  		for (clang::CXXRecordDecl::method_iterator
         	I = CD->method_begin(), E = CD->method_end(); I != E; ++I)  {
			if ( I->isThisDeclarationADefinition() ) {
				std::string buf;
	    			llvm::raw_string_ostream os(buf);
      				os << "Declaration of Method  " << *I << " in Class "<< *CD<<".";
//				BR.EmitBasicReport( CD, "Class Method Checker","ThreadSafety",os.str(), DLoc);

// 		This loop keeps causing EOF crashes in clang
//
//				clang::Stmt * S = I->getBody();
//				for (clang::Stmt::const_child_iterator
//					c = S->child_begin(), e=S->child_end();c !=e; ++c) {						
//						if ( llvm::isa<clang::CallExpr>(*c) ) {
//							const clang::CallExpr * ce = llvm::cast<clang::CallExpr>(*c);
//							clang::ento::PathDiagnosticLocation DLoc =
//   	    						clang::ento::PathDiagnosticLocation::createBegin
//							(ce->getCalleeDecl(), BR.getSourceManager());
// 							if ( ! m_exception.reportClass( DLoc, BR ) )
//								return; 
//     							std::string buf;
//	    						llvm::raw_string_ostream os(buf);
//							os << "Declaration of CallExpr in Method" << *CD << " .";
//							BR.EmitBasicReport(D, "Class Checker","ThreadSafety",os.str(), DLoc);
//						}
//				} 

    			} /* end declaration is definition */
		} /* end method loop */


  		for (clang::CXXRecordDecl::field_iterator
         	I = D->field_begin(), E = D->field_end(); I != E; ++I)  {
	    		std::string buf;
	    		llvm::raw_string_ostream os(buf);
      			os << "Declaration of Field  " << *I << " in Class "<<*D<<" .";
//			BR.EmitBasicReport(D, "Class Field Checker","ThreadSafety",os.str(), DLoc);
    			} 

	} /* has definition */
} /* end class */



void ClassCheckerCall::checkPostStmt(const clang::CXXMemberCallExpr *CE,
		clang::ento::CheckerContext &C) const 
{

	if (clang::ento::ExplodedNode *errorNode = C.generateSink()) {
		if (!BT)
			BT.reset(new clang::ento::BugType("Class Call Checker", "ThreadSafety"));
		clang::ento::BugReport *R = new clang::ento::BugReport(*BT, "CXXMemberCallExpression", errorNode);
		R->addRange(CE->getSourceRange());
	   	if ( ! m_exception.reportConstCast( *R, C ) )
			return;
		C.EmitReport(R);
	}

}



} /* end namespace */

