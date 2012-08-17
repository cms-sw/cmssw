#include "ClassChecker.h"

namespace clangcms {

void ClassChecker::checkASTDecl(const clang::CXXRecordDecl *D,
                    clang::ento::AnalysisManager &Mgr,
                    clang::ento::BugReporter &BR) const
{
	if ( D->isRecord() )
	{
	    clang::ento::PathDiagnosticLocation DLoc =
	    clang::ento::PathDiagnosticLocation::createBegin(D, BR.getSourceManager());

	    if ( ! m_exception.reportClass( DLoc, BR ) )
		return; 

	    std::string buf;
	    llvm::raw_string_ostream os(buf);
	    os << "Declaration of Class  " << *D << " .";
	    BR.EmitBasicReport(D, "Class Checker","ThreadSafety",os.str(), DLoc);

  		for (clang::CXXRecordDecl::method_iterator
         	I = D->method_begin(), E = D->method_end(); I != E; ++I)  {
			if ( (*I).hasBody() ) {
	    		std::string buf;
	    		llvm::raw_string_ostream os(buf);

      			os << "Declaration of Method  " << *I << " in Class "<<*D<<" .";
			BR.EmitBasicReport( D, "Class Checker","ThreadSafety",os.str(), DLoc);
    			}
		}

  		for (clang::CXXRecordDecl::field_iterator
         	I = D->field_begin(), E = D->field_end(); I != E; ++I)  {
	    		std::string buf;
	    		llvm::raw_string_ostream os(buf);
      			os << "Declaration of Field  " << *I << " in Class "<<*D<<" .";
			BR.EmitBasicReport(D, "Class Checker","ThreadSafety",os.str(), DLoc);
    			}

	
  		return;
	}

}


}

