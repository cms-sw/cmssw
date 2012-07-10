//== GlobalStaticChecker.cpp - Checks for non-const global statics --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#include "GlobalStaticChecker.h"

#include "CmsSupport.h"

namespace clangcms
{

void GlobalStaticChecker::checkASTDecl(const clang::VarDecl *D,
                    clang::ento::AnalysisManager &Mgr,
                    clang::ento::BugReporter &BR) const
{

	clang::QualType t =  D->getType();
	if ( (D->getStorageClass() == clang::SC_Static) &&
			  !D->isStaticDataMember() &&
			  !D->isStaticLocal() &&
			  !support::isConst( t ) )
	{
	    clang::ento::PathDiagnosticLocation DLoc = clang::ento::PathDiagnosticLocation::createBegin(D, BR.getSourceManager());
    	    clang::QualType t =  D->getType();

	    if ( ! m_exception.reportGlobalStaticForType( t, DLoc, BR ) )
		   return;

	    std::string buf;
	    llvm::raw_string_ostream os(buf);
	    os << "Non-const variable '" << *D << "' is static and might be thread-unsafe";

	    BR.EmitBasicReport(D, "Possibly Thread-Unsafe: non-const static variable",
	    					"ThreadSafety",
	                       os.str(), DLoc);
	    return;
	
	}

}

}
