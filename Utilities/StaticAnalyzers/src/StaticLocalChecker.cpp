//== StaticLocalChecker.cpp - Checks for non-const static locals --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#include "StaticLocalChecker.h"

#include "CmsSupport.h"
#include <iostream>

namespace clangcms {


void StaticLocalChecker::checkASTDecl(const clang::VarDecl *D,
                    clang::ento::AnalysisManager &Mgr,
                    clang::ento::BugReporter &BR) const
{
	clang::QualType t =  D->getType();

	if ( D->isStaticLocal() && ! support::isConst( t ) )
	{
	    clang::ento::PathDiagnosticLocation DLoc = clang::ento::PathDiagnosticLocation::createBegin(D, BR.getSourceManager());

	    if ( ! m_exception.reportGlobalStaticForType( t, DLoc, BR ) )
			return;

	    std::string buf;
	    llvm::raw_string_ostream os(buf);
	    os << "Non-const variable '" << *D << "' is local static and might be thread-unsafe";

	    BR.EmitBasicReport(D, "Possibly Thread-Unsafe: non-const static local variable",
	    					"ThreadSafety",
	                       os.str(), DLoc);
	    return;
	}
}



}

