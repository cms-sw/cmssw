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

	if ( ( (D->isStaticLocal() || D->isStaticDataMember() )  && D->getTSCSpec() != clang::ThreadStorageClassSpecifier::TSCS_thread_local) && ! support::isConst( t ) )
	{
	    clang::ento::PathDiagnosticLocation DLoc = clang::ento::PathDiagnosticLocation::createBegin(D, BR.getSourceManager());

	    if ( ! m_exception.reportGlobalStaticForType( t, DLoc, BR ) )
			return;
	    std::string vname = D->getCanonicalDecl()->getQualifiedNameAsString();
	    unsigned found = vname.find_last_of("::");
	    std::string cname = vname.substr(0,found);
//	    if ( ! support::isDataClass( cname) ) return;
	    if ( support::isSafeClassName( vname ) ) return;

	    std::string buf;
	    llvm::raw_string_ostream os(buf);
	    os << "Non-const variable '" <<t.getAsString()<<" "<< *D << "' is static local or static member data and might be thread-unsafe";

	    BR.EmitBasicReport(D, "Possibly Thread-Unsafe: non-const static variable",
	    					"ThreadSafety",
	                       os.str(), DLoc);
	    return;
	}
}



}

