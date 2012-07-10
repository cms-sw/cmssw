//== MutableMemberChecker.cpp - Checks for mutable members --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#include "MutableMemberChecker.h"

namespace clangcms {

void MutableMemberChecker::checkASTDecl(const clang::FieldDecl *D,
                    clang::ento::AnalysisManager &Mgr,
                    clang::ento::BugReporter &BR) const
{
	if ( D->isMutable() &&
			 D->getDeclContext()->isRecord() )
	{
	    clang::QualType t =  D->getType();
	    clang::ento::PathDiagnosticLocation DLoc =
	    clang::ento::PathDiagnosticLocation::createBegin(D, BR.getSourceManager());

	    if ( ! m_exception.reportMutableMember( t, DLoc, BR ) )
		return;

	    std::string buf;
	    llvm::raw_string_ostream os(buf);
	    os << "Mutable member'" << *D << "' in class, might be thread-unsafe when accessing via a const handle.";

	    BR.EmitBasicReport(D, "Possibly Thread-Unsafe: Mutable member",
	    					"ThreadSafety",
	                       os.str(), DLoc);
	    return;
	}

}


}

