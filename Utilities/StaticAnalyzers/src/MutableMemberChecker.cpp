//== MutableMemberChecker.cpp - Checks for mutable members --------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#include "MutableMemberChecker.h"

namespace clangcms {

void MutableMemberChecker::checkASTDecl(const FieldDecl *D,
                    AnalysisManager &Mgr,
                    BugReporter &BR) const
{
	if ( D->isMutable() &&
			// I *think* this means it is member of a class ...
			 D->getDeclContext()->isRecord() )
	{
	    QualType t =  D->getType();
	    PathDiagnosticLocation DLoc =
	    PathDiagnosticLocation::createBegin(D, BR.getSourceManager());

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

