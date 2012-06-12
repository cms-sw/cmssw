//== GlobalStaticChecker.cpp - Checks for non-const global statics --------------*- C++ -*--==//
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

#include "GlobalStaticChecker.h"

#include "CmsSupport.h"

namespace clangcms
{

void GlobalStaticChecker::checkASTDecl(const VarDecl *D,
                    AnalysisManager &Mgr,
                    BugReporter &BR) const
{

	QualType t =  D->getType();
	if ( (D->getStorageClass() == SC_Static) &&
			  !D->isStaticDataMember() &&
			  !D->isStaticLocal() &&
			  !support::isConst( t ) )
	{
	    PathDiagnosticLocation DLoc = PathDiagnosticLocation::createBegin(D, BR.getSourceManager());
    	    QualType t =  D->getType();

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
