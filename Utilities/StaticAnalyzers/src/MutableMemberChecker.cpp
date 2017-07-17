//== MutableMemberChecker.cpp - Checks for mutable members --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#include "MutableMemberChecker.h"
#include <clang/AST/Attr.h>
using namespace clang;
using namespace ento;
using namespace llvm;
namespace clangcms {

void MutableMemberChecker::checkASTDecl(const clang::FieldDecl *D,
                    clang::ento::AnalysisManager &Mgr,
                    clang::ento::BugReporter &BR) const
{
        if ( D->hasAttr<CMSThreadGuardAttr>() || D->hasAttr<CMSThreadSafeAttr>()) return;
	if ( D->isMutable() &&
			 D->getDeclContext()->isRecord() )
	{
	    clang::QualType t =  D->getType();
	    clang::ento::PathDiagnosticLocation DLoc =
	    clang::ento::PathDiagnosticLocation::createBegin(D, BR.getSourceManager());

	    if ( ! m_exception.reportMutableMember( t, DLoc, BR ) )
		return;
	    if ( support::isSafeClassName( t.getCanonicalType().getAsString() ) ) return;
	    if ( ! support::isDataClass( D->getParent()->getQualifiedNameAsString() ) ) return;
	    std::string buf;
	    llvm::raw_string_ostream os(buf);
	    os << "Mutable member'" <<t.getCanonicalType().getAsString()<<" "<<*D << "' in class '"<<D->getParent()->getQualifiedNameAsString()<<"', might be thread-unsafe when accessing via a const handle.";
	    BR.EmitBasicReport(D, this, "mutable member",
	    					"ThreadSafety", os.str(), DLoc);
	    return;
	}

}


}

