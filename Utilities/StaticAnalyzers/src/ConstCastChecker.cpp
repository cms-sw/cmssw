//== ConstCastChecker.cpp - Checks for const_cast<> --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//


#include "ConstCastChecker.h"

namespace clangcms {

void ConstCastChecker::checkPreStmt(const clang::CXXConstCastExpr *CE,
		clang::ento::CheckerContext &C) const
{
	if (clang::ento::ExplodedNode *errorNode = C.generateSink()) {
		if (!BT)
			BT.reset(
					new clang::ento::BugType("const_cast used",
							"ThreadSafety"));
		clang::ento::BugReport *R = new clang::ento::BugReport(*BT, "const_cast was used, this may result in thread-unsafe code.", errorNode);
		R->addRange(CE->getSourceRange());
	   	if ( ! m_exception.reportConstCast( *R, C ) )
			return;
		C.emitReport(R);
	}

}



}


