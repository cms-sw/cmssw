//== ConstCastAwayChecker.cpp - Checks for removed const qualfiers --------------*- C++ -*--==//
//
// Check in a generic way if an explicit cast removes a const qualifier.
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#include "ConstCastAwayChecker.h"
#include "CmsSupport.h" 

namespace clangcms
{


void ConstCastAwayChecker::checkPreStmt(const clang::ExplicitCastExpr *CE,
		clang::ento::CheckerContext &C) const 
{
	const clang::Expr *E = CE->getSubExpr();
	clang::ASTContext &Ctx = C.getASTContext();
	clang::QualType OrigTy = Ctx.getCanonicalType(E->getType());
	clang::QualType ToTy = Ctx.getCanonicalType(CE->getType());

	if ( support::isConst( OrigTy ) && ! support::isConst(ToTy) ) {
		if ( clang::ento::ExplodedNode *errorNode = C.generateSink()) {
			if (!BT)
				BT.reset(
						new clang::ento::BugType("const cast away",
								"ThreadSafety"));
			clang::ento::BugReport *R = new clang::ento::BugReport(*BT, 
					"const qualifier was removed via a cast, this may result in thread-unsafe code.", errorNode);
			R->addRange(CE->getSourceRange());
		   	if ( ! m_exception.reportConstCastAway( *R, C ) )
				return;
			C.emitReport(R);
		}
	}
}

}
