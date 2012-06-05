//== ConstCastChecker.cpp - Checks for const_cast<> --------------*- C++ -*--==//
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


#include "ConstCastChecker.h"

namespace clangcms {

void ConstCastChecker::checkPreStmt(const CXXConstCastExpr *CE,
		CheckerContext &C) const
{
	if (ExplodedNode *errorNode = C.generateSink()) {
		if (!BT)
			BT.reset(
					new BugType("const_cast used",
							"ThreadSafety"));
		BugReport *R = new BugReport(*BT, "const_cast was used, this may result in thread-unsafe code.", errorNode);
		R->addRange(CE->getSourceRange());

		C.EmitReport(R);
	}

}



}


