//===--- CmsSupport.h - Provides support functions ------------*- C++ -*-===//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ] and Patrick Gartung
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CMS_SUPPORT_H
#define LLVM_CLANG_STATICANALYZER_CMS_SUPPORT_H

#include <llvm/Support/Regex.h>

#include <clang/AST/Type.h>

namespace clangcms {

namespace support {


// The three cases
//
// const int var;
// int const& var;
// int const* var;
//
// have to be handled slightly different. This function implements the functionality to check
// for const qualifier for all of them.
//
inline bool isConst( clang::QualType const& qt )
{
	if ( qt->isReferenceType() )
	{
		// remove only the surounding reference type
		return qt.getNonReferenceType().isConstQualified();
	}
	if ( qt->isPointerType() )
	{
		clang::PointerType const* pt = qt->getAs<clang::PointerType>();
		return pt->getPointeeType().isConstQualified();
	}

	// regular type
	return qt.isConstQualified();
}

bool isCmsLocalFile(const char* file);

}
} 

#endif
