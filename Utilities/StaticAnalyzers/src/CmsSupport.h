//===--- CmsSupport.h - Provides support functions ------------*- C++ -*-===//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ] and Patrick Gartung
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CMS_SUPPORT_H
#define LLVM_CLANG_STATICANALYZER_CMS_SUPPORT_H

#include <clang/AST/Type.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <string>

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
std::string getQualifiedName(const clang::NamedDecl &d);
bool isSafeClassName(const std::string &d);
bool isDataClass(const std::string &d);
bool isInterestingLocation(const std::string &d);
bool isKnownThrUnsafeFunc(const std::string &name );
void writeLog(const std::string &ostring,const std::string &tfstring); 
}
} 

#endif
