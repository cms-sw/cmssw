//===--- CmsException.h - exceptions for bug reports ------------*- C++ -*-===//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ] and Patrick Gartung
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CMS_CMSEXCEPTION_H
#define LLVM_CLANG_STATICANALYZER_CMS_CMSEXCEPTION_H

#include <llvm/Support/Regex.h>

#include <clang/AST/Type.h>
#include <clang/Basic/SourceManager.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugReporter.h>

namespace clangcms {

class CmsException {
public:
	bool reportGlobalStaticForType( clang::QualType const& t,
				clang::ento::PathDiagnosticLocation const& path,
				clang::ento::BugReporter & BR  ) const;

	bool reportGlobalStatic( clang::QualType const& t,
				clang::ento::PathDiagnosticLocation const& path,
				clang::ento::BugReporter & BR  ) const;	

	bool reportMutableMember( clang::QualType const& t,
				clang::ento::PathDiagnosticLocation const& path,
				clang::ento::BugReporter & BR  ) const;	

	bool reportGeneral( clang::ento::PathDiagnosticLocation const& path, 
				clang::ento::BugReporter & BR ) const; 
private:
	/*typedef std::vector< llvm::Regex *> ExList;
	mutable ExList m_exceptions;*/
};

} 

#endif
