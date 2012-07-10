//== CmsException.cpp -                             --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ] and Patrick Gartung
//
//===----------------------------------------------------------------------===//


#include "CmsException.h"

namespace clangcms {

bool CmsException::reportGeneral( clang::ento::PathDiagnosticLocation const& path,
				clang::ento::BugReporter & BR ) const
{
// using external defines to allow to switch-off Cms-specific exceptions in standalone builds.
// both values will be defined via BuildFile.xml during a SCRAM build
#if defined(THREAD_CHECKS_USE_CMS_EXCEPTIONS) || defined(THREAD_CHECKS_NO_REPORT_SYSTEM)
	  clang::SourceLocation SL = path.asLocation();
	  if ( SL.isMacroID() ) {return false;}	

      const clang::SourceManager &SM = BR.getSourceManager();
	  clang::PresumedLoc PL = SM.getPresumedLoc(SL);
#endif

#ifdef THREAD_CHECKS_USE_CMS_EXCEPTIONS
	  llvm::StringRef FN = llvm::StringRef((PL.getFilename()));
	  size_t found = 0;
	  found += FN.count("xr.cc");
	  found += FN.count("xi.cc");
	  found += FN.count("LinkDef.cc");
	  found += FN.count("/external/");
	  found +=FN.count("/lcg/");
	  if ( found!=0 )  {return false;}
#endif

#ifdef THREAD_CHECKS_NO_REPORT_SYSTEM
	  if (SM.isInSystemHeader(SL) || SM.isInExternCSystemHeader(SL)) {return false;}
#endif 
 	  return true;

}


bool CmsException::reportGlobalStatic( clang::QualType const& t,
			clang::ento::PathDiagnosticLocation const& path,
			clang::ento::BugReporter & BR  ) const
{
	return reportGeneral ( path, BR );
}

bool CmsException::reportMutableMember( clang::QualType const& t,
			clang::ento::PathDiagnosticLocation const& path,
			clang::ento::BugReporter & BR  ) const
{
	return reportGeneral ( path, BR );
}

bool CmsException::reportGlobalStaticForType( clang::QualType const& t,
				clang::ento::PathDiagnosticLocation const& path,
				clang::ento::BugReporter & BR ) const
{
	return reportGeneral ( path, BR );
}
}


