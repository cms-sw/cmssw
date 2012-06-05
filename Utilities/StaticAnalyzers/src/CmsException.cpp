//== StaticLocalChecker.cpp - Checks for non-const static locals --------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ] and Patrick Gartung
//
//===----------------------------------------------------------------------===//


#include "CmsException.h"

namespace clangcms {


CmsException::CmsException()
{
	/*m_exceptions.push_back( new llvm::Regex( "edm::InputSourcePluginFactory::PMaker.*" ));
	m_exceptions.push_back( new llvm::Regex( "edm::InputSourcePluginFactory*" ));*/
}


CmsException::~CmsException()
{
	/*for ( ExList::iterator it = m_exceptions.begin();
			it != m_exceptions.end();
			++ it)
	{
		delete (*it);
	}*/
}

bool CmsException::reportGeneral( clang::ento::PathDiagnosticLocation const& path,
				clang::ento::BugReporter & BR ) const
{
	  clang::SourceLocation SL = path.asLocation();
	  if ( SL.isMacroID() ) {return false;}	

          const SourceManager &SM = BR.getSourceManager();
	  PresumedLoc PL = SM.getPresumedLoc(SL);
	  llvm::StringRef FN = llvm::StringRef((PL.getFilename()));
	  size_t found = 0;
	  found += FN.count("xr.cc");
	  found += FN.count("xi.cc");
	  found += FN.count("LinkDef.cc");
	  found += FN.count("/external/");
	  found +=FN.count("/lcg/");
	  if ( found!=0 )  {return false;}

	  if (SM.isInSystemHeader(SL) || SM.isInExternCSystemHeader(SL)) {return false;}
 	return true;
}


bool CmsException::reportGlobalStatic( QualType const& t,
			clang::ento::PathDiagnosticLocation const& path,
			clang::ento::BugReporter & BR  ) const
{
	return reportGeneral ( path, BR );
}

bool CmsException::reportMutableMember( QualType const& t,
			clang::ento::PathDiagnosticLocation const& path,
			clang::ento::BugReporter & BR  ) const
{
	return reportGeneral ( path, BR );
}

bool CmsException::reportGlobalStaticForType( QualType const& t,
				clang::ento::PathDiagnosticLocation const& path,
				clang::ento::BugReporter & BR ) const
{/*	not used yet
	//std::string t.getAsString()
	for ( ExList::iterator it = m_exceptions.begin();
			it != m_exceptions.end();
			++ it)
	{
		std::string s = t.getAsString();
		StringRef str_r(s);
		if  ( (*it)->match( str_r ))
			return false;
	}
*/
	return reportGeneral ( path, BR );
}
}


