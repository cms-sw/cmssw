//==                                                                     --==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ] 
//
//===----------------------------------------------------------------------===//



#include "ConstCastChecker.h"
#include "ConstCastAwayChecker.h"
#include "GlobalStaticChecker.h"
#include "StaticLocalChecker.h"
#include "MutableMemberChecker.h"
#include "ClassChecker.h"
#include "ClassDumper.h"
#include "edmChecker.h"
#include "getByChecker.h"
#include "FiniteMathChecker.h"
#include "CatchAll.h"
#include "UsingNamespace.h"
#include "ArgSizeChecker.h"

#include <clang/StaticAnalyzer/Core/CheckerRegistry.h>

// register all custom checkers with clang
// add new entries here if you want to create a new checker
extern "C" 
void clang_registerCheckers ( clang::ento::CheckerRegistry &registry) 
{ 

	registry.addChecker< clangcms::ConstCastAwayChecker>( "threadsafety.ConstCastAway",  "Checks for casts which remove const qualifier and might result in thread-unsafe code" );
	registry.addChecker< clangcms::ConstCastChecker>( "threadsafety.ConstCast", "Checks for casts which remove const qualifier and might result in thread-unsafe code" );
	registry.addChecker< clangcms::StaticLocalChecker>( "threadsafety.StaticLocal", "Checks for non-const method local statics which might not be thread-safe" );
	registry.addChecker< clangcms::MutableMemberChecker>( "threadsafety.MutableMember", "Checks for members with the mutable keyword which might not be thread-safe" );
	registry.addChecker< clangcms::GlobalStaticChecker>( "threadsafety.GlobalStatic", "Checks for global non-const statics which might not be thread-safe" );
	registry.addChecker< clangcms::ClassChecker>( "optional.ClassChecker", "Checks classes of interest" );
	registry.addChecker< clangcms::ClassDumperCT>( "optional.ClassDumperCT", "Dumps class info" );
	registry.addChecker< clangcms::ClassDumperFT>( "optional.ClassDumperFT", "Dumps class info" );
	registry.addChecker< clangcms::ClassDumperInherit>( "optional.ClassDumperInherit", "Dumps class inheritance info" );
	registry.addChecker< clangcms::FiniteMathChecker>( "cms.NonFiniteMath", "Reports usage of isnan and isinf." );
	registry.addChecker< clangcms::UsingNamespace>( "cms.CodeRules.UsingNamespace", "Checks for 'using namespace' or 'using std::' in header files" );
	registry.addChecker< clangcms::CatchAll>( "cms.CodeRules.CatchAll", "Checks for 'catch(...)' in source files" );
	registry.addChecker< clangcms::edmChecker>( "optional.edmChecker", "Checks classes inheriting from edm::Producer and edm::Filter" );
	registry.addChecker< clangcms::getByChecker>( "optional.getByChecker", "Checks for calls to edm::getByLabel or edm::getManyByType and reports edm::Handle type passed" );
	registry.addChecker< clangcms::ArgSizeChecker>( "cms.ArgSize", "Reports args passed by value with size>4k." );
}

extern "C"
const char clang_analyzerAPIVersionString[] = CLANG_ANALYZER_API_VERSION_STRING;




