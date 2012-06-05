


#include "ConstCastChecker.h"
#include "ConstCastAwayChecker.h"
#include "GlobalStaticChecker.h"
#include "StaticLocalChecker.h"
#include "MutableMemberChecker.h"


#include "clang/StaticAnalyzer/Core/CheckerRegistry.h"

extern "C" 
void clang_registerCheckers ( CheckerRegistry &registry) { 

	registry.addChecker< clangcms::ConstCastAwayChecker>( "threadsafety.ConstCastAway",  "Checks for casts which remove const qualifier and might result in thread-unsafe code" );
	registry.addChecker< clangcms::ConstCastChecker>( "threadsafety.ConstCast", "Checks for casts which remove const qualifier and might result in thread-unsafe code" );
	registry.addChecker< clangcms::StaticLocalChecker>( "threadsafety.StaticLocal", "Checks for non-const method local statics which might not be thread-safe" );
	registry.addChecker< clangcms::MutableMemberChecker>( "threadsafety.MutableMember", "Checks for members with the mutable keyword which might not be thread-safe" );
	registry.addChecker< clangcms::GlobalStaticChecker>( "threadsafety.GlobalStatic", "Checks for global non-const statics which might not be thread-safe" );

}

extern "C"
const char clang_analyzerAPIVersionString[] = CLANG_ANALYZER_API_VERSION_STRING;




