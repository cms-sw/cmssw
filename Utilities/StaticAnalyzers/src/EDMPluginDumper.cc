#include "EDMPluginDumper.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm> 

using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {

void EDMPluginDumper::checkASTDecl(const clang::ClassTemplateDecl *TD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {

	const clang::SourceManager &SM = BR.getSourceManager();
	std::string tname = TD->getTemplatedDecl()->getQualifiedNameAsString();
	if ( tname == "edm::WorkerMaker" ) {
		for ( auto I = TD->spec_begin(), E = TD->spec_end(); I != E; ++I) {
			for (unsigned J = 0, F = I->getTemplateArgs().size(); J!=F; ++J)  {
				llvm::SmallString<100> buf;
				llvm::raw_svector_ostream os(buf);
				I->getTemplateArgs().get(J).print(mgr.getASTContext().getPrintingPolicy(),os);
				std::string rname = os.str();
				const char * pPath = std::getenv("LOCALRT");
				std::string dname("");
				if ( pPath != NULL ) dname = std::string(pPath);
				std::string fname("/tmp/plugins.txt.unsorted");
				std::string tname = dname + fname;
				std::string ostring = rname +"\n";
				std::ofstream file(tname.c_str(),std::ios::app);
				file<<ostring;
			}
		}
	} 		

} //end class


}//end namespace


