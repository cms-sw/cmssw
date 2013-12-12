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
[[edm::thread_safe]] static boost::interprocess::interprocess_semaphore file_mutex(1);

void EDMPluginDumper::checkASTDecl(const clang::ClassTemplateDecl *TD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {
	const clang::SourceManager &SM = BR.getSourceManager();
	std::string tname = TD->getTemplatedDecl()->getQualifiedNameAsString();
	if ( tname == "edm::WorkerMaker" ) {
		for ( auto I = TD->spec_begin(), E = TD->spec_end(); I != E; ++I) 
			{
			for (unsigned J = 0, F = I->getTemplateArgs().size(); J!=F; ++J)
				{
				if (const clang::CXXRecordDecl * D = I->getTemplateArgs().get(J).getAsType().getTypePtr()->getAsCXXRecordDecl()) {
					const char * pPath = std::getenv("LOCALRT");
					std::string rname = D->getQualifiedNameAsString();
					std::string dname(""); 
					if ( pPath != NULL ) dname = std::string(pPath);
					std::string fname("/tmp/plugins.txt.unsorted");
					std::string tname = dname + fname;
					std::string ostring = "edmplugin type '"+ rname +"'\n";
					file_mutex.wait();
					std::fstream file(tname.c_str(),std::ios::in|std::ios::out|std::ios::app);
					std::string filecontents((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>() );
					if ( filecontents.find(ostring) == std::string::npos) {
						file<<ostring;
						file.close();
						file_mutex.post();
					} else {
						file.close();
						file_mutex.post();
					}
				}
				}
			} 		
		};
} //end class


}//end namespace


