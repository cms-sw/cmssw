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
		for ( auto I = TD->spec_begin(), E = TD->spec_end(); I != E; ++I) 
			{
			for (unsigned J = 0, F = I->getTemplateArgs().size(); J!=F; ++J)
				{
				if (const clang::CXXRecordDecl * RD = I->getTemplateArgs().get(J).getAsType().getTypePtr()->getAsCXXRecordDecl()) {
					const char * pPath = std::getenv("LOCALRT");
					std::string rname = RD->getQualifiedNameAsString();
					std::string dname(""); 
					if ( pPath != NULL ) dname = std::string(pPath);
					std::string fname("/tmp/plugins.txt.unsorted");
					std::string tname = dname + fname;
					std::string ostring = "edmplugin type '"+ rname +"'\n";
					std::ofstream file(tname.c_str(),std::ios::app);
					file<<ostring;
					if (const ClassTemplateSpecializationDecl *SD = dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
						for (unsigned J = 0, F = SD->getTemplateArgs().size(); J!=F; ++J) {
							if (SD->getTemplateArgs().get(J).getKind() == clang::TemplateArgument::Type && SD->getTemplateArgs().get(J).getAsType().getTypePtr()->isRecordType() ) 
							{
							const clang::CXXRecordDecl * D = SD->getTemplateArgs().get(J).getAsType().getTypePtr()->getAsCXXRecordDecl();
							std::string dname = D->getQualifiedNameAsString();
							std::string ostring = "edmplugin type '"+rname+"' template arg '"+ dname +"'\n";
							std::ofstream file(tname.c_str(),std::ios::app);
							file<<ostring;
							
							}
						}
					}	 
				}
			}
		} 		
	}
} //end class


}//end namespace


