#include "EDMPluginDumper.h"

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
				if (const clang::CXXRecordDecl * D = I->getTemplateArgs().get(J).getAsType().getTypePtr()->getAsCXXRecordDecl()) {
				llvm::errs()<<"edmplugin type '"<<D->getQualifiedNameAsString()<<"'\n";}
				}
			} 		
		};
} //end class


}//end namespace


