#include "edmChecker.h"
using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {

void edmChecker::checkASTDecl(const clang::CXXRecordDecl *RD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const {
	if (!RD->hasDefinition()) return;
	const clang::SourceManager &SM = BR.getSourceManager();
	for ( auto J=RD->bases_begin(), F=RD->bases_end();J != F; ++J) {
		auto BRD = J->getType()->getAsCXXRecordDecl();
		if (!BRD) continue;
		std::string bname = BRD->getQualifiedNameAsString();
		if (bname=="edm::EDProducer" || bname=="edm::EDFilter" || bname=="edm::EDAnalyzer" || bname=="edm::OutputModule" ) {
			llvm::SmallString<100> buf;
			llvm::raw_svector_ostream os(buf);
			os << RD->getQualifiedNameAsString() << " inherits from edm::EDProducer,edm::EDFilter,edm::EDAnalyzer, or edm::OutputModule";
			os << "\n";
			clang::ento::PathDiagnosticLocation ELoc =clang::ento::PathDiagnosticLocation::createBegin( RD, SM );
			BR.EmitBasicReport(RD, this, "inherits from edm::EDProducer,edm::EDFilter,edm::EDAnalyzer, or edm::OutputModule","ThreadSafety",os.str(),ELoc);
		}
	}
} //end of class

}
