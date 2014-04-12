#include "edmChecker.h"
using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {

void edmChecker::checkASTDecl(const clang::CXXRecordDecl *RD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const {

	const clang::SourceManager &SM = BR.getSourceManager();
	clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( RD, SM );
	if (  !m_exception.reportClass( DLoc, BR ) ) return;
// Check the class methods (member methods).
	for (clang::CXXRecordDecl::method_iterator
		I = RD->method_begin(), E = RD->method_end(); I != E; ++I)  
	{      
		if ( !llvm::isa<clang::CXXMethodDecl>((*I)) ) continue;
		clang::CXXMethodDecl * MD = llvm::cast<clang::CXXMethodDecl>((*I));
			if ( MD->getNameAsString() == "beginRun" 
				|| MD->getNameAsString() == "endRun" 
				|| MD->getNameAsString() == "beginLuminosityBlock" 
				|| MD->getNameAsString() == "endLuminosityBlock" )
			{
//				llvm::errs()<<MD->getQualifiedNameAsString()<<"\n";	
				for (auto J=RD->bases_begin(), F=RD->bases_end();J != F; ++J)
					{  
					std::string name = J->getType()->castAs<RecordType>()->getDecl()->getQualifiedNameAsString();
//					llvm::errs()<<RD->getQualifiedNameAsString()<<"\n";	
//					llvm::errs() << "inherits from " <<name<<"\n";
					if (name=="edm::EDProducer" || name=="edm::EDFilter")
						{
						llvm::SmallString<100> buf;
						llvm::raw_svector_ostream os(buf);
						os << RD->getQualifiedNameAsString() << " inherits from edm::EDProducer or edm::EDFilter";
						os << "\n";
						llvm::errs()<<os.str();

						CXXMethodDecl::param_iterator I = MD->param_begin();
						ParmVarDecl * PVD = *(I);
						QualType PQT = PVD->getType();
						if ( PQT->isReferenceType() ) {
							QualType RQT = PQT->getPointeeType();
							if (RQT.isConstQualified()) continue;
						}
						clang::ento::PathDiagnosticLocation ELoc =clang::ento::PathDiagnosticLocation::createBegin( MD, SM );
						clang::SourceLocation SL = MD->getLocStart();
						BR.EmitBasicReport(MD, "Class Checker : inherits from edm::EDProducer or edm::EDFilter","optional",os.str(),ELoc,SL);
						}
					}
			}
	}
} //end of class


}
