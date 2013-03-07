#include "edmChecker.h"
using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {

void EDMChecker::checkASTDecl(const CXXRecordDecl *RD, AnalysisManager& mgr,
                    BugReporter &BR) const {
	return;
	const SourceManager &SM = BR.getSourceManager();
	PathDiagnosticLocation DLoc =PathDiagnosticLocation::createBegin( RD, SM );
//	if (  !m_exception.reportClass( DLoc, BR ) ) return;
// Check the class methods (member methods).
	for (CXXRecordDecl::method_iterator
		I = RD->method_begin(), E = RD->method_end(); I != E; ++I)  
	{      
		if ( !llvm::isa<CXXMethodDecl>((*I)) ) continue;
		CXXMethodDecl * MD = llvm::cast<CXXMethodDecl>((*I));
			if ( MD->getNameAsString() == "beginRun" 
				|| MD->getNameAsString() == "endRun" 
				|| MD->getNameAsString() == "beginLuminosityBlock" 
				|| MD->getNameAsString() == "endLuminosityBlock" )
			{
				llvm::errs()<<MD->getQualifiedNameAsString()<<"\n";	
				for (auto J=RD->bases_begin(), F=RD->bases_end();J != F; ++J)
					{  
					std::string name = J->getType()->castAs<RecordType>()->getDecl()->getQualifiedNameAsString();
					llvm::errs()<<RD->getQualifiedNameAsString()<<"\n";	
					llvm::errs() << "inherits from " <<name<<"\n";
					if (name=="edm::EDProducer" || name=="edm::EDFilter")
						{
						llvm::SmallString<100> buf;
						llvm::raw_svector_ostream os(buf);
						os << RD->getQualifiedNameAsString() << " inherits from edm::EDProducer or edm::EDFilter";
						os << "\n";
						llvm::errs()<<os.str();
						PathDiagnosticLocation ELoc =PathDiagnosticLocation::createBegin( MD, SM );
						SourceLocation SL = MD->getLocStart();
						BR.EmitBasicReport(MD, "Class Checker : inherits from edm::EDProducer or edm::EDFilter","optional",os.str(),ELoc,SL);
						}
					}
			}
	}
} //end of class


void EDMChecker::checkASTDecl(const CXXMethodDecl *MD, AnalysisManager& mgr,
                    BugReporter &BR) const {
	const SourceManager &SM = BR.getSourceManager();
	PathDiagnosticLocation DLoc =PathDiagnosticLocation::createBegin( MD, SM );
//	if (  !m_exception.reportClass( DLoc, BR ) ) return;
	std::string mname = MD->getQualifiedNameAsString();
	if (mname == "edm::EDProducer::produce" || 
		mname == "edm::EDFilter::filter" || 
		mname == "edm::EDAnalyzer::analyze" ||
		mname == "edm::Event::getByLabel" ||
		mname == "edm::PrincipalGetAdapter::getByLabel" ) return;
	for ( CXXMethodDecl::param_const_iterator I=MD->param_begin(), E=MD->param_end(); I != E; ++I) {
		QualType QT = (*I)->getOriginalType();
		const CXXRecordDecl * RD = QT->getPointeeCXXRecordDecl();
		if (RD) {
			std::string name = RD->getQualifiedNameAsString();
			if (name =="edm::Event" || name == "edm::Handle") {
				llvm::SmallString<100> buf;
				llvm::raw_svector_ostream os(buf);
					
				os<<"function "<<MD->getQualifiedNameAsString()<<"\t";
				os<<"parameter type "<<name<<"\t";
				std::string name = (*I)->getNameAsString();
				if (name.length() != 0) {
					os<<"parameter name ";
					os<<name;
					}
				llvm::errs()<<os.str()<<"\tfunction with edm::Event or edm::Handle parameter type\n\n";
				PathDiagnosticLocation ELoc =PathDiagnosticLocation::createBegin( MD, SM );
				SourceLocation SL = MD->getLocation();
				BR.EmitBasicReport(MD, "function with edm::Event or edm::Handle parameter type","optional",os.str(),ELoc,SL);

			}
		}
	}


}

void EDMChecker::checkPreStmt(const CXXMemberCallExpr *CE, CheckerContext &C) const {

	const Decl * D = C.getCurrentAnalysisDeclContext()->getDecl();
	CXXMethodDecl * MD = CE->getMethodDecl();
	if (!MD) return;
	std::string name = MD->getQualifiedNameAsString();
	LangOptions LangOpts;
	LangOpts.CPlusPlus = true;
	PrintingPolicy Policy(LangOpts);
	llvm::SmallString<100> buf;
	llvm::raw_svector_ostream os(buf);

	if ( name == "edm::Event::getByLabel" || name == "edm::Event::getManyByType" ) {
		std::string dname = llvm::dyn_cast<NamedDecl>(D)->getQualifiedNameAsString();
//			if (const CXXRecordDecl * RD = llvm::dyn_cast<CXXMethodDecl>(D)->getParent() ) {
//				llvm::errs()<<"class "<<RD->getQualifiedNameAsString()<<"\n";
//				llvm::errs()<<"\n";
//				}
			os<<"function "<<dname<<"\t";
			os<<"call expression ";
			CE->printPretty(os,0,Policy);
			os<<"\t";
			QualType QT;
			os<<"argument type ";
			if (name == "edm::Event::getByLabel") { QT = (CE->arg_begin()+1)->getType();}
			else {	QT = (CE->arg_begin())->getType();}
			os<<QT.getAsString();
			llvm::errs()<<os.str()<<"\tedm::getByLabel or edm::getManyByType called"<<"\n\n";
			ExplodedNode *errorNode = C.generateSink();
			BugType * BT = new BugType("edm::getByLabel or edm::getManyByType called","optional") ;
			BugReport * R = new BugReport(*BT,os.str(),errorNode);
			R->addRange(CE->getSourceRange());
			C.emitReport(R);

	} 
	else {
		std::string dname = llvm::dyn_cast<NamedDecl>(D)->getQualifiedNameAsString();
		for (auto I=CE->arg_begin(), E=CE->arg_end(); I != E; ++I) {
			QualType QT = (*I)->getType();
			const CXXRecordDecl * RD = QT->getAsCXXRecordDecl();
			if ( RD && ( RD->getQualifiedNameAsString()=="edm::Event" ||
				RD->getQualifiedNameAsString()=="edm::Handle")  ) {
				os<<"function "<<dname<<"\t";
				os<<"call expression ";
				CE->printPretty(os,0,Policy);
				os<<"\t";
				os<<"argument type "<<QT.getAsString();
				llvm::errs()<<os.str()<<"\targument of type edm::Event or edm::Handle passed to function call"<<"\n\n";
				ExplodedNode *errorNode = C.generateSink();
 				BugType * BT = new BugType("argument of type edm::Event or edm::Handle passed to function call ","optional");
				BugReport * R = new BugReport(*BT,os.str(),errorNode);
				R->addRange(CE->getSourceRange());
				C.emitReport(R);
				}
		}
	}
		
		
}


}
