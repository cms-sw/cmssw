#include "getByChecker.h"
using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

class Walker : public clang::StmtVisitor<Walker> {
  const CheckerBase *Checker;
  clang::ento::BugReporter &BR;
  clang::AnalysisDeclContext *AC;

public:
  Walker( const CheckerBase *checker, clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac )
    : Checker(checker),
      BR(br),
      AC(ac) {}

  void VisitChildren(clang::Stmt *S );
  void VisitStmt( clang::Stmt *S) { VisitChildren(S); }
  void VisitCXXMemberCallExpr( clang::CXXMemberCallExpr *CE );
 
};

void Walker::VisitChildren( clang::Stmt *S) {
  for (clang::Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (clang::Stmt *child = *I) {
      Visit(child);
    }
}



void Walker::VisitCXXMemberCallExpr( CXXMemberCallExpr *CE ) {
	LangOptions LangOpts;
	LangOpts.CPlusPlus = true;
	PrintingPolicy Policy(LangOpts);
	const Decl * D = AC->getDecl();
	std::string dname =""; 
	if (const NamedDecl * ND = llvm::dyn_cast_or_null<NamedDecl>(D)) dname = ND->getQualifiedNameAsString();
	CXXMethodDecl * MD = CE->getMethodDecl();
	if (!MD) return;
	std::string mname = MD->getQualifiedNameAsString();
//	llvm::errs()<<"Parent Decl: '"<<dname<<"'\n";
//	llvm::errs()<<"Method Decl: '"<<mname<<"'\n";
//	llvm::errs()<<"call expression '";
//	CE->printPretty(llvm::errs(),0,Policy);
//	llvm::errs()<<"'\n";
//	if (!MD) return;
	llvm::SmallString<100> buf;
	llvm::raw_svector_ostream os(buf);
	if ( mname == "edm::Event::getByLabel" || mname == "edm::Event::getManyByType" ) {
//			if (const CXXRecordDecl * RD = llvm::dyn_cast_or_null<CXXMethodDecl>(D)->getParent() ) {
//				llvm::errs()<<"class "<<RD->getQualifiedNameAsString()<<"\n";
//				llvm::errs()<<"\n";
//				}
			os<<"function '";
			llvm::dyn_cast<CXXMethodDecl>(D)->getNameForDiagnostic(os,Policy,1);
			os<<"' ";
//			os<<"call expression '";
//			CE->printPretty(os,0,Policy);
//			os<<"' ";
		if (mname == "edm::Event::getByLabel") {
			os <<"calls edm::Event::getByLabel with arguments '";
			QualType QT;
			for ( auto I=CE->arg_begin(), E=CE->arg_end(); I != E; ++I) {
				QT=(*I)->getType();
				std::string qtname = QT.getCanonicalType().getAsString();
				if ( qtname.substr(0,17)=="class edm::Handle" ) {
//					os<<"argument name '";
//					(*I)->printPretty(os,0,Policy);
//					os<<"' ";
					const CXXRecordDecl * RD = QT->getAsCXXRecordDecl();
					std::string rname = RD->getQualifiedNameAsString();
					os << rname<<" ";
					const ClassTemplateSpecializationDecl *SD = dyn_cast<ClassTemplateSpecializationDecl>(RD);
					for (unsigned J = 0, F = SD->getTemplateArgs().size(); J!=F; ++J) {
						SD->getTemplateArgs().data()[J].print(Policy,os);
						os<<", ";
						}
				}
				else { 
					os<<" "<< qtname <<" ";
					(*I)->printPretty(os,0,Policy);
					os <<", ";
				}
			}
			os <<"'\n";	
		} else {
			os <<"calls edm::Event::getManyByType with argument '";
			QualType QT = CE->arg_begin()->getType();
			const CXXRecordDecl * RD = QT->getAsCXXRecordDecl();
			os << "getManyByType , ";
			const ClassTemplateSpecializationDecl *SD = dyn_cast<ClassTemplateSpecializationDecl>(RD);
			const TemplateArgument TA = SD->getTemplateArgs().data()[0];
			const QualType AQT = TA.getAsType();
			const CXXRecordDecl * SRD = AQT->getAsCXXRecordDecl();
			os << SRD->getQualifiedNameAsString()<<" ";
			const ClassTemplateSpecializationDecl *SVD = dyn_cast<ClassTemplateSpecializationDecl>(SRD);
			for (unsigned J = 0, F = SVD->getTemplateArgs().size(); J!=F; ++J) {
				SVD->getTemplateArgs().data()[J].print(Policy,os);
				os<<", ";
				}
			
		}

//			llvm::errs()<<os.str()<<"\n";
			PathDiagnosticLocation CELoc = 
				PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
			BugType * BT = new BugType(Checker,"edm::getByLabel or edm::getManyByType called","optional") ;
			BugReport * R = new BugReport(*BT,os.str(),CELoc);
			R->addRange(CE->getSourceRange());
			BR.emitReport(R);
	} 
	else {
		for (auto I=CE->arg_begin(), E=CE->arg_end(); I != E; ++I) {
			QualType QT = (*I)->getType();
			std::string qtname = QT.getAsString();
//			if (qtname.find(" edm::Event") != std::string::npos ) llvm::errs()<<"arg type '" << qtname <<"'\n";
			if ( qtname=="edm::Event" || qtname=="const edm::Event" ||
				qtname=="edm::Event *" || qtname=="const edm::Event *" )  {
				std::string tname;
				os<<"function '"<<dname<<"' ";
				os<<"calls '";
				MD->getNameForDiagnostic(os,Policy,1);
				os<<"' with argument of type '"<<qtname<<"'\n";
//				llvm::errs()<<"\n";
//				llvm::errs()<<"call expression passed edm::Event ";
//				CE->printPretty(llvm::errs(),0,Policy);
//				llvm::errs()<<" argument name ";
//				(*I)->printPretty(llvm::errs(),0,Policy);
//				llvm::errs()<<" "<<qtname<<"\n";
				PathDiagnosticLocation CELoc = 
					PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(),AC);
 				BugType * BT = new BugType(Checker,"function call with argument of type edm::Event","optional");
				BugReport * R = new BugReport(*BT,os.str(),CELoc);
				R->addRange(CE->getSourceRange());
				BR.emitReport(R);
				}
		}
	}
		
		
}

void getByChecker::checkASTDecl(const CXXMethodDecl *MD, AnalysisManager& mgr,
                    BugReporter &BR) const {
       	const SourceManager &SM = BR.getSourceManager();
       	PathDiagnosticLocation DLoc =PathDiagnosticLocation::createBegin( MD, SM );
	if ( SM.isInSystemHeader(DLoc.asLocation()) || SM.isInExternCSystemHeader(DLoc.asLocation()) ) return;
       	if (!MD->doesThisDeclarationHaveABody()) return;
	clangcms::Walker walker(this,BR, mgr.getAnalysisDeclContext(MD));
	walker.Visit(MD->getBody());
       	return;
} 

void getByChecker::checkASTDecl(const FunctionTemplateDecl *TD, AnalysisManager& mgr,
                    BugReporter &BR) const {
	const clang::SourceManager &SM = BR.getSourceManager();
	clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( TD, SM );
	if ( SM.isInSystemHeader(DLoc.asLocation()) || SM.isInExternCSystemHeader(DLoc.asLocation()) ) return;

	for (FunctionTemplateDecl::spec_iterator I = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_begin(), 
			E = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_end(); I != E; ++I) 
		{
			if (I->doesThisDeclarationHaveABody()) {
				clangcms::Walker walker(this,BR, mgr.getAnalysisDeclContext(*I));
				walker.Visit(I->getBody());
				}
		}	
	return;
}



}
