#include "ClassDumper.h"

using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {




void ClassDumper::checkASTDecl(const clang::CXXRecordDecl *RD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {

	const clang::SourceManager &SM = BR.getSourceManager();
	clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( RD, SM );
//	if (  !m_exception.reportClass( DLoc, BR ) ) return;
//Dump the template name and args
	if (const ClassTemplateSpecializationDecl *SD = dyn_cast<ClassTemplateSpecializationDecl>(RD))
		{
			for (unsigned J = 0, F = SD->getTemplateArgs().size(); J!=F; ++J)
			{
//			llvm::errs()<<"\nTemplate "<<SD->getSpecializedTemplate()->getQualifiedNameAsString()<<";";
//			llvm::errs()<<"Template Argument ";
//			llvm::errs()<<SD->getTemplateArgs().get(J).getAsType().getAsString();
//			llvm::errs()<<"\n\n\t";
			if (SD->getTemplateArgs().get(J).getKind() == clang::TemplateArgument::Type && SD->getTemplateArgs().get(J).getAsType().getTypePtr()->isRecordType() )
				{
				const clang::CXXRecordDecl * D = SD->getTemplateArgs().get(J).getAsType().getTypePtr()->getAsCXXRecordDecl();
				checkASTDecl( D, mgr, BR );
				}
			}

		}
	
// Dump the class members.
	std::string err;
	std::string fname("/tmp/classes.txt.unsorted");
	llvm::raw_fd_ostream output(fname.c_str(),err,llvm::raw_fd_ostream::F_Append);
//	llvm::errs() <<"class " <<RD->getQualifiedNameAsString()<<"\n";
	output <<"class " <<RD->getQualifiedNameAsString()<<"\n";
	for (clang::RecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end(); I != E; ++I)
	{
		clang::QualType qual;
		if (I->getType().getTypePtr()->isAnyPointerType()) 
			qual = I->getType().getTypePtr()->getPointeeType();
		else 
			qual = I->getType().getNonReferenceType();

		if (!qual.getTypePtr()->isRecordType()) return;
//		llvm::errs() <<"Class Member ";
//		if (I->getType() == qual)
//			{
//			llvm::errs() <<"; "<<I->getType().getCanonicalType().getTypePtr()->getTypeClassName();
//			}
//		else
//			{
//			llvm::errs() <<"; "<<qual.getCanonicalType().getTypePtr()->getTypeClassName()<<" "<<I->getType().getCanonicalType().getTypePtr()->getTypeClassName();
//			}
//		llvm::errs() <<"; "<<I->getType().getCanonicalType().getAsString();
//		llvm::errs() <<"; "<<I->getType().getAsString();
//		llvm::errs() <<"; "<< I->getQualifiedNameAsString();

//		llvm::errs() <<"\n\n";
		if (const CXXRecordDecl * TRD = I->getType().getTypePtr()->getAsCXXRecordDecl()) 
			{
			if (RD->getNameAsString() == TRD->getNameAsString())
				{
				checkASTDecl( TRD, mgr, BR );
				}
			}
	}

} //end class


void ClassDumperCT::checkASTDecl(const clang::ClassTemplateDecl *TD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {
	const clang::SourceManager &SM = BR.getSourceManager();
	clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( TD, SM );
	if ( SM.isInSystemHeader(DLoc.asLocation()) || SM.isInExternCSystemHeader(DLoc.asLocation()) ) return;
	if (TD->getTemplatedDecl()->getQualifiedNameAsString() == "edm::Wrapper" ) 
		{
//		llvm::errs()<<"\n";
		for (ClassTemplateDecl::spec_iterator I = const_cast<clang::ClassTemplateDecl *>(TD)->spec_begin(), 
			E = const_cast<clang::ClassTemplateDecl *>(TD)->spec_end(); I != E; ++I) 
			{
			for (unsigned J = 0, F = I->getTemplateArgs().size(); J!=F; ++J)
				{
//				llvm::errs()<<"template class "<< TD->getTemplatedDecl()->getQualifiedNameAsString()<<"<" ;
//				llvm::errs()<<I->getTemplateArgs().get(J).getAsType().getAsString();
//				llvm::errs()<<">\n";
				if (const clang::CXXRecordDecl * D = I->getTemplateArgs().get(J).getAsType().getTypePtr()->getAsCXXRecordDecl())
					{
					ClassDumper dumper;
					dumper.checkASTDecl( D, mgr, BR );
					}
				}
			} 		
		};
} //end class

void ClassDumperFT::checkASTDecl(const clang::FunctionTemplateDecl *TD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {
	const clang::SourceManager &SM = BR.getSourceManager();
	clang::ento::PathDiagnosticLocation DLoc =clang::ento::PathDiagnosticLocation::createBegin( TD, SM );
	if ( SM.isInSystemHeader(DLoc.asLocation()) || SM.isInExternCSystemHeader(DLoc.asLocation()) ) return;
	if (TD->getTemplatedDecl()->getQualifiedNameAsString().find("typelookup") != std::string::npos ) 
		{
//		llvm::errs()<<"\n";
		for (FunctionTemplateDecl::spec_iterator I = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_begin(), 
				E = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_end(); I != E; ++I) 
			{
			for (unsigned J = 0, F = (*I)->getTemplateSpecializationArgs()->size(); J!=F;++J)
				{
//				llvm::errs()<<"template function " << TD->getTemplatedDecl()->getQualifiedNameAsString()<<"<";
//				llvm::errs()<<(*I)->getTemplateSpecializationArgs()->get(J).getAsType().getAsString();
//				llvm::errs()<<">\n";
				if (const clang::CXXRecordDecl * D = (*I)->getTemplateSpecializationArgs()->get(J).getAsType().getTypePtr()->getAsCXXRecordDecl()) 
					{
					ClassDumper dumper;
					dumper.checkASTDecl( D, mgr, BR );
					}
				}
	
			} 		
		};
} //end class

void ClassDumperInherit::checkASTDecl(const clang::CXXRecordDecl *RD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const {

	const clang::SourceManager &SM = BR.getSourceManager();
	if (!RD->hasDefinition()) return;

	clang::FileSystemOptions FSO;
	clang::FileManager FM(FSO);
	if (!FM.getFile("/tmp/classes.txt.dumperft") ) {
		llvm::errs()<<"\n\nChecker cannot find /tmp/classes.txt.dumperft \n";
		exit(1);
		}
	llvm::MemoryBuffer * buffer = FM.getBufferForFile(FM.getFile("/tmp/classes.txt.dumperft"));
//	llvm::errs()<<"class "<<RD->getQualifiedNameAsString()<<"\n";

	for (clang::CXXRecordDecl::base_class_const_iterator J=RD->bases_begin(), F=RD->bases_end();J != F; ++J)
	{  
		const clang::CXXRecordDecl * BRD = J->getType()->getAsCXXRecordDecl();
		if (!BRD) continue;
		std::string name = BRD->getQualifiedNameAsString();
//		llvm::errs() << " class " << RD->getQualifiedNameAsString() << " inherits from "<<name <<"\n";
		llvm::StringRef Rname("class "+name);
		if (buffer->getBuffer().find(Rname) != llvm::StringRef::npos )
			{
			std::string err;
			std::string fname("/tmp/classes.txt.unsorted");
			llvm::raw_fd_ostream output(fname.c_str(),err,llvm::raw_fd_ostream::F_Append);
			output <<"class " <<RD->getQualifiedNameAsString()<<"\n";

			llvm::SmallString<100> buf;
			llvm::raw_svector_ostream os(buf);
			os << " class " << RD->getQualifiedNameAsString() << " inherits from "<<name <<"\n";
			llvm::errs()<<os.str();
//			clang::ento::PathDiagnosticLocation ELoc =clang::ento::PathDiagnosticLocation::createBegin( RD, SM );
//			clang::SourceLocation SL = RD->getLocStart();
//			BR.EmitBasicReport(RD, "Class Checker : inherits from TYPELOOKUP_DATA_REG class","optional",os.str(),ELoc,SL);
			}
			
	}
} //end of class


}//end namespace


