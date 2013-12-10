#include "ClassDumper.h"

using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {

void ClassDumper::checkASTDecl(const clang::CXXRecordDecl *RD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {
//Dump the template name and args
	if (const ClassTemplateSpecializationDecl *SD = dyn_cast<ClassTemplateSpecializationDecl>(RD))
		{
			for (unsigned J = 0, F = SD->getTemplateArgs().size(); J!=F; ++J)
			{
			if (SD->getTemplateArgs().get(J).getKind() == clang::TemplateArgument::Type && SD->getTemplateArgs().get(J).getAsType().getTypePtr()->isRecordType() )
				{
				const clang::CXXRecordDecl * D = SD->getTemplateArgs().get(J).getAsType().getTypePtr()->getAsCXXRecordDecl();
				checkASTDecl( D, mgr, BR );
				}
			}

		}
	
// Dump the class members.
	std::string err;
	const char * pPath = std::getenv("LOCALRT");
	std::string dname(""); 
	if ( pPath != NULL ) dname = std::string(pPath);
	std::string fname("/tmp/classes.txt.unsorted");
	std::string tname = dname + fname;
	std::string rname = RD->getQualifiedNameAsString();
	llvm::StringRef Rname("class "+rname);
	llvm::raw_fd_ostream output(tname.c_str(),err,llvm::raw_fd_ostream::F_Append);
	output << Rname.str() <<"\n";
	for (clang::RecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end(); I != E; ++I)
		{
			FieldDecl * FD = *I;
			clang::QualType qual;
			if (FD->getType().getTypePtr()->isAnyPointerType()) 
				qual = FD->getType().getTypePtr()->getPointeeType();
			else 
				qual = FD->getType().getNonReferenceType();
			if (!qual.getTypePtr()->isRecordType()) return;
			if (const CXXRecordDecl * TRD = qual->getAsCXXRecordDecl()) checkASTDecl( TRD, mgr, BR );
		}

} //end class

void ClassDumperCT::checkASTDecl(const clang::ClassTemplateDecl *TD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {
 	const char *sfile=BR.getSourceManager().getPresumedLoc(TD->getLocation()).getFilename();
   	if (!support::isCmsLocalFile(sfile)) return;

	std::string tname = TD->getTemplatedDecl()->getQualifiedNameAsString();
	if ( tname == "edm::Wrapper" || tname == "edm::RunCache" || tname == "edm::LuminosityBlockCache" || tname == "edm::GlobalCache" ) 
		{
		for (ClassTemplateDecl::spec_iterator I = const_cast<clang::ClassTemplateDecl *>(TD)->spec_begin(), 
			E = const_cast<clang::ClassTemplateDecl *>(TD)->spec_end(); I != E; ++I) 
			{
			for (unsigned J = 0, F = I->getTemplateArgs().size(); J!=F; ++J)
				{
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

 	const char *sfile=BR.getSourceManager().getPresumedLoc(TD->getLocation()).getFilename();
   	if (!support::isCmsLocalFile(sfile)) return;

	if (TD->getTemplatedDecl()->getQualifiedNameAsString().find("typelookup") != std::string::npos ) 
		{
		for (FunctionTemplateDecl::spec_iterator I = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_begin(), 
				E = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_end(); I != E; ++I) 
			{
			for (unsigned J = 0, F = (*I)->getTemplateSpecializationArgs()->size(); J!=F;++J)
				{
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

 	const char *sfile=BR.getSourceManager().getPresumedLoc(RD->getLocation()).getFilename();
   	if (!support::isCmsLocalFile(sfile)) return;

	if (!RD->hasDefinition()) return;

	clang::FileSystemOptions FSO;
	clang::FileManager FM(FSO);
	const char * pPath = std::getenv("LOCALRT");
	std::string dname(""); 
	if ( pPath != NULL ) dname = std::string(pPath);
	std::string fname("/tmp/classes.txt.dumperft");
	std::string tname = dname + fname;
	if (!FM.getFile(tname) ) {
		llvm::errs()<<"\n\nChecker cannot find $LOCALRT/tmp/classes.txt.dumperft \n";
		exit(1);
		}
	llvm::MemoryBuffer * buffer = FM.getBufferForFile(FM.getFile(tname));

	for (clang::CXXRecordDecl::base_class_const_iterator J=RD->bases_begin(), F=RD->bases_end();J != F; ++J)
	{  
		const clang::CXXRecordDecl * BRD = J->getType()->getAsCXXRecordDecl();
		if (!BRD) continue;
		std::string name = BRD->getQualifiedNameAsString();
		std::string ename = "edm::global::";
		llvm::StringRef Rname("class "+name);
		if ((buffer->getBuffer().find(Rname) != llvm::StringRef::npos )|| (name.substr(0,ename.length()) == ename) )
			{
			std::string err;
			const char * pPath = std::getenv("LOCALRT");
			std::string dname(""); 
			if ( pPath != NULL ) dname = std::string(pPath);
			std::string fname("/tmp/classes.txt.unsorted");
			std::string tname = dname + fname;
			llvm::raw_fd_ostream output(tname.c_str(),err,llvm::raw_fd_ostream::F_Append);
			output <<"class " <<RD->getQualifiedNameAsString()<<"\n";
			ClassDumper dumper;
			dumper.checkASTDecl( RD, mgr, BR );
			}
			
	}
} //end of class


}//end namespace


