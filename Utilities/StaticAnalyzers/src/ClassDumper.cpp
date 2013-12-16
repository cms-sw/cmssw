#include "ClassDumper.h"
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

void ClassDumper::checkASTDecl(const clang::CXXRecordDecl *RD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR, std::string &tname ) const {
//Dump the template name and args
	if (const ClassTemplateSpecializationDecl *SD = dyn_cast<ClassTemplateSpecializationDecl>(RD))
		{
			for (unsigned J = 0, F = SD->getTemplateArgs().size(); J!=F; ++J)
			{
			if (SD->getTemplateArgs().get(J).getKind() == clang::TemplateArgument::Type && SD->getTemplateArgs().get(J).getAsType().getTypePtr()->isRecordType() )
				{
				const clang::CXXRecordDecl * D = SD->getTemplateArgs().get(J).getAsType().getTypePtr()->getAsCXXRecordDecl();
				checkASTDecl( D, mgr, BR, tname);
				}
			}

		}
	
// Dump the class members.
	std::string err;
	std::string rname = RD->getQualifiedNameAsString();
	std::string crname = "class "+rname+"\n";
	file_mutex.wait();
	std::fstream file(tname.c_str(),std::ios::in|std::ios::out|std::ios::app);
	std::string filecontents((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>() );
	if ( filecontents.find(crname) == std::string::npos )  {
		file << crname;
		file.close();
		file_mutex.post();
		for (clang::RecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end(); I != E; ++I)
			{
				clang::QualType qual;
				if (I->getType().getTypePtr()->isAnyPointerType()) 
					qual = I->getType().getTypePtr()->getPointeeType();
				else 
					qual = I->getType().getNonReferenceType();
				if (!qual.getTypePtr()->isRecordType()) return;
				if (const CXXRecordDecl * TRD = qual.getTypePtr()->getAsCXXRecordDecl()) checkASTDecl( TRD, mgr, BR, tname );
			}
	} else {
		file.close();
		file_mutex.post();
	}
} //end class

void ClassDumperCT::checkASTDecl(const clang::ClassTemplateDecl *TD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {

	const char * pPath = std::getenv("LOCALRT");
	std::string pname(""); 
	if ( pPath != NULL ) pname = std::string(pPath);
	pname += "/tmp/classes.txt.dumperct.unsorted";
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
					dumper.checkASTDecl( D, mgr, BR,pname );
					}
				}
			} 		
		};
} //end class

void ClassDumperFT::checkASTDecl(const clang::FunctionTemplateDecl *TD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {

	const char * pPath = std::getenv("LOCALRT");
	std::string pname(""); 
	if ( pPath != NULL ) pname = std::string(pPath);
	pname += "/tmp/classes.txt.dumperft.unsorted";
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
					dumper.checkASTDecl( D, mgr, BR,pname );
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
	std::string iname(""); 
	if ( pPath != NULL ) iname = std::string(pPath);
	iname += "/tmp/classes.txt.dumperft";
	std::ifstream ifile;
	ifile.open(iname.c_str(),std::ifstream::in);
	if (!ifile.good() ) {
		llvm::errs()<<"\n\nChecker cannot find $LOCALRT/tmp/classes.txt.dumperft \n";
		exit(1);
		}
	std::string ifilecontents((std::istreambuf_iterator<char>(ifile)),std::istreambuf_iterator<char>() );
	for (clang::CXXRecordDecl::base_class_const_iterator J=RD->bases_begin(), F=RD->bases_end();J != F; ++J) {  
		const clang::CXXRecordDecl * BRD = J->getType()->getAsCXXRecordDecl();
		if (!BRD) continue;
		std::string bname = BRD->getQualifiedNameAsString();
		std::string ename = "edm::global::";
		std::string rname = RD->getQualifiedNameAsString();
		std::string crname = "class "+rname+"\n";
		std::string cbname = "class "+bname+"\n";
		if (ifilecontents.find(cbname) != std::string::npos ) {
			std::string err;
			const char * pPath = std::getenv("LOCALRT");
			std::string pname(""); 
			if ( pPath != NULL ) pname = std::string(pPath);
			pname += "/tmp/classes.txt.inherits.unsorted";
			file_mutex.wait();
			std::fstream file(pname.c_str(),std::ios::in|std::ios::out|std::ios::app);
			std::string filecontents((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>() );
			file.close();
			file_mutex.post();
			if (filecontents.find(crname) == std::string::npos) {
				ClassDumper dumper;
				dumper.checkASTDecl( RD, mgr, BR, pname );
			}
			
		}
	}
} //end of class


}//end namespace


