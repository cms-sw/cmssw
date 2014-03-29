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

void writeLog(std::string ostring,std::string tfstring) {
	const char * pPath = std::getenv("LOCALRT");
	std::string tname = "";
	if ( pPath != NULL ) tname += std::string(pPath);
	tname+=tfstring;
	std::fstream file;
	file.open(tname.c_str(),std::ios::in|std::ios::out|std::ios::app);
	file<<ostring<<"\n";
	file.close();
//	std::cout<<ostring<<"\n";
	return;
}

void ClassDumper::checkASTDecl(const clang::CXXRecordDecl *RD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR, std::string tname ) const {

	if (!RD->hasDefinition()) return;
	std::string rname = RD->getQualifiedNameAsString();
	clang::LangOptions LangOpts;
	LangOpts.CPlusPlus = true;
	clang::PrintingPolicy Policy(LangOpts);
	std::string stdname("std::");
	std::string rootname("ROOT::");
	std::string edmname("edm::");
	std::string crname("class '");
	const ClassTemplateSpecializationDecl *SD = dyn_cast<ClassTemplateSpecializationDecl>(RD);
	if (SD) {
		std::string buf;
		llvm::raw_string_ostream os(buf);
		SD->getNameForDiagnostic(os,Policy,1);
		crname = crname+os.str()+"'";
		writeLog(crname, tname);
	} else {
// Dump the class name
		crname = crname+rname+"'";
		writeLog(crname,tname);

	}

// Dump the class member classes
		for (clang::RecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end(); I != E; ++I) {
				clang::QualType qual;
				if (I->getType().getTypePtr()->isAnyPointerType())
					qual = I->getType().getTypePtr()->getPointeeType();
				else
					qual = I->getType().getNonReferenceType();
				if (!qual.getTypePtr()->isRecordType()) continue;
				if (const CXXRecordDecl * TRD = qual.getTypePtr()->getAsCXXRecordDecl()) {
					std::string fname = TRD->getQualifiedNameAsString();
					const ClassTemplateSpecializationDecl *SD = dyn_cast<ClassTemplateSpecializationDecl>(TRD);
					if (SD) {
							std::string buf;
							llvm::raw_string_ostream os(buf);
							SD->getNameForDiagnostic(os,Policy,1);
							std::string cfname ="member data class '"+os.str()+"'";
							writeLog(crname+" "+cfname,tname);
					// Recurse the template args
							for (unsigned J = 0, F = SD->getTemplateArgs().size(); J!=F; ++J) {
								if (SD->getTemplateArgs().get(J).getKind() == clang::TemplateArgument::Type
								&& SD->getTemplateArgs().get(J).getAsType().getTypePtr()->isRecordType() ) {
								const clang::CXXRecordDecl * TAD = SD->getTemplateArgs().get(J).getAsType().getTypePtr()->getAsCXXRecordDecl();
								std::string taname = TAD->getQualifiedNameAsString();
								std::string sdname = SD->getQualifiedNameAsString();
								std::string cfname = "templated member data class '"+sdname+"' template type class '"+taname+"'";
								writeLog(crname+" "+cfname,tname);
								}
							}
					} else {
							std::string cfname ="member data class '"+fname+"' ";
							writeLog(crname+" "+cfname,tname);
					}
				}
			}





// Dump the base classes

		for (clang::CXXRecordDecl::base_class_const_iterator J=RD->bases_begin(), F=RD->bases_end();J != F; ++J) {
			const clang::CXXRecordDecl * BRD = J->getType()->getAsCXXRecordDecl();
			if (!BRD) continue;
			std::string bname = BRD->getQualifiedNameAsString();
			std::string cbname = "base class '"+bname+"'";
			writeLog(crname+" "+cbname,tname);
		}


} //end class

void ClassDumperCT::checkASTDecl(const clang::ClassTemplateDecl *TD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {

	std::string pname = "/tmp/classes.txt.dumperct.unsorted";
	std::string tname = TD->getTemplatedDecl()->getQualifiedNameAsString();
	if ( tname == "edm::Wrapper" || tname == "edm::RunCache" || tname == "edm::LuminosityBlockCache" || tname == "edm::GlobalCache" ) {
		for (ClassTemplateDecl::spec_iterator I = const_cast<clang::ClassTemplateDecl *>(TD)->spec_begin(),
			E = const_cast<clang::ClassTemplateDecl *>(TD)->spec_end(); I != E; ++I)
			{
			for (unsigned J = 0, F = I->getTemplateArgs().size(); J!=F; ++J)
                               {
                               if (const clang::CXXRecordDecl * D = I->getTemplateArgs().get(J).getAsType()->getAsCXXRecordDecl() )
                                       {
                                       ClassDumper dumper;
                                       dumper.checkASTDecl( D, mgr, BR,pname );
                                       }
                               }
		}
	}

} //end class

void ClassDumperFT::checkASTDecl(const clang::FunctionTemplateDecl *TD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {

	std::string pname = "/tmp/classes.txt.dumperft.unsorted";
	if (TD->getTemplatedDecl()->getQualifiedNameAsString().find("typelookup") != std::string::npos )
		{
		for (FunctionTemplateDecl::spec_iterator I = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_begin(),
				E = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_end(); I != E; ++I)
			{
			for (unsigned J = 0, F = (*I)->getTemplateSpecializationArgs()->size(); J!=F;++J)
				{
				if (const clang::CXXRecordDecl * D = (*I)->getTemplateSpecializationArgs()->get(J).getAsType()->getAsCXXRecordDecl())
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
		std::string cbname = "class '"+bname+"'\n";
		if (ifilecontents.find(cbname) != std::string::npos ) {
			std::string pname = "/tmp/classes.txt.inherits.unsorted";
			std::string rname = RD->getQualifiedNameAsString();
			std::string crname = "class '"+rname+"'";
			writeLog(crname, pname);
			ClassDumper dumper;
			dumper.checkASTDecl( RD, mgr, BR, pname );
		}
	}
} //end of class


}//end namespace


