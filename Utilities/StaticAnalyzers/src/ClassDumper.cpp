#include "ClassDumper.h"
#include "CmsSupport.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm>
using namespace clang;
using namespace clang::ento;
using namespace llvm;

namespace clangcms {


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
	const ClassTemplateSpecializationDecl *SD = dyn_cast_or_null<ClassTemplateSpecializationDecl>(RD);
	if (SD) {
		std::string buf;
		llvm::raw_string_ostream os(buf);
		SD->getNameForDiagnostic(os,Policy,1);
		crname = crname+os.str()+"'";
		support::writeLog(crname, tname);
		for (unsigned J = 0, F = SD->getTemplateArgs().size(); J!=F; ++J) {
			if (SD->getTemplateArgs().get(J).getKind() == clang::TemplateArgument::Type) {
				std::string taname;
				auto tt = SD->getTemplateArgs().get(J).getAsType().getTypePtr();
				if (tt->isRecordType()) {
					auto TAD = tt->getAsCXXRecordDecl();
					if (TAD) taname = TAD->getQualifiedNameAsString();
				}
				if (tt->isPointerType() || tt->isReferenceType() ) {
					auto TAD = tt->getPointeeCXXRecordDecl();
					if (TAD) taname = TAD->getQualifiedNameAsString();
				}
				if ( ! ( taname == "")  ) {
					std::string sdname = SD->getQualifiedNameAsString();
					std::string cfname = "templated data class '"+sdname+"' template type class '"+taname+"'";
					support::writeLog(crname+" "+cfname,tname);
				}
			}
		}

	} else {
// Dump the class name
		crname = crname+rname+"'";
		support::writeLog(crname,tname);

	}

// Dump the class member classes
		for ( auto I = RD->field_begin(), E = RD->field_end(); I != E; ++I) {
				clang::QualType qual;
				if (I->getType().getTypePtr()->isAnyPointerType())
					qual = I->getType().getTypePtr()->getPointeeType();
				else
					qual = I->getType().getNonReferenceType();
				if (!qual.getTypePtr()->isRecordType()) continue;
				if (const CXXRecordDecl * TRD = qual.getTypePtr()->getAsCXXRecordDecl()) {
					std::string fname = TRD->getQualifiedNameAsString();
					const ClassTemplateSpecializationDecl *SD = dyn_cast_or_null<ClassTemplateSpecializationDecl>(TRD);
					if (SD) {
							std::string buf;
							llvm::raw_string_ostream os(buf);
							SD->getNameForDiagnostic(os,Policy,1);
							std::string cfname ="member data class '"+os.str()+"'";
							support::writeLog(crname+" "+cfname,tname);
					// Recurse the template args
							for (unsigned J = 0, F = SD->getTemplateArgs().size(); J!=F; ++J) {
								if (SD->getTemplateArgs().get(J).getKind() == clang::TemplateArgument::Type) {
									std::string taname;
									const Type * tt = SD->getTemplateArgs().get(J).getAsType().getTypePtr();
									if ( tt->isRecordType() ) {
										const clang::CXXRecordDecl * TAD = tt->getAsCXXRecordDecl();
										if (TAD) taname = TAD->getQualifiedNameAsString();
									}
									if ( tt->isPointerType() || tt->isReferenceType() ) {
										const clang::CXXRecordDecl * TAD = tt->getPointeeCXXRecordDecl();
										if (TAD) taname = TAD->getQualifiedNameAsString();
									}
									if (!(taname == "")) {
										std::string sdname = SD->getQualifiedNameAsString();
										std::string cfname = "templated member data class '"+sdname+"' template type class '"+taname+"'";
										support::writeLog(crname+" "+cfname,tname);
									}
								}
							}
					} else {
							std::string cfname ="member data class '"+fname+"' ";
							support::writeLog(crname+" "+cfname,tname);
					}
				}
			}





// Dump the base classes

		for ( auto J=RD->bases_begin(), F=RD->bases_end();J != F; ++J) {
			auto BRD = J->getType()->getAsCXXRecordDecl();
			if (!BRD) continue;
			std::string bname = BRD->getQualifiedNameAsString();
			std::string cbname = "base class '"+bname+"'";
			support::writeLog(crname+" "+cbname,tname);
		}


} //end class

void ClassDumperCT::checkASTDecl(const clang::ClassTemplateDecl *TD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {

 	const char *sfile=BR.getSourceManager().getPresumedLoc(TD->getLocation()).getFilename();
 	if (!support::isCmsLocalFile(sfile)) return;

	std::string pname = "classes.txt.dumperct.unsorted";
	std::string tname = TD->getTemplatedDecl()->getQualifiedNameAsString();
	if ( tname == "edm::Wrapper" || tname == "edm::RunCache" || tname == "edm::LuminosityBlockCache" || tname == "edm::GlobalCache" ) {
		for ( auto I = TD->spec_begin(),
			E = TD->spec_end(); I != E; ++I) {
			for ( unsigned J = 0, F = I->getTemplateArgs().size(); J!=F; ++J) {
                               		if (auto D = I->getTemplateArgs().get(J).getAsType()->getAsCXXRecordDecl() ) {
                                       		if (D) {ClassDumper dumper; dumper.checkASTDecl( D, mgr, BR,pname );}
                                       	}
                               		if (auto D = I->getTemplateArgs().get(J).getAsType()->getPointeeCXXRecordDecl() ) {
                                       		if (D) {ClassDumper dumper; dumper.checkASTDecl( D, mgr, BR,pname );}
                               		}
			}
		}
	}

} //end class

void ClassDumperFT::checkASTDecl(const clang::FunctionTemplateDecl *TD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {

 	const char *sfile=BR.getSourceManager().getPresumedLoc(TD->getLocation()).getFilename();
 	if (!support::isCmsLocalFile(sfile)) return;

	std::string pname = "classes.txt.dumperft.unsorted";
	if (TD->getTemplatedDecl()->getQualifiedNameAsString().find("typelookup") != std::string::npos ) {
		for ( auto I = TD->spec_begin(),
				E = TD->spec_end(); I != E; ++I) {
			for (unsigned J = 0, F = (*I)->getTemplateSpecializationArgs()->size(); J!=F;++J){
				if (auto D = (*I)->getTemplateSpecializationArgs()->get(J).getAsType()->getAsCXXRecordDecl()) {
					if (D) {ClassDumper dumper; dumper.checkASTDecl( D, mgr, BR,pname );}
				}
				if (auto D = (*I)->getTemplateSpecializationArgs()->get(J).getAsType()->getPointeeCXXRecordDecl()) {
					if (D) {ClassDumper dumper; dumper.checkASTDecl( D, mgr, BR,pname );}
				}
			}
		}
	}
} //end class

void ClassDumperInherit::checkASTDecl(const clang::CXXRecordDecl *RD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const {
  return;
} //end of class


}//end namespace


