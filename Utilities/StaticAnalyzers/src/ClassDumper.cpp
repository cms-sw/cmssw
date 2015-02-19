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
 	const std::string anon_ns = "(anonymous namespace)";
	if (rname.substr(0, anon_ns.size()) == anon_ns ) {
 		const char* fname = BR.getSourceManager().getPresumedLoc(RD->getLocation()).getFilename();
		const char* sname = "/src/";
		const char* filename = std::strstr(fname, sname);
		rname = rname.substr(0, anon_ns.size() - 1)+" in "+filename+")"+rname.substr(anon_ns.size());
		}
	clang::LangOptions LangOpts;
	LangOpts.CPlusPlus = true;
	clang::PrintingPolicy Policy(LangOpts);
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
					std::string sdname = SD->getQualifiedNameAsString();
					std::string cfname = "templated data class '"+sdname+"' template type class '"+taname+"'";
					support::writeLog(crname+" "+cfname,tname);
				}
				if (tt->isPointerType() || tt->isReferenceType() ) {
					auto TAD = tt->getPointeeCXXRecordDecl();
					if (TAD) taname = TAD->getQualifiedNameAsString();
					std::string sdname = SD->getQualifiedNameAsString();
					std::string cfname = "templated data class '"+sdname+"' template type class '"+taname+"'";
					std::string cbname = "templated data class 'bare_ptr' template type class '"+taname+"'";
					support::writeLog(crname+" "+cfname,tname);
					support::writeLog(crname+" "+cbname,tname);
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
				clang::QualType type = I->getType();
				if (type.getTypePtr()->isAnyPointerType())
					qual = type.getTypePtr()->getPointeeType();
				else
					qual = type.getNonReferenceType();
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
										std::string sdname = SD->getQualifiedNameAsString();
										std::string cfname = "templated member data class '"+sdname+"' template type class '"+taname+"'";
										support::writeLog(crname+" "+cfname,tname);
									}
									if ( tt->isPointerType() || tt->isReferenceType() ) {
										const clang::CXXRecordDecl * TAD = tt->getPointeeCXXRecordDecl();
										if (TAD) taname = TAD->getQualifiedNameAsString();
										std::string sdname = SD->getQualifiedNameAsString();
										std::string cfname = "templated member data class '"+sdname+"' template type class '"+taname+"'";
										std::string cbname = "templated member data class 'bare_ptr' template type class '"+taname+"'";
										support::writeLog(crname+" "+cfname,tname);
										support::writeLog(crname+" "+cbname,tname);
									}
								}
							}
					} else {
						if (type.getTypePtr()->isRecordType()) {
							std::string cfname ="member data class '"+fname+"' ";
							support::writeLog(crname+" "+cfname,tname);
							}
						if (type.getTypePtr()->isAnyPointerType()) {
							std::string cfname = "templated member data class 'bare_ptr' template type class '"+fname+"'";
							support::writeLog(crname+" "+cfname,tname);
							}
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

	std::string crname("class '");
	std::string pname = "classes.txt.dumperct.unsorted";
	std::string tname = TD->getTemplatedDecl()->getQualifiedNameAsString();
	if ( tname == "edm::Wrapper" || tname == "edm::RunCache" || tname == "edm::LuminosityBlockCache" || tname == "edm::GlobalCache" ) {
		for ( auto I = TD->spec_begin(),
			E = TD->spec_end(); I != E; ++I) {
			for ( unsigned J = 0, F = I->getTemplateArgs().size(); J!=F; ++J) {
					auto D = I->getTemplateArgs().get(J).getAsType()->getAsCXXRecordDecl();  
					if (D) {
							ClassDumper dumper; dumper.checkASTDecl( D, mgr, BR,pname );
							std::string taname = D->getQualifiedNameAsString();
							std::string tdname = TD->getQualifiedNameAsString();
							std::string cfname = "templated class '"+tdname+"' template type class '"+taname+"'";
							support::writeLog(cfname,pname);
							}
					auto E = I->getTemplateArgs().get(J).getAsType()->getPointeeCXXRecordDecl(); 
					if (E) {
							ClassDumper dumper; dumper.checkASTDecl( E, mgr, BR,pname );
							std::string taname = E->getQualifiedNameAsString();
							std::string tdname = TD->getQualifiedNameAsString();
							std::string cfname = "templated class '"+tdname+"' template type class '"+taname+"'";
							support::writeLog(cfname,pname);
							std::string cbname = "templated class 'bare_ptr' template type class '"+taname+"'";
							support::writeLog(crname+" "+cbname,pname);
							}
			}
		}
	}

} //end class

void ClassDumperFT::checkASTDecl(const clang::FunctionTemplateDecl *TD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const {

 	const char *sfile=BR.getSourceManager().getPresumedLoc(TD->getLocation()).getFilename();
 	if (!support::isCmsLocalFile(sfile)) return;

	std::string crname("class '");
	std::string pname = "classes.txt.dumperft.unsorted";
	if (TD->getTemplatedDecl()->getQualifiedNameAsString().find("typelookup::className") != std::string::npos ) {
		for ( auto I = TD->spec_begin(),
				E = TD->spec_end(); I != E; ++I) {
			auto * SD = (*I); 
			for (unsigned J = 0, F = SD->getTemplateSpecializationArgs()->size(); J!=F;++J) {
				auto D = SD->getTemplateSpecializationArgs()->get(J).getAsType()->getAsCXXRecordDecl();
				if (D) {
					ClassDumper dumper; dumper.checkASTDecl( D, mgr, BR,pname );
					std::string taname = D->getQualifiedNameAsString();
					std::string sdname = SD->getQualifiedNameAsString();
					std::string cfname = "templated function '"+sdname+"' template type class '"+taname+"'";
					support::writeLog(cfname,pname);
					}
				auto E = SD->getTemplateSpecializationArgs()->get(J).getAsType()->getPointeeCXXRecordDecl();
				if (E) {
					ClassDumper dumper; dumper.checkASTDecl( E, mgr, BR,pname );
					std::string taname = E->getQualifiedNameAsString();
					std::string sdname = SD->getQualifiedNameAsString();
					std::string cfname = "templated function '"+sdname+"' template type class '"+taname+"'";
					support::writeLog(cfname,pname);
					std::string cbname = "templated function 'bare_ptr' template type class '"+taname+"'";
					support::writeLog(crname+" "+cbname,pname);
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


