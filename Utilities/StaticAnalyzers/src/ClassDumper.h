#ifndef Utilities_StaticAnalyzers_MemberDumper_h
#define Utilities_StaticAnalyzers_MemberDumper_h
#include <clang/AST/DeclCXX.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/ParentMap.h>
#include <clang/Analysis/CFGStmtMap.h>
#include <llvm/Support/SaveAndRestore.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h>
#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugReporter.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include <llvm/ADT/SmallString.h>
#include <string>
#include "CmsException.h"
#include "CmsSupport.h"

namespace clangcms {

class ClassDumper : public clang::ento::Checker<clang::ento::check::ASTDecl<clang::CXXRecordDecl> > {

public:
  void checkASTDecl(const clang::CXXRecordDecl *CRD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR, std::string tname ) const ;

  void checkASTDecl(const clang::CXXRecordDecl *RD,clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const {
	std::string pname = "classes.txt.dumperall.unsorted";
	checkASTDecl(RD,mgr,BR,pname);
}

private:
  CmsException m_exception;

};

class ClassDumperCT : public clang::ento::Checker<clang::ento::check::ASTDecl<clang::ClassTemplateDecl> > {

public:

  void checkASTDecl(const clang::ClassTemplateDecl *TD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const ;

private:
  CmsException m_exception;

};

class ClassDumperFT : public clang::ento::Checker<clang::ento::check::ASTDecl<clang::FunctionTemplateDecl> > {

public:

  void checkASTDecl(const clang::FunctionTemplateDecl *TD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const ;

private:
  CmsException m_exception;

};

class ClassDumperInherit : public clang::ento::Checker<clang::ento::check::ASTDecl<clang::CXXRecordDecl> > {

public:
  void checkASTDecl(const clang::CXXRecordDecl *CRD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const ;

private:
  CmsException m_exception;

};


}
#endif
