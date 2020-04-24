#ifndef Utilities_StaticAnalyzers_FunctionDumper_h
#define Utilities_StaticAnalyzers_FunctionDumper_h
#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>

namespace clangcms {

class FunctionDumper : public clang::ento::Checker< clang::ento::check::ASTDecl<clang::CXXMethodDecl>,
						clang::ento::check::ASTDecl<clang::FunctionDecl> , 
						clang::ento::check::ASTDecl<clang::FunctionTemplateDecl> > 
{


public:

  void checkASTDecl(const clang::CXXMethodDecl *CMD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const ;

  void checkASTDecl(const clang::FunctionDecl *MD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const ;

  void checkASTDecl(const clang::FunctionTemplateDecl *TD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR) const ;


private:
  CmsException m_exception;
};

}
#endif
