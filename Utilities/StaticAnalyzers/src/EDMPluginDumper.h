#ifndef Utilities_StaticAnalyzers_EDMPluginDumper_h
#define Utilities_StaticAnalyzers_EDMPluginDumper_h
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

#include "CmsException.h"
#include "CmsSupport.h"

namespace clangcms {

class EDMPluginDumper : public clang::ento::Checker<clang::ento::check::ASTDecl<clang::ClassTemplateDecl> > {

public:

  void checkASTDecl(const clang::ClassTemplateDecl *TD, clang::ento::AnalysisManager& mgr,
                    clang::ento::BugReporter &BR ) const ;

private:
  CmsException m_exception;

};


}
#endif
