#include "FunctionDumper.h"
using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

class FDumper : public clang::StmtVisitor<FDumper> {
  clang::ento::BugReporter &BR;
  clang::AnalysisDeclContext *AC;

public:
  FDumper(clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac )
    : BR(br),
      AC(ac) {}

  const clang::Stmt * ParentStmt(const Stmt *S) {
  	const Stmt * P = AC->getParentMap().getParentIgnoreParens(S);
	if (!P) return 0;
	return P;
  }


  void VisitChildren(clang::Stmt *S );
  void VisitStmt( clang::Stmt *S) { VisitChildren(S); }
  void VisitCXXMemberCallExpr( CXXMemberCallExpr *CE ); 
 
};

void FDumper::VisitChildren( clang::Stmt *S) {
  for (clang::Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (clang::Stmt *child = *I) {
      Visit(child);
    }
}


void FDumper::VisitCXXMemberCallExpr( CXXMemberCallExpr *CE ) {
	LangOptions LangOpts;
	LangOpts.CPlusPlus = true;
	PrintingPolicy Policy(LangOpts);
	const Decl * D = AC->getDecl();
	std::string dname =""; 
	if (const NamedDecl * ND = llvm::dyn_cast<NamedDecl>(D)) dname = support::getQualifiedName(*ND);
	CXXMethodDecl * MD = dyn_cast<CXXMethodDecl>(CE->getMethodDecl()->getMostRecentDecl());
	if (!MD) return;
 	const char *sfile=BR.getSourceManager().getPresumedLoc(CE->getExprLoc()).getFilename();
  	if (!support::isCmsLocalFile(sfile)) return;
 	std::string mname = support::getQualifiedName(*MD);
	llvm::SmallString<1000> buf;
	llvm::raw_svector_ostream os(buf);
	os<<"function '"<<dname<<"' ";
	os<<"calls function '"<<mname;
//	MD->getNameForDiagnostic(os,Policy,1);
	os<<"' \n\n";
	for (auto I = MD->begin_overridden_methods(), E = MD->end_overridden_methods(); I!=E; ++I) {
		os<<"function '"<<mname<<"' ";
		os<<"overrides function '";
//		(*I)->getNameForDiagnostic(os,Policy,1);
		os<<support::getQualifiedName(*(*I));
		os<<"' \n\n";	
	} 
        llvm::errs()<<os.str();
}

void FunctionDumper::checkASTDecl(const CXXMethodDecl *MD, AnalysisManager& mgr,
                    BugReporter &BR) const {

 	const char *sfile=BR.getSourceManager().getPresumedLoc(MD->getLocation()).getFilename();
   	if (!support::isCmsLocalFile(sfile)) return;
  
      	if (!MD->doesThisDeclarationHaveABody()) return;
	FDumper walker(BR, mgr.getAnalysisDeclContext(MD));
	walker.Visit(MD->getBody());
       	return;
} 

void FunctionDumper::checkASTDecl(const FunctionTemplateDecl *TD, AnalysisManager& mgr,
                    BugReporter &BR) const {

 	const char *sfile=BR.getSourceManager().getPresumedLoc(TD->getLocation ()).getFilename();
   	if (!support::isCmsLocalFile(sfile)) return;
  
	for (FunctionTemplateDecl::spec_iterator I = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_begin(), 
			E = const_cast<clang::FunctionTemplateDecl *>(TD)->spec_end(); I != E; ++I) 
		{
			if (I->doesThisDeclarationHaveABody()) {
				FDumper walker(BR, mgr.getAnalysisDeclContext(*I));
				walker.Visit(I->getBody());
				}
		}	
	return;
}



}
