#include "ArgSizeChecker.h"
#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/DeclGroup.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Expr.h>
#include <clang/AST/CharUnits.h>
#include <llvm/ADT/SmallString.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugReporter.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>


#include "CmsSupport.h"
#include <iostream>
using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {


void ArgSizeChecker::checkPreStmt(const CXXMemberCallExpr *CE, CheckerContext &ctx) const
{

  CmsException m_exception;
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

  const ProgramStateRef state = ctx.getState();
  const LocationContext *LC = ctx.getLocationContext();
  const Expr *Callee = CE->getCallee();
  const FunctionDecl *FD = state->getSVal(Callee, LC).getAsFunctionDecl();

  if (!FD)
    return;
 const clang::ento::PathDiagnosticLocation ELoc =
   clang::ento::PathDiagnosticLocation::createBegin(FD, ctx.getSourceManager());

  if ( ! m_exception.reportGeneral( ELoc, ctx.getBugReporter() ) )  return;

  for ( int I = 0, E=CE->getNumArgs(); I != E; ++I ) {
  	llvm::SmallString<100> buf;
  	llvm::raw_svector_ostream os(buf);
	QualType QT = CE->getArg(I)->getType();
	if (QT->isIncompleteType()) continue;
	const clang::ParmVarDecl *PVD=llvm::dyn_cast<clang::ParmVarDecl>(FD->getParamDecl(I));
	clang::QualType PQT = PVD->getOriginalType().getCanonicalType();
	if (PQT->isReferenceType() || PQT->isPointerType() || PQT->isMemberFunctionPointerType() || PQT->isArrayType() ) continue;
	uint64_t size_arg = ctx.getASTContext().getTypeSize(QT);
	uint64_t size_param = ctx.getASTContext().getTypeSize(PQT);

//	CE->dump();
//	(*I)->dump();
//	QT->dump();
//	llvm::errs()<<"size of arg :"<<size_arg.getQuantity()<<" bits\n";

	int64_t max_bits=64;
	if ( size_param > max_bits ) {
	  ExplodedNode *N = ctx.generateSink();
	  if (!N) continue;
	const Decl * D = ctx.getCurrentAnalysisDeclContext()->getDecl();
	std::string dname =""; 
	if (const NamedDecl * ND = llvm::dyn_cast<NamedDecl>(D)) dname = ND->getQualifiedNameAsString();
	std::string pname = PQT.getAsString();
	std::string bpname = "class boost::shared_ptr<";
	std::string cbpname = "const class boost::shared_ptr<";
	if ( pname.substr(0,bpname.length()) == bpname || pname.substr(0,cbpname.length()) == cbpname ) continue;
	  os<<"Argument passed by value with size of parameter '"<<size_param
		<<"' bits > max size '"<<max_bits
	  	<<"'\n";
	  llvm::errs()<<"Argument passed by value with size of parameter '"<<size_param
		<<"' bits > max size '"<<max_bits
		<<"' bits parameter type '"<<pname
	  	<<"' function '"<<FD->getQualifiedNameAsString()
		<<"' parent function '"<< dname
		<<"'\n";
	  BugType * BT = new BugType("Argument passed by value to parameter with size > max", "ArgSize");
//	  BugReport *report = new BugReport(*BT, os.str() , N);
	  BugReport *report = new BugReport(*BT, os.str() , ELoc);
	  ctx.emitReport(report);
//	  
	}
  }
}

void ArgSizeChecker::checkASTDecl(const CXXMethodDecl *MD, AnalysisManager& mgr,
                    BugReporter &BR) const {
       	const SourceManager &SM = BR.getSourceManager();
	CmsException m_exception;
       	PathDiagnosticLocation DLoc =PathDiagnosticLocation::createBegin( MD, SM );
	
 
//       	return;
	if ( ! m_exception.reportGeneral( DLoc, BR ) )  return;

	for (CXXMethodDecl::param_const_iterator I = MD->param_begin(), E = MD->param_end(); I!=E; I++) {
		llvm::SmallString<100> buf;
		llvm::raw_svector_ostream os(buf);
		QualType QT = (*I)->getOriginalType();
		if (QT->isIncompleteType()||QT->isDependentType()) continue;
		clang::QualType PQT = QT.getCanonicalType();
		if (PQT->isReferenceType() || PQT->isPointerType() || PQT->isMemberFunctionPointerType() || PQT->isArrayType() ) continue;
		uint64_t size_param = mgr.getASTContext().getTypeSize(PQT);
		int64_t max_bits=64;
		if ( size_param <= max_bits ) continue;
		std::string pname = PQT.getAsString();
		std::string bpname = "class boost::shared_ptr<";
		std::string cbpname = "const class boost::shared_ptr<";
		if ( pname.substr(0,bpname.length()) == bpname || pname.substr(0,cbpname.length()) == cbpname ) continue;
	  	os<<"Function parameter passed by value with size of parameter '"<<size_param
			<<"' bits > max size '"<<max_bits
	  		<<"'\n";
	  	llvm::errs()<< "Function parameter passed by value with size of parameter '"<<size_param
			<<"' bits > max size '"<<max_bits
			<<"' bits parameter type '"<<pname
	  		<<"' function '"<<MD->getQualifiedNameAsString()
			<<"'\n";
		BugType * BT = new BugType("Function parameter with size > max", "ArgSize");
	  	BugReport *report = new BugReport(*BT, os.str() , DLoc);
	  	BR.emitReport(report);
	}
} 


}
