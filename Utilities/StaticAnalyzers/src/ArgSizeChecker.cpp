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
#include <clang/AST/ParentMap.h>

#include "CmsSupport.h"
#include <iostream>
using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {


void ArgSizeChecker::checkPreStmt(const CXXConstructExpr *E, CheckerContext &ctx) const
{

  CmsException m_exception;
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

 const clang::ento::PathDiagnosticLocation ELoc =
   clang::ento::PathDiagnosticLocation::createBegin(E, ctx.getSourceManager(),ctx.getLocationContext());

  if ( ! m_exception.reportGeneral( ELoc, ctx.getBugReporter() ) )  return;
  	llvm::SmallString<100> buf;
  	llvm::raw_svector_ostream os(buf);

//	llvm::errs()<<E->getStmtClassName()<<";\t";
//	E->dumpPretty(ctx.getASTContext());
//	llvm::errs()<<"\n";
//	E->dump();
//	llvm::errs()<<"\n";

	for (clang::Stmt::const_child_iterator I = E->child_begin(), F = E->child_end(); I!=F; ++I) {
		const Expr * child = llvm::dyn_cast<Expr>(*I);
		if (! child) continue;
		if ( llvm::isa<DeclRefExpr>(child->IgnoreImpCasts())) {
//			(*I)->dump();
//			llvm::errs()<<"\n";
			const NamedDecl * ND = llvm::cast<DeclRefExpr>(child->IgnoreImpCasts())->getFoundDecl();
			if ( llvm::isa<ParmVarDecl>(ND)) 
				{
//				ND->dump();
//				llvm::errs()<<"\n";
				const ParmVarDecl * PVD = llvm::cast<ParmVarDecl>(ND);
				QualType QT = PVD->getOriginalType();
//				QT.dump();
//				llvm::errs()<<"\n";
				if (QT->isIncompleteType()||QT->isDependentType()) continue;
				clang::QualType PQT = QT.getCanonicalType();
				PQT.removeLocalConst();
				if (PQT->isReferenceType() || PQT->isPointerType() 
					|| PQT->isMemberFunctionPointerType() || PQT->isArrayType()
					|| PQT->isBuiltinType() || PQT->isUnionType() || PQT->isVectorType() ) continue;
				uint64_t size_param = ctx.getASTContext().getTypeSize(PQT);
				int64_t max_bits=128;
				if ( size_param <= max_bits ) continue;
				std::string qname = QT.getAsString();
				std::string pname = PQT.getAsString();
				std::string bpname = "class boost::";
				std::string cbpname = "const class boost::";
				std::string sfname = "class std::function<";
				std::string ehname = "class edm::Handle<";
				std::string cehname = "const class edm::Handle<";
				std::string xname = "class xoap::";
				std::string ername = "class edm::Ptr<";
				std::string epname = "class edm::Ref<";
				std::string cername = "const class edm::Ptr<";
				std::string cepname = "const class edm::Ref<";
				std::string erviname = "class edm::RefVectorIterator<";
				const CXXMethodDecl * MD = llvm::dyn_cast<CXXMethodDecl>(ctx.getCurrentAnalysisDeclContext()->getDecl()) ;
//				if ( pname.substr(0,bpname.length()) == bpname || pname.substr(0,cbpname.length()) == cbpname 
//					|| pname.substr(0,ehname.length()) == ehname || pname.substr(0,cehname.length()) == cehname
//					|| pname.substr(0,epname.length()) == epname || pname.substr(0,cepname.length()) == cepname
//					|| pname.substr(0,ername.length()) == ername || pname.substr(0,cername.length()) == cername
//					|| pname.substr(0,sfname.length()) == sfname || pname.substr(0,xname.length()) == xname 
//					|| pname.substr(0,erviname.length()) == erviname ) continue;
				os<<"Function parameter copied by value with size '"<<size_param
					<<"' bits > max size '"<<max_bits
					<<"' bits parameter type '"<<pname
					<<"' function '";
				std::string fname = MD->getNameAsString();
				if (MD) { os<< fname <<"' class '"<< MD->getParent()->getNameAsString(); }
				os << "'\n";

				std::string oname = "operator"; 
//				if ( fname.substr(0,oname.length()) == oname ) continue;

				const clang::ento::PathDiagnosticLocation DLoc =
			   		clang::ento::PathDiagnosticLocation::createBegin(PVD, ctx.getSourceManager());

				BugType * BT = new BugType("Function parameter copied by value with size > max","ArgSize");
				BugReport *report = new BugReport(*BT, os.str() , DLoc);
				report->addRange(PVD->getSourceRange());
	 			ctx.emitReport(report);
				}	
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
		PQT.removeLocalConst();
		if (PQT->isReferenceType() || PQT->isPointerType() || PQT->isMemberFunctionPointerType() 
			|| PQT->isArrayType()|| PQT->isBuiltinType() || PQT->isUnionType() || PQT->isVectorType()  ) continue;
		uint64_t size_param = mgr.getASTContext().getTypeSize(PQT);
		int64_t max_bits=128;
		if ( size_param <= max_bits ) continue;
				std::string qname = QT.getAsString();
		std::string pname = PQT.getAsString();
		std::string bpname = "class boost::";
		std::string cbpname = "const class boost::";
		std::string sfname = "class std::function<";
		std::string ehname = "class edm::Handle<";
		std::string cehname = "const class edm::Handle<";
		std::string xname = "class xoap::";
		std::string ername = "class edm::Ptr<";
		std::string epname = "class edm::Ref<";
		std::string cername = "const class edm::Ptr<";
		std::string cepname = "const class edm::Ref<";
		std::string erviname = "class edm::RefVectorIterator<";

//		if ( pname.substr(0,bpname.length()) == bpname || pname.substr(0,cbpname.length()) == cbpname 
//			|| pname.substr(0,ehname.length()) == ehname || pname.substr(0,cehname.length()) == cehname
//			|| pname.substr(0,epname.length()) == epname || pname.substr(0,cepname.length()) == cepname
//			|| pname.substr(0,ername.length()) == ername || pname.substr(0,cername.length()) == cername
//			|| pname.substr(0,sfname.length()) == sfname || pname.substr(0,xname.length()) == xname
//			|| pname.substr(0,erviname.length()) == erviname ) continue;
		std::string fname = MD->getNameAsString();
	  	os<<"Function parameter passed by value with size of parameter '"<<size_param
			<<"' bits > max size '"<<max_bits
//	  		<<"'\n";
//	  	llvm::errs()<< "Function parameter passed by value with size of parameter '"<<size_param
//			<<"' bits > max size '"<<max_bits
			<<"' bits parameter type '"<<pname
	  		<<"' function '"<<fname
	  		<<"' class '"<<MD->getParent()->getNameAsString()
			<<"'\n";
		std::string oname = "operator"; 
//		if ( fname.substr(0,oname.length()) == oname ) continue;

		BugType * BT = new BugType("Function parameter with size > max", "ArgSize");
	  	BugReport *report = new BugReport(*BT, os.str() , DLoc);
	  	BR.emitReport(report);
	}
} 


}
