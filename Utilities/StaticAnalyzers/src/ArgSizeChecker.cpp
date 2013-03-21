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

  llvm::SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);
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
   clang::ento::PathDiagnosticLocation::createBegin(CE, ctx.getSourceManager(),ctx.getCurrentAnalysisDeclContext());

  if ( ! m_exception.reportGeneral( ELoc, ctx.getBugReporter() ) )  return;

  for ( int I = 0, E=CE->getNumArgs(); I != E; ++I ) {
	QualType QT = CE->getArg(I)->getType();
	if (QT->isIncompleteType()) continue;
	const clang::ParmVarDecl *PVD=llvm::dyn_cast<clang::ParmVarDecl>(FD->getParamDecl(I));
	clang::QualType PQT = PVD->getOriginalType();
	if (PQT->isReferenceType() || PQT->isPointerType()) continue;
	CharUnits size_arg = ctx.getASTContext().getTypeSizeInChars(PQT);

//	CE->dump();
//	(*I)->dump();
//	QT->dump();
//	llvm::errs()<<"size of arg :"<<size_arg.getQuantity()<<" bits\n";

	int max_bits=64;
	if ( size_arg > CharUnits::fromQuantity(max_bits) && !QT->isPointerType() ) {
	  ExplodedNode *N = ctx.generateSink();
	  if (!N) continue;
	  os<<"Argument passed by value to paramater type '"<<PQT.getAsString();
	  os<<"' size of arg '"<<size_arg.getQuantity()<<"' bits > max size '"<<max_bits<<"' bits\n";
	  BugType * BT = new BugType("Argument passed by value with size > max", "ArgSize");
	  BugReport *report = new BugReport(*BT, os.str() , N);
	  ctx.emitReport(report);
	}
  }
}

}
