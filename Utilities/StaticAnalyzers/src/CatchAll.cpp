//==-- CatchAll.cpp - Checks for using namespace and using std:: in headers --------------*- C++ -*--==//
//
// by Shahzad Malik MUZAFFAR [ Shahzad.Malik.MUZAFFAR@cern.ch ]
//
//===----------------------------------------------------------------------===//

#include "CatchAll.h"
#include "clang/Basic/SourceManager.h"
#include "CmsSupport.h"
using namespace clangcms;

void CatchAll::checkASTCodeBody(const clang::Decl*& D, clang::ento::AnalysisManager& AM, clang::ento::BugReporter& BR) const
{
  const char *sfile=BR.getSourceManager().getPresumedLoc(D->getLocation()).getFilename();
  if ((!sfile) || (!support::isCmsLocalFile(sfile))) return;
  const clang::Stmt* s=D->getBody();
  if (!s) return;
  s=process(s);
  if (!s) return;
  clang::ento::LocationOrAnalysisDeclContext x(AM.getAnalysisDeclContext(D));
  clang::ento::PathDiagnosticLocation DLoc = clang::ento::PathDiagnosticLocation::createBegin(s, BR.getSourceManager(),x);
  BR.EmitBasicReport(D, "'catch(...)' in sources","CMS code rules","using 'catch(...)' is forbidden", DLoc,s->getSourceRange ());
  
}

const clang::Stmt* CatchAll::process(const clang::Stmt* S) const
{
  if (clang::CXXCatchStmt::classof(S) && 
      checkCatchAll(static_cast<const clang::CXXCatchStmt*>(S)))
    return S;
  clang::Stmt::const_child_iterator b=S->child_begin ();
  clang::Stmt::const_child_iterator e=S->child_end ();
  const clang::Stmt* catchAll = 0;
  while(b!=e)
  {
    if (*b)
    {
      catchAll = process(*b);
      if (catchAll!=0) break;
    }
    b++;
  }
  return catchAll;
}
