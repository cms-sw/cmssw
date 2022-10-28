#include "PsetExistsFCallChecker.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>

using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

  class PEFWalker : public clang::StmtVisitor<PEFWalker> {
    const CheckerBase *Checker;
    clang::ento::BugReporter &BR;
    clang::AnalysisDeclContext *AC;

  public:
    PEFWalker(const CheckerBase *checker, clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac)
        : Checker(checker), BR(br), AC(ac) {}

    void VisitChildren(clang::Stmt *S);
    void VisitCallExpr(CallExpr *CE);
    void VisitStmt(clang::Stmt *S) { VisitChildren(S); }
    void VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *CE);
    void VisitCXXConstructExpr(CXXConstructExpr *CCE);
    void Report(const std::string &mname, const std::string &pname, const Expr *CE) const;
  };

  void PEFWalker::VisitChildren(clang::Stmt *S) {
    for (clang::Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I)
      if (clang::Stmt *child = *I) {
        Visit(child);
      }
  }

  void PEFWalker::VisitCXXConstructExpr(CXXConstructExpr *CCE) {
    CXXConstructorDecl *CCD = CCE->getConstructor();
    if (!CCD)
      return;
    const char *sfile = BR.getSourceManager().getPresumedLoc(CCE->getExprLoc()).getFilename();
    std::string sname(sfile);
    if (!support::isInterestingLocation(sname))
      return;
    std::string mname = support::getQualifiedName(*CCD);
    const NamedDecl *PD = llvm::dyn_cast_or_null<NamedDecl>(AC->getDecl());
    if (!PD)
      return;
    std::string pname = support::getQualifiedName(*PD);
    Report(mname, pname, CCE);

    VisitChildren(CCE);
  }

  void PEFWalker::VisitCXXMemberCallExpr(CXXMemberCallExpr *CE) {
    CXXMethodDecl *MD = CE->getMethodDecl();
    if (!MD)
      return;
    const NamedDecl *PD = llvm::dyn_cast_or_null<NamedDecl>(AC->getDecl());
    if (!PD)
      return;
    std::string mname = support::getQualifiedName(*MD);
    std::string pname = support::getQualifiedName(*PD);
    Report(mname, pname, CE);
  }

  void PEFWalker::VisitCallExpr(CallExpr *CE) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    LangOptions LangOpts;
    LangOpts.CPlusPlus = true;
    PrintingPolicy Policy(LangOpts);
    FunctionDecl *FD = CE->getDirectCallee();
    if (!FD)
      return;
    const char *sfile = BR.getSourceManager().getPresumedLoc(CE->getExprLoc()).getFilename();
    std::string sname(sfile);
    if (!support::isInterestingLocation(sname))
      return;
    std::string mname;
    mname = support::getQualifiedName(*FD);
    const NamedDecl *PD = llvm::dyn_cast_or_null<NamedDecl>(AC->getDecl());
    if (!PD)
      return;
    std::string pname = support::getQualifiedName(*PD);
    Report(mname, pname, CE);

    VisitChildren(CE);
  }

  void PEFWalker::Report(const std::string &mname, const std::string &pname, const Expr *CE) const {
    PathDiagnosticLocation CELoc = PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(), AC);
    std::string ename = "edm::ParameterSet::exists";
    std::string eaname = "edm::ParameterSet::existsAs";
    std::string pdname = "edm::ParameterDescription";
    llvm::SmallString<100> buf;
    llvm::raw_svector_ostream os(buf);
    if (mname.find(ename) != std::string::npos || mname.find(eaname) != std::string::npos) {
      if (pname.find(pdname) != std::string::npos)
        return;
      os << "function " << mname << " is called in function " << pname;
      BugType *BT = new BugType(
          Checker, "Function edm::ParameterSet::exists() or edm::ParameterSet::existsAs<>() called", "CMS code rules");
      std::unique_ptr<BasicBugReport> R = std::make_unique<BasicBugReport>(*BT, os.str(), CELoc);
      R->setDeclWithIssue(AC->getDecl());
      R->addRange(CE->getExprLoc());
      BR.emitReport(std::move(R));
      std::string tname = "function-checker.txt.unsorted";
      std::string ostring = "function '" + pname + "' calls function '" + mname + "'.\n";
      support::writeLog(ostring, tname);
    }
  }

  void PsetExistsFCallChecker::checkASTDecl(const CXXMethodDecl *MD, AnalysisManager &mgr, BugReporter &BR) const {
    const SourceManager &SM = BR.getSourceManager();
    PathDiagnosticLocation DLoc = PathDiagnosticLocation::createBegin(MD, SM);
    if (SM.isInSystemHeader(DLoc.asLocation()) || SM.isInExternCSystemHeader(DLoc.asLocation()))
      return;
    if (!MD->doesThisDeclarationHaveABody())
      return;
    clangcms::PEFWalker walker(this, BR, mgr.getAnalysisDeclContext(MD));
    walker.Visit(MD->getBody());
    return;
  }

  void PsetExistsFCallChecker::checkASTDecl(const FunctionTemplateDecl *TD,
                                            AnalysisManager &mgr,
                                            BugReporter &BR) const {
    const clang::SourceManager &SM = BR.getSourceManager();
    clang::ento::PathDiagnosticLocation DLoc = clang::ento::PathDiagnosticLocation::createBegin(TD, SM);
    if (SM.isInSystemHeader(DLoc.asLocation()) || SM.isInExternCSystemHeader(DLoc.asLocation()))
      return;

    for (auto I = TD->spec_begin(), E = TD->spec_end(); I != E; ++I) {
      if (I->doesThisDeclarationHaveABody()) {
        clangcms::PEFWalker walker(this, BR, mgr.getAnalysisDeclContext(*I));
        walker.Visit(I->getBody());
      }
    }
    return;
  }

}  // namespace clangcms
