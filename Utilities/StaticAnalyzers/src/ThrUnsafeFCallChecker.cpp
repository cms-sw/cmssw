#include "ThrUnsafeFCallChecker.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>

using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

  class TUFWalker : public clang::StmtVisitor<TUFWalker> {
    const CheckerBase *Checker;
    clang::ento::BugReporter &BR;
    clang::AnalysisDeclContext *AC;

  public:
    TUFWalker(const CheckerBase *checker, clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac)
        : Checker(checker), BR(br), AC(ac) {}

    void VisitChildren(clang::Stmt *S);
    void VisitStmt(clang::Stmt *S) { VisitChildren(S); }
    void VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *CE);
  };

  void TUFWalker::VisitChildren(clang::Stmt *S) {
    for (clang::Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I)
      if (clang::Stmt *child = *I) {
        Visit(child);
      }
  }

  void TUFWalker::VisitCXXMemberCallExpr(CXXMemberCallExpr *CE) {
    CXXMethodDecl *MD = CE->getMethodDecl();
    if (!MD)
      return;
    const CXXMethodDecl *PD = llvm::dyn_cast_or_null<CXXMethodDecl>(AC->getDecl());
    if (!PD)
      return;
    std::string mname = support::getQualifiedName(*MD);
    std::string pname = support::getQualifiedName(*PD);
    llvm::SmallString<100> buf;
    llvm::raw_svector_ostream os(buf);
    const std::string tfname = "TFileService::";
    const std::string eoname = "edm::one";
    if (mname.substr(0, tfname.length()) == tfname) {
      for (auto I = PD->begin_overridden_methods(), E = PD->end_overridden_methods(); I != E; ++I) {
        std::string oname = support::getQualifiedName(*(*I));
        if (oname.substr(0, eoname.length()) != eoname) {
          os << "TFileService function " << mname << " is called in function " << pname;
          PathDiagnosticLocation CELoc = PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(), AC);
          BugType *BT = new BugType(Checker, "TFileService function called ", "ThreadSafety");
          std::unique_ptr<BasicBugReport> R = std::make_unique<BasicBugReport>(*BT, os.str(), CELoc);
          R->setDeclWithIssue(AC->getDecl());
          R->addRange(CE->getSourceRange());
          BR.emitReport(std::move(R));
          std::string tname = "function-checker.txt.unsorted";
          std::string ostring = "function '" + pname + "' known thread unsafe function '" + mname + "'.\n";
          support::writeLog(ostring, tname);
        }
      }
    } else if (support::isKnownThrUnsafeFunc(mname)) {
      os << "Known thread unsafe function " << mname << " is called in function " << pname;
      PathDiagnosticLocation CELoc = PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(), AC);
      BugType *BT = new BugType(Checker, "known thread unsafe function called", "ThreadSafety");
      std::unique_ptr<BasicBugReport> R = std::make_unique<BasicBugReport>(*BT, os.str(), CELoc);
      R->setDeclWithIssue(AC->getDecl());
      R->addRange(CE->getSourceRange());
      BR.emitReport(std::move(R));
      std::string tname = "function-checker.txt.unsorted";
      std::string ostring = "function '" + pname + "' known thread unsafe function '" + mname + "'.\n";
      support::writeLog(ostring, tname);
    }
  }

  void ThrUnsafeFCallChecker::checkASTDecl(const CXXMethodDecl *MD, AnalysisManager &mgr, BugReporter &BR) const {
    const SourceManager &SM = BR.getSourceManager();
    PathDiagnosticLocation DLoc = PathDiagnosticLocation::createBegin(MD, SM);
    if (SM.isInSystemHeader(DLoc.asLocation()) || SM.isInExternCSystemHeader(DLoc.asLocation()))
      return;
    if (!MD->doesThisDeclarationHaveABody())
      return;
    clangcms::TUFWalker walker(this, BR, mgr.getAnalysisDeclContext(MD));
    walker.Visit(MD->getBody());
    return;
  }

  void ThrUnsafeFCallChecker::checkASTDecl(const FunctionTemplateDecl *TD,
                                           AnalysisManager &mgr,
                                           BugReporter &BR) const {
    const clang::SourceManager &SM = BR.getSourceManager();
    clang::ento::PathDiagnosticLocation DLoc = clang::ento::PathDiagnosticLocation::createBegin(TD, SM);
    if (SM.isInSystemHeader(DLoc.asLocation()) || SM.isInExternCSystemHeader(DLoc.asLocation()))
      return;

    for (auto I = TD->spec_begin(), E = TD->spec_end(); I != E; ++I) {
      if (I->doesThisDeclarationHaveABody()) {
        clangcms::TUFWalker walker(this, BR, mgr.getAnalysisDeclContext(*I));
        walker.Visit(I->getBody());
      }
    }
    return;
  }

}  // namespace clangcms
