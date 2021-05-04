#include "ESRecordGetChecker.h"
using namespace clang;
using namespace ento;
using namespace llvm;

namespace clangcms {

  class ESRWalker : public clang::StmtVisitor<ESRWalker> {
    const CheckerBase *Checker;
    clang::ento::BugReporter &BR;
    clang::AnalysisDeclContext *AC;

  public:
    ESRWalker(const CheckerBase *checker, clang::ento::BugReporter &br, clang::AnalysisDeclContext *ac)
        : Checker(checker), BR(br), AC(ac) {}

    void VisitChildren(clang::Stmt *S);
    void VisitStmt(clang::Stmt *S) { VisitChildren(S); }
    void VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *CE);
  };

  void ESRWalker::VisitChildren(clang::Stmt *S) {
    for (clang::Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I)
      if (clang::Stmt *child = *I) {
        Visit(child);
      }
  }

  void ESRWalker::VisitCXXMemberCallExpr(CXXMemberCallExpr *CE) {
    LangOptions LangOpts;
    LangOpts.CPlusPlus = true;
    PrintingPolicy Policy(LangOpts);
    const Decl *D = AC->getDecl();
    std::string dname = "";
    if (const NamedDecl *ND = llvm::dyn_cast_or_null<NamedDecl>(D))
      dname = ND->getQualifiedNameAsString();
    if (dname == "edm::eventsetup::EventSetupRecord::get")
      return;
    CXXMethodDecl *MD = CE->getMethodDecl();
    if (!MD)
      return;
    std::string mname = MD->getQualifiedNameAsString();
    if (mname == "edm::eventsetup::EventSetupRecord::get") {
      std::string mname = MD->getQualifiedNameAsString();
      llvm::SmallString<100> buf;
      llvm::raw_svector_ostream os(buf);
      os << "function '";
      llvm::dyn_cast<FunctionDecl>(D)->getNameForDiagnostic(os, Policy, true);
      os << "' ";
      os << "calls function '";
      MD->getNameForDiagnostic(os, Policy, true);
      for (auto I = CE->arg_begin(), E = CE->arg_end(); I != E; ++I) {
        QualType QT = (*I)->getType();
        std::string qtname = QT.getAsString();
        os << "' with argument of type '" << qtname;
      }
      os << "'";
      os << ". Direct call of function EventSetupRecord::get(ESHandle&) is deprecated and should be replaced with a "
            "call to EventSetup::getHandle(ESGetToken&). To use ESGetToken see "
            "https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideHowToGetDataFromES#In_ED_module To get data with "
            "the token see "
            "https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideHowToGetDataFromES#Getting_data_from_EventSetup_wit";
      PathDiagnosticLocation CELoc = PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(), AC);
      BugType *BT = new BugType(Checker, "EventSetupRecord::get function called", "ThreadSafety");
      std::unique_ptr<BasicBugReport> R = std::make_unique<BasicBugReport>(*BT, llvm::StringRef(os.str()), CELoc);
      R->addRange(CE->getSourceRange());
      BR.emitReport(std::move(R));
    }
  }

  void ESRGetChecker::checkASTDecl(const CXXMethodDecl *MD, AnalysisManager &mgr, BugReporter &BR) const {
    const SourceManager &SM = BR.getSourceManager();
    PathDiagnosticLocation DLoc = PathDiagnosticLocation::createBegin(MD, SM);
    if (SM.isInSystemHeader(DLoc.asLocation()) || SM.isInExternCSystemHeader(DLoc.asLocation()))
      return;
    if (!MD->doesThisDeclarationHaveABody())
      return;
    ESRWalker walker(this, BR, mgr.getAnalysisDeclContext(MD));
    walker.Visit(MD->getBody());
    return;
  }

  void ESRGetChecker::checkASTDecl(const FunctionTemplateDecl *TD, AnalysisManager &mgr, BugReporter &BR) const {
    const clang::SourceManager &SM = BR.getSourceManager();
    clang::ento::PathDiagnosticLocation DLoc = clang::ento::PathDiagnosticLocation::createBegin(TD, SM);
    if (SM.isInSystemHeader(DLoc.asLocation()) || SM.isInExternCSystemHeader(DLoc.asLocation()))
      return;

    for (auto I = TD->spec_begin(), E = TD->spec_end(); I != E; ++I) {
      if (I->doesThisDeclarationHaveABody()) {
        ESRWalker walker(this, BR, mgr.getAnalysisDeclContext(*I));
        walker.Visit(I->getBody());
      }
    }
    return;
  }

  void ESRGetChecker::checkASTDecl(const FunctionDecl *FD, AnalysisManager &mgr, BugReporter &BR) const {
    const SourceManager &SM = BR.getSourceManager();
    PathDiagnosticLocation DLoc = PathDiagnosticLocation::createBegin(FD, SM);
    if (SM.isInSystemHeader(DLoc.asLocation()) || SM.isInExternCSystemHeader(DLoc.asLocation()))
      return;
    if (!FD->doesThisDeclarationHaveABody())
      return;
    ESRWalker walker(this, BR, mgr.getAnalysisDeclContext(FD));
    walker.Visit(FD->getBody());
    return;
  }

}  // namespace clangcms
