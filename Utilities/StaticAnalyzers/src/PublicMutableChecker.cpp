#include <clang/AST/ASTContext.h>
#include <clang/AST/DeclCXX.h>
#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include "PublicMutableChecker.h"
#include "CmsSupport.h"

namespace clangcms {
  void PublicMutableChecker::checkASTDecl(const clang::FieldDecl *D,
                                          clang::ento::AnalysisManager &Mgr,
                                          clang::ento::BugReporter &BR) const {
    if (D->hasAttr<clang::CMSThreadGuardAttr>() || D->hasAttr<clang::CMSThreadSafeAttr>() ||
        D->hasAttr<clang::CMSSaAllowAttr>())
      return;
    if (D->isMutable() && D->getDeclContext()->isRecord() && D->getAccess() != clang::AS_private) {
      clang::QualType t = D->getType();
      clang::ento::PathDiagnosticLocation DLoc = clang::ento::PathDiagnosticLocation::create(D, BR.getSourceManager());

      if (!m_exception.reportMutableMember(DLoc, BR))
        return;
      std::string mname = t.getCanonicalType().getAsString();
      if (support::isSafeClassName(mname))
        return;
      std::string buf;
      llvm::raw_string_ostream os(buf);
      std::string pname = D->getParent()->getQualifiedNameAsString();
      os << "Class '" << pname << "' has publically-accessible mutable member '" << D->getQualifiedNameAsString()
         << "', this is not allowed.";
      if (!BT)
        BT = std::make_unique<clang::ento::BugType>(this, "public mutable member", "ConstThreadSafety");
      std::unique_ptr<clang::ento::BasicBugReport> R =
          std::make_unique<clang::ento::BasicBugReport>(*BT, llvm::StringRef(os.str()), DLoc);
      R->setDeclWithIssue(D);
      R->addRange(D->getSourceRange());
      BR.emitReport(std::move(R));
      std::string tname = "mutablemember-checker.txt.unsorted";
      std::string ostring = "flagged class '" + pname + "' mutable member '" + D->getQualifiedNameAsString() + "'";
      support::writeLog(ostring, tname);
    }
  }
};  // namespace clangcms
