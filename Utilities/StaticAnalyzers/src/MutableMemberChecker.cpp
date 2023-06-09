//== MutableMemberChecker.cpp - Checks for mutable members --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#include "MutableMemberChecker.h"
#include <clang/AST/Attr.h>
using namespace clang;
using namespace ento;
using namespace llvm;
namespace clangcms {

  void MutableMemberChecker::checkASTDecl(const clang::FieldDecl *D,
                                          clang::ento::AnalysisManager &Mgr,
                                          clang::ento::BugReporter &BR) const {
    if (D->hasAttr<CMSThreadGuardAttr>() || D->hasAttr<CMSThreadSafeAttr>() || D->hasAttr<CMSSaAllowAttr>())
      return;
    if (D->isMutable() && D->getDeclContext()->isRecord()) {
      clang::QualType t = D->getType();
      clang::ento::PathDiagnosticLocation DLoc = clang::ento::PathDiagnosticLocation::create(D, BR.getSourceManager());

      if (!m_exception.reportMutableMember(t, DLoc, BR))
        return;
      std::string mname = t.getCanonicalType().getAsString();
      if (support::isSafeClassName(mname))
        return;
      std::string buf;
      llvm::raw_string_ostream os(buf);
      std::string pname = D->getParent()->getQualifiedNameAsString();
      os << "Mutable member '" << D->getQualifiedNameAsString() << "' in class '" << pname
         << "', might be thread-unsafe when accessing via a const pointer.";
      if (!BT)
        BT = std::make_unique<clang::ento::BugType>(
            this, "mutable member if accessed via const pointer", "ConstThreadSafety");
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

}  // namespace clangcms
