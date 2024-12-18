#include "UnnecessaryMutableChecker.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Path.h"
#include "CmsSupport.h"

namespace clangcms {

  // Find all classes/structs
  void UnnecessaryMutableChecker::checkASTDecl(const clang::CXXRecordDecl *RD,
                                               clang::ento::AnalysisManager &Mgr,
                                               clang::ento::BugReporter &BR) const {
    if (!RD->hasDefinition())
      return;

    // == Only process main file (one passed on the command line) ==
    const clang::SourceManager &SM = Mgr.getASTContext().getSourceManager();
    auto MainFileID = SM.getMainFileID();
    if (MainFileID.isInvalid()) {
      llvm::errs() << "Main file is invalid, skipping\n";
      return;
    }
    std::string MainFileStr = SM.getFileEntryRefForID(MainFileID)->getName().str();  // Main file name
    // Get the base name of the main file (without path or extension)
    std::string BaseNameStr =
        llvm::sys::path::parent_path(MainFileStr).str() + "/" + llvm::sys::path::stem(MainFileStr).str();
    clang::StringRef CurrentFile = SM.getFilename(RD->getLocation());
    if (!CurrentFile.ends_with(BaseNameStr + ".cpp") && !CurrentFile.ends_with(BaseNameStr + ".h")) {
      return;
    }

    // == Filter out classes with "safe" names ==
    std::string ClassName = RD->getNameAsString();
    if (support::isSafeClassName(ClassName)) {
      return;  // Skip checking for this class
    }

    // == Check if this is a cmssw local file ==
    clang::ento::PathDiagnosticLocation PathLoc = clang::ento::PathDiagnosticLocation::createBegin(RD, SM);

    if (!m_exception.reportMutableMember(PathLoc, BR)) {
      return;
    }

    // == Iterate over fields ==
    for (const auto *Field : RD->fields()) {
      if (!Field->isMutable()) {
        continue;
      }
      // == Skip atmoic mutables, these are thread-safe by design ==
      if (support::isStdAtomic(Field)) {
        return;  // Skip if it's a mutable std::atomic
      }

      // == Check if a field is marked with special attributes ==
      const clang::AttrVec &FAttrs = Field->getAttrs();
      for (const auto *A : FAttrs) {
        if (clang::isa<clang::CMSThreadGuardAttr>(A) || clang::isa<clang::CMSThreadSafeAttr>(A) ||
            clang::isa<clang::CMSSaAllowAttr>(A)) {
          return;
        }
      }

      // == Check for modifications ==
      if (!isMutableMemberModified(Field, RD)) {
        clang::SourceLocation Loc = Field->getLocation();
        if (Loc.isValid()) {
          if (!BT) {
            BT = std::make_unique<clang::ento::BugType>(this, "Unnecessarily Mutable Member", "Coding Practices");
          }
          BR.EmitBasicReport(
              Field,
              this,
              "Useless mutable field",
              "ConstThreadSafety",
              "The mutable field '" + Field->getQualifiedNameAsString() + "' is not modified in any const methods",
              PathLoc);
        }
      }
    }
  }

  bool UnnecessaryMutableChecker::isMutableMemberModified(const clang::FieldDecl *Field,
                                                          const clang::CXXRecordDecl *RD) const {
    for (const auto *Method : RD->methods()) {
      if (!Method->isConst())
        continue;

      if (const auto *Body = Method->getBody()) {
        for (const auto *Stmt : Body->children()) {
          if (Stmt) {
            if (analyzeStmt(Stmt, Field))
              return true;
          }
        }
      }
    }

    return false;
  }

  bool UnnecessaryMutableChecker::analyzeStmt(const clang::Stmt *S, const clang::FieldDecl *Field) const {
    if (const auto *UnaryOp = clang::dyn_cast<clang::UnaryOperator>(S)) {  // x++, x--, ++x, --x
      if (UnaryOp->isIncrementDecrementOp()) {
        if (const auto *ME = clang::dyn_cast<clang::MemberExpr>(UnaryOp->getSubExpr())) {
          if (const auto *FD = clang::dyn_cast<clang::FieldDecl>(ME->getMemberDecl())) {
            if (FD == Field) {
              return true;
            }
          }
        }
      }
    } else if (const auto *BinaryOp = clang::dyn_cast<clang::BinaryOperator>(S)) {  // x = y
      if (BinaryOp->isAssignmentOp() || BinaryOp->isCompoundAssignmentOp()) {
        if (const auto *LHS = clang::dyn_cast<clang::MemberExpr>(BinaryOp->getLHS())) {
          if (const auto *FD = clang::dyn_cast<clang::FieldDecl>(LHS->getMemberDecl())) {
            if (FD == Field) {
              return true;
            }
          }
        }
      }
    } else if (const auto *Call = clang::dyn_cast<clang::CXXMemberCallExpr>(S)) {  // x.const_method()
      if (const auto *Callee = Call->getMethodDecl()) {
        if (!Callee->isConst()) {
          if (const auto *ImplicitObj = Call->getImplicitObjectArgument()) {
            if (const auto *ME = clang::dyn_cast<clang::MemberExpr>(ImplicitObj)) {
              if (const auto *FD = clang::dyn_cast<clang::FieldDecl>(ME->getMemberDecl())) {
                if (FD == Field) {
                  return true;
                }
              }
            }
          }
        }
      }
    } else if (const auto *OpCall = clang::dyn_cast<clang::CXXOperatorCallExpr>(S)) {  // x.operator=()
      if (OpCall->isAssignmentOp()) {
        if (const auto *ME = llvm::dyn_cast<clang::MemberExpr>(OpCall->getArg(0))) {
          if (const auto *FD = clang::dyn_cast<clang::FieldDecl>(ME->getMemberDecl())) {
            if (FD == Field) {
              return true;
            }
          }
        }
      }
    }

    // Recursively analyze child statements
    for (const auto *Child : S->children()) {
      if (Child) {
        if (analyzeStmt(Child, Field)) {
          return true;
        }
      }
    }

    return false;
  }
}  // namespace clangcms
