//== MutableMemberModificationChecker.cpp - Checks for accessing mutable members via const pointer --------------*- C++ -*--==//
//
// By Thomas Hauth [ Thomas.Hauth@cern.ch ], updated by Ivan Razumov <ivan.razumov@cern.ch>
//
//===----------------------------------------------------------------------===//

#include "MutableMemberModificationChecker.h"
#include <clang/AST/Decl.h>
#include <clang/AST/Type.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/ParentMap.h>
#include <clang/Analysis/AnalysisDeclContext.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>

namespace clangcms {
  void MutableMemberModificationChecker::checkPreStmt(const clang::MemberExpr *ME,
                                                      clang::ento::CheckerContext &C) const {
    // Common checks

    // == Filter out classes with "safe" names ==
    const auto *RD = llvm::dyn_cast<clang::CXXRecordDecl>(ME->getMemberDecl()->getDeclContext());
    if (RD) {
      std::string ClassName = RD->getNameAsString();
      if (support::isSafeClassName(ClassName)) {
        return;  // Skip checking for this class
      }
    }

    // == Check attributes ==
    const clang::FunctionDecl *FuncD = C.getLocationContext()->getStackFrame()->getDecl()->getAsFunction();
    const clang::AttrVec &Attrs = FuncD->getAttrs();
    for (const auto *A : Attrs) {
      if (clang::isa<clang::CMSThreadGuardAttr>(A) || clang::isa<clang::CMSThreadSafeAttr>(A) ||
          clang::isa<clang::CMSSaAllowAttr>(A)) {
        return;
      }
    }

    // == Check if this is a cmssw local file ==
    // Create a PathDiagnosticLocation for reporting
    clang::ento::PathDiagnosticLocation PathLoc =
        clang::ento::PathDiagnosticLocation::createBegin(ME, C.getSourceManager(), C.getLocationContext());

    // Get the BugReporter instance from the CheckerContext
    clang::ento::BugReporter &BR = C.getBugReporter();

    if (!m_exception.reportMutableMember(PathLoc, BR)) {
      return;
    }

    // == Only proceed if the member is mutable ==
    const auto *FD = llvm::dyn_cast<clang::FieldDecl>(ME->getMemberDecl());
    if (!FD || !FD->isMutable()) {
      return;  // Skip if it's not a mutable field
    }

    // == Skip non-private mutables, we deal with them elsewhere
    if (FD->getAccess() != clang::AS_private) {
      return;
    }

    // == Skip atomic mutables, these are thread-safe by design ==
    if (support::isStdAtomic(FD)) {
      return;  // Skip if it's a mutable std::atomic
    }

    // == Check if a field is marked with special attributes ==
    const clang::AttrVec &FAttrs = FD->getAttrs();
    for (const auto *A : FAttrs) {
      if (clang::isa<clang::CMSThreadGuardAttr>(A) || clang::isa<clang::CMSThreadSafeAttr>(A) ||
          clang::isa<clang::CMSSaAllowAttr>(A)) {
        return;
      }
    }

    // == Check if we are inside a const-qualified member function ==
    const auto *MethodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(FuncD);
    if (!MethodDecl || !MethodDecl->isConst()) {
      return;
    }

    // == Check if we are modifying the mutable ==
    bool ret;
    ret = checkAssignToMutable(ME, C, FuncD);
    if (!ret) {
      ret = checkCallNonConstOfMutable(ME, C);
    }

    if (ret) {
      if (RD) {
        std::string ClassName = RD->getNameAsString();
        std::string MemberName = ME->getMemberDecl()->getNameAsString();
        std::string FunctionName = MethodDecl->getNameAsString();
        std::string tname = "mutablemember-checker.txt.unsorted";
        std::string ostring = "flagged class '" + ClassName + "' modifying mutable member '" + MemberName +
                              "' in function '" + FunctionName + "'";
        support::writeLog(ostring, tname);
      }
    }
  }  // checkPreStmt

  // Check direct modifications of mutable (assign, compound stmt, increment/decrement)
  bool MutableMemberModificationChecker::checkAssignToMutable(const clang::MemberExpr *ME,
                                                              clang::ento::CheckerContext &C,
                                                              const clang::FunctionDecl *FuncD) const {
    // == Check if this is a modifying statement ==
    bool isModification = false;

    // Retrieve the parent statement of the MemberExpr
    const clang::LocationContext *LC = C.getLocationContext();
    const clang::ParentMap &PM = LC->getParentMap();
    const clang::Stmt *ParentStmt = PM.getParent(ME);

    if (!ParentStmt) {
      return false;
    }

    // Check if it is an assignment operator (binary operator)
    const auto *BO = llvm::dyn_cast<clang::BinaryOperator>(ParentStmt);
    if (BO) {
      const auto *LHSAsME = llvm::dyn_cast<clang::MemberExpr>(BO->getLHS());
      if (LHSAsME) {
        if (BO->isAssignmentOp() && LHSAsME == ME) {
          // The MemberExpr is on the left-hand side of an assignment
          isModification = true;
        }
      }
    }

    // Check if it is an overloaded assignment operator
    const auto *CO = llvm::dyn_cast<clang::CXXOperatorCallExpr>(ParentStmt);
    if (CO) {
      const auto *LHSAsME = llvm::dyn_cast<clang::MemberExpr>(CO->getArg(0));
      if (LHSAsME) {
        if (CO->isAssignmentOp() && LHSAsME == ME) {
          // The MemberExpr is on the left-hand side of an assignment
          isModification = true;
        }
      }
    }

    // Check for increment/decrement
    if (const auto *UO = llvm::dyn_cast<clang::UnaryOperator>(ParentStmt)) {
      if (UO->isIncrementDecrementOp() && UO->getSubExpr() == ME) {
        isModification = true;
      }
    }

    if (!isModification) {
      return false;
    }

    // == Report a bug if none of the above conditions allow access. ==
    std::string MutableMemberName = ME->getMemberDecl()->getQualifiedNameAsString();
    if (!BT) {
      BT = std::make_unique<clang::ento::BugType>(
          this, "Mutable member modification in const member function", "ConstThreadSafety");
    }
    std::string Description =
        "Modifying mutable member '" + MutableMemberName + "' in const member function is potentially thread-unsafe ";
    auto Report = std::make_unique<clang::ento::PathSensitiveBugReport>(*BT, Description, C.generateErrorNode());
    Report->addRange(ME->getSourceRange());
    C.emitReport(std::move(Report));
    return true;
  }  // checkAssignToMutable

  // Check for indirect modifications of mutable (calling non-const method)
  bool MutableMemberModificationChecker::checkCallNonConstOfMutable(const clang::MemberExpr *ME,
                                                                    clang::ento::CheckerContext &C) const {
    // Traverse upwards to check if the MemberExpr is part of a CXXMemberCallExpr
    const clang::Expr *E = ME;
    while (E) {
      if (const clang::CXXMemberCallExpr *Call = llvm::dyn_cast<clang::CXXMemberCallExpr>(E->IgnoreParenCasts())) {
        const clang::CXXMethodDecl *CalledMethod = Call->getMethodDecl();
        if (CalledMethod && !CalledMethod->isConst()) {
          // Get the name of the mutable member
          std::string MutableMemberName = ME->getMemberDecl()->getQualifiedNameAsString();

          // Get the name of the called method
          std::string CalledMethodName = CalledMethod->getQualifiedNameAsString();
          // Report an issue
          if (!BT) {
            BT = std::make_unique<clang::ento::BugType>(
                this, "Mutable member modification in const member function", "ConstThreadSafety");
          }
          std::string Description = "Calling non-const method '" + CalledMethodName + "' of mutable member '" +
                                    MutableMemberName + "' in a const member function is potentially thread-unsafe.";
          auto Report = std::make_unique<clang::ento::PathSensitiveBugReport>(*BT, Description, C.generateErrorNode());
          Report->addRange(ME->getSourceRange());
          C.emitReport(std::move(Report));
          return true;
        }
      }
      // Move up to the parent expression
      const clang::Stmt *ParentStmt = C.getLocationContext()->getParentMap().getParent(E);
      E = llvm::dyn_cast_or_null<clang::Expr>(ParentStmt);
    }
    return false;
  }  // checkCallNonConstOfMutable

}  // namespace clangcms
