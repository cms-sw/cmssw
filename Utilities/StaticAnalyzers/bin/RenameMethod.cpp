//=- examples/rename-method/RenameMethod.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A small example tool that uses AST matchers to find calls to the Get() method
// in subclasses of ElementsBase and replaces them with calls to Front().
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/Support/CommandLine.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

cl::opt<std::string> BuildPath(
  cl::Positional,
  cl::desc("<build-path>"));

cl::list<std::string> SourcePaths(
  cl::Positional,
  cl::desc("<source0> [... <sourceN>]"),
  cl::OneOrMore);

// Implements a callback that replaces the calls for the AST
// nodes we matched.
class CallRenamer : public MatchFinder::MatchCallback {
public:
  CallRenamer(Replacements *Replace) : Replace(Replace) {}

  // This method is called every time the registered matcher matches
  // on the AST.
	virtual void run(const MatchFinder::MatchResult &Result) {
    const MemberExpr *M = Result.Nodes.getStmtAs<MemberExpr>("member");
    // We can assume M is non-null, because the ast matchers guarantee
    // that a node with this type was bound, as the matcher would otherwise
    // not match.

    Replace->insert(
      // Replacements are a source manager independent way to express
      // transformation on the source.
      Replacement(*Result.SourceManager,
                  // Replace the range of the member name...
                  CharSourceRange::getTokenRange(
                    SourceRange(M->getMemberLoc())),
                  // ... with "Front".
                  "Front"));
  }

private:
  // Replacements are the RefactoringTool's way to keep track of code
  // transformations, deduplicate them and apply them to the code when
  // the tool has finished with all translation units.
  Replacements *Replace;
};

// Implements a callback that replaces the decls for the AST
// nodes we matched.
class DeclRenamer : public MatchFinder::MatchCallback {
public:
  DeclRenamer(Replacements *Replace) : Replace(Replace) {}

  // This method is called every time the registered matcher matches
  // on the AST.
	virtual void run(const MatchFinder::MatchResult &Result) {
    const CXXMethodDecl *D = Result.Nodes.getDeclAs<CXXMethodDecl>("method");
    // We can assume D is non-null, because the ast matchers guarantee
    // that a node with this type was bound, as the matcher would otherwise
    // not match.

    Replace->insert(
      // Replacements are a source manager independent way to express
      // transformation on the source.
      Replacement(*Result.SourceManager,
                  // Replace the range of the declarator identifier...
                  CharSourceRange::getTokenRange(
                    SourceRange(D->getLocation())),
                  // ... with "Front".
                  "Front"));
  }

private:
  // Replacements are the RefactoringTool's way to keep track of code
  // transformations, deduplicate them and apply them to the code when
  // the tool has finished with all translation units.
  Replacements *Replace;
};

int main(int argc, const char **argv) {
  // First see if we can create the compile command line from the
  // positional parameters after "--".
  OwningPtr<CompilationDatabase> Compilations(
    FixedCompilationDatabase::loadFromCommandLine(argc, argv));

  // Do normal command line handling from the rest of the arguments.
  cl::ParseCommandLineOptions(argc, argv);

  if (!Compilations) {
    // If the caller didn't specify a compile command line to use, try to
    // load it from a build directory. For example when running cmake, use
    // CMAKE_EXPORT_COMPILE_COMMANDS=ON to prepare your build directory to
    // be useable with clang tools.
    std::string ErrorMessage;
    Compilations.reset(CompilationDatabase::loadFromDirectory(BuildPath,
                                                              ErrorMessage));
    if (!Compilations)
      llvm::report_fatal_error(ErrorMessage);
  }

  RefactoringTool Tool(*Compilations, SourcePaths);
  ast_matchers::MatchFinder Finder;
  CallRenamer CallCallback(&Tool.getReplacements());
  Finder.addMatcher(
    // Match calls...
    memberCallExpr(
      // Where the callee is a method called "Get"...
      callee(methodDecl(hasName("Get"))),
      // ... and the class on which the method is called is derived
      // from ElementsBase ...
      thisPointerType(recordDecl(
        isDerivedFrom("ElementsBase"))),
      // ... and bind the member expression to the ID "member", under which
      // it can later be found in the callback.
      callee(id("member", memberExpr()))),
    &CallCallback);

  DeclRenamer DeclCallback(&Tool.getReplacements());
  Finder.addMatcher(
    // Match declarations...
    id("method", methodDecl(hasName("Get"),
                        ofClass(isDerivedFrom("ElementsBase")))),
    &DeclCallback);

  return Tool.run(newFrontendActionFactory(&Finder));
}
