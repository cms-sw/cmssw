//== CmsException.cpp -                             --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ] and Patrick Gartung
//
//===----------------------------------------------------------------------===//

#include "CmsException.h"
#include "CmsSupport.h"

namespace clangcms {

  bool CmsException::reportGeneral(clang::ento::PathDiagnosticLocation const& path,
                                   clang::ento::BugReporter& BR) const {
    const char* sfile = BR.getSourceManager().getPresumedLoc(path.asLocation()).getFilename();
    if ((!sfile) || (!support::isCmsLocalFile(sfile)))
      return false;
    return true;
  }

  bool CmsException::reportConstCast(const clang::ento::BugReport& R, clang::ento::CheckerContext& C) const {
    clang::ento::BugReporter& BR = C.getBugReporter();
    const clang::SourceManager& SM = BR.getSourceManager();
    clang::ento::PathDiagnosticLocation const& path = R.getLocation(SM);
    return reportGeneral(path, BR);
  }

  bool CmsException::reportConstCastAway(const clang::ento::BugReport& R, clang::ento::CheckerContext& C) const {
    clang::ento::BugReporter& BR = C.getBugReporter();
    const clang::SourceManager& SM = BR.getSourceManager();
    clang::ento::PathDiagnosticLocation const& path = R.getLocation(SM);
    return reportGeneral(path, BR);
  }

  bool CmsException::reportGlobalStatic(clang::QualType const& t,
                                        clang::ento::PathDiagnosticLocation const& path,
                                        clang::ento::BugReporter& BR) const {
    return reportGeneral(path, BR);
  }

  bool CmsException::reportMutableMember(clang::QualType const& t,
                                         clang::ento::PathDiagnosticLocation const& path,
                                         clang::ento::BugReporter& BR) const {
    return reportGeneral(path, BR);
  }

  bool CmsException::reportClass(clang::ento::PathDiagnosticLocation const& path, clang::ento::BugReporter& BR) const {
    return reportGeneral(path, BR);
  }

  bool CmsException::reportGlobalStaticForType(clang::QualType const& t,
                                               clang::ento::PathDiagnosticLocation const& path,
                                               clang::ento::BugReporter& BR) const {
    return reportGeneral(path, BR);
  }
}  // namespace clangcms
