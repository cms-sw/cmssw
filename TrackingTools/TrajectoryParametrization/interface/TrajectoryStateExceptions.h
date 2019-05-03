/** Exceptions thrown from TrajectoryParametrization and dependent subsystems.
 */

#ifndef TrajectoryState_Exceptions_H
#define TrajectoryState_Exceptions_H

#include "FWCore/Utilities/interface/Exception.h"
#include <string>

/// Common base class

class TrajectoryStateException : public cms::Exception {
public:
  TrajectoryStateException(const std::string &message) throw()
      : cms::Exception(message) {}
  ~TrajectoryStateException() throw() override {}
};

#endif
