/** Exceptions thrown from TrajectoryParametrization and dependent subsystems.
 */

#ifndef MeasurementDetExceptions_H
#define MeasurementDetExceptions_H

#include "FWCore/Utilities/interface/Exception.h"
#include <string>

/// Common base class

class MeasurementDetException : public cms::Exception {
public:
  MeasurementDetException(const std::string &message) throw()
      : cms::Exception(message) {}
  ~MeasurementDetException() throw() override {}
};

#endif
