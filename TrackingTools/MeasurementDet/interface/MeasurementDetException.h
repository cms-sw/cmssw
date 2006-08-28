/** Exceptions thrown from TrajectoryParametrization and dependent subsystems.
 */

#ifndef MeasurementDetExceptions_H
#define MeasurementDetExceptions_H

#include <string>
#include "FWCore/Utilities/interface/Exception.h"

/// Common base class

class MeasurementDetException : public cms::Exception {
public:
  MeasurementDetException( const std::string& message) throw() :  cms::Exception(message)  {}
  virtual ~MeasurementDetException() throw() {}

};

#endif
