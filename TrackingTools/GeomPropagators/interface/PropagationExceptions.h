/** Exceptions thrown from TrajectoryParametrization and dependent subsystems.
 */

#ifndef Propagation_Exceptions_H
#define Propagation_Exceptions_H

//#include <exception>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

#include <string>

/// Common base class

class PropagationException final : public cms::Exception {
public:
  PropagationException( const std::string& message) throw() :  cms::Exception(message)  {}
  ~PropagationException() throw() override {}
private:
};

#endif
