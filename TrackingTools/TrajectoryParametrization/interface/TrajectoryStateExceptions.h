/** Exceptions thrown from TrajectoryParametrization and dependent subsystems.
 */

#ifndef TrajectoryState_Exceptions_H
#define TrajectoryState_Exceptions_H

#include <string>
#include "FWCore/Utilities/interface/Exception.h"

/// Common base class

class TrajectoryStateException : public cms::Exception{
public:
  TrajectoryStateException( const std::string& message) throw() :  cms::Exception(message){}
  virtual ~TrajectoryStateException() throw() {}
};

#endif
