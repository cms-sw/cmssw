/** Exceptions thrown from TrackingTools/DetLayers and dependent subsystems.
 */

#ifndef DetLayers_DetLayerException_h
#define DetLayers_DetLayerException_h

#include <exception>
#include <string>

/// Common base class
#include "FWCore/Utilities/interface/Exception.h"

class DetLayerException : public cms::Exception {
public:
  DetLayerException( const std::string& message) throw() : cms::Exception(message)  {}
  virtual ~DetLayerException() throw() {}
private:
};

#endif
