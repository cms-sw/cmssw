#ifndef DetLayers_DetLayerException_h
#define DetLayers_DetLayerException_h

/** \class DetLayerException
 *  Exceptions thrown from TrackingTools/DetLayers and dependent subsystems.
 *
 *  $Date: 2007/03/07 16:28:39 $
 *  $Revision: 1.3 $
 */

/// Common base class
#include "FWCore/Utilities/interface/Exception.h"

#include <exception>
#include <string>

class DetLayerException : public cms::Exception {
public:
  DetLayerException( const std::string& message) throw() : cms::Exception(message)  {}
  virtual ~DetLayerException() throw() {}
private:
};

#endif
