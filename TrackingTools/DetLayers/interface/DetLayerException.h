/** Exceptions thrown from TrackingTools/DetLayers and dependent subsystems.
 */

#ifndef DetLayers_DetLayerException_h
#define DetLayers_DetLayerException_h

#include <exception>
#include <string>

/// Common base class

class DetLayerException : public std::exception {
public:
  DetLayerException() throw() {}
  DetLayerException( const std::string& message) throw() : theMessage(message) {}
  virtual ~DetLayerException() throw() {}
  virtual const char* what() const throw() { return theMessage.c_str();}
private:
  std::string theMessage;
};

#endif
