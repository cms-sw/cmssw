/** Exceptions thrown from TrajectoryParametrization and dependent subsystems.
 */

#ifndef MeasurementDetExceptions_H
#define MeasurementDetExceptions_H

#include <exception>
#include <string>

/// Common base class

class MeasurementDetException : public std::exception {
public:
  MeasurementDetException() throw() {}
  MeasurementDetException( const std::string& message) throw() : theMessage(message) {}
  virtual ~MeasurementDetException() throw() {}
  virtual const char* what() const throw() { return theMessage.c_str();}
private:
  std::string theMessage;
};

#endif
