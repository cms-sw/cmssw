/** Exceptions thrown from TrajectoryParametrization and dependent subsystems.
 */

#ifndef Propagation_Exceptions_H
#define Propagation_Exceptions_H

#include <exception>
#include <string>

/// Common base class

class PropagationException : public std::exception {
public:
  PropagationException() throw() {}
  PropagationException( const std::string& message) throw() : theMessage(message) {}
  virtual ~PropagationException() throw() {}
  virtual const char* what() const throw() { return theMessage.c_str();}
private:
  std::string theMessage;
};

#endif
