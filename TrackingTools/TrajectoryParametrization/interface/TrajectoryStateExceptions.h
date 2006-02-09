/** Exceptions thrown from TrajectoryParametrization and dependent subsystems.
 */

#ifndef TrajectoryState_Exceptions_H
#define TrajectoryState_Exceptions_H

#include <exception>
#include <string>

/// Common base class

class TrajectoryStateException : public std::exception {
public:
  TrajectoryStateException() throw() {}
  TrajectoryStateException( const std::string& message) throw() : theMessage(message) {}
  virtual ~TrajectoryStateException() throw() {}
  virtual const char* what() const throw() { return theMessage.c_str();}
private:
  std::string theMessage;
};

#endif
