#ifndef SimG4Core_SimG4Exception_H
#define SimG4Core_SimG4Exception_H

#include <exception>
#include <string>

/** Generic mantis exception. 
 *  Can be thrown directly, or derived from.
 *  SimG4 should (ideally) only throw exceptions derived from 
 *  this class.
 */

class SimG4Exception : public std::exception {
public:
  SimG4Exception(const std::string& message) : error_(message) {}
  ~SimG4Exception() throw() override {}
  const char* what() const throw() override { return error_.c_str(); }

private:
  std::string error_;
};

#endif
