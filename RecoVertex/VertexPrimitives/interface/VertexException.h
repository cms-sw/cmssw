/** Exceptions thrown from RecoVertex dependent subsystems.
 */

#ifndef Vertex_Exceptions_H
#define Vertex_Exceptions_H

#include <exception>
#include <string>

/// Common base class

class VertexException : public std::exception {
public:
  VertexException() throw() {}
  VertexException( const std::string& message) throw() : theMessage(message) {}
  ~VertexException() throw() override {}
  const char* what() const throw() override { return theMessage.c_str();}
private:
  std::string theMessage;
};

#endif
