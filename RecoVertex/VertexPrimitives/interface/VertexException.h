/** Exceptions thrown from RecoVertex dependent subsystems.
 */

#ifndef Vertex_Exceptions_H
#define Vertex_Exceptions_H

#include "FWCore/Utilities/interface/Exception.h"
#include <string>

/// Common base class

class VertexException : public cms::Exception {
public:
  VertexException() throw() : cms::Exception("VertexException") {}
  VertexException(const std::string& message) throw() : cms::Exception("VertexException"), theMessage(message) {}
  ~VertexException() throw() override {}
  const char* what() const throw() override { return theMessage.c_str(); }

private:
  std::string theMessage;
};

#endif
