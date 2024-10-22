/** Exceptions thrown from RecoVertex dependent subsystems.
 */

#ifndef Vertex_Exceptions_H
#define Vertex_Exceptions_H

#include "FWCore/Utilities/interface/Exception.h"
#include <string>

/// Common base class

class VertexException : public cms::Exception {
public:
  VertexException() : cms::Exception("VertexException") {}
  explicit VertexException(const std::string& message) : cms::Exception("VertexException", message) {}
};

#endif
