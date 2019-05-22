#include "SimDataFormats/Vertex/interface/CoreSimVertex.h"

std::ostream& operator<<(std::ostream& o, const CoreSimVertex& v) {
  o << v.position();
  return o;
}
