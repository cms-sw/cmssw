#ifndef TrivialSerialisation_Common_SerialiserBase_h
#define TrivialSerialisation_Common_SerialiserBase_h

#include "DataFormats/Common/interface/WrapperBase.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiserBase.h"

namespace ngt {
  class SerialiserBase {
  public:
    SerialiserBase() = default;

    virtual std::unique_ptr<TrivialSerialiserBase> initialize(edm::WrapperBase& wrapper) = 0;
    virtual std::unique_ptr<const TrivialSerialiserBase> initialize(const edm::WrapperBase& wrapper) = 0;

    virtual ~SerialiserBase() = default;
  };
}  // namespace ngt

#endif  // TrivialSerialisation_Common_SerialiserBase_h
