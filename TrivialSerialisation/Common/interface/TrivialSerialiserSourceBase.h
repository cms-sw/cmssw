#ifndef TrivialSerialisation_Common_TrivialSerialiserSourceBase_h
#define TrivialSerialisation_Common_TrivialSerialiserSourceBase_h

#include "DataFormats/Common/interface/WrapperBase.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiserBase.h"

namespace ngt {
  class TrivialSerialiserSourceBase {
  public:
    TrivialSerialiserSourceBase() = default;

    virtual std::unique_ptr<TrivialSerialiserBase> initialize(edm::WrapperBase& wrapper) = 0;
    virtual std::unique_ptr<const TrivialSerialiserBase> initialize(const edm::WrapperBase& wrapper) = 0;

    virtual ~TrivialSerialiserSourceBase() = default;
  };
}  // namespace ngt

#endif  // TrivialSerialisation_Common_TrivialSerialiserSourceBase_h
