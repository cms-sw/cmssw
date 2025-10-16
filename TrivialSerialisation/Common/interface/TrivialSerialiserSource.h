#ifndef TrivialSerialisation_Common_TrivialSerialiserSource_h
#define TrivialSerialisation_Common_TrivialSerialiserSource_h

#include "TrivialSerialisation/Common/interface/TrivialSerialiserSourceBase.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiser.h"

namespace ngt {
  template <typename T>
  class TrivialSerialiserSource : public TrivialSerialiserSourceBase {
  public:
    std::unique_ptr<TrivialSerialiserBase> initialize(edm::WrapperBase& wrapper) override {
      edm::Wrapper<T>& w = dynamic_cast<edm::Wrapper<T>&>(wrapper);
      return std::make_unique<TrivialSerialiser<T>>(w);
    }
    std::unique_ptr<const TrivialSerialiserBase> initialize(edm::WrapperBase const& wrapper) override {
      edm::Wrapper<T> const& w = dynamic_cast<edm::Wrapper<T> const&>(wrapper);
      return std::make_unique<const TrivialSerialiser<T>>(w);
    }
  };

}  // namespace ngt

#endif  // TrivialSerialisation_Common_TrivialSerialiserSource_h
