#ifndef TrivialSerialisation_Common_interface_TrivialSerialiserBase_h
#define TrivialSerialisation_Common_interface_TrivialSerialiserBase_h

#include "DataFormats/Common/interface/AnyBuffer.h"
#include "DataFormats/Common/interface/WrapperBase.h"

#include <span>
#include <vector>

namespace ngt {
  class TrivialSerialiserBase {
  public:
    TrivialSerialiserBase(const edm::WrapperBase* ptr) : ptr_(ptr) {}

    virtual void initialize(edm::AnyBuffer const& args) = 0;
    virtual edm::AnyBuffer parameters() const = 0;
    virtual std::vector<std::span<const std::byte>> regions() const = 0;
    virtual std::vector<std::span<std::byte>> regions() = 0;
    virtual void trivialCopyFinalize() = 0;

    const edm::WrapperBase* getWrapperBasePtr() const { return ptr_; }

    virtual ~TrivialSerialiserBase() = default;

  private:
    const edm::WrapperBase* ptr_;
  };

}  // namespace ngt

#endif  // TrivialSerialisation_Common_interface_TrivialSerialiserBase_h
