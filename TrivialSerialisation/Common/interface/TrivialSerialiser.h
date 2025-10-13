#ifndef TrivialSerialisation_Common_TrivialSerialiser_h
#define TrivialSerialisation_Common_TrivialSerialiser_h

#include <cstdio>
#include "DataFormats/Common/interface/Wrapper.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiserBase.h"
#include "TrivialSerialisation/Common/interface/TrivialCopyTraits.h"

// defines all methods of TrivialSerialiserBase

namespace ngt {

  template <typename T>
  class TrivialSerialiser : public TrivialSerialiserBase {
  public:
    using WrapperType = edm::Wrapper<T>;
    TrivialSerialiser(WrapperType const& obj_) : TrivialSerialiserBase(&obj_) {}

    bool hasTrivialCopyTraits() const override;
    bool hasTrivialCopyProperties() const override;
    void trivialCopyInitialize(edm::AnyBuffer const& args) override;
    edm::AnyBuffer trivialCopyParameters() const override;
    std::vector<std::span<const std::byte>> trivialCopyRegions() const override;
    std::vector<std::span<std::byte>> trivialCopyRegions() override;
    void trivialCopyFinalize() override;

  private:
    const T& getWrappedObj_(WrapperType const& w) const {
      if (not w.isPresent()) {
        throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
      }
      return *w.product();
    }

    T& getWrappedObj_(WrapperType& w) {
      if (not w.isPresent()) {
        throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
      }
      return w.bareProduct();
    }
  };

  template <typename T>
  inline bool TrivialSerialiser<T>::hasTrivialCopyTraits() const {
    if constexpr (requires(T& t) { edm::TrivialCopyTraits<T>::regions(t); }) {
      return true;
    }
    return false;
  }

  template <typename T>
  inline bool TrivialSerialiser<T>::hasTrivialCopyProperties() const {
    if constexpr (requires { typename edm::TrivialCopyTraits<T>::Properties; }) {
      printf("TrivialCopyTraits<%s>::Properties is defined\n", edm::typeDemangle(typeid(T).name()).c_str());
      return true;
    }
    return false;
  }

  template <typename T>
  void TrivialSerialiser<T>::trivialCopyInitialize([[maybe_unused]] edm::AnyBuffer const& args) {
    if constexpr (not requires { typename edm::TrivialCopyTraits<T>::Properties; }) {
      // if edm::TrivialCopyTraits<T>::Properties is not defined, do not call initialize()
      return;
    } else {
      auto& w = static_cast<WrapperType const&>(*getWrapperBasePtr());
      // Each serialiser stores a pointer to the wrapper used to initialize it as "const edm::WrapperBase*"
      // For serialisers used for writing, that were initialized with a non-const wrapper, the const_cast below is safe because the wrapper was originally non-const.
      // For serialisers used for reading, that were initialized with a const wrapper, this function cannot be called because it is not marked as const.
      if constexpr (std::is_same_v<typename edm::TrivialCopyTraits<T>::Properties, void>) {
        // if edm::TrivialCopyTraits<T>::Properties is void, call initialize() without any additional arguments
        edm::TrivialCopyTraits<T>::initialize(const_cast<WrapperType&>(w).bareProduct());
      } else {
        // if edm::TrivialCopyTraits<T>::Properties is not void, cast args to Properties and pass it as an additional argument to initialize()
        edm::TrivialCopyTraits<T>::initialize(const_cast<WrapperType&>(w).bareProduct(),
                                              args.cast_to<typename edm::TrivialCopyTraits<T>::Properties>());
      }
    }
  }

  template <typename T>
  inline edm::AnyBuffer TrivialSerialiser<T>::trivialCopyParameters() const {
    auto& w = static_cast<WrapperType const&>(*getWrapperBasePtr());
    const T& obj = getWrappedObj_(w);
    if constexpr (not requires { typename edm::TrivialCopyTraits<T>::Properties; }) {
      // if edm::TrivialCopyTraits<T>::Properties is not defined, do not call properties()
      return {};
    } else if constexpr (std::is_same_v<typename edm::TrivialCopyTraits<T>::Properties, void>) {
      // if edm::TrivialCopyTraits<T>::Properties is void, do not call properties()
      return {};
    } else {
      // if edm::TrivialCopyTraits<T>::Properties is not void, call properties() and wrap the result in an edm::AnyBuffer
      typename edm::TrivialCopyTraits<T>::Properties p = edm::TrivialCopyTraits<T>::properties(obj);
      return edm::AnyBuffer(p);
    }
  }

  template <typename T>
  inline std::vector<std::span<const std::byte>> TrivialSerialiser<T>::trivialCopyRegions() const {
    if constexpr (requires(T const& t) { edm::TrivialCopyTraits<T>::regions(t); }) {
      auto& w = static_cast<WrapperType const&>(*getWrapperBasePtr());
      const T& obj = getWrappedObj_(w);
      return edm::TrivialCopyTraits<T>::regions(obj);
    } else {
      throw edm::Exception(edm::errors::LogicError)
          << "edm::TrivialCopyTraits<T>::regions(const T&) is not defined for type "
          << edm::typeDemangle(typeid(T).name());
      return {};
    }
  }

  template <typename T>
  inline std::vector<std::span<std::byte>> TrivialSerialiser<T>::trivialCopyRegions() {
    if constexpr (requires(T& t) { edm::TrivialCopyTraits<T>::regions(t); }) {
      auto& w = const_cast<WrapperType&>(static_cast<WrapperType const&>(*getWrapperBasePtr()));
      T& obj = getWrappedObj_(w);
      return edm::TrivialCopyTraits<T>::regions(obj);
    } else {
      throw edm::Exception(edm::errors::LogicError)
          << "edm::TrivialCopyTraits<T>::regions(const T&) is not defined for type "
          << edm::typeDemangle(typeid(T).name());
      return {};
    }
  }

  template <typename T>
  inline void TrivialSerialiser<T>::trivialCopyFinalize() {
    if constexpr (requires(T& t) { edm::TrivialCopyTraits<T>::finalize(t); }) {
      auto& w = static_cast<WrapperType const&>(*getWrapperBasePtr());
      const T& obj = getWrappedObj_(w);
      edm::TrivialCopyTraits<T>::finalize(obj);
    }
  }

}  // namespace ngt

#endif  // TrivialSerialisation_Common_TrivialSerialiser_h
