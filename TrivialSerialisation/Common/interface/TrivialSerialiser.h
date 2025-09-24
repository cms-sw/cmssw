#ifndef TrivialSerialisation_Common_TrivialSerialiser_h
#define TrivialSerialisation_Common_TrivialSerialiser_h

#include <cstdio>
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiserBase.h"
#include "TrivialSerialisation/Common/interface/TrivialCopyTraits.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiserSourceBase.h"
#include "DataFormats/Common/interface/Uninitialized.h"

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
    constexpr T construct_() {
      if constexpr (requires { T(); }) {
        return T();
      } else {
        return T(edm::kUninitialized);
      }
    }

    const T& getWrappedObj_(WrapperType const& w) const {
      if (not w.isPresent()) {
        throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper aa";
      }
      return *w.product();
    }

    T& getWrappedObj_(WrapperType& w) {
      if (not w.isPresent()) {
        throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper bb";
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
      auto& w = const_cast<edm::Wrapper<T>&>(static_cast<edm::Wrapper<T> const&>(*getWrapperBasePtr()));
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

#endif
