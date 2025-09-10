


#include "TrivialSerialisation/Common/interface/TrivialSerialiserBase.h"
#include "TrivialSerialisation/Common/interface/TrivialCopyTraits.h"









namespace ngt {
  template <typename T>
  class TrivialSerialiser : public TrivialSerialiserBase {
  public:
    TrivialSerialiser() : TrivialSerialiserBase(), present(false) {}

    bool hasTrivialCopyTraits_() const override;
    bool hasTrivialCopyProperties_() const override;
    void trivialCopyInitialize_(edm::AnyBuffer const& args) override;
    edm::AnyBuffer trivialCopyParameters_() const override;
    std::vector<std::span<const std::byte>> trivialCopyRegions_() const override;
    std::vector<std::span<std::byte>> trivialCopyRegions_() override;
    void trivialCopyFinalize_() override;


  };

  template <typename T>
  inline bool Wrapper<T>::hasTrivialCopyTraits_() const {
    if constexpr (requires(T& t) { edm::TrivialCopyTraits<T>::regions(t); }) {
      return true;
    }
    return false;
  }

  template <typename T>
  inline bool Wrapper<T>::hasTrivialCopyProperties_() const {
    if constexpr (requires { typename edm::TrivialCopyTraits<T>::Properties; }) {
      return true;
    }
    return false;
  }

  template <typename T>
  void Wrapper<T>::trivialCopyInitialize_([[maybe_unused]] edm::AnyBuffer const& args) {
    if (not present) {
      throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
    }
    // if edm::TrivialCopyTraits<T>::Properties is not defined, do not call initialize()
    if constexpr (not requires { typename edm::TrivialCopyTraits<T>::Properties; }) {
      return;
    } else
      // if edm::TrivialCopyTraits<T>::Properties is void, call initialize() without any additional arguments
      if constexpr (std::is_same_v<typename edm::TrivialCopyTraits<T>::Properties, void>) {
        edm::TrivialCopyTraits<T>::initialize(obj);
      } else
      // if edm::TrivialCopyTraits<T>::Properties is not void, cast args to Properties and pass it as an additional argument to initialize()
      {
        edm::TrivialCopyTraits<T>::initialize(obj, args.cast_to<typename edm::TrivialCopyTraits<T>::Properties>());
      }
  }

  template <typename T>
  inline edm::AnyBuffer Wrapper<T>::trivialCopyParameters_() const {
    if (not present) {
      throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
    }
    // if edm::TrivialCopyTraits<T>::Properties is not defined, do not call properties()
    if constexpr (not requires { typename edm::TrivialCopyTraits<T>::Properties; }) {
      return {};
    } else
      // if edm::TrivialCopyTraits<T>::Properties is void, do not call properties()
      if constexpr (std::is_same_v<typename edm::TrivialCopyTraits<T>::Properties, void>) {
        return {};
      } else
      // if edm::TrivialCopyTraits<T>::Properties is not void, call properties() and wrap the result in an edm::AnyBuffer
      {
        typename edm::TrivialCopyTraits<T>::Properties p = edm::TrivialCopyTraits<T>::properties(obj);
        return edm::AnyBuffer(p);
      }
  }

  template <typename T>
  inline std::vector<std::span<const std::byte>> Wrapper<T>::trivialCopyRegions_() const {
    if (not present) {
      throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
    }
    if constexpr (requires(T const& t) { edm::TrivialCopyTraits<T>::regions(t); }) {
      return edm::TrivialCopyTraits<T>::regions(obj);
    } else {
      throw edm::Exception(edm::errors::LogicError)
          << "edm::TrivialCopyTraits<T>::regions(const T&) is not defined for type "
          << edm::typeDemangle(typeid(T).name());
      return {};
    }
  }

  template <typename T>
  inline std::vector<std::span<std::byte>> Wrapper<T>::trivialCopyRegions_() {
    if (not present) {
      throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
    }
    if constexpr (requires(T& t) { edm::TrivialCopyTraits<T>::regions(t); }) {
      return edm::TrivialCopyTraits<T>::regions(obj);
    } else {
      throw edm::Exception(edm::errors::LogicError)
          << "edm::TrivialCopyTraits<T>::regions(const T&) is not defined for type "
          << edm::typeDemangle(typeid(T).name());
      return {};
    }
  }

  template <typename T>
  inline void Wrapper<T>::trivialCopyFinalize_() {
    if (not present) {
      throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
    }
    if constexpr (requires(T& t) { edm::TrivialCopyTraits<T>::finalize(t); }) {
      edm::TrivialCopyTraits<T>::finalize(obj);
    }
  }

}  // namespace ngt