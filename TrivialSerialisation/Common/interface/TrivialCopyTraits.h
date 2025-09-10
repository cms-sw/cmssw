#ifndef DataFormats_Common_interface_TrivialCopyTraits_h
#define DataFormats_Common_interface_TrivialCopyTraits_h

#include <cassert>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace edm {

  // This struct should be specialised for each type that can be safely memcpy'ed.
  //
  // The specialisation shall have two static methods
  //
  //   static std::vector<std::span<std::byte>> regions(T& object);
  //   static std::vector<std::span<const std::byte>> regions(T const& object);
  //
  // that return a vector of address, size pairs.
  // A type that supports this interface can be copied by doing a memcpy of all
  // the address, size pairs from a source object to a destination object.
  //
  //
  // A specialisation may implement the type alias Properties to describe the
  // properties of an object that can be queried from an existing object via the
  // properties() method, and used to initialise a newly allocated copy of the
  // object via the initialize() method.
  //
  // If Properties is void, the properties() method should not be implemented,
  // and the initialize() method takes a single argument:
  //
  //   using Properties = void;
  //   static void initialise(T& object);
  //
  // If Properties is a concrete type, the properties() method should return an
  // instance of Properties, and the initialise() method should take as a second
  // parameter a const reference to a Properties object:
  //
  //   using Properties = ...;
  //   static Properties properties(T const& object);
  //   static void initialise(T& object, Properties const& args);
  //
  //
  // A specialisation can optionally provide a static method
  //
  //   static void finalize(T& object);
  //
  // If present, it should be called to restore the object invariants after a
  // memcpy operation.

  template <typename T>
  struct TrivialCopyTraits;

  // Specialisation for arithmetic types
  template <typename T>
    requires std::is_arithmetic_v<T>
  struct TrivialCopyTraits<T> {
    using value_type = T;

    static std::vector<std::span<std::byte>> regions(value_type& object) {
      return {{reinterpret_cast<std::byte*>(&object), sizeof(value_type)}};
    }

    static std::vector<std::span<const std::byte>> regions(value_type const& object) {
      return {{reinterpret_cast<std::byte const*>(&object), sizeof(value_type)}};
    }
  };

  // Specialisation for vectors of arithmetic types
  template <typename T>
    requires(std::is_arithmetic_v<T> and not std::is_same_v<T, bool>)
  struct TrivialCopyTraits<std::vector<T>> {
    using value_type = std::vector<T>;

    using Properties = std::vector<T>::size_type;

    static Properties properties(value_type const& object) { return object.size(); }

    static void initialize(value_type& object, Properties const& size) { object.resize(size); }

    static std::vector<std::span<std::byte>> regions(value_type& object) {
      return {{reinterpret_cast<std::byte*>(object.data()), object.size() * sizeof(T)}};
    }

    static std::vector<std::span<const std::byte>> regions(value_type const& object) {
      return {{reinterpret_cast<std::byte const*>(object.data()), object.size() * sizeof(T)}};
    }
  };

}  // namespace edm

#endif  // DataFormats_Common_interface_TrivialCopyTraits_h
