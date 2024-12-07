#ifndef SimDataFormats_Associations_TICLAssociationMap_h
#define SimDataFormats_Associations_TICLAssociationMap_h

#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <iostream>
#include <cassert>

#include <limits>

// CMSSW specific includes
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "FWCore/Framework/interface/Event.h"

namespace ticl {

  // Define wrapper types to differentiate between fraction and shared energy
  struct FractionType {
    float value;
    FractionType(float v = 0.0f) : value(v) {}
    FractionType& operator+=(float v) {
      value += v;
      return *this;
    }
  };

  struct SharedEnergyType {
    float value;
    SharedEnergyType(float v = 0.0f) : value(v) {}
    SharedEnergyType& operator+=(float v) {
      value += v;
      return *this;
    }
  };

  // AssociationElement class to store index and value, and provide methods directly
  template <typename V>
  class AssociationElement {
  public:
    using value_type = V;
    AssociationElement() : index_(std::numeric_limits<unsigned int>::max()) {
      if constexpr (std::is_same_v<V, FractionType> || std::is_same_v<V, SharedEnergyType>) {
        value_.value = -1.f;
      } else if constexpr (std::is_same_v<V, std::pair<FractionType, float>> ||
                           std::is_same_v<V, std::pair<SharedEnergyType, float>>) {
        value_.first.value = -1.f;
      }
    }
    AssociationElement(unsigned int index, const V& value) : index_(index), value_(value) {}

    unsigned int index() const { return index_; }

    bool isValid() const {
      if constexpr (std::is_same_v<V, FractionType> || std::is_same_v<V, SharedEnergyType>) {
        return value_.value >= 0.f;
      } else if constexpr (std::is_same_v<V, std::pair<FractionType, float>> ||
                           std::is_same_v<V, std::pair<SharedEnergyType, float>>) {
        return value_.first.value >= 0.f;
      }
    }

    // Enable fraction() if ValueType is FractionType
    template <typename T = V, typename std::enable_if_t<std::is_same_v<T, FractionType>, int> = 0>
    float fraction() const {
      return value_.value;
    }

    // Enable sharedEnergy() if ValueType is SharedEnergyType
    template <typename T = V, typename std::enable_if_t<std::is_same_v<T, SharedEnergyType>, int> = 0>
    float sharedEnergy() const {
      return value_.value;
    }

    // Enable fraction() and score() if ValueType is std::pair<FractionType, float>
    template <typename T = V, typename std::enable_if_t<std::is_same_v<T, std::pair<FractionType, float>>, int> = 0>
    float fraction() const {
      return value_.first.value;
    }
    template <typename T = V, typename std::enable_if_t<std::is_same_v<T, std::pair<FractionType, float>>, int> = 0>
    float score() const {
      return value_.second;
    }

    // Enable sharedEnergy() and score() if ValueType is std::pair<SharedEnergyType, float>
    template <typename T = V, typename std::enable_if_t<std::is_same_v<T, std::pair<SharedEnergyType, float>>, int> = 0>
    float sharedEnergy() const {
      return value_.first.value;
    }
    template <typename T = V, typename std::enable_if_t<std::is_same_v<T, std::pair<SharedEnergyType, float>>, int> = 0>
    float score() const {
      return value_.second;
    }

    // Method to accumulate values
    void accumulate(const V& other_value) {
      if constexpr (std::is_same_v<V, FractionType> || std::is_same_v<V, SharedEnergyType>) {
        value_.value += other_value.value;
      } else if constexpr (std::is_same_v<V, std::pair<FractionType, float>> ||
                           std::is_same_v<V, std::pair<SharedEnergyType, float>>) {
        value_.first.value += other_value.first.value;
        value_.second += other_value.second;
      }
    }
    bool operator==(const AssociationElement& other) const {
      return index_ == other.index_ && value_.value == other.value_.value;
    }

    bool operator!=(const AssociationElement& other) const { return !(*this == other); }

  private:
    unsigned int index_;
    V value_;
  };

  // Type traits to differentiate between one-to-one and one-to-many maps
  template <typename T>
  struct MapTraits;

  template <typename V>
  struct MapTraits<std::vector<AssociationElement<V>>> {
    static constexpr bool is_one_to_one = true;
    using AssociationElementType = AssociationElement<V>;
    using ValueType = V;
  };

  template <typename V>
  struct MapTraits<std::vector<std::vector<AssociationElement<V>>>> {
    static constexpr bool is_one_to_one = false;
    using AssociationElementType = AssociationElement<V>;
    using ValueType = V;
  };

  // Trait to check if V is a std::pair (i.e., has a score)
  template <typename T>
  struct IsValueTypeWithScore : std::false_type {};

  template <typename First>
  struct IsValueTypeWithScore<std::pair<First, float>> : std::true_type {};

  // Define map types using AssociationElement and container types
  using mapWithFraction = std::vector<std::vector<AssociationElement<FractionType>>>;
  using mapWithFractionAndScore = std::vector<std::vector<AssociationElement<std::pair<FractionType, float>>>>;
  using oneToOneMapWithFraction = std::vector<AssociationElement<FractionType>>;
  using oneToOneMapWithFractionAndScore = std::vector<AssociationElement<std::pair<FractionType, float>>>;

  using mapWithSharedEnergy = std::vector<std::vector<AssociationElement<SharedEnergyType>>>;
  using mapWithSharedEnergyAndScore = std::vector<std::vector<AssociationElement<std::pair<SharedEnergyType, float>>>>;
  using oneToOneMapWithSharedEnergy = std::vector<AssociationElement<SharedEnergyType>>;
  using oneToOneMapWithSharedEnergyAndScore = std::vector<AssociationElement<std::pair<SharedEnergyType, float>>>;

  // AssociationMap class templated on MapType
  template <typename MapType, typename Collection1 = void, typename Collection2 = void>
  class AssociationMap {
  private:
    MapType map_;

    // Type alias for conditionally including collectionRefProds
    using CollectionRefProdType =
        typename std::conditional_t<std::is_void_v<Collection1> || std::is_void_v<Collection2>,
                                    std::monostate,
                                    std::pair<edm::RefProd<Collection1>, edm::RefProd<Collection2>>>;

    CollectionRefProdType collectionRefProds;

    // Traits to deduce AssociationElementType and ValueType
    using Traits = MapTraits<MapType>;
    using AssociationElementType = typename Traits::AssociationElementType;
    using V = typename Traits::ValueType;
    static constexpr bool is_one_to_one = Traits::is_one_to_one;

  public:
    AssociationMap() : collectionRefProds() {}

    // Constructor for generic use
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<std::is_void_v<C1> && std::is_void_v<C2>, int> = 0>
    AssociationMap(const unsigned int size1 = 0) {
      map_.resize(size1);
    }

    // Constructor for CMSSW-specific use
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    AssociationMap(const edm::RefProd<C1>& id1, const edm::RefProd<C2>& id2, const edm::Event& event)
        : collectionRefProds(std::make_pair(id1, id2)) {
      resize(event);
    }

    // Constructor for CMSSW-specific use
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    AssociationMap(const edm::Handle<C1>& handle1, const edm::Handle<C2>& handle2, const edm::Event& event)
        : collectionRefProds(std::make_pair(edm::RefProd<C1>(handle1), edm::RefProd<C2>(handle2))) {
      resize(event);
    }

    MapType& getMap() { return map_; }

    const MapType& getMap() const { return map_; }

    const auto size() const { return map_.size(); }

    // CMSSW-specific method to get references
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    edm::Ref<C1> getRefFirst(unsigned int index) const {
      return edm::Ref<C1>(collectionRefProds.first, index);
    }

    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    edm::Ref<C2> getRefSecond(unsigned int index) const {
      return edm::Ref<C2>(collectionRefProds.second, index);
    }

    // Method to get collection IDs for CMSSW-specific use
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    std::pair<const edm::RefProd<C1>, const edm::RefProd<C2>> getCollectionIDs() const {
      return collectionRefProds;
    }

    void insert(unsigned int index1, unsigned int index2, float fraction_or_energy, float score = 0.0f) {
      assert(index1 < map_.size());
      V value;
      if constexpr (IsValueTypeWithScore<V>::value) {
        using FirstType = typename V::first_type;
        value = V(FirstType(fraction_or_energy), score);
      } else {
        value = V(fraction_or_energy);
      }
      AssociationElementType element(index2, value);

      if constexpr (is_one_to_one) {
        map_[index1] = element;
      } else {
        auto& vec = map_[index1];
        auto it =
            std::find_if(vec.begin(), vec.end(), [&](const AssociationElementType& e) { return e.index() == index2; });
        if (it != vec.end()) {
          // Accumulate value
          it->accumulate(value);
        } else {
          vec.push_back(element);
        }
      }
    }

    // Overload of insert for CMSSW-specific use
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    void insert(const edm::Ref<C1>& ref1, const edm::Ref<C2>& ref2, float fraction_or_energy, float score = 0.0f) {
      insert(ref1.key(), ref2.key(), fraction_or_energy, score);
    }

    void sort(bool byScore = false) {
      if constexpr (is_one_to_one) {
        // Sorting not applicable for one-to-one maps
      } else {
        for (auto& vec : map_) {
          if (byScore && IsValueTypeWithScore<V>::value) {
            std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
              if (a.score() != b.score()) {
                return a.score() > b.score();
              } else {
                return a.index() < b.index();
              }
            });
          } else {
            if constexpr (std::is_same_v<V, FractionType> || std::is_same_v<V, std::pair<FractionType, float>>) {
              std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
                if (a.fraction() != b.fraction()) {
                  return a.fraction() > b.fraction();
                } else {
                  return a.index() < b.index();
                }
              });
            } else {
              std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
                if (a.sharedEnergy() != b.sharedEnergy()) {
                  return a.sharedEnergy() > b.sharedEnergy();
                } else {
                  return a.index() < b.index();
                }
              });
            }
          }
        }
      }
    }

    // Overload of sort() that accepts a custom comparator
    template <typename Compare>
    void sort(Compare comp) {
      if constexpr (is_one_to_one) {
        // Sorting not applicable for one-to-one maps
      } else {
        for (auto& vec : map_) {
          std::sort(vec.begin(), vec.end(), comp);
        }
      }
    }

    // Access methods
    const auto& operator[](unsigned int index1) const { return map_[index1]; }

    auto& operator[](unsigned int index1) { return map_[index1]; }

    const auto& at(unsigned int index1) const {
      const auto& elem = map_.at(index1);
      if (!elem.isValid()) {
        throw std::out_of_range("Attempted to access an unset element in AssociationMap. Element index: " +
                                std::to_string(index1));
      }
      return elem;
    }

    auto& at(unsigned int index1) {
      auto& elem = map_.at(index1);
      if (!elem.isValid()) {
        throw std::out_of_range("Attempted to access an unset element in AssociationMap. Element index: " +
                                std::to_string(index1));
      }
      return elem;
    }

    // CMSSW-specific resize method
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    void resize(const edm::Event& event) {
      map_.resize(collectionRefProds.first->size());
    }

    // Generic resize method
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<std::is_void_v<C1> && std::is_void_v<C2>, int> = 0>
    void resize(const unsigned int size1) {
      map_.resize(size1);
    }

    // Method to print the entire map
    void print(std::ostream& os) const {
      for (size_t i = 0; i < map_.size(); ++i) {
        if constexpr (is_one_to_one) {
          const auto& elem = map_[i];
          if (!elem.isValid()) {
            continue;
          }
          os << "Index " << i << ":\n";

          os << "  (" << elem.index() << ", ";
          if constexpr (IsValueTypeWithScore<V>::value) {
            if constexpr (std::is_same_v<typename V::first_type, FractionType>) {
              os << "Fraction: " << elem.fraction() << ", Score: " << elem.score();
            } else {
              os << "SharedEnergy: " << elem.sharedEnergy() << ", Score: " << elem.score();
            }
          } else {
            if constexpr (std::is_same_v<V, FractionType>) {
              os << "Fraction: " << elem.fraction();
            } else if constexpr (std::is_same_v<V, SharedEnergyType>) {
              os << "SharedEnergy: " << elem.sharedEnergy();
            }
          }
          os << ")\n";
        } else {
          os << "Index " << i << ":\n";
          for (const auto& elem : map_[i]) {
            os << "  (" << elem.index() << ", ";
            if constexpr (IsValueTypeWithScore<V>::value) {
              if constexpr (std::is_same_v<typename V::first_type, FractionType>) {
                os << "Fraction: " << elem.fraction() << ", Score: " << elem.score();
              } else {
                os << "SharedEnergy: " << elem.sharedEnergy() << ", Score: " << elem.score();
              }
            } else {
              if constexpr (std::is_same_v<V, FractionType>) {
                os << "Fraction: " << elem.fraction();
              } else if constexpr (std::is_same_v<V, SharedEnergyType>) {
                os << "SharedEnergy: " << elem.sharedEnergy();
              }
            }
            os << ")\n";
          }
        }
      }
    }
  };

}  // namespace ticl

#endif
