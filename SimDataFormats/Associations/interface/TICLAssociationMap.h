#ifndef SimDataFormats_Associations_TICLAssociationMap_h
#define SimDataFormats_Associations_TICLAssociationMap_h

#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <iostream>

// CMSSW specific includes
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "FWCore/Framework/interface/Event.h"

namespace ticl {

  // Define the possible map types
  using mapWithFraction = std::vector<std::vector<std::pair<unsigned int, float>>>;
  using mapWithFractionAndScore = std::vector<std::vector<std::pair<unsigned int, std::pair<float, float>>>>;
  using oneToOneMapWithFraction = std::vector<std::pair<unsigned int, float>>;
  using oneToOneMapWithFractionAndScore = std::vector<std::pair<unsigned int, std::pair<float, float>>>;

  template <typename MapType, typename Collection1 = void, typename Collection2 = void>
  class AssociationMap {
  private:
    // Type alias for conditionally including collectionRefProds
    using CollectionRefProdType =
        typename std::conditional_t<std::is_void_v<Collection1> || std::is_void_v<Collection2>,
                                    std::monostate,
                                    std::pair<edm::RefProd<Collection1>, edm::RefProd<Collection2>>>;

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

    void insert(unsigned int index1, unsigned int index2, float fraction, float score = 0.0f) {
      if constexpr (std::is_same<MapType, mapWithFraction>::value) {
        if (index1 >= map_.size()) {
          map_.resize(index1 + 1);
        }
        auto& vec = map_[index1];
        auto it = std::find_if(vec.begin(), vec.end(), [&](const auto& pair) { return pair.first == index2; });
        if (it != vec.end()) {
          it->second += fraction;
        } else {
          vec.emplace_back(index2, fraction);
        }
      } else if constexpr (std::is_same<MapType, mapWithFractionAndScore>::value) {
        if (index1 >= map_.size()) {
          map_.resize(index1 + 1);
        }
        auto& vec = map_[index1];
        auto it = std::find_if(vec.begin(), vec.end(), [&](const auto& pair) { return pair.first == index2; });
        if (it != vec.end()) {
          it->second.first += fraction;
          it->second.second += score;
        } else {
          vec.emplace_back(index2, std::make_pair(fraction, score));
        }
      } else if constexpr (std::is_same<MapType, oneToOneMapWithFraction>::value) {
        auto it = std::find_if(map_.begin(), map_.end(), [&](const auto& pair) { return pair.first == index1; });
        if (it != map_.end()) {
          it->second += fraction;
        } else {
          map_.emplace_back(index1, fraction);
        }
      } else if constexpr (std::is_same<MapType, oneToOneMapWithFractionAndScore>::value) {
        auto it = std::find_if(map_.begin(), map_.end(), [&](const auto& pair) { return pair.first == index1; });
        if (it != map_.end()) {
          it->second.first += fraction;
          it->second.second += score;
        } else {
          map_.emplace_back(index1, std::make_pair(fraction, score));
        }
      }
    }

    // Overload of insert for CMSSW-specific use
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    void insert(const edm::Ref<C1>& ref1, const edm::Ref<C2>& ref2, float fraction, float score = 0.0f) {
      insert(ref1.key(), ref2.key(), fraction, score);
    }

    void sort(bool byScore = false) {
      static_assert(!std::is_same_v<MapType, oneToOneMapWithFraction> &&
                        !std::is_same_v<MapType, oneToOneMapWithFractionAndScore>,
                    "Sort is not applicable for one-to-one maps");

      if constexpr (std::is_same_v<MapType, mapWithFraction>) {
        for (auto& vec : map_) {
          std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        }
      } else if constexpr (std::is_same_v<MapType, mapWithFractionAndScore>) {
        for (auto& vec : map_) {
          if (byScore) {
            std::sort(
                vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
          } else {
            std::sort(
                vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
          }
        }
      }
    }

    auto& operator[](unsigned int index1) { return map_[index1]; }

    const auto& operator[](unsigned int index1) const { return map_[index1]; }

    const auto& at(unsigned int index1) const {
      if (index1 >= map_.size()) {
        throw std::out_of_range("Index out of range");
      }
      return map_[index1];
    }

    auto& at(unsigned int index1) {
      if (index1 >= map_.size()) {
        throw std::out_of_range("Index out of range");
      }
      return map_[index1];
    }
    // CMSSW-specific resize method
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    void resize(const edm::Event& event) {
      map_.resize(collectionRefProds.first->size());
    }
    // Constructor for generic use
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<std::is_void_v<C1> && std::is_void_v<C2>, int> = 0>
    void resize(const unsigned int size1) {
      map_.resize(size1);
    }

    // Method to print the entire map
    void print(std::ostream& os) const {
      for (size_t i = 0; i < map_.size(); ++i) {
        os << "Index " << i << ":\n";
        for (const auto& pair : map_[i]) {
          os << "  (" << pair.first << ", ";
          if constexpr (std::is_same<MapType, mapWithFractionAndScore>::value ||
                        std::is_same<MapType, oneToOneMapWithFractionAndScore>::value) {
            os << pair.second.first << ", " << pair.second.second;
          } else {
            os << pair.second;
          }
          os << ")\n";
        }
      }
    }

  private:
    // For CMSSW-specific use
    CollectionRefProdType collectionRefProds;
    MapType map_;  // Store the map directly
  };

}  // namespace ticl

#endif
