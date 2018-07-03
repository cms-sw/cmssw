#ifndef TrackingTools_TransientTrackingRecHit_SeedingLayerSetsLooper
#define TrackingTools_TransientTrackingRecHit_SeedingLayerSetsLooper

#include <vector>
#include <cstddef>

/**
 * Class to loop over arbitrary containers according to seeding layer sets indices
 */
class SeedingLayerSetsLooper {
public:
  using LayerSetIndex = unsigned short;

  template <typename T>
  class LayerSet {
  public:
    using internal_iterator_type = std::vector<LayerSetIndex>::const_iterator;
    using value_type = typename T::value_type;
    using difference_type = typename internal_iterator_type::difference_type;

    LayerSet(const T *container, internal_iterator_type begin, internal_iterator_type end):
      container_(container), begin_(begin), end_(end) {}

    LayerSetIndex size() const { return end_-begin_; }

    const value_type& operator[](size_t i) const { return (*container_)[*(begin_+i)]; }

  private:
    const T *container_ = nullptr;
    internal_iterator_type begin_;
    internal_iterator_type end_;
  };

  template <typename T>
  class LayerSetRange {
  public:
    LayerSetRange(const T *container, const SeedingLayerSetsLooper *info):
      container_(container), info_(info)
    {}

    class const_iterator {
    public:
      using internal_iterator_type = std::vector<LayerSetIndex>::const_iterator;
      using value_type = LayerSet<T>;
      using difference_type = typename internal_iterator_type::difference_type;

      //const_iterator() = default;
      const_iterator(const T *container, const SeedingLayerSetsLooper *info, internal_iterator_type iter):
        container_(container), info_(info), iter_(iter) {}

      value_type operator*() const { return value_type(container_, iter_, iter_+info_->nlayers_); }

      const_iterator& operator++() { std::advance(iter_, info_->nlayers_); return *this; }
      const_iterator operator++(int) {
        const_iterator clone(*this);
        ++(*this);
        return clone;
      }

      bool operator==(const const_iterator& other) const { return iter_ == other.iter_; }
      bool operator!=(const const_iterator& other) const { return !operator==(other); }

    private:
      const T *container_ = nullptr;
      const SeedingLayerSetsLooper *info_ = nullptr;
      internal_iterator_type iter_;
    };

    const_iterator begin() const { return const_iterator(container_, info_, info_->layerSetIndices_->begin()); }
    const_iterator cbegin() const { return begin(); }
    const_iterator end() const { return const_iterator(container_, info_, info_->layerSetIndices_->end()); }
    const_iterator cend() const { return end(); }

  private:
    const T *container_ = nullptr;
    const SeedingLayerSetsLooper *info_;
  };



  SeedingLayerSetsLooper() = default;

  /**
   * Constructor.
   *
   * \param nlayers         Number of layers in each SeedingLayerSet
   * \param layerSetIndices Pointer to a vector holding the indices of layer sets (pointer to vector is stored)
   */
  SeedingLayerSetsLooper(unsigned short nlayers,
                         const std::vector<LayerSetIndex> *layerSetIndices):
    nlayers_(nlayers),
    layerSetIndices_(layerSetIndices)
  {}

  template <typename T>
  LayerSetRange<T> makeRange(const T& container) const {
    return LayerSetRange<T>(&container, this);
  }

private:
  /// Number of layers in a SeedingLayerSet
  unsigned short nlayers_ = 0;

  /**
   * Stores SeedingLayerSets as nlayers_ consecutive layer indices.
   * Layer indices point to layerHitRanges_, layerNames_, and
   * layerDets_. Hence layerSetIndices.size() == nlayers_*"number of layer sets"
   */
  const std::vector<LayerSetIndex> *layerSetIndices_ = nullptr;
};

#endif
