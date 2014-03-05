#ifndef TrackingTools_TransientTrackingRecHit_SeedingLayerSetsHits
#define TrackingTools_TransientTrackingRecHit_SeedingLayerSetsHits

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <vector>
#include <string>
#include <utility>

class DetLayer;

/**
 * Class to store TransientTrackingRecHits, names, and DetLayer
 * pointers of each ctfseeding::SeedingLayer as they come from
 * SeedingLayerSetsBuilder.
 *
 * In contrast to ctfseeding::SeedingLayerSets, this class requires
 * that all contained SeedingLayerSets have the same number of layers
 * in each set.
 *
 * This class was created in part for SeedingLayer getByToken
 * migration, and in part as a performance improvement.
 */
class SeedingLayerSetsHits {
public:
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
  typedef std::vector<ConstRecHitPointer> Hits;

  typedef unsigned short LayerSetIndex;
  typedef unsigned short LayerIndex;
  typedef unsigned int HitIndex;

  /**
   * Auxiliary class to represent a single SeedingLayer. Holds a
   * pointer to SeedingLayerSetsHits and the index of the
   * SeedingLayer. All calls are forwarded to SeedingLayerSetsHits.
   */
  class SeedingLayer {
  public:
    SeedingLayer(): seedingLayerSets_(0), index_(0) {}
    SeedingLayer(const SeedingLayerSetsHits *sls, LayerIndex index):
      seedingLayerSets_(sls), index_(index) {}

    /**
     * Index of the SeedingLayer within SeedingLayerSetsHits.
     *
     * The index is unique within a SeedingLayerSetsHits object, and
     * is the same for all SeedingLayers with the same name.
     */
    LayerIndex index() const { return index_; }
    const std::string& name() const { return (*seedingLayerSets_->layerNames_)[index_]; }
    const DetLayer *detLayer() const { return seedingLayerSets_->layerDets_[index_]; }
    Hits hits() const { return seedingLayerSets_->hits(index_); }

  private:
    const SeedingLayerSetsHits *seedingLayerSets_;
    LayerIndex index_;
  };

  /**
   * Auxiliary class to represent a set of SeedingLayers (e.g. BPIX1+BPIX2+BPIX3).
   *
   * Holds a pointer to SeedingLayerSetsHits, and iterators to
   * SeedingLayerSetsHits::layerSetIndices_ to for the first and last+1
   * layer of the set.
   */
  class SeedingLayerSet {
  public:
    class const_iterator {
    public:
      typedef std::vector<LayerSetIndex>::const_iterator internal_iterator_type;
      typedef SeedingLayer value_type;
      typedef internal_iterator_type::difference_type difference_type;

      const_iterator(): seedingLayerSets_(0) {}
      const_iterator(const SeedingLayerSetsHits *sls, internal_iterator_type iter): seedingLayerSets_(sls), iter_(iter) {}

      value_type operator*() const { return SeedingLayer(seedingLayerSets_, *iter_); }

      const_iterator& operator++() { ++iter_; return *this; }
      const_iterator operator++(int) {
        const_iterator clone(*this);
        ++clone;
        return clone;
      }

      bool operator==(const const_iterator& other) const { return iter_ == other.iter_; }
      bool operator!=(const const_iterator& other) const { return !operator==(other); }

    private:
      const SeedingLayerSetsHits *seedingLayerSets_;
      internal_iterator_type iter_;
    };

    SeedingLayerSet(): seedingLayerSets_(0) {}
    SeedingLayerSet(const SeedingLayerSetsHits *sls, std::vector<LayerSetIndex>::const_iterator begin, std::vector<LayerSetIndex>::const_iterator end):
      seedingLayerSets_(sls), begin_(begin), end_(end) {}

    /// Number of layers in this set
    LayerSetIndex size() const { return end_-begin_; }

    /// Get a given SeedingLayer (index is between 0 and size()-1)
    SeedingLayer operator[](LayerSetIndex index) const {
      return SeedingLayer(seedingLayerSets_, *(begin_+index));
    }

    // iterators for range-for
    const_iterator begin() const { return const_iterator(seedingLayerSets_, begin_); }
    const_iterator cbegin() const { return begin(); }
    const_iterator end() const { return const_iterator(seedingLayerSets_, end_); }
    const_iterator cend() const { return end(); }

  private:
    const SeedingLayerSetsHits *seedingLayerSets_;
    std::vector<LayerSetIndex>::const_iterator begin_; // Iterator to SeedingLayerSetsHits::layerSetIndices_, first layer
    std::vector<LayerSetIndex>::const_iterator end_;   // Iterator to SeedingLayerSetsHits::layerSetIndices_, last+1 layer
  };

  class const_iterator {
  public:
    typedef std::vector<LayerSetIndex>::const_iterator internal_iterator_type;
    typedef SeedingLayerSet value_type;
    typedef internal_iterator_type::difference_type difference_type;

    const_iterator(): seedingLayerSets_(0) {}
    const_iterator(const SeedingLayerSetsHits *sls, internal_iterator_type iter): seedingLayerSets_(sls), iter_(iter) {}

    value_type operator*() const { return SeedingLayerSet(seedingLayerSets_, iter_, iter_+seedingLayerSets_->nlayers_); }

    const_iterator& operator++() { std::advance(iter_, seedingLayerSets_->nlayers_); return *this; }
    const_iterator operator++(int) {
      const_iterator clone(*this);
      ++clone;
      return clone;
    }

    bool operator==(const const_iterator& other) const { return iter_ == other.iter_; }
    bool operator!=(const const_iterator& other) const { return !operator==(other); }

  private:
    const SeedingLayerSetsHits *seedingLayerSets_;
    internal_iterator_type iter_;
  };


  SeedingLayerSetsHits();

  /**
   * Constructor.
   *
   * \param nlayers         Number of layers in each SeedingLayerSet
   * \param layerSetIndices Pointer to a vector holding the indices of layer sets (pointer to vector is stored)
   * \param layerNames      Pointer to a vector holding the layer names (pointer to vector is stored)
   * \param layerDets       Vector of pointers to layer DetLayer objects (vector is copied, i.e. DetLayer pointers are stored)
   */
  SeedingLayerSetsHits(unsigned short nlayers,
                       const std::vector<LayerSetIndex> *layerSetIndices,
                       const std::vector<std::string> *layerNames,
                       const std::vector<const DetLayer *>& layerDets);

  ~SeedingLayerSetsHits();

  void swapHits(std::vector<HitIndex>& layerHitIndices,  Hits& hits);


  /// Get number of layers in each SeedingLayerSets
  unsigned short numberOfLayersInSet() const { return nlayers_; }
  /// Get the number of SeedingLayerSets
  unsigned short size() const { return nlayers_ > 0 ? layerSetIndices_->size() / nlayers_ : 0; }

  /// Get the SeedingLayerSet at a given index
  SeedingLayerSet operator[](LayerSetIndex index) const {
    std::vector<LayerSetIndex>::const_iterator begin = layerSetIndices_->begin()+nlayers_*index;
    std::vector<LayerSetIndex>::const_iterator end = begin+nlayers_;
    return SeedingLayerSet(this, begin, end);
  }

  // iterators for range-for
  const_iterator begin() const { return const_iterator(this, layerSetIndices_->begin()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator end() const { return const_iterator(this, layerSetIndices_->end()); }
  const_iterator cend() const { return end(); }

  // for more efficient edm::Event::put()
  void swap(SeedingLayerSetsHits& other) {
    std::swap(nlayers_, other.nlayers_);
    std::swap(layerSetIndices_, other.layerSetIndices_);
    layerHitIndices_.swap(other.layerHitIndices_);
    std::swap(layerNames_, other.layerNames_);
    layerDets_.swap(other.layerDets_);
    rechits_.swap(other.rechits_);
  }

  void print() const;

private:
  Hits hits(LayerIndex layerIndex) const;

  /// Number of layers in a SeedingLayerSet
  unsigned short nlayers_;

  /**
   * Stores SeedingLayerSets as nlayers_ consecutive layer indices.
   * Layer indices point to layerHitRanges_, layerNames_, and
   * layerDets_. Hence layerSetIndices.size() == nlayers_*"number of layer sets"
   */
  const std::vector<LayerSetIndex> *layerSetIndices_;

  // following are indexed by LayerIndex
  std::vector<HitIndex> layerHitIndices_; // Indices to first hits in rechits_
  const std::vector<std::string> *layerNames_; // Names of the layers
  std::vector<const DetLayer *> layerDets_; // Pointers to corresponding DetLayer objects

  /**
   * List of RecHits of all SeedingLayers. Hits of each layer are
   * identified by the begin indices in layerHitIndices_.
   */
  std::vector<ConstRecHitPointer> rechits_;
};

#endif
