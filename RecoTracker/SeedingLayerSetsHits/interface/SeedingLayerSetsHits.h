#ifndef RecoTracker_SeedingLayerSet_SeedingLayerSetsHits
#define RecoTracker_SeedingLayerSet_SeedingLayerSetsHits

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

  typedef unsigned int LayerSetIndex;
  typedef unsigned int LayerIndex;

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
    const std::string& name() const { return seedingLayerSets_->layerNames_[index_]; }
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
      typedef std::vector<unsigned int>::const_iterator internal_iterator_type;
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
    SeedingLayerSet(const SeedingLayerSetsHits *sls, std::vector<unsigned int>::const_iterator begin, std::vector<unsigned int>::const_iterator end):
      seedingLayerSets_(sls), begin_(begin), end_(end) {}

    /// Number of layers in this set
    unsigned int size() const { return end_-begin_; }

    /// Get a given SeedingLayer (index is between 0 and size()-1)
    SeedingLayer operator[](unsigned int index) const {
      return SeedingLayer(seedingLayerSets_, *(begin_+index));
    }

    // iterators for range-for
    const_iterator begin() const { return const_iterator(seedingLayerSets_, begin_); }
    const_iterator cbegin() const { return begin(); }
    const_iterator end() const { return const_iterator(seedingLayerSets_, end_); }
    const_iterator cend() const { return end(); }

  private:
    const SeedingLayerSetsHits *seedingLayerSets_;
    std::vector<unsigned int>::const_iterator begin_; // Iterator to SeedingLayerSetsHits::layerSetIndices_, first layer
    std::vector<unsigned int>::const_iterator end_;   // Iterator to SeedingLayerSetsHits::layerSetIndices_, last+1 layer
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
   * \param nlayers_  Number of layers in each SeedingLayerSet
   */
  explicit SeedingLayerSetsHits(unsigned int nlayers_);

  ~SeedingLayerSetsHits();

  /**
   * Insert a layer
   *
   * \param layerName  Name of the layer
   * \param layerDet   Pointer to the corresponding DetLayer object
   *
   * \return Pair of the layer index and boolean indicating if the
   * layer was inserted or not. If the boolean is true, the hits
   * should be inserted with insertLayerHits()
   */
  std::pair<LayerIndex, bool> insertLayer(const std::string& layerName, const DetLayer *layerDet);
  /**
   * Insert hits for a layer
   *
   * \param layerIndex   Index of the layer
   * \param hits         Hits to insert
   *
   * Should be called if the layer was truly inserted by
   * insertLayer(). The layerIndex should be the index returned by
   * insertLayer.
   */
  void insertLayerHits(LayerIndex layerIndex, const Hits& hits);

  /// Get number of layers in each SeedingLayerSets
  unsigned int numberOfLayersInSet() const { return nlayers_; }
  /// Get the number of SeedingLayerSets
  unsigned int size() const { return layerSetIndices_.size() / nlayers_; }

  /// Get the SeedingLayerSet at a given index
  SeedingLayerSet operator[](LayerSetIndex index) const {
    std::vector<unsigned int>::const_iterator begin = layerSetIndices_.begin()+nlayers_*index;
    std::vector<unsigned int>::const_iterator end = begin+nlayers_;
    return SeedingLayerSet(this, begin, end);
  }

  // iterators for range-for
  const_iterator begin() const { return const_iterator(this, layerSetIndices_.begin()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator end() const { return const_iterator(this, layerSetIndices_.end()); }
  const_iterator cend() const { return end(); }

  // for more efficient edm::Event::put()
  void swap(SeedingLayerSetsHits& other) {
    std::swap(nlayers_, other.nlayers_);
    layerSetIndices_.swap(other.layerSetIndices_);
    layerHitRanges_.swap(other.layerHitRanges_);
    layerNames_.swap(other.layerNames_);
    layerDets_.swap(other.layerDets_);
    rechits_.swap(other.rechits_);
  }

  void print() const;

private:
  std::pair<LayerIndex, bool> insertLayer_(const std::string& layerName, const DetLayer *layerDet);
  Hits hits(LayerIndex layerIndex) const;

  /// Number of layers in a SeedingLayerSet
  unsigned int nlayers_;

  /**
   * Stores SeedingLayerSets as nlayers_ consecutive layer indices.
   * Layer indices point to layerHitRanges_, layerNames_, and
   * layerDets_. Hence layerSetIndices.size() == nlayers_*"number of layer sets"
   */
  std::vector<LayerSetIndex> layerSetIndices_;

  // following are indexed by LayerIndex
  typedef std::pair<unsigned int, unsigned int> Range;
  /**
   * Pair of indices (begin, end) to rechits_ for the list of RecHits
   * for the layer.
   */
  std::vector<Range> layerHitRanges_;
  std::vector<std::string> layerNames_; // Names of the layers
  std::vector<const DetLayer *> layerDets_; // Pointers to corresponding DetLayer objects

  /**
   * List of RecHits of all SeedingLayers. Hits of each layer are
   * identified by the (begin, end) index pairs in layerHitRanges_.
   */
  std::vector<ConstRecHitPointer> rechits_;
};

#endif
