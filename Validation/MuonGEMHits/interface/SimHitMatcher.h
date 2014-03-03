#ifndef GEMValidation_SimHitMatcher_h
#define GEMValidation_SimHitMatcher_h

/**\class SimHitMatcher

 Description: Matching of SimHit for SimTrack in CSC & GEM

 Original Author:  "Vadim Khotilovich"
 $Id: SimHitMatcher.h,v 1.3 2013/02/17 15:55:16 khotilov Exp $
*/

#include "BaseMatcher.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <vector>
#include <map>
#include <set>

class GEMGeometry;

class SimHitMatcher : public BaseMatcher
{
public:
  
  SimHitMatcher(const SimTrack& t, const SimVertex& v,
      const edm::ParameterSet& ps, const edm::Event& ev, const edm::EventSetup& es, const GEMGeometry* gem_geo);
  
  ~SimHitMatcher();

  /// access to all the GEM SimHits
  const edm::PSimHitContainer& simHitsGEM() const {return gem_hits_;}
  /// GEM partitions' detIds with SimHits
  std::set<unsigned int> detIdsGEM() const;

  /// GEM detid's with hits in 2 layers of coincidence pads
  /// those are layer==1 only detid's
  std::set<unsigned int> detIdsGEMCoincidences() const;

  /// GEM chamber detIds with SimHits
  std::set<unsigned int> chamberIdsGEM() const;

  /// GEM superchamber detIds with SimHits
  std::set<unsigned int> superChamberIdsGEM() const;
  /// GEM superchamber detIds with SimHits 2 layers of coincidence pads
  std::set<unsigned int> superChamberIdsGEMCoincidences() const;

  /// simhits from a particular partition (GEM)/layer (CSC), chamber or superchamber
  const edm::PSimHitContainer& hitsInDetId(unsigned int) const;
  const edm::PSimHitContainer& hitsInChamber(unsigned int) const;
  const edm::PSimHitContainer& hitsInSuperChamber(unsigned int) const;

  /// #layers with hits
  /// for CSC: "super-chamber" means chamber
  int nLayersWithHitsInSuperChamber(unsigned int) const;

  /// How many pads with simhits in GEM did this simtrack get?
  int nPadsWithHits() const;
  /// How many coincidence pads with simhits in GEM did this simtrack get?
  int nCoincidencePadsWithHits() const;

  /// calculate Global average position for a provided collection of simhits
  GlobalPoint simHitsMeanPosition(const edm::PSimHitContainer& sim_hits) const;

  /// calculate average strip (strip for GEM, half-strip for CSC) number for a provided collection of simhits
  float simHitsMeanStrip(const edm::PSimHitContainer& sim_hits) const;

  std::set<int> hitStripsInDetId(unsigned int, int margin_n_strips = 0) const;  // GEM or CSC
  std::set<int> hitPadsInDetId(unsigned int) const; // GEM
  std::set<int> hitCoPadsInDetId(unsigned int) const; // GEM coincidence pads with hits

  // what unique partitions numbers were hit by this simtrack?
  std::set<int> hitPartitions() const; // GEM

private:

  void init();

  std::vector<unsigned int> getIdsOfSimTrackShower(unsigned  trk_id,
      const edm::SimTrackContainer& simTracks, const edm::SimVertexContainer& simVertices);

  void matchSimHitsToSimTrack(std::vector<unsigned int> track_ids, const edm::PSimHitContainer& gem_hits);

  bool simMuOnlyGEM_;
  bool discardEleHitsGEM_;
  std::string simInputLabel_;

  const GEMGeometry* gem_geo_;
  std::map<unsigned int, unsigned int> trkid_to_index_;

  edm::PSimHitContainer no_hits_;
  edm::PSimHitContainer gem_hits_;
  std::map<unsigned int, edm::PSimHitContainer > gem_detid_to_hits_;
  std::map<unsigned int, edm::PSimHitContainer > gem_chamber_to_hits_;
  std::map<unsigned int, edm::PSimHitContainer > gem_superchamber_to_hits_;

  // detids with hits in pads
  std::map<unsigned int, std::set<int> > gem_detids_to_pads_;
  // detids with hits in 2-layer pad coincidences
  std::map<unsigned int, std::set<int> > gem_detids_to_copads_;
};

#endif
