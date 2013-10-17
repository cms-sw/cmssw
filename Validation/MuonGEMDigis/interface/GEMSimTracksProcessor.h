#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/Frameworkfwd.h>

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include <vector>
#include <map>
#include <set>


/// \class GEMSimTracksProcessor
///
/// keeps information of SimTracks with their SimHits in GEM and ME11
class GEMSimTracksProcessor
{
public:

  /// structure to keep data for simtracks
  struct SimTrackExtra
  {
    SimTrackExtra(): trk_(nullptr), vtx_(nullptr) {}
    SimTrackExtra(const SimTrack * trk, const SimVertex * vtx): trk_(trk), vtx_(vtx) {}

    const SimTrack * trk_;
    const SimVertex * vtx_;
    std::set<uint32_t> gem_ids_;
    std::set<uint32_t> csc_ids_;
  };

  // what chamber number track hits
  enum ChamberType {NONE = 0, ODD = 1, EVEN, BOTH};

  /// need the config to make the minTrackPt_ configurable
  GEMSimTracksProcessor(const edm::ParameterSet& iConfig);
  ~GEMSimTracksProcessor() {}

  /// to be called at each beginRun
  void init(const edm::EventSetup& iSetup);

  /// Initialize the tracks
  void fillTracks(const edm::SimTrackContainer &trks, const edm::SimVertexContainer &vtxs);

  ///
  void addSimHit(const PSimHit &hit, GlobalPoint &hit_gp);

  /// number of simtracks
  size_t size() {return trackids_.size();}

  /// access SimTrack number itrk
  const SimTrack * track(size_t itrk);

  /// for SimTrack number itrk, return list of GEM detector ids in which it left hits
  std::set<uint32_t> getDetIdsGEM(size_t itrk, ChamberType odd_even = BOTH);

  /// for SimTrack number itrk, return list of csc detector ids in which it left hits
  std::set<uint32_t> getDetIdsCSC(size_t itrk, ChamberType odd_even = BOTH);

  /// does it hit odd GEM chamber number, even, both or none?
  ChamberType chamberTypesHitGEM(size_t itrk, int layer);

  /// does simtrack hit odd ME1/1 chamber number, even, both or none?
  ChamberType chamberTypesHitCSC(size_t itrk);

  /// for simtrack number itrk, calculate average global position of its simhits in GEM
  GlobalPoint meanSimHitsPositionGEM(size_t itrk, int layer, ChamberType odd_even);

  /// for simtrack number itrk, calculate average global position of its simhits in CSC
  GlobalPoint meanSimHitsPositionCSC(size_t itrk, ChamberType odd_even);

  /// for simtrack number itrk, calculate its propagated intersection point
  /// with a z-plane defined by average position of its simhits in GEM
  GlobalPoint propagatedPositionGEM(size_t itrk, int layer, ChamberType odd_even);

  /// for simtrack number itrk, calculate its propagated intersection point
  /// with a z-plane defined by average position of its simhits in CSC
  GlobalPoint propagatedPositionCSC(size_t itrk, ChamberType odd_even);

private:

  void addSimTrack(const SimTrack * trk, const SimVertex * vtx);

  /// for simtrack number itrk, calculate its propagated intersection point with a plane at z
  GlobalPoint propagateToZ(size_t itrk, float z);

  edm::ESHandle<MagneticField> magfield_;
  edm::ESHandle<Propagator> propagator_;
  edm::ESHandle<Propagator> propagatorOpposite_;

  std::vector<unsigned int> trackids_;
  std::map<unsigned int, SimTrackExtra> trackid_to_trackextra_;
  std::map<uint32_t, std::vector<GlobalPoint> > gemid_to_simhits_;
  std::map<uint32_t, std::vector<GlobalPoint> > cscid_to_simhits_;

  /// minimum track momentum
  double minTrackPt_;
};
