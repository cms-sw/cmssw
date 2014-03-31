#include "Validation/MuonGEMHits/interface/SimTrackMatchManager.h"

SimTrackMatchManager::SimTrackMatchManager(const SimTrack& t, const SimVertex& v,
      const edm::ParameterSet& ps, const edm::Event& ev, const edm::EventSetup& es, const GEMGeometry* gem_geo)
: simhits_(t, v, ps, ev, es,gem_geo)
, gem_digis_(simhits_)
{}

SimTrackMatchManager::~SimTrackMatchManager() {}
