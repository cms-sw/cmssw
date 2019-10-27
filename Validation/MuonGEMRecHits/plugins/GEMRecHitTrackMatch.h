#ifndef Validation_MuonGEMRecHits_GEMRecHitTrackMatch_H
#define Validation_MuonGEMRecHits_GEMRecHitTrackMatch_H

#include "Validation/MuonGEMHits/interface/GEMTrackMatch.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Validation/MuonGEMRecHits/interface/GEMRecHitMatcher.h"


class GEMRecHitTrackMatch : public GEMTrackMatch
{
public:
  explicit GEMRecHitTrackMatch(const edm::ParameterSet& ps);
  ~GEMRecHitTrackMatch() override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const &) override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;

 private:

  std::unique_ptr<GEMRecHitMatcher> gemRecHitMatcher_;

  MonitorElement* track_eta[3];
  MonitorElement* track_phi[3][3];

  MonitorElement* rh_eta[4][3];
  MonitorElement* rh_sh_eta[4][3];

  MonitorElement* rh_phi[4][3][3];

  MonitorElement* rh_sh_phi[4][3][3];

  edm::ESHandle<GEMGeometry> hGeom;
  const GEMGeometry* gemGeometry_;
};

#endif
