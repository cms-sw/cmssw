#ifndef Validation_MuonGEMDigis_GEMDigiTrackMatch_H
#define Validation_MuonGEMDigis_GEMDigiTrackMatch_H

#include "Validation/MuonGEMHits/interface/GEMTrackMatch.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Validation/MuonHits/interface/MuonSimHitMatcher.h"
#include "Validation/MuonGEMDigis/interface/GEMDigiMatcher.h"


class GEMDigiTrackMatch : public GEMTrackMatch
{

 public:
  explicit GEMDigiTrackMatch(const edm::ParameterSet& ps);
  ~GEMDigiTrackMatch() override;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const &) override;

  void analyze(const edm::Event& e, const edm::EventSetup&) override;

 private:

  std::unique_ptr<GEMDigiMatcher> gemDigiMatcher_;

  MonitorElement* track_eta[3];
  MonitorElement* track_phi[3][3];

  MonitorElement* dg_eta[4][3];
  MonitorElement* dg_sh_eta[4][3];
  MonitorElement* pad_eta[4][3];
  MonitorElement *cluster_eta[4][3];
  MonitorElement* copad_eta[4][3];

  MonitorElement* dg_phi[4][3][3];
  MonitorElement* dg_sh_phi[4][3][3];
  MonitorElement* pad_phi[4][3][3];
  MonitorElement *cluster_phi[4][3][3];
  MonitorElement* copad_phi[4][3][3];

  edm::ESHandle<GEMGeometry> hGeom;
  const GEMGeometry* geom;
};

#endif
