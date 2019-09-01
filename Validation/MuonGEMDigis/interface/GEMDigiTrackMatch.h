#ifndef GEMDigiTrackMatch_H
#define GEMDigiTrackMatch_H

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Validation/MuonGEMDigis/interface/GEMDigiMatcher.h"
#include "Validation/MuonGEMHits/interface/GEMTrackMatch.h"
#include "Validation/MuonGEMHits/interface/SimHitMatcher.h"

class GEMDigiTrackMatch : public GEMTrackMatch {
public:
  explicit GEMDigiTrackMatch(const edm::ParameterSet &ps);
  ~GEMDigiTrackMatch() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &e, const edm::EventSetup &) override;

private:
  MonitorElement *track_eta[3];
  MonitorElement *track_phi[3][3];

  MonitorElement *dg_eta[4][3];
  MonitorElement *dg_sh_eta[4][3];
  MonitorElement *pad_eta[4][3];
  MonitorElement *copad_eta[4][3];

  MonitorElement *dg_phi[4][3][3];
  MonitorElement *dg_sh_phi[4][3][3];
  MonitorElement *pad_phi[4][3][3];
  MonitorElement *copad_phi[4][3][3];

  edm::EDGetToken gem_digiToken_;
  edm::EDGetToken gem_padToken_;
  edm::EDGetToken gem_copadToken_;

  //  std::map< UInt_t , MonitorElement* > theStrip_dcEta;
  //  std::map< UInt_t , MonitorElement* > thePad_dcEta;
  //  std::map< UInt_t , MonitorElement* > theCoPad_dcEta;
};

#endif
