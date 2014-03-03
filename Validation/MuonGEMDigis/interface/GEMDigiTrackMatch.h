#ifndef GEMDigiTrackMatch_H
#define GEMDigiTrackMatch_H

#include "Validation/MuonGEMHits/interface/GEMTrackMatch.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

class GEMDigiTrackMatch : public GEMTrackMatch 
{
public:
  GEMDigiTrackMatch(DQMStore* , std::string , edm::ParameterSet);
  ~GEMDigiTrackMatch();
  void analyze(const edm::Event& e, const edm::EventSetup&);
  void bookHisto();

 private:

  MonitorElement* track_eta;
  MonitorElement* track_phi;

  MonitorElement* track_dg_eta;
  MonitorElement* track_sh_eta;

  MonitorElement* dg_eta[4];
  MonitorElement* dg_sh_eta[4]; 


  MonitorElement* dg_phi[4];
  MonitorElement* dg_sh_phi[4]; 

  MonitorElement* pad_eta[4];
  MonitorElement* pad_phi[4];


  MonitorElement* dg_lx_even;
  MonitorElement* dg_lx_even_l1;
  MonitorElement* dg_lx_even_l2;
  MonitorElement* dg_lx_even_l1or2;
  MonitorElement* dg_lx_even_l1and2;

  MonitorElement* dg_ly_even;
  MonitorElement* dg_ly_even_l1;
  MonitorElement* dg_ly_even_l2;
  MonitorElement* dg_ly_even_l1or2;
  MonitorElement* dg_ly_even_l1and2;

  MonitorElement* dg_lx_odd;
  MonitorElement* dg_lx_odd_l1;
  MonitorElement* dg_lx_odd_l2;
  MonitorElement* dg_lx_odd_l1or2;
  MonitorElement* dg_lx_odd_l1and2;

  MonitorElement* dg_ly_odd;
  MonitorElement* dg_ly_odd_l1;
  MonitorElement* dg_ly_odd_l2;
  MonitorElement* dg_ly_odd_l1or2;
  MonitorElement* dg_ly_odd_l1and2;

};

#endif
