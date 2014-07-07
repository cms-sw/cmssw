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
  GEMDigiTrackMatch(DQMStore* , edm::EDGetToken& ,edm::EDGetToken&, edm::ParameterSet);
  ~GEMDigiTrackMatch();
  void analyze(const edm::Event& e, const edm::EventSetup&);
  void bookHisto(const GEMGeometry* geom);
	void FillWithTrigger( MonitorElement* me[4][3], bool array[3][2], Float_t value);
 private:

  MonitorElement* track_eta[3];
  MonitorElement* track_phi[3];

  //MonitorElement* track_dg_eta;
  //MonitorElement* track_sh_eta;

  MonitorElement* dg_eta[4][3];
  MonitorElement* dg_sh_eta[4][3]; 


  MonitorElement* dg_phi[4][3];
  MonitorElement* dg_sh_phi[4][3]; 

  MonitorElement* pad_eta[4][3];
  MonitorElement* pad_phi[4][3];

/*
  MonitorElement* dg_lx_even;
  MonitorElement* dg_lx_odd;
  MonitorElement* dg_ly_even;
  MonitorElement* dg_ly_odd;
////
  MonitorElement* dg_lx_even[4][3][2];
  MonitorElement* dg_ly_even[4][3][2];
*/

};

#endif
