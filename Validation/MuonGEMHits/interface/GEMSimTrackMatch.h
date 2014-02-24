#ifndef GEMSimTrackMatch_H
#define GEMSimTrackMatch_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"


#include "Validation/MuonGEMHits/interface/SimTrackMatchManager.h"


class GEMSimTrackMatch 
{
public:
  GEMSimTrackMatch(DQMStore* , std::string , edm::ParameterSet);
  ~GEMSimTrackMatch();
  void analyze(const edm::Event& e, const edm::EventSetup&);

  void buildLUT();
  std::pair<int,int> getClosestChambers(int region, float phi);
  bool isSimTrackGood(const SimTrack& );
  void setGeometry(const GEMGeometry* geom); 



 private:

  edm::ParameterSet cfg_;
  std::string simInputLabel_;
  DQMStore* dbe_; 
  const GEMGeometry* theGEMGeometry;   

  
  MonitorElement* track_eta;
  MonitorElement* track_eta_l1;
  MonitorElement* track_eta_l2;
  MonitorElement* track_eta_l1or2;
  MonitorElement* track_eta_l1and2;

  MonitorElement* track_phi;
  MonitorElement* track_phi_l1;
  MonitorElement* track_phi_l2;
  MonitorElement* track_phi_l1or2;
  MonitorElement* track_phi_l1and2;

  MonitorElement* gem_lx_even;
  MonitorElement* gem_lx_even_l1;
  MonitorElement* gem_lx_even_l2;
  MonitorElement* gem_lx_even_l1or2;
  MonitorElement* gem_lx_even_l1and2;

  MonitorElement* gem_ly_even;
  MonitorElement* gem_ly_even_l1;
  MonitorElement* gem_ly_even_l2;
  MonitorElement* gem_ly_even_l1or2;
  MonitorElement* gem_ly_even_l1and2;

  MonitorElement* gem_lx_odd;
  MonitorElement* gem_lx_odd_l1;
  MonitorElement* gem_lx_odd_l2;
  MonitorElement* gem_lx_odd_l1or2;
  MonitorElement* gem_lx_odd_l1and2;

  MonitorElement* gem_ly_odd;
  MonitorElement* gem_ly_odd_l1;
  MonitorElement* gem_ly_odd_l2;
  MonitorElement* gem_ly_odd_l1or2;
  MonitorElement* gem_ly_odd_l1and2;





  
  std::pair<std::vector<float>,std::vector<int> > positiveLUT_;
  std::pair<std::vector<float>,std::vector<int> > negativeLUT_;

  edm::Handle<edm::SimTrackContainer> sim_tracks;
  edm::Handle<edm::SimVertexContainer> sim_vertices;
  
  float minPt_;
  float minEta_;
  float maxEta_;
  float radiusCenter_, chamberHeight_;


};

#endif
