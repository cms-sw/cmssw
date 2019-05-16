#ifndef Validation_MuonGEMHits_GEMTrackMatch_H
#define Validation_MuonGEMHits_GEMTrackMatch_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"

struct MySimTrack
{
  Float_t pt, eta, phi;
  Char_t gem_sh_layer1, gem_sh_layer2;
  Char_t gem_dg_layer1, gem_dg_layer2;
  Char_t gem_pad_layer1, gem_pad_layer2;
  Char_t has_gem_dg_l1, has_gem_dg_l2;
  Char_t has_gem_pad_l1, has_gem_pad_l2;
  Char_t has_gem_sh_l1, has_gem_sh_l2;
  bool gem_sh[3][2] ;
  bool gem_dg[3][2] ;
  bool gem_pad[3][2] ;
  bool gem_rh[3][2] ;
  bool hitOdd[3];
  bool hitEven[3];
};

class GEMTrackMatch : public DQMEDAnalyzer
{
public:
  explicit GEMTrackMatch( const edm::ParameterSet&  cfg);
  ~GEMTrackMatch() override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override = 0 ;

  void buildLUT(const int maxChamberId);
  std::pair<int,int> getClosestChambers(const int maxChamberId, int region, float phi);
  std::pair<double, double> getEtaRangeForPhi( int station );
  bool isSimTrackGood(const SimTrack& );
  void setGeometry(const GEMGeometry& geom); 
  std::pair<double,double> getEtaRange(int station, int chamber ) ;

  void FillWithTrigger( MonitorElement* me[3], Float_t eta);
  void FillWithTrigger( MonitorElement* me[3][3], Float_t eta, Float_t phi, bool odd[3], bool even[3]);
  void FillWithTrigger( MonitorElement* me[4][3], bool array[3][2], Float_t value);
  void FillWithTrigger( MonitorElement* me[4][3][3], bool array[3][2], Float_t eta, Float_t phi, bool odd[3], bool even[3]);


 protected:
  edm::ParameterSet cfg_;
  edm::EDGetToken simHitsToken_;
  edm::EDGetToken simTracksToken_;
  edm::EDGetToken simVerticesToken_;

  std::pair<std::vector<float>,std::vector<int> > positiveLUT_;
  std::pair<std::vector<float>,std::vector<int> > negativeLUT_;

  std::vector< double > etaRangeForPhi;

  
  float minPt_;
  float minEta_;
  float maxEta_;
  float radiusCenter_, chamberHeight_;
  int useRoll_;
  const GEMGeometry* gem_geom_;
  unsigned int nstation;
  bool detailPlot_;
};

#endif
