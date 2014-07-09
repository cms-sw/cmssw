#ifndef GEMTrackMatch_H
#define GEMTrackMatch_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"


#include "Validation/MuonGEMHits/interface/SimTrackMatchManager.h"


class GEMTrackMatch 
{
public:
  GEMTrackMatch(DQMStore* dbe, edm::EDGetToken&, edm::EDGetToken& , edm::ParameterSet cfg);
  virtual ~GEMTrackMatch();
  virtual void analyze(const edm::Event& e, const edm::EventSetup&) = 0 ;

  void buildLUT();
  std::pair<int,int> getClosestChambers(int region, float phi);
	std::pair<double, double> getEtaRangeForPhi( int station );
  bool isSimTrackGood(const SimTrack& );
  void setGeometry(const GEMGeometry* geom); 
  virtual void bookHisto(const GEMGeometry* geom) = 0 ;


 protected:

  edm::ParameterSet cfg_;
  edm::EDGetToken simTracksToken_;
  edm::EDGetToken simVerticesToken_;
  DQMStore* dbe_; 
  const GEMGeometry* theGEMGeometry;   

  std::pair<std::vector<float>,std::vector<int> > positiveLUT_;
  std::pair<std::vector<float>,std::vector<int> > negativeLUT_;

	std::vector< double > etaRangeForPhi;

  edm::Handle<edm::SimTrackContainer> sim_tracks;
  edm::Handle<edm::SimVertexContainer> sim_vertices;
  
  float minPt_;
  float minEta_;
  float maxEta_;
  float radiusCenter_, chamberHeight_;
	int useRoll_;
  unsigned int nstation;
};

#endif
