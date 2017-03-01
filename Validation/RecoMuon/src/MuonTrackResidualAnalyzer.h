#ifndef Validation_RecoMuon_MuonTrackResidualAnalyzer_H
#define Validation_RecoMuon_MuonTrackResidualAnalyzer_H

/** \class MuonTrackResidualAnalyzer
 *  No description available.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */ 

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"


#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>


namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}
class DQMStore;
class HTracks;
class HResolution;

class MuonServiceProxy;
class KFUpdator;
class MeasurementEstimator;
class HResolution1DRecHit;

class MuonTrackResidualAnalyzer: public DQMEDAnalyzer {
  
 public:
  enum EtaRange{all,barrel,endcap};

public:
  /// Constructor
  MuonTrackResidualAnalyzer(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~MuonTrackResidualAnalyzer();
  
  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup) override;

  virtual void beginJob() ;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;

protected:

private:
  bool isInTheAcceptance(double eta);
  
  std::map<DetId, const PSimHit*> mapMuSimHitsPerId(edm::Handle<edm::PSimHitContainer> dtSimhits,
						    edm::Handle<edm::PSimHitContainer> cscSimhits,
						    edm::Handle<edm::PSimHitContainer> rpcSimhits);
  
  void mapMuSimHitsPerId(edm::Handle<edm::PSimHitContainer> simhits, 
			 std::map<DetId,const PSimHit*> &hitIdMap);
  
  void computeResolution(Trajectory &trajectory, 
			 std::map<DetId,const PSimHit*> &hitIdMap,
			 HResolution1DRecHit *histos);
  
 private:
  
  DQMStore* dbe_;
  std::string dirName_;
  std::string subsystemname_;
  edm::ParameterSet pset;
  std::string out;
  
  edm::InputTag theDataType;
  edm::EDGetTokenT<edm::SimTrackContainer> theDataTypeToken;
  EtaRange theEtaRange;
  
  edm::InputTag theMuonTrackLabel;
  edm::InputTag cscSimHitLabel;
  edm::InputTag dtSimHitLabel;
  edm::InputTag rpcSimHitLabel;

  edm::EDGetTokenT<reco::TrackCollection> theMuonTrackToken;
  edm::EDGetTokenT<std::vector<PSimHit> > theCSCSimHitToken;
  edm::EDGetTokenT<std::vector<PSimHit> > theDTSimHitToken;
  edm::EDGetTokenT<std::vector<PSimHit> > theRPCSimHitToken;

  MuonServiceProxy *theService;
  KFUpdator *theUpdator;
  MeasurementEstimator *theEstimator;

 private:
  MonitorElement *hDPtRef;
 
  // Resolution wrt the 1D Rec Hits
  HResolution1DRecHit *h1DRecHitRes;
  
  // Resolution wrt the 1d Sim Hits
  HResolution1DRecHit  *h1DSimHitRes;

  MonitorElement *hSimHitsPerTrack;
  MonitorElement *hSimHitsPerTrackVsEta; 
  MonitorElement *hDeltaPtVsEtaSim;
  MonitorElement *hDeltaPtVsEtaSim2;

  int theMuonSimHitNumberPerEvent;
  
  unsigned int theSimTkId;


  std::vector<const PSimHit*> theSimHitContainer;


  struct RadiusComparatorInOut{

    RadiusComparatorInOut(edm::ESHandle<GlobalTrackingGeometry> tg):theTG(tg){}
    
    bool operator()(const PSimHit *a,
		    const PSimHit *b) const{ 
      
      const GeomDet *geomDetA = theTG->idToDet(DetId(a->detUnitId()));
      const GeomDet *geomDetB = theTG->idToDet(DetId(b->detUnitId()));
      
      double distA = geomDetA->toGlobal(a->localPosition()).mag();
      double distB = geomDetB->toGlobal(b->localPosition()).mag();
      
      return distA < distB; 
    }

    edm::ESHandle<GlobalTrackingGeometry> theTG;
  };


};
#endif

