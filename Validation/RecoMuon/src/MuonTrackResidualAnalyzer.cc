#include "Validation/RecoMuon/src/MuonTrackResidualAnalyzer.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "Validation/RecoMuon/src/Histograms.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TDirectory.h"

using namespace std;
using namespace edm;

/// Constructor
MuonTrackResidualAnalyzer::MuonTrackResidualAnalyzer(const edm::ParameterSet& pset){
  
    // service parameters
  ParameterSet serviceParameters = pset.getParameter<ParameterSet>("ServiceParameters");
  // the services
  theService = new MuonServiceProxy(serviceParameters);
  
  theMuonTrackLabel = pset.getParameter<InputTag>("MuonTrack");
  theSeedCollectionLabel = pset.getParameter<InputTag>("MuonSeed");

  cscSimHitLabel = pset.getParameter<InputTag>("CSCSimHit");
  dtSimHitLabel = pset.getParameter<InputTag>("DTSimHit");
  rpcSimHitLabel = pset.getParameter<InputTag>("RPCSimHit");

  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");
  
  // Sim or Real
  theDataType = pset.getParameter<InputTag>("DataType"); 
  if(theDataType.label() != "RealData" && theDataType.label() != "SimData")
    cout<<"Error in Data Type!!"<<endl;

  theEtaRange = (EtaRange) pset.getParameter<int>("EtaRange");

  theUpdator = new KFUpdator();
  theEstimator = new Chi2MeasurementEstimator(100000.);

  theMuonSimHitNumberPerEvent = 0;
}

/// Destructor
MuonTrackResidualAnalyzer::~MuonTrackResidualAnalyzer(){
  delete theUpdator;
  delete theEstimator;
  delete theService;
}

// Operations
void MuonTrackResidualAnalyzer::beginJob(const edm::EventSetup& eventSetup){
  cout<<"Begin Job"<<endl;
  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

  hDPtRef = new TH1F("DeltaPtRef","P^{in}_{t}-P^{in ref}",10000,-20,20);

  // Resolution wrt the 1D Rec Hits
  //  h1DRecHitRes = new HResolution1DRecHit("TotalRec");

  // Resolution wrt the 1d Sim Hits
  //  h1DSimHitRes = new HResolution1DRecHit("TotalSim");

  hSimHitsPerTrack  = new TH1F("SimHitsPerTrack","Number of sim hits per track",55,0,55);
  hSimHitsPerTrackVsEta  = new TH2F("SimHitsPerTrackVsEta","Number of sim hits per track VS #eta",
				    120,-3.,3.,55,0,55);
  hDeltaPtVsEtaSim = new TH2F("DeltaPtVsEtaSim","#Delta P_{t} vs #eta gen, sim quantity",120,-3.,3.,500,-250.,250.);
  hDeltaPtVsEtaSim2 = new TH2F("DeltaPtVsEtaSim2","#Delta P_{t} vs #eta gen, sim quantity",120,-3.,3.,500,-250.,250.);
}

void MuonTrackResidualAnalyzer::endJob(){
  theFile->cd();
  // Write the histos to file

  hDPtRef->Write();
  hDeltaPtVsEtaSim->Write();
  hDeltaPtVsEtaSim2->Write();
  // h1DRecHitRes->Write();
  // h1DSimHitRes->Write();
  hSimHitsPerTrack->Write();
  hSimHitsPerTrackVsEta->Write();
  theFile->Close();
}
 

void MuonTrackResidualAnalyzer::analyze(const edm::Event & event, const edm::EventSetup& eventSetup){
  cout<<"Analyze"<<endl;

  // Update the services
  theService->update(eventSetup);
  MuonPatternRecoDumper debug;
  theMuonSimHitNumberPerEvent = 0;

  // Get the SimHit collection from the event
  Handle<PSimHitContainer> dtSimHits;
  event.getByLabel(dtSimHitLabel.instance(),dtSimHitLabel.label(), dtSimHits);

  Handle<PSimHitContainer> cscSimHits;
  event.getByLabel(cscSimHitLabel.instance(),cscSimHitLabel.label(), cscSimHits);

  Handle<PSimHitContainer> rpcSimHits;
  event.getByLabel(rpcSimHitLabel.instance(),rpcSimHitLabel.label(), rpcSimHits);

  Handle<SimTrackContainer> simTracks;

  // FIXME Add the tracker one??

  // Map simhits per DetId
  map<DetId, const PSimHit* > muonSimHitsPerId =
    mapMuSimHitsPerId(dtSimHits,cscSimHits,rpcSimHits);
  
  hSimHitsPerTrack->Fill(theMuonSimHitNumberPerEvent);
  
  double etaSim=0;

  if(theDataType.label() == "SimData"){
    
    // Get the SimTrack collection from the event
    event.getByLabel(theDataType.instance(),simTracks);
    
    // Loop over the Sim tracks
    SimTrackContainer::const_iterator simTrack;
    
    for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack)
      if (abs((*simTrack).type()) == 13){
	hSimHitsPerTrackVsEta->Fill((*simTrack).momentum().eta(),theMuonSimHitNumberPerEvent);
	etaSim = (*simTrack).momentum().eta();
	theSimTkId = (*simTrack).trackId();

      }
  }

  
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> muonTracks;
  event.getByLabel(theMuonTrackLabel, muonTracks);

  reco::TrackCollection::const_iterator muonTrack;
  
  // Loop over the Rec tracks
  for (muonTrack = muonTracks->begin(); muonTrack != muonTracks->end(); ++muonTrack) {
    
    reco::TransientTrack track(*muonTrack,&*theService->magneticField(),theService->trackingGeometry());   

    TrajectoryStateOnSurface outerTSOS = track.outermostMeasurementState();
    TrajectoryStateOnSurface innerTSOS = track.innermostMeasurementState();

    TransientTrackingRecHit::ConstRecHitContainer result;
    
    trackingRecHit_iterator rhend = track.recHitsBegin()-1;
    trackingRecHit_iterator rhbegin = track.recHitsEnd()-2;
    
    // SimHit Energy loss analysis  
    double momAtEntry = -150., momAtExit = -150.;

    if(theSimHitContainer.size()>1){

      const GeomDet *geomDetA = theService->trackingGeometry()->idToDet(DetId(theSimHitContainer.front()->detUnitId()));
      double distA = geomDetA->toGlobal(theSimHitContainer.front()->localPosition()).mag();

      const GeomDet *geomDetB = theService->trackingGeometry()->idToDet(DetId(theSimHitContainer.back()->detUnitId()));
      double distB = geomDetB->toGlobal(theSimHitContainer.back()->localPosition()).mag();

      cout<<"Inner SimHit: "<<theSimHitContainer.front()->particleType()
	  <<" Pt: "<<theSimHitContainer.front()->momentumAtEntry().perp()
	  <<" E: "<<theSimHitContainer.front()->momentumAtEntry().perp()
	  <<" R: "<<distA<<endl;
      cout<<"Outer SimHit: "<<theSimHitContainer.back()->particleType()
	  <<" Pt: "<<theSimHitContainer.back()->momentumAtEntry().perp()
	  <<" E: "<<theSimHitContainer.front()->momentumAtEntry().perp()
	  <<" R: "<<distB<<endl;

      momAtEntry = theSimHitContainer.front()->momentumAtEntry().perp();
      momAtExit =  theSimHitContainer.back()->momentumAtEntry().perp();
    }

    trackingRecHit_iterator rhFirst = track.recHitsBegin();
    trackingRecHit_iterator rhLast = track.recHitsEnd()-1;
    map<DetId,const PSimHit*>::const_iterator itFirst = muonSimHitsPerId.find((*rhFirst)->geographicalId());
    map<DetId,const PSimHit*>::const_iterator itLast = muonSimHitsPerId.find((*rhLast)->geographicalId());
    
    double momAtEntry2 = -150, momAtExit2 = -150.;
    if (itFirst != muonSimHitsPerId.end() )
      momAtEntry2 = itFirst->second->momentumAtEntry().perp();
    else {
      cout<<"No first sim hit found"<<endl;
      ++rhFirst;
      itFirst = muonSimHitsPerId.find((*rhFirst)->geographicalId());
      if (itFirst != muonSimHitsPerId.end() )
	momAtEntry2 = itFirst->second->momentumAtEntry().perp();
      else{
	cout<<"No second sim hit found"<<endl;
	// continue;
      }
    }
    
    if (itLast != muonSimHitsPerId.end() )
      momAtExit2 = itLast->second->momentumAtEntry().perp();
    else {
      cout<<"No last sim hit found"<<endl;
      --rhLast;
      itLast = muonSimHitsPerId.find((*rhLast)->geographicalId());
      if (itLast != muonSimHitsPerId.end() )
	momAtExit2 = itLast->second->momentumAtEntry().perp();
      else{
	cout<<"No last but one sim hit found"<<endl;
	// continue;
      }
    }
    
    if(etaSim){
      if(momAtEntry >=0 && momAtExit >= 0)
	hDeltaPtVsEtaSim->Fill(etaSim,momAtEntry-momAtExit);
      if(momAtEntry2 >=0 && momAtExit2 >= 0)
	hDeltaPtVsEtaSim2->Fill(etaSim,momAtEntry2-momAtExit2);
    }
    else
      cout<<"NO SimTrack'eta"<<endl;
    //

    // computeResolution(trajectoryBW,muonSimHitsPerId,h1DSimHitRes);
    // computeResolution(smoothed,muonSimHitsPerId,h1DSimHitRes);
    
  }
}

bool MuonTrackResidualAnalyzer::isInTheAcceptance(double eta){
  switch(theEtaRange){
  case all:
    return ( abs(eta) <= 2.4 ) ? true : false;
  case barrel:
    return ( abs(eta) < 1.1 ) ? true : false;
  case endcap:
    return ( abs(eta) >= 1.1 && abs(eta) <= 2.4 ) ? true : false;  
  default:
    {cout<<"No correct Eta range selected!! "<<endl; return false;}
  }
}

// map the muon simhits by id
map<DetId,const PSimHit*>
MuonTrackResidualAnalyzer::mapMuSimHitsPerId(Handle<PSimHitContainer> dtSimhits,
					     Handle<PSimHitContainer> cscSimhits,
					     Handle<PSimHitContainer> rpcSimhits) {
  
  MuonPatternRecoDumper debug;
  
  map<DetId,const PSimHit*> hitIdMap;
  theSimHitContainer.clear();

  mapMuSimHitsPerId(dtSimhits,hitIdMap);
  mapMuSimHitsPerId(cscSimhits,hitIdMap);
  mapMuSimHitsPerId(rpcSimhits,hitIdMap);

  if(theSimHitContainer.size() >1)
    stable_sort(theSimHitContainer.begin(),theSimHitContainer.end(),RadiusComparatorInOut(theService->trackingGeometry()));

  cout<<"Sim Hit list"<<endl;
  int count=1;
  for(vector<const PSimHit*>::const_iterator it = theSimHitContainer.begin(); 
      it != theSimHitContainer.end(); ++it){
    cout<<count 
	<< " " 
	<< " Process Type: " << (*it)->processType()
	<< " " 
	<< debug.dumpMuonId(DetId( (*it)->detUnitId() ))<<endl;
  }  

  return hitIdMap;
}


void MuonTrackResidualAnalyzer::mapMuSimHitsPerId(Handle<PSimHitContainer> simhits, 
						  map<DetId,const PSimHit*> &hitIdMap){
  
  for(PSimHitContainer::const_iterator simhit = simhits->begin();
      simhit != simhits->end(); ++simhit) {
    
    if ( abs(simhit->particleType()) != 13 && theSimTkId != simhit->trackId()) continue; 
    
    theSimHitContainer.push_back(&*simhit);
    DetId id = DetId(simhit->detUnitId());
    
    if(id.subdetId() == MuonSubdetId::DT){
      DTLayerId lId(id.rawId());
      id = DetId(lId.rawId());
    }

    map<DetId,const PSimHit*>::const_iterator it = hitIdMap.find(id);
    
    if (it == hitIdMap.end() )
      hitIdMap[id] = &*simhit;
    else
      cout<<"TWO muons in the same sensible volume!!"<<endl;
    
    ++theMuonSimHitNumberPerEvent;
  }
}


void MuonTrackResidualAnalyzer::computeResolution(Trajectory &trajectory, 
						  map<DetId,const PSimHit*> &hitIdMap,
						  HResolution1DRecHit *histos){
  
  Trajectory::DataContainer data = trajectory.measurements();

  for(Trajectory::DataContainer::const_iterator datum = data.begin();
      datum != data.end(); ++datum){
    
    GlobalPoint fitPoint = datum->updatedState().globalPosition();

    // FIXME!
    //     double errX = datum->updatedState().cartesianError().matrix()[0][0];
    //     double errY = datum->updatedState().cartesianError().matrix()[1][1];
    //     double errZ = datum->updatedState().cartesianError().matrix()[2][2];
    //
    double errX = datum->updatedState().localError().matrix()[3][3];
    double errY = datum->updatedState().localError().matrix()[4][4];
    double errZ = 1.;

    map<DetId,const PSimHit*>::const_iterator it = hitIdMap.find(datum->recHit()->geographicalId());
 
    
    if (it == hitIdMap.end() ) continue; // FIXME! Put a counter
    
    const PSimHit* simhit = it->second;
    
    LocalPoint simHitPoint = simhit->localPosition();

    const GeomDet* geomDet = theService->trackingGeometry()->idToDet(DetId(simhit->detUnitId()));

    LocalPoint fitLocalPoint = geomDet->toLocal(fitPoint);
    
    LocalVector diff = fitLocalPoint-simHitPoint;

    
    cout << "SimHit position "<< simHitPoint << endl;
    cout << "Fit position "<<  fitLocalPoint << endl;
    cout << "Fit position2 "<< datum->updatedState().localPosition() << endl;
    cout << "Errors on the fit position: (" << errX << "," << errY << "," << errZ << ")"<<endl;
    cout << "Resolution on x: " << diff.x()/abs(simHitPoint.x()) << endl;
    cout << "Resolution on y: " << diff.y()/abs(simHitPoint.y()) << endl;
    cout << "Resolution on z: " << diff.z()/abs(simHitPoint.z()) << endl;

    cout << "Eta direction: "<< simhit->momentumAtEntry().eta() <<" eta position: " << simHitPoint.eta() << endl;
    cout << "Phi direction: "<< simhit->momentumAtEntry().phi() <<" phi position: " << simHitPoint.phi() << endl;

    
    histos->Fill( simHitPoint.x(), simHitPoint.y(), simHitPoint.z(), 
		  diff.x(), diff.y(), diff.z(),
		  errX,  errY, errZ,
		  simhit->momentumAtEntry().eta(), simhit->momentumAtEntry().phi());
		  // simHitPoint.eta(), simHitPoint.phi() ); // FIXME!
  }


}
