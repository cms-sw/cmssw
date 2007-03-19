#include "Validation/RecoMuon/src/RecoMuonValidator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TMath.h"

using namespace reco;
using namespace edm;
using namespace std;

typedef TrajectoryStateOnSurface TSOS;

RecoMuonValidator::RecoMuonValidator(const ParameterSet& pset)
{
  outputFileName_ = pset.getUntrackedParameter<string>("outputFileName");

  minPt_ = pset.getUntrackedParameter<double>("minPt");
  maxPt_ = pset.getUntrackedParameter<double>("maxPt");

  nBinEta_ = pset.getUntrackedParameter<unsigned int>("nBinEta");
  minEta_  = pset.getUntrackedParameter<double>("minEta");
  maxEta_  = pset.getUntrackedParameter<double>("maxEta");

  nBinPhi_ = pset.getUntrackedParameter<unsigned int>("nBinPhi");
  minPhi_  = pset.getUntrackedParameter<double>("minPhi", -TMath::Pi());
  maxPhi_  = pset.getUntrackedParameter<double>("maxPhi",  TMath::Pi());

  // (recQ/recPt-simQ/simPt)/(simQ/simPt)
  nBinErrQPt_      = pset.getUntrackedParameter<unsigned int>("nBinErrQPt"); 
  widthStaErrQPt_  = pset.getUntrackedParameter<double>("widthStaErrQPt");
  widthGlbErrQPt_  = pset.getUntrackedParameter<double>("widthGlbErrQPt");
  widthSeedErrQPt_ = pset.getUntrackedParameter<double>("widthSeedErrQPt");

  nBinPull_ = pset.getUntrackedParameter<unsigned int>("nBinPull");
  widthPull_ = pset.getUntrackedParameter<double>("widthPull");

  nHits_ = pset.getUntrackedParameter<unsigned int>("nHits");
 
  // Track Labels
  simTrackLabel_ = pset.getParameter<InputTag>("SimTrack");
  staTrackLabel_ = pset.getParameter<InputTag>("StaTrack");
  glbTrackLabel_ = pset.getParameter<InputTag>("GlbTrack");
  tkTrackLabel_  = pset.getParameter<InputTag>("TkTrack");
  seedLabel_     = pset.getParameter<InputTag>("Seed");

  // the service parameters
  ParameterSet serviceParameters 
    = pset.getParameter<ParameterSet>("ServiceParameters");
  theMuonService_ = new MuonServiceProxy(serviceParameters);

  seedPropagatorName_ = pset.getParameter<string>("SeedPropagator");

  //theDQM_ = edm::Service<DaqMonitorBEInterface>().operator->();
}

RecoMuonValidator::~RecoMuonValidator()
{
  delete theMuonService_;
//  hSimEtaVsPhi_ ->Delete();
//  hStaEtaVsPhi_ ->Delete();
//  hGlbEtaVsPhi_ ->Delete();
//  hSeedEtaVsPhi_->Delete();

//  hEtaVsNDtSimHits_ ->Delete();
//  hEtaVsNCSCSimHits_->Delete();
//  hEtaVsNRPCSimHits_->Delete();
//  hEtaVsNSimHits_   ->Delete();

//  hSeedEtaVsNHits_->Delete();
//  hStaEtaVsNHits_ ->Delete();
//  hGlbEtaVsNHits_ ->Delete();

  delete hStaResol_ ;
  delete hGlbResol_ ;
  delete hSeedResol_;

//  outputFile_->Delete();
}

void RecoMuonValidator::beginJob(const EventSetup& eventSetup)
{
  // Start Histogram booking
  outputFile_ = new TFile(outputFileName_.c_str(), "RECREATE");
  outputFile_->cd();

  hSimEtaVsPhi_  = new TH2F("SimEtaVsPhi", "Sim #eta vs #phi",
                            nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
  hStaEtaVsPhi_  = new TH2F("StaEtaVsPhi", "Sta #eta vs #phi",
                            nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
  hGlbEtaVsPhi_  = new TH2F("GlbEtaVsPhi", "Glb #eta vs #phi",
                            nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
  hTkEtaVsPhi_   = new TH2F("TkEtaVsPhi" , "Tk #eta vs #phi",
                            nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
  hSeedEtaVsPhi_ = new TH2F("SeedEtaVsPhi", "Seed #eta vs #phi",
                            nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);

  hEtaVsNDtSimHits_  = new TH2F("SimEtaVsNDtHits", "Sim #eta vs number of DT SimHits",
                                nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));
  hEtaVsNCSCSimHits_ = new TH2F("SimEtaVsNCSCHits", "Sim #eta vs number of CSC SimHits",
                                nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));
  hEtaVsNRPCSimHits_ = new TH2F("SimEtaVsNRPCHits", "Sim #eta vs number of RPC SimHits",
                                nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));
  hEtaVsNSimHits_    = new TH2F("SimEtaVsNHits", "Sim #eta vs number of Hits",
                                nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));

  hSeedEtaVsNHits_ = new TH2F("SeedEtaVsNHits", "Seed #eta vs NHits",
                              nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));
  hStaEtaVsNHits_  = new TH2F("StaEtaVsNHits", "Sta #eta vs NHits",
                              nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));
  hGlbEtaVsNHits_  = new TH2F("GlbEtaVsNHits", "Glb #eta vs NHits",
                              nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));

  hStaResol_  = new HResolution("Sta", 
                                nBinErrQPt_, widthStaErrQPt_, nBinPull_, widthPull_, 
                                nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
  hGlbResol_  = new HResolution("Glb", 
                                nBinErrQPt_, widthGlbErrQPt_, nBinPull_, widthPull_, 
                                nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
  hSeedResol_ = new HResolution("Seed", 
                                nBinErrQPt_, widthSeedErrQPt_, nBinPull_, widthPull_, 
                                nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
}

void RecoMuonValidator::endJob()
{
  outputFile_->cd();

  hSimEtaVsPhi_ ->Write();
  hStaEtaVsPhi_ ->Write();
  hGlbEtaVsPhi_ ->Write();
  hTkEtaVsPhi_  ->Write();
  hSeedEtaVsPhi_->Write();

  hEtaVsNDtSimHits_ ->Write();
  hEtaVsNCSCSimHits_->Write();
  hEtaVsNRPCSimHits_->Write();
  hEtaVsNSimHits_   ->Write();

  hSeedEtaVsNHits_->Write();
  hStaEtaVsNHits_ ->Write();
  hGlbEtaVsNHits_ ->Write();

  hStaResol_ ->write();
  hGlbResol_ ->write();
  hSeedResol_->write();

  outputFile_->Close();
}

void RecoMuonValidator::analyze(const Event& event, const EventSetup& eventSetup)
{
  theMuonService_->update(eventSetup);

  // get a SimMuon Track from the event.
  int nSimMuon = 0;
  Handle<SimTrackContainer> simTracks;
  event.getByLabel(simTrackLabel_, simTracks);
  SimTrackContainer::const_iterator candSimMuon = simTracks->end();
  for ( SimTrackContainer::const_iterator iSimTrack = simTracks->begin();
        iSimTrack!=simTracks->end(); iSimTrack++ ) {
    if ( abs(iSimTrack->type()) != 13 ) continue;
    candSimMuon = iSimTrack;
    nSimMuon++;
  }
  if ( nSimMuon >= 2 ) LogInfo("EventInfo") << "More than 1 simMuon, n = " << nSimMuon;
  if ( nSimMuon == 0 ) {
    LogInfo("EventInfo") << "No SimTrack!!"; 
    return;
  }

  SimTrack simTrack = *candSimMuon;
  const double simPt  = simTrack.momentum().perp();
  const double simEta = simTrack.momentum().eta();
  const double simPhi = simTrack.momentum().phi();
  if ( simPt  < minPt_  || simPt  > maxPt_  ) return;
  if ( simEta < minEta_ || simEta > maxEta_ ) return;
  if ( simPhi < minPhi_ || simPhi > maxPhi_ ) return;

  hSimEtaVsPhi_->Fill(simEta, simPhi);
  
  // Get and fill Number of Hits
  int nDtSimHits  = getNSimHits(event, "MuonDTHits" , simTrack.trackId());
  int nCSCSimHits = getNSimHits(event, "MuonCSCHits", simTrack.trackId()); 
  int nRPCSimHits = getNSimHits(event, "MuonRPCHits", simTrack.trackId());
  int nSimHits = nDtSimHits+nCSCSimHits+nRPCSimHits;

  hEtaVsNDtSimHits_ ->Fill(simEta, nDtSimHits );
  hEtaVsNCSCSimHits_->Fill(simEta, nCSCSimHits);
  hEtaVsNRPCSimHits_->Fill(simEta, nRPCSimHits);
  hEtaVsNSimHits_->Fill(simEta, nSimHits);

  Handle<TrajectorySeedCollection> seeds;
  event.getByLabel(seedLabel_, seeds);
  if ( seeds->size() > 0 ) {
    pair<TSOS, TrajectorySeed> seedInfo = matchTrack(simTrack, seeds);
    hSeedEtaVsPhi_->Fill(simEta, simPhi);
    hSeedEtaVsNHits_->Fill(simEta, seedInfo.second.nHits());
    hSeedResol_->fillInfo(simTrack, seedInfo.first);
  }

  Handle<TrackCollection> staTracks;
  event.getByLabel(staTrackLabel_, staTracks);
  if ( staTracks->size() > 0 ) {
    pair<TSOS, TransientTrack> staInfo = matchTrack(simTrack, staTracks);
    hStaEtaVsPhi_->Fill(simEta, simPhi);
    hStaEtaVsNHits_->Fill(simEta, staInfo.second.numberOfValidHits());
    hStaResol_->fillInfo(simTrack, staInfo.first);

    Handle<TrackCollection> tkTracks;
    event.getByLabel(tkTrackLabel_, tkTracks);
    if ( tkTracks->size() > 0 
         && staInfo.second.pt() > 1.0
         && staInfo.second.innerMomentum().Rho() > 1.0
         && staInfo.second.innerMomentum().R() > 2.5 ) {  
      pair<TSOS, TransientTrack> tkInfo = matchTrack(simTrack, tkTracks);
      if ( tkInfo.second.p() > 2.5 && tkInfo.second.pt() > 1.0 ) {
        hTkEtaVsPhi_->Fill(simEta, simPhi);
      }
    }
  }

  Handle<TrackCollection> glbTracks;
  event.getByLabel(glbTrackLabel_, glbTracks);
  if ( glbTracks->size() > 0 ) {
    pair<TSOS, TransientTrack> glbInfo = matchTrack(simTrack, glbTracks);
    hGlbEtaVsPhi_->Fill(simEta, simPhi);
    hGlbEtaVsNHits_->Fill(simEta, glbInfo.second.numberOfValidHits());
    hGlbResol_->fillInfo(simTrack, glbInfo.first);
  }
}

pair<TSOS, TransientTrack> RecoMuonValidator::matchTrack(const SimTrack& simTrack, 
                                                          Handle<TrackCollection> recTracks)
{
  double candDeltaR = -999.0;

  TransientTrack candTrack;
  TSOS candTSOS;

  for ( TrackCollection::const_iterator recTrack = recTracks->begin();
        recTrack != recTracks->end(); recTrack++ ) { 
    TransientTrack track(*recTrack, &*theMuonService_->magneticField(), 
                         theMuonService_->trackingGeometry());
    TSOS tsos = track.impactPointState();

    GlobalVector tsosVect = tsos.globalMomentum();
    Hep3Vector trackVect = Hep3Vector(tsosVect.x(), tsosVect.y(), tsosVect.z());
    double deltaR = trackVect.deltaR(simTrack.momentum().vect());

    if ( candDeltaR < 0 || deltaR < candDeltaR ) {
      LogDebug("RecoMuonValidator") << "Matching Track with DeltaR = " << deltaR;
      candDeltaR = deltaR;
      candTrack  = track;
      candTSOS   = tsos;
    }
  }
  pair<TSOS, TransientTrack> retVal(candTSOS, candTrack);
  return retVal;
}

pair<TSOS, TrajectorySeed> RecoMuonValidator::matchTrack(const SimTrack& simTrack,
                                                          Handle<TrajectorySeedCollection> seeds)
{
  double candDeltaR = -999.0;

  TrajectorySeed candSeed;
  TSOS candTSOS;

  for ( TrajectorySeedCollection::const_iterator iSeed = seeds->begin();
        iSeed != seeds->end(); iSeed++ ) {
    TSOS tsos = getSeedTSOS(*iSeed);

    GlobalVector tsosVect = tsos.globalMomentum();
    Hep3Vector seedVect(tsosVect.x(), tsosVect.y(), tsosVect.z());
    double deltaR = seedVect.deltaR(simTrack.momentum().vect());

    if ( candDeltaR < 0 || deltaR < candDeltaR ) {
      LogDebug("RecoMuonValidator") << "Matching Track with DeltaR = " << deltaR;
      candDeltaR = deltaR;
      candSeed   = *iSeed;
      candTSOS   = tsos;
    }
  }
  return pair<TSOS, TrajectorySeed>(candTSOS, candSeed);
}

TSOS RecoMuonValidator::getSeedTSOS(const TrajectorySeed& seed)
{
  // Get the Trajectory State on Det (persistent version of a TSOS) from the seed
  PTrajectoryStateOnDet pTSOD = seed.startingState();

  // Transform it in a TrajectoryStateOnSurface
  TrajectoryStateTransform tsTransform;

  DetId seedDetId(pTSOD.detId());

  const GeomDet* gdet = theMuonService_->trackingGeometry()->idToDet( seedDetId );

  TSOS initialState = tsTransform.transientState(pTSOD, &(gdet->surface()), &*theMuonService_->magneticField());

  // Get the layer on which the seed relies
  const DetLayer *initialLayer = theMuonService_->detLayerGeometry()->idToLayer( seedDetId );

  PropagationDirection detLayerOrder = oppositeToMomentum;

  // ask for compatible layers
  vector<const DetLayer*> detLayers;
  //  if(theNavigationType == "Standard")
    detLayers = initialLayer->compatibleLayers( *initialState.freeState(),detLayerOrder);
    //   else if (theNavigationType == "Direct"){
    //     DirectMuonNavigation navigation( &*theMuonService_->detLayerGeometry() );
    //     detLayers = navigation.compatibleLayers( *initialState.freeState(),detLayerOrder);
    //   }
    //   else
    //     edm::LogError(metname) << "No Properly Navigation Selected!!"<<endl;
  TSOS result = initialState;
  if(detLayers.size()){
    const DetLayer* finalLayer = detLayers.back();
    const TSOS propagatedState = theMuonService_->propagator(seedPropagatorName_)->propagate(initialState, finalLayer->surface());
    if(propagatedState.isValid())
      result = propagatedState;
  }

  return result;
}

int RecoMuonValidator::getNSimHits(const Event& event, string simHitLabel, unsigned int trackId)
{
  int nSimHits = 0;
  Handle<PSimHitContainer> simHits;
  event.getByLabel("g4SimHits", simHitLabel.c_str(), simHits);
  for ( PSimHitContainer::const_iterator iSimHit = simHits->begin();
        iSimHit != simHits->end(); iSimHit++ ) {
    if ( iSimHit->trackId() == trackId ) {
      nSimHits++;
    }
  }
  return nSimHits;
}
