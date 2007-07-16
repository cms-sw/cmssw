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

  // Track Cuts
  staMinPt_  = pset.getParameter<double>("staMinPt");
  staMinRho_ = pset.getParameter<double>("staMinRho");
  staMinR_   = pset.getParameter<double>("staMinR");

  tkMinPt_ = pset.getParameter<double>("tkMinPt");
  tkMinP_  = pset.getParameter<double>("tkMinP");

  seedPropagatorName_ = pset.getParameter<string>("SeedPropagator");

  // the service parameters
  ParameterSet serviceParameters 
    = pset.getParameter<ParameterSet>("ServiceParameters");
  theMuonService_ = new MuonServiceProxy(serviceParameters);

  theDQMService_ = 0;
  theDQMService_ = Service<DaqMonitorBEInterface>().operator->();

  subDir_ = pset.getParameter<string>("subDir");
}

RecoMuonValidator::~RecoMuonValidator()
{
  delete theMuonService_;

  delete hStaResol_ ;
  delete hGlbResol_ ;
  delete hSeedResol_;
}

void RecoMuonValidator::beginJob(const EventSetup& eventSetup)
{
  if ( theDQMService_ ) {
    theDQMService_->cd();

    string dir = "RecoMuonTask/";
    dir+=subDir_;

    theDQMService_->setCurrentFolder(dir.c_str());

    hSimEtaVsPhi_  = theDQMService_->book2D("SimEtaVsPhi", "Sim #eta vs #phi",
                                      nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
    hStaEtaVsPhi_  = theDQMService_->book2D("StaEtaVsPhi", "Sta #eta vs #phi",
                                      nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
    hGlbEtaVsPhi_  = theDQMService_->book2D("GlbEtaVsPhi", "Glb #eta vs #phi",
                                      nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
    hTkEtaVsPhi_   = theDQMService_->book2D("TkEtaVsPhi" , "Tk #eta vs #phi",
                                      nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
    hSeedEtaVsPhi_ = theDQMService_->book2D("SeedEtaVsPhi", "Seed #eta vs #phi",
                                      nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);

    hSeedSim_effEta = theDQMService_->book1D("SeedSim_effEta","SeedSim Efficiency",nBinEta_,minEta_,maxEta_);
    hStaSim_effEta = theDQMService_->book1D("StaSim_effEta","StaSim Efficiency",nBinEta_,minEta_,maxEta_);
    hStaSeed_effEta = theDQMService_->book1D("StaSeed_effEta","StaSeed Efficiency",nBinEta_,minEta_,maxEta_);
    hGlbSim_effEta = theDQMService_->book1D("GlbSim_effEta","GlbSim Efficiency",nBinEta_,minEta_,maxEta_);
    hGlbTk_effEta = theDQMService_->book1D("GlbTk_effEta","GlbTk Efficiency",nBinEta_,minEta_,maxEta_);
    hGlbSta_effEta = theDQMService_->book1D("GlbSta_effEta","GlbSta Efficiency",nBinEta_,minEta_,maxEta_);
    hGlbSeed_effEta = theDQMService_->book1D("GlbSeed_effEta","GlbSeed Efficiency",nBinEta_,minEta_,maxEta_);
 
    hEtaVsNDtSimHits_  = theDQMService_->book2D("SimEtaVsNDtHits", "Sim #eta vs number of DT SimHits",
                                          nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));
    hEtaVsNCSCSimHits_ = theDQMService_->book2D("SimEtaVsNCSCHits", "Sim #eta vs number of CSC SimHits",
                                          nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));
    hEtaVsNRPCSimHits_ = theDQMService_->book2D("SimEtaVsNRPCHits", "Sim #eta vs number of RPC SimHits",
                                          nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));
    hEtaVsNSimHits_    = theDQMService_->book2D("SimEtaVsNHits", "Sim #eta vs number of Hits",
                                          nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));
 
    hSeedEtaVsNHits_ = theDQMService_->book2D("SeedEtaVsNHits", "Seed #eta vs NHits",
                                        nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));
    hStaEtaVsNHits_  = theDQMService_->book2D("StaEtaVsNHits", "Sta #eta vs NHits",
                                        nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));
    hGlbEtaVsNHits_  = theDQMService_->book2D("GlbEtaVsNHits", "Glb #eta vs NHits",
                                        nBinEta_, minEta_, maxEta_, nHits_, 0, static_cast<float>(nHits_));
 
    hStaResol_  = new HResolution(theDQMService_, "Sta", 
                                   nBinErrQPt_, widthStaErrQPt_, nBinPull_, widthPull_, 
                                   nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
    hGlbResol_  = new HResolution(theDQMService_, "Glb", 
                                   nBinErrQPt_, widthGlbErrQPt_, nBinPull_, widthPull_, 
                                   nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
    hSeedResol_ = new HResolution(theDQMService_, "Seed", 
                                   nBinErrQPt_, widthSeedErrQPt_, nBinPull_, widthPull_, 
                                   nBinEta_, minEta_, maxEta_, nBinPhi_, minPhi_, maxPhi_);
  }
}

void RecoMuonValidator::endJob()
{

  computeEfficiency(hSeedSim_effEta,hSeedEtaVsPhi_,hSimEtaVsPhi_);
  computeEfficiency(hStaSim_effEta,hStaEtaVsPhi_,hSimEtaVsPhi_);
  computeEfficiency(hStaSeed_effEta,hStaEtaVsPhi_,hSeedEtaVsPhi_);
  computeEfficiency(hGlbSim_effEta,hGlbEtaVsPhi_,hSimEtaVsPhi_);
  computeEfficiency(hGlbTk_effEta,hGlbEtaVsPhi_,hTkEtaVsPhi_);
  computeEfficiency(hGlbSta_effEta,hGlbEtaVsPhi_,hStaEtaVsPhi_);
  computeEfficiency(hGlbSeed_effEta,hGlbEtaVsPhi_,hSeedEtaVsPhi_);

  if ( theDQMService_ ) theDQMService_->save(outputFileName_);
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
         && staInfo.second.track().pt() > staMinPt_
         && staInfo.second.track().innerMomentum().Rho() > staMinRho_
         && staInfo.second.track().innerMomentum().R() > staMinR_ ) {  
      pair<TSOS, TransientTrack> tkInfo = matchTrack(simTrack, tkTracks);
      if ( tkInfo.second.track().p() > tkMinP_ && tkInfo.second.track().pt() > tkMinPt_ ) {
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


void RecoMuonValidator::computeEfficiency(MonitorElement *effHist, MonitorElement *recoTH2, MonitorElement *simTH2){
  TH2F * h1 =dynamic_cast<TH2F*>(&(**((MonitorElementRootH2 *)recoTH2)));
  TH1D* reco = h1->ProjectionX();

  TH2F * h2 =dynamic_cast<TH2F*>(&(**((MonitorElementRootH2 *)simTH2)));
  TH1D* sim  = h2 ->ProjectionX();

    
  TH1F *hEff = (TH1F*) reco->Clone();  
  
  hEff->Divide(sim);
  
  hEff->SetName("tmp_"+TString(reco->GetName()));
  
  // Set the error accordingly to binomial statistics
  int nBinsEta = hEff->GetNbinsX();
  for(int bin = 1; bin <=  nBinsEta; bin++) {
    float nSimHit = sim->GetBinContent(bin);
    float eff = hEff->GetBinContent(bin);
    float error = 0;
    if(nSimHit != 0 && eff <= 1) {
      error = sqrt(eff*(1-eff)/nSimHit);
    }
    hEff->SetBinError(bin, error);
    effHist->setBinContent(bin,eff);
    effHist->setBinError(bin,error);
  }
  
}
