#include "Validation/RecoMuon/src/RecoMuonValidator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <map>
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "TMath.h"

using namespace reco;
using namespace edm;
using namespace std;

RecoMuonValidator::RecoMuonValidator(const ParameterSet& pset)
{
  // Set Number of Bins in Histograms
  nBinPt_  = pset.getUntrackedParameter<unsigned int>("nBinPt" ); // # of Bins for pT Histo
  nBinEta_ = pset.getUntrackedParameter<unsigned int>("nBinEta"); // # of Bins for eta
  nBinPhi_ = pset.getUntrackedParameter<unsigned int>("nBinPhi"); // # of Bins for phi
  nBinDPt_ = pset.getUntrackedParameter<unsigned int>("nBinDPt"); // # of Bins for (recPt-simPt)/simPt
  nBinQPt_ = pset.getUntrackedParameter<unsigned int>("nBinQPt"); // # of Bins for (recQ/recPt-simQ/simPt)/(simQ/simPt)

  // pT, eta simTrack cut range
  minPt_   = pset.getUntrackedParameter<double>("minPt" );
  maxPt_   = pset.getUntrackedParameter<double>("maxPt" );
  minEta_  = pset.getUntrackedParameter<double>("minEta");
  maxEta_  = pset.getUntrackedParameter<double>("maxEta");
  minPhi_  = pset.getUntrackedParameter<double>("minPhi", -TMath::Pi());
  maxPhi_  = pset.getUntrackedParameter<double>("maxPhi",  TMath::Pi());

  // PTDR |sigma(q/pT)| = |(q'/pT'-q/pT)/(q/pT)|
  maxStaQPt_ = pset.getUntrackedParameter<double>("maxStaQPt");
  maxGlbQPt_ = pset.getUntrackedParameter<double>("maxGlbQPt");

  // Maximum Delta(pT)/pT = (pT'-pT)/pT  (Minimum = -1)
  maxStaDPt_ = pset.getUntrackedParameter<double>("maxStaDPt");
  maxGlbDPt_ = pset.getUntrackedParameter<double>("maxGlbDPt");

  outFileName_ = pset.getUntrackedParameter<string>("outputFileName");

  // Track Labels
  staTrackLabel_ = pset.getParameter<InputTag>("StaTrack");
  glbTrackLabel_ = pset.getParameter<InputTag>("GlbTrack");
  simTrackLabel_ = pset.getParameter<InputTag>("SimTrack");

  // the services
  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);

  //theDQM_ = edm::Service<DaqMonitorBEInterface>().operator->();
}

RecoMuonValidator::~RecoMuonValidator()
{
//  if ( hSimPt_ ) hSimPt_->Delete();
//  if ( hRecPt_  || !hRecPt_ ->IsZombie() ) delete hRecPt_ ;
//  if ( hSimEta_ || !hSimEta_->IsZombie() ) delete hSimEta_;
//  if ( hRecEta_ || !hRecEta_->IsZombie() ) delete hRecEta_;

//  if ( hDeltaPtVsEta_ || !hDeltaPtVsEta_->IsZombie() ) delete hDeltaPtVsEta_;
//  if ( hResolPtVsEta_ /*|| !hResolPtVsEta_->IsZombie()*/ ) delete hResolPtVsEta_;
//  if ( outFile_ || !outFile_->IsZombie() ) delete outFile_;
}

void RecoMuonValidator::beginJob(const EventSetup& eventSetup)
{
  // Start Histogram booking
  outFile_ = new TFile(outFileName_.c_str(), "RECREATE");
  outFile_->cd();
//  theDQM_->cd();

  hGenPt_  = new TH1F("GenPtHist" , "Generated p_{T}", nBinPt_, minPt_ , maxPt_ );
  hSimPt_  = new TH1F("SimPtHist" , "Simulated p_{T}", nBinPt_, minPt_ , maxPt_ );
  hStaPt_  = new TH1F("StaPtHist" , "Sta Muon p_{T}" , nBinPt_, minPt_ , maxPt_ );
  hGlbPt_  = new TH1F("GlbPtHist" , "Glb Muon p_{T}" , nBinPt_, minPt_ , maxPt_ );

  hGenPhiVsEta_ = new TH2F("GenPhiVsEtaHist", "Generated #phi vs #eta",
                                  nBinPhi_, minPhi_, maxPhi_, nBinEta_, minEta_, maxEta_);
  hSimPhiVsEta_ = new TH2F("SimPhiVsEtaHist", "Simulated #phi vs #eta",
                                  nBinPhi_, minPhi_, maxPhi_, nBinEta_, minEta_, maxEta_);
  hStaPhiVsEta_ = new TH2F("StaPhiVsEtaHist", "Sta Muons #phi vs #eta",
                                  nBinPhi_, minPhi_, maxPhi_, nBinEta_, minEta_, maxEta_);
  hGlbPhiVsEta_ = new TH2F("GlbPhiVsEtaHist", "Glb Muons #phi vs #eta",
                                  nBinPhi_, minPhi_, maxPhi_, nBinEta_, minEta_, maxEta_);

  hStaEtaVsDeltaPt_ = new TH2F("StaEtaVsDeltaPtHist", "Sta #eta vs #Delta(p_{T})/p_{T}",
                                      nBinEta_, minEta_, maxEta_, nBinDPt_, -1.0, maxStaDPt_);
  hGlbEtaVsDeltaPt_ = new TH2F("GlbEtaVsDeltaPtHist", "Glb #eta vs #Delta(p_{T})/p_{T}",
                                      nBinEta_, minEta_, maxEta_, nBinDPt_, -1.0, maxGlbDPt_);

  hStaEtaVsResolPt_ = new TH2F("StaEtaVsResolPtHist", "Sta #eta vs #sigma(q/p_{T})",
                                      nBinEta_, minEta_, maxEta_, nBinQPt_, -maxStaQPt_, maxStaQPt_);
  hGlbEtaVsResolPt_ = new TH2F("GlbEtaVsResolPtHist", "Glb #eta vs #sigma(q/p_{T})",
                                      nBinEta_, minEta_, maxEta_, nBinQPt_, -maxGlbQPt_, maxGlbQPt_);
}

void RecoMuonValidator::endJob()
{

  outFile_->cd();

  hGenPt_ ->Write();
  hSimPt_ ->Write();
  hStaPt_ ->Write();
  hGlbPt_ ->Write();

  hGenPhiVsEta_->Write();
  hSimPhiVsEta_->Write();
  hStaPhiVsEta_->Write();
  hGlbPhiVsEta_->Write();

  hStaEtaVsDeltaPt_->Write();
  hGlbEtaVsDeltaPt_->Write();

  hStaEtaVsResolPt_->Write();
  hGlbEtaVsResolPt_->Write();

  outFile_->Close();

  //if ( theDQM_ ) theDQM_->save(outFileName_);
}

TrackCollection::const_iterator RecoMuonValidator::matchTrack(SimTrackContainer::const_iterator simTrack,
                                                              Handle<TrackCollection>& recTracks)
{
  // RecoTrack - MC SimTrack Matching
  // This routine can produce some bugs, because this matching can be duplicated
  //  when nSimTrack > 1 ( matching 2(or more) simTracks to a common recTrack )
  double candDeltaR = 5;
  TrackCollection::const_iterator candTrack = recTracks->end();

  // Find a recTrack that gives smallest deltaR wrt the simTrack
  for ( TrackCollection::const_iterator recTrack = recTracks->begin();
        recTrack != recTracks->end(); recTrack++ ) {
    Hep3Vector trackVect(recTrack->px(), recTrack->py(), recTrack->pz());
    const double deltaR = trackVect.deltaR(simTrack->momentum().vect());
    if ( deltaR < candDeltaR ) {
      LogDebug("RecoMuonValidator") << "Matching Track with DeltaR = " << deltaR;
      candDeltaR = deltaR;
      candTrack  = recTrack;
    }
  }
  // candTrack points the minimum deltaR recTrack
  //  or, recTracks->end()
  return candTrack;
}

void RecoMuonValidator::analyze(const Event& event, const EventSetup& eventSetup)
{
  // Update the services
  theService->update(eventSetup);
  
  const static int muonPID = 13;

  // Grab all muon simTracks
  Handle<SimTrackContainer> simTracks;
  event.getByLabel(simTrackLabel_, simTracks);

  // Collect muon simHits to remove no-simHit muon tracks
  map<unsigned int, vector<const PSimHit*> > mapOfMuonSimHits;

  Handle<PSimHitContainer> dtSimHits;
  event.getByLabel("g4SimHits", "MuonDTHits", dtSimHits);
  for(PSimHitContainer::const_iterator simHit = dtSimHits->begin();
      simHit != dtSimHits->end(); simHit++) {
    if ( abs(simHit->particleType()) != muonPID ) continue;
    mapOfMuonSimHits[simHit->trackId()].push_back(&*simHit);
  }

  Handle<PSimHitContainer> cscSimHits;
  event.getByLabel("g4SimHits", "MuonCSCHits", cscSimHits);
  for(PSimHitContainer::const_iterator simHit = cscSimHits->begin();
      simHit != cscSimHits->end(); simHit++) {
    if ( abs(simHit->particleType()) != muonPID ) continue;
    mapOfMuonSimHits[simHit->trackId()].push_back(&*simHit);
  }

  Handle<PSimHitContainer> rpcSimHits;
  event.getByLabel("g4SimHits", "MuonRPCHits", rpcSimHits);
  for(PSimHitContainer::const_iterator simHit = rpcSimHits->begin();
      simHit != rpcSimHits->end(); simHit++) {
    if ( abs(simHit->particleType()) != muonPID ) continue;
    mapOfMuonSimHits[simHit->trackId()].push_back(&*simHit);
  }

  // Take the seed collection
  Handle<TrajectorySeedCollection> seeds; 
  event.getByLabel(theSeedCollectionLabel,seeds);
 
  // Grab all standalone muon tracks
  Handle<TrackCollection> staTracks;
  event.getByLabel(staTrackLabel_, staTracks);

  // Grab all global muon tracks
  Handle<TrackCollection> glbTracks;
  event.getByLabel(glbTrackLabel_, glbTracks);

  // Now, loop over all simTracks, Find matched Sta, Glb muon recTracks
  //  and fill their information to histograms
  unsigned int nSimMuon = 0;
  for ( SimTrackContainer::const_iterator simTrack = simTracks->begin();
        simTrack != simTracks->end(); simTrack++ ) {
    // Skip for other particle species
    if ( abs(simTrack->type()) != muonPID ) continue;
    nSimMuon++;

    const double simPt  = simTrack->momentum().perp();
    //const double simEta = simTrack->momentum().eta();
    const double simEta = fabs(simTrack->momentum().eta());
    const double simPhi = simTrack->momentum().phi();

    if ( simPt  < minPt_  || simPt  > maxPt_  ) continue;
    if ( simEta < minEta_ || simEta > maxEta_ ) continue;
    if ( simPhi < minPhi_ || simPhi > maxPhi_ ) continue;

    hGenPt_ ->Fill(simPt );
    hGenPhiVsEta_->Fill(simPhi, simEta);

    // If no associated simHits for the simTrack, continue to next.
    if ( mapOfMuonSimHits.find(simTrack->trackId()) == mapOfMuonSimHits.end() ) continue;

    hSimPt_ ->Fill(simPt );
    hSimPhiVsEta_->Fill(simPhi, simEta);

    // Find Standalone tracks and match it to simTrack
    TrackCollection::const_iterator staTrack = matchTrack(simTrack, staTracks);
    if ( staTrack != staTracks->end() ) {
      // Fill Matched RecTrack Histo
      const double simQ  = simTrack->charge();
      const double staQ  = staTrack->charge();
      const double staPt = staTrack->pt();
      hStaPt_          ->Fill(staPt);
      hStaPhiVsEta_    ->Fill(simPhi, simEta);
      hStaEtaVsDeltaPt_->Fill(simEta, (staPt-simPt)/simPt);
      hStaEtaVsResolPt_->Fill(simEta, (staQ/staPt-simQ/simPt)/(simQ/simPt));
    }

    // Find Global muon tracks and match it to simTrack
    TrackCollection::const_iterator glbTrack = matchTrack(simTrack, glbTracks);
    if ( glbTrack != glbTracks->end() ) {
      // Fill Matched RecTrack Histo
      const double simQ  = simTrack->charge();
      const double glbQ  = glbTrack->charge();
      const double glbPt = glbTrack->pt();
      hGlbPt_          ->Fill(glbPt);
      hGlbPhiVsEta_    ->Fill(simPhi, simEta);
      hGlbEtaVsDeltaPt_->Fill(simEta, (glbPt-simPt)/simPt);
      hGlbEtaVsResolPt_->Fill(simEta, (glbQ/glbPt-simQ/simPt)/(simQ/simPt));
    }
    
    
    //  Loop over the seed
    // call seedTSOS
    // wite a matchTrack suitable for a TSOS
    // fill the histos

  }
  if ( nSimMuon == 0 ) LogError("RecoMuonValidator") << "No SimMuonTrack found!!";
  if ( nSimMuon >  1 ) LogInfo("RecoMuonValidator") << "1 < nSimMuon = " << nSimMuon;
  if ( staTracks->size() >  1 ) LogInfo("RecoMuonValidator") << "Multiple StaRecMuon, n = " << staTracks->size();
  if ( glbTracks->size() >  1 ) LogInfo("RecoMuonValidator") << "Multiple GlbRecMuon, n = " << glbTracks->size();
}


//////////////////////////////////////////////////////////////////////

//get the seed's TSOS:
TrajectoryStateOnSurface
seedTSOS(const TrajectorySeed& seed){

  // Get the Trajectory State on Det (persistent version of a TSOS) from the seed
  PTrajectoryStateOnDet pTSOD = seed.startingState();
  
  // Transform it in a TrajectoryStateOnSurface
  TrajectoryStateTransform tsTransform;
  
  DetId seedDetId(pTSOD.detId());

  const GeomDet* gdet = theService->trackingGeometry()->idToDet( seedDetId );

  TrajectoryStateOnSurface initialState = tsTransform.transientState(pTSOD, &(gdet->surface()), 
								     &*theService->magneticField());
  

  // Get the layer on which the seed relies
  const DetLayer *initialLayer = theService->detLayerGeometry()->idToLayer( seedDetId );

  PropagationDirection detLayerOrder = oppositeToMomentum;

  // ask for compatible layers
  vector<const DetLayer*> detLayers;

  //  if(theNavigationType == "Standard")
    detLayers = initialLayer->compatibleLayers( *initialState.freeState(),detLayerOrder); 
    //   else if (theNavigationType == "Direct"){
    //     DirectMuonNavigation navigation( &*theService->detLayerGeometry() );
    //     detLayers = navigation.compatibleLayers( *initialState.freeState(),detLayerOrder);
    //   }
    //   else
    //     edm::LogError(metname) << "No Properly Navigation Selected!!"<<endl;

 
  TrajectoryStateOnSurface result = initialState;

  if(detLayers.size()){

    const DetLayer* finalLayer = detLayers.back();
    
    const TrajectoryStateOnSurface propagatedState = 
      theService->propagator(theSeedPropagatorName)->propagate(initialState,
							       finalLayer->surface());

    if(propagatedState.isValid())
      result = propagatedState;
  }
  
  return result;
}
