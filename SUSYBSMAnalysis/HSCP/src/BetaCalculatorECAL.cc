#include "SUSYBSMAnalysis/HSCP/interface/BetaCalculatorECAL.h"

BetaCalculatorECAL::BetaCalculatorECAL(const edm::ParameterSet& iConfig){
   edm::ParameterSet trkParameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   parameters_.loadParameters( trkParameters ); 
   trackAssociator_.useDefaultPropagator();
}


void BetaCalculatorECAL::addInfoToCandidate(HSCParticle& candidate, edm::Handle<reco::TrackCollection>& tracks, edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // the calo info object
  CaloBetaMeasurement result;
  
  // select the track
  reco::Track track;
  if(      candidate.hasMuonRef() && candidate.muonRef()->combinedMuon()  .isNonnull()){ track=*(candidate.muonRef()->combinedMuon());
  }else if(candidate.hasMuonRef() && candidate.muonRef()->innerTrack()    .isNonnull()){ track=*(candidate.muonRef()->innerTrack());
  }else if(candidate.hasMuonRef() && candidate.muonRef()->standAloneMuon().isNonnull()){ track=*(candidate.muonRef()->standAloneMuon());
  }else return;
/*
  if(candidate.hasMuonCombinedTrack()) {
    track = candidate.combinedTrack();
  } else if(candidate.hasTrackerTrack()) {
    track = candidate.trackerTrack();
  } else if(candidate.hasMuonStaTrack()) {
    track = candidate.staTrack();
  } else return;  
*/

  // compute the track isolation
  result.trkisodr=100;
  for(reco::TrackCollection::const_iterator ndTrack = tracks->begin(); ndTrack != tracks->end(); ++ndTrack) {
      double dr=sqrt(pow((track.outerEta()-ndTrack->outerEta()),2)+pow((track.outerPhi()-ndTrack->outerPhi()),2));
      if(dr>0.00001 && dr<result.trkisodr) result.trkisodr=dr;
  }

  // use the track associator to propagate to the calo
  TrackDetMatchInfo info = trackAssociator_.associate( iEvent, iSetup, 
                                                       trackAssociator_.getFreeTrajectoryState(iSetup, track),
                                                       parameters_ );

  // extract various quantities
  result.ecalenergy = info.crossedEnergy(TrackDetMatchInfo::EcalRecHits);
  DetId centerId = info.findMaxDeposition(TrackDetMatchInfo::EcalRecHits);
  GlobalPoint position = info.getPosition(centerId);
  double matchedR = sqrt(pow(position.x(),2)+pow(position.y(),2)+pow(position.z(),2));
  result.ecal5by5dir = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 1);
  result.ecal5by5dir = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 2);
  for(std::vector<const EcalRecHit*>::const_iterator hit = info.crossedEcalRecHits.begin(); 
      hit != info.crossedEcalRecHits.end(); ++hit) {
    result.ecaltime += (*hit)->time();	
  }
  if(info.crossedEcalRecHits.size()) {
    result.ecaltime /= info.crossedEcalRecHits.size();
    result.ecalbeta = (matchedR+23)/(result.ecaltime*25.*30+matchedR);
  }
  result.hcalenergy = info.crossedEnergy(TrackDetMatchInfo::HcalRecHits);
  result.hoenergy = info.crossedEnergy(TrackDetMatchInfo::HORecHits);
  centerId = info.findMaxDeposition(TrackDetMatchInfo::HcalRecHits);
  result.hcal3by3dir = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 1);
  result.hcal5by5dir = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 2);
  // conclude by putting all that in the candidate
  candidate.setCalo(result);
}

