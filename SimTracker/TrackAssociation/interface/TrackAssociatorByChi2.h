#ifndef TRACKASSOCIATORBYCHI2_H
#define TRACKASSOCIATORBYCHI2_H

#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociation.h"

#include<map>

class TrackAssociatorByChi2 {

 public:
  typedef std::map<double,  SimTrack> Chi2SimMap;
  typedef std::pair< reco::Track, Chi2SimMap> RecoToSimPair;
  typedef std::vector< RecoToSimPair > RecoToSimPairAssociation;

  TrackAssociatorByChi2(const edm::EventSetup& es){
    es.get<IdealMagneticFieldRecord>().get(theMF);  
  }

  ~TrackAssociatorByChi2(){
  }

  double compareTracksParam ( reco::TrackCollection::const_iterator, 
			      edm::SimTrackContainer::const_iterator, 
			      const HepLorentzVector, 
			      GlobalVector,
			      reco::TrackBase::CovarianceMatrix) ;

  RecoToSimPairAssociation compareTracksParam(const reco::TrackCollection&, 
					      const edm::SimTrackContainer&, 
					      const edm::SimVertexContainer&) ;
 
  reco::RecoToSimCollection compareTracksParam (edm::Handle<reco::TrackCollection>&, 
						edm::Handle<TrackingParticleCollection>& ) ;
  


 private:
  edm::ESHandle<MagneticField> theMF;
};



#endif
