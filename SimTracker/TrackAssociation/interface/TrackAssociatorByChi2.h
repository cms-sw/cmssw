#ifndef TrackAssociatorByChi2_h
#define TrackAssociatorByChi2_h

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include<map>

class TrackAssociatorByChi2 : public TrackAssociatorBase {

 public:
  typedef std::map<double,  SimTrack> Chi2SimMap;
  typedef std::pair< reco::Track, Chi2SimMap> RecoToSimPair;
  typedef std::vector< RecoToSimPair > RecoToSimPairAssociation;

  TrackAssociatorByChi2(const edm::ESHandle<MagneticField> mF){
    theMF=mF;  
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
 
  reco::RecoToSimCollection associateRecoToSim (edm::Handle<reco::TrackCollection>&, 
						edm::Handle<TrackingParticleCollection>& ) ;

  reco::SimToRecoCollection associateSimToReco (edm::Handle<reco::TrackCollection>&, 
						edm::Handle<TrackingParticleCollection>& );

 private:
  edm::ESHandle<MagneticField> theMF;
};

#endif
