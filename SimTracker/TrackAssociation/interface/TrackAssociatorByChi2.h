#ifndef TrackAssociatorByChi2_h
#define TrackAssociatorByChi2_h

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<map>

class TrackAssociatorByChi2 : public TrackAssociatorBase {

 public:
  typedef std::map<double,  SimTrack> Chi2SimMap;
  typedef std::pair< reco::Track, Chi2SimMap> RecoToSimPair;
  typedef std::vector< RecoToSimPair > RecoToSimPairAssociation;

  TrackAssociatorByChi2(const edm::ESHandle<MagneticField> mF, edm::ParameterSet conf):
    chi2cut(conf.getParameter<double>("chi2cut")){
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
						edm::Handle<TrackingParticleCollection>&, 
						const edm::Event * event = 0) ;

  reco::SimToRecoCollection associateSimToReco (edm::Handle<reco::TrackCollection>&, 
						edm::Handle<TrackingParticleCollection>& ,
						const edm::Event * event = 0);

  reco::TrackBase::ParameterVector parametersAtClosestApproach2Order (Basic3DVector<double>,// vertex
								Basic3DVector<double>,// momAtVtx
								int);// charge

  reco::TrackBase::ParameterVector parametersAtClosestApproachGeom (Basic3DVector<double>,// vertex
								    Basic3DVector<double>,// momAtVtx
								    int);// charge

 private:
  edm::ESHandle<MagneticField> theMF;
  double chi2cut;
};

#endif
