#ifndef TrackGenAssociatorBase_h
#define TrackGenAssociatorBase_h

/** \class TrackGenAssociatorBase
 *  Base class for TrackGenAssociators. Methods take as input the handle of Track and GenParticle collections and return an AssociationMap (oneToManyWithQuality)
 *
 *  \author cerati, magni
 */

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

#include<map>

//Note that the Association Map is filled with -ch2 and not chi2 because it is ordered using std::greater:
//the track with the lowest association chi2 will be the first in the output map.

namespace reco{
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric
    <reco::GenParticleCollection, edm::View<reco::Track>, double> >
    GenToRecoCollection;  
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric 
    <edm::View<reco::Track>, reco::GenParticleCollection, double> >
    RecoToGenCollection;    
}


class TrackGenAssociatorBase  {

 public:
  /// Constructor
  TrackGenAssociatorBase();
  virtual ~TrackGenAssociatorBase();

  /// Association Sim To Reco with Collections (Gen Particle version)
  virtual reco::RecoToGenCollection associateRecoToGen(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<reco::GenParticleCollection>&,
					       const edm::Event * event = 0,
					       const edm::EventSetup * setup = 0 ) const = 0;
  /// Association Sim To Reco with Collections (Gen Particle version)
  virtual reco::GenToRecoCollection associateGenToReco(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<reco::GenParticleCollection>&,
					       const edm::Event * event = 0,
					       const edm::EventSetup * setup = 0 ) const = 0 ;

  /// compare reco to sim the handle of reco::Track and GenParticle collections
  virtual reco::RecoToGenCollection associateRecoToGen(edm::Handle<edm::View<reco::Track> >& tCH, 
						       edm::Handle<reco::GenParticleCollection>& tPCH, 
						       const edm::Event * event = 0,
                                                       const edm::EventSetup * setup = 0) const = 0;
  
  /// compare reco to sim the handle of reco::Track and GenParticle collections
  virtual reco::GenToRecoCollection associateGenToReco(edm::Handle<edm::View<reco::Track> >& tCH, 
						       edm::Handle<reco::GenParticleCollection>& tPCH,
						       const edm::Event * event = 0,
                                                       const edm::EventSetup * setup = 0) const =0;


 private:

};

#endif
