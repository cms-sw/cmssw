#ifndef SimDataFormats_Associations_TrackToGenParticleAssociatorBaseImpl_h
#define SimDataFormats_Associations_TrackToGenParticleAssociatorBaseImpl_h

/** \class TrackToGenParticleAssociatorBaseImpl
 *  Base class for implementations of a Track to GenParticle associator
 *
 *  \author cerati, magni
 */

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

#include "SimDataFormats/Associations/interface/TrackToGenParticleAssociatorBaseImpl.h"

namespace reco{
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric
    <reco::GenParticleCollection, edm::View<reco::Track>, double> >
    GenToRecoCollection;  
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric 
    <edm::View<reco::Track>, reco::GenParticleCollection, double> >
    RecoToGenCollection;    

  
  
  class TrackToGenParticleAssociatorBaseImpl  {
    
  public:
    /// Constructor
    TrackToGenParticleAssociatorBaseImpl();
    virtual ~TrackToGenParticleAssociatorBaseImpl();
    
    /// Association Sim To Reco with Collections (Gen Particle version)
    virtual reco::RecoToGenCollection associateRecoToGen(const edm::RefToBaseVector<reco::Track>& tracks,
                                                         const edm::RefVector<reco::GenParticleCollection>& gens) const = 0;
    
    /// Association Sim To Reco with Collections (Gen Particle version)
    virtual reco::GenToRecoCollection associateGenToReco(const edm::RefToBaseVector<reco::Track>& tracks,
                                                         const edm::RefVector<reco::GenParticleCollection>& gens) const =0;
    
    
    /// compare reco to sim the handle of reco::Track and GenParticle collections
    virtual reco::RecoToGenCollection associateRecoToGen(const edm::Handle<edm::View<reco::Track> >& tCH, 
                                                         const edm::Handle<reco::GenParticleCollection>& tPCH) const = 0;
    
    /// compare reco to sim the handle of reco::Track and GenParticle collections
    virtual reco::GenToRecoCollection associateGenToReco(const edm::Handle<edm::View<reco::Track> >& tCH, 
                                                         const edm::Handle<reco::GenParticleCollection>& tPCH) const = 0;
    
    
  private:
    TrackToGenParticleAssociatorBaseImpl(const TrackToGenParticleAssociatorBaseImpl&); // stop default
    
    const TrackToGenParticleAssociatorBaseImpl& operator=(const TrackToGenParticleAssociatorBaseImpl&); // stop default
    
  };
}

#endif
