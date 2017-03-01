#ifndef SimDataFormats_Associations_TrackToGenParticleAssociator_h
#define SimDataFormats_Associations_TrackToGenParticleAssociator_h

/** \class TrackToGenParticleAssociator
 *  Interface for accessing a Track to GenParticle associator
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

#include <memory>

//Note that the Association Map is filled with -ch2 and not chi2 because it is ordered using std::greater:
//the track with the lowest association chi2 will be the first in the output map.

namespace reco{
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric
    <reco::GenParticleCollection, edm::View<reco::Track>, double> >
    GenToRecoCollection;  
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric 
    <edm::View<reco::Track>, reco::GenParticleCollection, double> >
    RecoToGenCollection;

  
  class TrackToGenParticleAssociator  {
    
  public:
    /// Constructor
    TrackToGenParticleAssociator();
#ifndef __GCCXML__
    TrackToGenParticleAssociator( std::unique_ptr<reco::TrackToGenParticleAssociatorBaseImpl>);
#endif
    ~TrackToGenParticleAssociator();
    
    /// Association Sim To Reco with Collections (Gen Particle version)
    reco::RecoToGenCollection associateRecoToGen(const edm::RefToBaseVector<reco::Track>& tracks,
                                                 const edm::RefVector<reco::GenParticleCollection>& gens) const {
      return m_impl->associateRecoToGen(tracks ,gens);
    }
    /// Association Sim To Reco with Collections (Gen Particle version)
    reco::GenToRecoCollection associateGenToReco(const edm::RefToBaseVector<reco::Track>& tracks,
                                                 const edm::RefVector<reco::GenParticleCollection>& gens) const {
      return m_impl->associateGenToReco(tracks,gens);
    }
    
    /// compare reco to sim the handle of reco::Track and GenParticle collections
    reco::RecoToGenCollection associateRecoToGen(const edm::Handle<edm::View<reco::Track> >& tCH, 
                                                 const edm::Handle<reco::GenParticleCollection>& tPCH) const {
      return m_impl->associateRecoToGen(tCH, tPCH);
    }
    
    /// compare reco to sim the handle of reco::Track and GenParticle collections
    reco::GenToRecoCollection associateGenToReco(const edm::Handle<edm::View<reco::Track> >& tCH, 
                                                 const edm::Handle<reco::GenParticleCollection>& tPCH) const {
      return m_impl->associateGenToReco(tCH,tPCH);
    }
    
    void swap(TrackToGenParticleAssociator& iOther) {
      std::swap(m_impl, iOther.m_impl);
    }

    
  private:
    TrackToGenParticleAssociator(const TrackToGenParticleAssociator&); // stop default
    
    const TrackToGenParticleAssociator& operator=(const TrackToGenParticleAssociator&); // stop default
    
    TrackToGenParticleAssociatorBaseImpl* m_impl;
    
    
  };
}

#endif
