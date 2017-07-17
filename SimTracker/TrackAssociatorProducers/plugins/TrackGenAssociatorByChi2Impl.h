#ifndef TrackGenAssociatorByChi2Impl_h
#define TrackGenAssociatorByChi2Impl_h

/** \class TrackGenAssociatorByChi2Impl
 *  Class that performs the association of reco::Tracks and TrackingParticles evaluating the chi2 of reco tracks parameters and sim tracks parameters. The cut can be tuned from the config file: see data/TrackGenAssociatorByChi2.cfi. Note that the Association Map is filled with -ch2 and not chi2 because it is ordered using std::greater: the track with the lowest association chi2 will be the first in the output map.It is possible to use only diagonal terms (associator by pulls) seeting onlyDiagonal = true in the PSet 
 *
 *  \author cerati, magni
 */

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"
#include "SimDataFormats/Associations/interface/TrackToGenParticleAssociatorBaseImpl.h"

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


class TrackGenAssociatorByChi2Impl : public reco::TrackToGenParticleAssociatorBaseImpl {

 public:
  typedef std::map<double,  SimTrack> Chi2SimMap;
  typedef std::pair< reco::Track, Chi2SimMap> RecoToSimPair;
  typedef std::vector< RecoToSimPair > RecoToSimPairAssociation;

  /// Constructor with PSet

  /// Constructor with magnetic field, double, bool and InputTag
  TrackGenAssociatorByChi2Impl(const MagneticField& mF, 
                               const reco::BeamSpot& bs, 
                               double chi2Cut, bool onlyDiag):
  theMF(&mF),
    theBeamSpot(&bs),
    chi2cut(chi2Cut),
    onlyDiagonal(onlyDiag)
    {
    }

  /// Association Sim To Reco with Collections (Gen Particle version)
  reco::RecoToGenCollection associateRecoToGen(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<reco::GenParticleCollection>&) const override;
  
  /// Association Sim To Reco with Collections (Gen Particle version)
  reco::GenToRecoCollection associateGenToReco(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<reco::GenParticleCollection>&) const override;

  /// compare reco to sim the handle of reco::Track and GenParticle collections
  virtual reco::RecoToGenCollection associateRecoToGen(const edm::Handle<edm::View<reco::Track> >& tCH, 
						       const edm::Handle<reco::GenParticleCollection>& tPCH) const override { 
    edm::RefToBaseVector<reco::Track> tc(tCH);
    for (unsigned int j=0; j<tCH->size();j++)
      tc.push_back(edm::RefToBase<reco::Track>(tCH,j));

    edm::RefVector<reco::GenParticleCollection> tpc(tPCH.id());
    for (unsigned int j=0; j<tPCH->size();j++)
      tpc.push_back(edm::Ref<reco::GenParticleCollection>(tPCH,j));

    return associateRecoToGen(tc,tpc);
  }
  
  /// compare reco to sim the handle of reco::Track and GenParticle collections
  virtual reco::GenToRecoCollection associateGenToReco(const edm::Handle<edm::View<reco::Track> >& tCH, 
						       const edm::Handle<reco::GenParticleCollection>& tPCH) const override {
    edm::RefToBaseVector<reco::Track> tc(tCH);
    for (unsigned int j=0; j<tCH->size();j++)
      tc.push_back(edm::RefToBase<reco::Track>(tCH,j));

    edm::RefVector<reco::GenParticleCollection> tpc(tPCH.id());
    for (unsigned int j=0; j<tPCH->size();j++)
      tpc.push_back(edm::Ref<reco::GenParticleCollection>(tPCH,j));

    return associateGenToReco(tc,tpc);
  }  


 private:

  /// basic method where chi2 is computed
  double getChi2(const reco::TrackBase::ParameterVector& rParameters,
		 const reco::TrackBase::CovarianceMatrix& recoTrackCovMatrix,
		 const Basic3DVector<double>& momAtVtx,
		 const Basic3DVector<double>& vert,
		 int charge,
		 const reco::BeamSpot&) const;

  /// compare reco::TrackCollection and TrackingParticleCollection iterators: returns the chi2
  double associateRecoToSim(reco::TrackCollection::const_iterator,
			    TrackingParticleCollection::const_iterator,
			    const reco::BeamSpot&) const;


  const MagneticField* theMF;
  const reco::BeamSpot* theBeamSpot;
  double chi2cut;
  bool onlyDiagonal;
};

#endif
