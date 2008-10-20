#ifndef TopQuarkAnalysis_TopPairBSM_CATopJetProducer_h
#define TopQuarkAnalysis_TopPairBSM_CATopJetProducer_h


/* *********************************************************
  \class CATopJetProducer

  \brief Jet producer to produce top jets using the C-A algorithm to break
         jets into subjets as described here:
         "Top-tagging: A Method for Identifying Boosted Hadronic Tops"
         David E. Kaplan, Keith Rehermann, Matthew D. Schwartz, Brock Tweedie
         arXiv:0806.0848v1 [hep-ph] 

  \author   Salvatore Rappoccio
  \version  

         Notes on implementation:

	 Because the BaseJetProducer only allows the user to produce
	 one jet collection at a time, this algorithm cannot
	 fit into that paradigm. Since it relies on the output of
	 the fastjet objects themselves, it needs to simultaneously
	 write out two collections at once (subjets, and hard jets).

	 All of the "hard" jets are of type BasicJet, since
	 they are "jets of jets". The subjets will be either
	 CaloJets, GenJets, etc.

	 In order to avoid a templatization of the entire
	 EDProducer itself, we only use a templated method
	 to write out the subjets to the event record,
	 and to use that information to write out the
	 hard jets to the event record.

	 This templated method is called "write_outputs". It
	 relies on a second templated method called "write_specific",
	 which relies on some template specialization to create
	 different specific objects (i.e. CaloJets, BasicJets, GenJets, etc). 

 ************************************************************/




#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "RecoJets/JetProducers/src/BaseJetProducer.h"
#include "TopQuarkAnalysis/TopPairBSM/interface/CATopJetAlgorithm.h"

namespace cms
{
  class CATopJetProducer : public edm::EDProducer
  {
  public:

    CATopJetProducer(const edm::ParameterSet& ps);

    virtual ~CATopJetProducer() {}

    //Produces the EDM products
    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    edm::InputTag            src_;         /// Input constituents
    std::string              jetType_;     /// Jet type for the subjets
    CATopJetAlgorithm        alg_;         /// The algorithm to do the work

    /// function template to write out the outputs
    template<class T>
    void write_outputs( edm::Event & e, 
			edm::EventSetup const & c,
			std::vector<CATopPseudoJet> const & outputs,
			edm::Handle< edm::View<reco::Candidate> > const & fInput );

    /// Function template to write out the specific information for subjets.
    /// This function needs a template specialization because the construction interfaces
    /// for the various flavors of reco::Jet are not the same. 
    template<class T>
      T write_specific( reco::Particle::LorentzVector const & p4,
			reco::Particle::Point const & point, 
			std::vector<reco::CandidatePtr> const & constituents,
			edm::EventSetup const & c );
  };



/// function template to write out the outputs
template< class T>
void CATopJetProducer::write_outputs( edm::Event & e, 
				      edm::EventSetup const & c,
				      std::vector<CATopPseudoJet> const & outputs,
				      edm::Handle< edm::View<reco::Candidate> > const & fInputHandle )
{


  // get a list of output jets
  std::auto_ptr<reco::BasicJetCollection>  jetCollection( new reco::BasicJetCollection() );
  // get a list of output subjets
  std::auto_ptr<std::vector<T> >  subjetCollection( new std::vector<T>() );

  // This will store the handle for the subjets after we write them
  edm::OrphanHandle< std::vector<T> > subjetHandleAfterPut;
  // this is the mapping of subjet to hard jet
  std::vector< std::vector<int> > indices;
  // this is the list of hardjet 4-momenta
  std::vector<math::XYZTLorentzVector> p4_hardJets;

  // Loop over the hard jets
  std::vector<CATopPseudoJet>::const_iterator it = outputs.begin(),
    iEnd = outputs.end(),
    iBegin = outputs.begin();
  indices.resize( outputs.size() );
  for ( ; it != iEnd; ++it ) {
    int jetIndex = it - iBegin;
    fastjet::PseudoJet localJet = it->hardJet();
    // Get the 4-vector for the hard jet
    p4_hardJets.push_back( math::XYZTLorentzVector(localJet.px(), localJet.py(), localJet.pz(), localJet.e() ));

    // create the subjet list
    std::vector<CATopPseudoSubJet>::const_iterator itSubJetBegin = it->subjets().begin(),
      itSubJet = itSubJetBegin, itSubJetEnd = it->subjets().end();
    for (; itSubJet != itSubJetEnd; ++itSubJet ){

      fastjet::PseudoJet subjet = itSubJet->subjet();
      math::XYZTLorentzVector p4Subjet(subjet.px(), subjet.py(), subjet.pz(), subjet.e() );
      reco::Particle::Point point(0,0,0);

      // This will hold ptr's to the subjets
      std::vector<reco::CandidatePtr> subjetConstituents;

      // Get the transient subjet constituents from fastjet
      std::vector<int> const & subjetFastjetConstituentIndices = itSubJet->constituents();
      std::vector<int>::const_iterator fastSubIt = subjetFastjetConstituentIndices.begin(),
	transConstBegin = subjetFastjetConstituentIndices.begin(),
	transConstEnd = subjetFastjetConstituentIndices.end();
      for ( ; fastSubIt != transConstEnd; ++fastSubIt ) {
	// Add a ptr to this constituent
	if ( *fastSubIt >= 0 && *fastSubIt < fInputHandle->size() ) 
	  subjetConstituents.push_back(reco::CandidatePtr(fInputHandle, *fastSubIt));
      }

      // This holds the subjet-to-hardjet mapping
      indices[jetIndex].push_back( subjetCollection->size() );      


      // Add the concrete subjet type to the subjet list to write to event record
      subjetCollection->push_back( write_specific<T>( p4Subjet, point, subjetConstituents, c) );

    }
  }
  // put subjets into event record
  subjetHandleAfterPut = e.put( subjetCollection, "caTopSubJets" );
  
  
  // Now create the hard jets with ptr's to the subjets as constituents
  std::vector<math::XYZTLorentzVector>::const_iterator ip4 = p4_hardJets.begin(),
    ip4Begin = p4_hardJets.begin(),
    ip4End = p4_hardJets.end();

  for ( ; ip4 != ip4End; ++ip4 ) {
    int p4_index = ip4 - ip4Begin;
    std::vector<int> & ind = indices[p4_index];
    std::vector<reco::CandidatePtr> i_hardJetConstituents;
    // Add the subjets to the hard jet
    for( std::vector<int>::const_iterator isub = ind.begin();
	 isub != ind.end(); ++isub ) {
      reco::CandidatePtr candPtr( subjetHandleAfterPut, *isub, false );
      i_hardJetConstituents.push_back( candPtr );
    }   
    reco::Particle::Point point(0,0,0);
    jetCollection->push_back( reco::BasicJet( *ip4, point, i_hardJetConstituents) );
  }
  

  // put hard jets into event record
  e.put( jetCollection);

}



/// Function template to write out the specific information for subjets.
/// This function needs a template specialization because the construction interfaces
/// for the various flavors of reco::Jet are not the same. 


/// Specialization: CaloJet
template<>
reco::CaloJet CATopJetProducer::write_specific<reco::CaloJet>(reco::Particle::LorentzVector const & p4,
							      reco::Particle::Point const & point, 
							      std::vector<reco::CandidatePtr> const & constituents,
							      edm::EventSetup const & c  )
{
  return reco::CaloJet( p4, point, reco::CaloJet::Specific(), constituents);  
}

/// Specialization: BasicJet
template<>
reco::BasicJet CATopJetProducer::write_specific<reco::BasicJet>(reco::Particle::LorentzVector const & p4,
								reco::Particle::Point const & point, 
								std::vector<reco::CandidatePtr> const & constituents,
								edm::EventSetup const & c  )
{
  return reco::BasicJet( p4, point, constituents);  
}

/// Specialization: GenJet
template<>
reco::GenJet CATopJetProducer::write_specific<reco::GenJet>(reco::Particle::LorentzVector const & p4,
							      reco::Particle::Point const & point, 
							      std::vector<reco::CandidatePtr> const & constituents,
							    edm::EventSetup const & c  )
{
  return reco::GenJet( p4, point, reco::GenJet::Specific(), constituents);  
}

/// Specialization: PFJet
template<>
reco::PFJet CATopJetProducer::write_specific<reco::PFJet>(reco::Particle::LorentzVector const & p4,
							      reco::Particle::Point const & point, 
							      std::vector<reco::CandidatePtr> const & constituents,
							    edm::EventSetup const & c  )
{
  return reco::PFJet( p4, point, reco::PFJet::Specific(), constituents);  
}



}


#endif
