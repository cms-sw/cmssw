/* L1ExtraMaker Creates L1 Extra Objects from Clusters and jets

   M.Bachtis,S.Dasu University of Wisconsin-Madison */


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"

#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include "SimDataFormats/SLHC/interface/L1CaloJetFwd.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

class L1CaloGeometry;

class L1ExtraTranslator:public edm::EDProducer
{
  public:
	explicit L1ExtraTranslator( const edm::ParameterSet & );
	~L1ExtraTranslator(  );

  private:

	virtual void produce( edm::Event &, const edm::EventSetup & );
	virtual void endJob(  );

	edm::InputTag mClusters;
	edm::InputTag mJets;
	std::size_t mNparticles;			// Number of Objects to produce
	std::size_t mNjets;					// Number of Objects to produce

};





L1ExtraTranslator::L1ExtraTranslator( const edm::ParameterSet & iConfig ):
mClusters( iConfig.getParameter < edm::InputTag > ( "Clusters" ) ),
mJets( iConfig.getParameter < edm::InputTag > ( "Jets" ) ), 
mNparticles( iConfig.getParameter < unsigned int >( "NParticles" ) ), 
mNjets( iConfig.getParameter < unsigned int >( "NJets" ) )
{
	// Register Product
	produces < l1extra::L1EmParticleCollection > ( "EGamma" );
	produces < l1extra::L1EmParticleCollection > ( "IsoEGamma" );
	produces < l1extra::L1JetParticleCollection > ( "Taus" );
	produces < l1extra::L1JetParticleCollection > ( "IsoTaus" );
	produces < l1extra::L1JetParticleCollection > ( "Jets" );

}


L1ExtraTranslator::~L1ExtraTranslator(  )
{
}


void L1ExtraTranslator::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
	edm::Handle < l1slhc::L1CaloClusterCollection > clusters;
	if ( iEvent.getByLabel( mClusters, clusters ) )
	{

		std::auto_ptr < l1extra::L1EmParticleCollection > l1EGamma( new l1extra::L1EmParticleCollection );
		std::auto_ptr < l1extra::L1EmParticleCollection > l1IsoEGamma( new l1extra::L1EmParticleCollection );
		std::auto_ptr < l1extra::L1JetParticleCollection > l1Tau( new l1extra::L1JetParticleCollection );
		std::auto_ptr < l1extra::L1JetParticleCollection > l1IsoTau( new l1extra::L1JetParticleCollection );

		l1slhc::L1CaloClusterCollection finalClusters ( *clusters );
		finalClusters.sort(  );

		for ( l1slhc::L1CaloClusterCollection::const_iterator i = finalClusters.begin(  ); i != finalClusters.end(  ); ++i )
		{
			// EGamma
			if ( l1EGamma->size() != mNparticles )
			{
				if ( i->isEGamma(  ) )
				{
					l1EGamma->push_back( l1extra::L1EmParticle( i->p4(  ) ) );
				}
			}

			// Isolated EGamma
			if ( l1IsoEGamma->size() != mNparticles )
			{
				if ( i->isIsoEGamma(  ) )
				{
					l1IsoEGamma->push_back( l1extra::L1EmParticle( i->p4(  ) ) );
				}
			}

			if( abs( i->iEta(  ) ) <= 26 )
			{
				// Taus
				if ( l1Tau->size() != mNparticles )
				{
					if ( i->isTau(  ) )
					{
						l1Tau->push_back( l1extra::L1JetParticle( i->p4(  ) ) );
					}
				}

				// IsoTaus
				if ( l1IsoTau->size() != mNparticles )
				{
					if ( i->isIsoTau(  ) )
					{
						l1IsoTau->push_back( l1extra::L1JetParticle( i->p4(  ) ) );
					}
				}
			}

		}

		iEvent.put( l1EGamma, "EGamma" );
		iEvent.put( l1IsoEGamma, "IsoEGamma" );
		iEvent.put( l1Tau, "Taus" );
		iEvent.put( l1IsoTau, "IsoTaus" );
	}



	// Jets
	edm::Handle < l1slhc::L1CaloJetCollection > jets;
	if ( iEvent.getByLabel( mJets, jets ) )
	{
		std::auto_ptr < l1extra::L1JetParticleCollection > l1Jet( new l1extra::L1JetParticleCollection );

		l1slhc::L1CaloJetCollection lJets = *jets;
		lJets.sort(  );

		for ( l1slhc::L1CaloJetCollection::const_iterator i = lJets.begin(  ); i != lJets.end(  ) ; ++i )
		{
			l1Jet->push_back( l1extra::L1JetParticle( i->p4(  ) ) );
			if( l1Jet->size() == mNjets ) break;
		}

		iEvent.put( l1Jet, "Jets" );
	}




}


// ------------ method called once each job just after ending the event loop ------------
void L1ExtraTranslator::endJob(  )
{
}

// #define DEFINE_ANOTHER_FWK_MODULE(type) DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<type>,#type); DEFINE_FWK_PSET_DESC_FILLER(type)
DEFINE_EDM_PLUGIN( edm::MakerPluginFactory, edm::WorkerMaker < L1ExtraTranslator >, "L1ExtraTranslator" );
DEFINE_FWK_PSET_DESC_FILLER( L1ExtraTranslator );
// DEFINE_ANOTHER_FWK_MODULE(L1ExtraTranslator);
