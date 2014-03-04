/* L1ExtraMaker Creates L1 Extra Objects from Clusters 
 
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
#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeed.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeedFwd.h"


#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

class L1CaloGeometry;

class L1NewEgammaExtraTranslator:public edm::EDProducer
{
  public:
	explicit L1NewEgammaExtraTranslator( const edm::ParameterSet & );
	~L1NewEgammaExtraTranslator(  );

  private:

	virtual void produce( edm::Event &, const edm::EventSetup & );
	virtual void endJob(  );

	edm::InputTag mClusters;
	std::size_t mNparticles;			// Number of Objects to produce

};





L1NewEgammaExtraTranslator::L1NewEgammaExtraTranslator( const edm::ParameterSet & iConfig ):
mClusters( iConfig.getParameter < edm::InputTag > ( "Clusters" ) ),
mNparticles( iConfig.getParameter < unsigned int >( "NParticles" ) )
{
	// Register Product
	produces < l1extra::L1EmParticleCollection > ( "EGamma" );
	produces < l1extra::L1EmParticleCollection > ( "IsoEGamma" );

}


L1NewEgammaExtraTranslator::~L1NewEgammaExtraTranslator(  )
{
}


void L1NewEgammaExtraTranslator::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
	edm::Handle < l1slhc::L1CaloClusterWithSeedCollection > clusters;
	if ( iEvent.getByLabel( mClusters, clusters ) )
    {

        std::auto_ptr < l1extra::L1EmParticleCollection > l1EGamma( new l1extra::L1EmParticleCollection );
        std::auto_ptr < l1extra::L1EmParticleCollection > l1IsoEGamma( new l1extra::L1EmParticleCollection );

		l1slhc::L1CaloClusterWithSeedCollection finalClusters ( *clusters );
		finalClusters.sort(  );

		for ( l1slhc::L1CaloClusterWithSeedCollection::const_iterator i = finalClusters.begin(  ); i != finalClusters.end(  ); ++i )
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

        }

        iEvent.put( l1EGamma, "EGamma" );
        iEvent.put( l1IsoEGamma, "IsoEGamma" );
	}






}


// ------------ method called once each job just after ending the event loop ------------
void L1NewEgammaExtraTranslator::endJob(  )
{
}

// #define DEFINE_ANOTHER_FWK_MODULE(type) DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<type>,#type); DEFINE_FWK_PSET_DESC_FILLER(type)
DEFINE_EDM_PLUGIN( edm::MakerPluginFactory, edm::WorkerMaker < L1NewEgammaExtraTranslator >, "L1NewEgammaExtraTranslator" );
DEFINE_FWK_PSET_DESC_FILLER( L1NewEgammaExtraTranslator );
// DEFINE_ANOTHER_FWK_MODULE(L1NewEgammaExtraTranslator);
