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

#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"



class L1CaloGeometry;

typedef math::PtEtaPhiMLorentzVector LorentzVector;

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
  edm::InputTag mTowers;
  std::size_t mNparticles;			// Number of Objects to produce
  std::size_t mNjets;					// Number of Objects to produce
  double maxJetTowerEta;
};





L1ExtraTranslator::L1ExtraTranslator( const edm::ParameterSet & iConfig ):
mClusters( iConfig.getParameter < edm::InputTag > ( "Clusters" ) ),
mJets( iConfig.getParameter < edm::InputTag > ( "Jets" ) ), 
mTowers( iConfig.getParameter < edm::InputTag > ( "Towers" ) ), 
mNparticles( iConfig.getParameter < unsigned int >( "NParticles" ) ), 
mNjets( iConfig.getParameter < unsigned int >( "NJets" ) ),
maxJetTowerEta( iConfig.getParameter < double >( "maxJetTowerEta" ) )
{

	// Register Product
	produces < l1extra::L1EmParticleCollection > ( "EGamma" );
	produces < l1extra::L1EmParticleCollection > ( "IsoEGamma" );
	produces < l1extra::L1JetParticleCollection > ( "Taus" );
	produces < l1extra::L1JetParticleCollection > ( "IsoTaus" );
	produces < l1extra::L1JetParticleCollection > ( "Jets" );
	produces < l1extra::L1EtMissParticleCollection > ( "MHT" );
	produces < l1extra::L1EtMissParticleCollection > ( "MET" );
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
	  LorentzVector Htvec(0,0,0,0);
	  std::auto_ptr < l1extra::L1JetParticleCollection > l1Jet( new l1extra::L1JetParticleCollection );
	  std::auto_ptr < l1extra::L1EtMissParticleCollection> l1MHt( new l1extra::L1EtMissParticleCollection );
	 
		l1slhc::L1CaloJetCollection lJets = *jets;
		lJets.sort(  );

		for ( l1slhc::L1CaloJetCollection::const_iterator i = lJets.begin(  ); i != lJets.end(  ) ; ++i )
		{
		  
		  l1Jet->push_back( l1extra::L1JetParticle( i->p4(  ) ) );
			
			if ( fabs(i->p4().eta())< maxJetTowerEta ) {
			LorentzVector jet4v(0,0,0,0);
			jet4v.SetCoordinates(i->p4().Pt(), 0.0, i->p4().Phi(), 0.0);
			Htvec += jet4v;
			}
			
			if( l1Jet->size() == mNjets ) break;
			
		}

		double Ht = Htvec.Et();
		Htvec.SetCoordinates(Htvec.Pt(), 0.0, Htvec.Phi(), 0.0);
		
		l1MHt->push_back( l1extra::L1EtMissParticle( 
							    -Htvec,
							    l1extra::L1EtMissParticle::kMHT,
							    Ht,
							    edm::Ref< L1GctEtMissCollection >(),
							    edm::Ref< L1GctEtTotalCollection >(),
							    edm::Ref< L1GctHtMissCollection >(),
							    edm::Ref< L1GctEtHadCollection >(),
							    0 ) );
		
		iEvent.put( l1Jet, "Jets" );
		iEvent.put( l1MHt, "MHT" );
	
	}

	
	edm::Handle < l1slhc::L1CaloTowerCollection > towers;
	if ( iEvent.getByLabel( mTowers, towers ) )
	  {
	    LorentzVector Etvec(0,0,0,0);
	    
	    std::auto_ptr < l1extra::L1EtMissParticleCollection> l1Met( new l1extra::L1EtMissParticleCollection );
	    
	    l1slhc::L1CaloTowerCollection lTowers = *towers;
	    
	    for( l1slhc::L1CaloTowerCollection::const_iterator i = lTowers.begin() ; i != lTowers.end() ; ++i ){
	      
	      if ( fabs(0.087*i->iEta())< maxJetTowerEta ) {
	      LorentzVector twr4v(0,0,0,0);

	      twr4v.SetCoordinates((i->E()+i->H())/cosh(0.087*i->iEta()), 0.0, 0.087*i->iPhi(), 0.0);
	      Etvec += twr4v;      
	      }
	    }

	    double Et = Etvec.Et();
	    Etvec.SetCoordinates(Etvec.Pt(), 0.0, Etvec.Phi(), 0.0);
	    l1Met->push_back( l1extra::L1EtMissParticle(   -Etvec,
							   l1extra::L1EtMissParticle::kMET,
							   Et,
							   edm::Ref< L1GctEtMissCollection >(),
							   edm::Ref< L1GctEtTotalCollection >(),
							   edm::Ref< L1GctHtMissCollection >(),
							   edm::Ref< L1GctEtHadCollection >(),
							   0 ) );
	    iEvent.put( l1Met, "MET" );
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
