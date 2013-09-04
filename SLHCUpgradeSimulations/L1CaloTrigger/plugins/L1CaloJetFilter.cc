
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include "SimDataFormats/SLHC/interface/L1CaloJetFwd.h"

class L1CaloJetFilter:
public L1CaloAlgoBase < l1slhc::L1CaloJetCollection, l1slhc::L1CaloJetCollection >
{
  public:
	L1CaloJetFilter( const edm::ParameterSet & );
	 ~L1CaloJetFilter(  );

	// void initialize( );

	void algorithm( const int &, const int & );

  private:

};

L1CaloJetFilter::L1CaloJetFilter( const edm::ParameterSet & aConfig ):L1CaloAlgoBase < l1slhc::L1CaloJetCollection, l1slhc::L1CaloJetCollection > ( aConfig )
{
	// mPhiOffset = 0;
	mEtaOffset = -3;
	mPhiIncrement = 4;
	mEtaIncrement = 4;
}

L1CaloJetFilter::~L1CaloJetFilter(  )
{
}

/* 
   void L1CaloJetFilter::initialize( ) { } */

void L1CaloJetFilter::algorithm( const int &aEta, const int &aPhi )
{

	// Look if there is a cluster here
	l1slhc::L1CaloJetCollection::const_iterator lJetItr = fetch( aEta, aPhi );
	if ( lJetItr != mInputCollection->end(  ) )
	{
		l1slhc::L1CaloJet lFilteredJet( *lJetItr );

		// Set lCentralFlag bit
		bool lCentralFlag = true;

		// right
		l1slhc::L1CaloJetCollection::const_iterator lNeighbourItr = fetch( aEta + 4, aPhi );
		// If lNeighbourItr exists
		if ( lNeighbourItr != mInputCollection->end(  ) )
		{
			// Compare the energies and prune if the lNeighbourItr has higher Et
			if ( lJetItr->E(  ) <= lNeighbourItr->E(  ) )
			{
				lFilteredJet.removeConstituent( 4, 0 );
				lFilteredJet.removeConstituent( 4, 4 );
				lCentralFlag = false;
			}
		}

		// right-down
		lNeighbourItr = fetch( aEta + 4, aPhi + 4 );
		// If lNeighbourItr exists
		if ( lNeighbourItr != mInputCollection->end(  ) )
		{
			// Compare the energies and prune if the lNeighbourItr has higher Et
			if ( lJetItr->E(  ) <= lNeighbourItr->E(  ) )
			{
				lFilteredJet.removeConstituent( 4, 4 );
				lCentralFlag = false;
			}
		}

		// down
		lNeighbourItr = fetch( aEta, aPhi + 4 );
		// If lNeighbourItr exists
		if ( lNeighbourItr != mInputCollection->end(  ) )
		{
			// Compare the energies and prune if the lNeighbourItr has higher Et
			if ( lJetItr->E(  ) <= lNeighbourItr->E(  ) )
			{

				lFilteredJet.removeConstituent( 0, 4 );
				lFilteredJet.removeConstituent( 4, 4 );
				lCentralFlag = false;
			}
		}

		// down-left
		lNeighbourItr = fetch( aEta - 4, aPhi + 4 );
		// If lNeighbourItr exists
		if ( lNeighbourItr != mInputCollection->end(  ) )
		{
			// Compare the energies and prune if the lNeighbourItr has higher Et
			if ( lJetItr->E(  ) <= lNeighbourItr->E(  ) )
			{
				lFilteredJet.removeConstituent( 0, 4 );
				lCentralFlag = false;
			}
		}

		// left
		lNeighbourItr = fetch( aEta - 4, aPhi );
		// If lNeighbourItr exists
		if ( lNeighbourItr != mInputCollection->end(  ) )
		{
			// Compare the energies and prune if the lNeighbourItr has higher Et
			if ( lJetItr->E(  ) < lNeighbourItr->E(  ) )
			{
				lFilteredJet.removeConstituent( 0, 0 );
				lFilteredJet.removeConstituent( 0, 4 );
				lCentralFlag = false;
			}
		}

		// left-up
		lNeighbourItr = fetch( aEta - 4, aPhi - 4 );
		// If lNeighbourItr exists
		if ( lNeighbourItr != mInputCollection->end(  ) )
		{
			// Compare the energies and prune if the lNeighbourItr has higher Et
			if ( lJetItr->E(  ) < lNeighbourItr->E(  ) )
			{
				lFilteredJet.removeConstituent( aEta, aPhi );
				lCentralFlag = false;
			}
		}

		// up
		lNeighbourItr = fetch( aEta, aPhi - 4 );
		// If lNeighbourItr exists
		if ( lNeighbourItr != mInputCollection->end(  ) )
		{
			// Compare the energies and prune if the lNeighbourItr has higher Et
			if ( lJetItr->E(  ) < lNeighbourItr->E(  ) )
			{
				lFilteredJet.removeConstituent( 0, 0 );
				lFilteredJet.removeConstituent( 4, 0 );
				lCentralFlag = false;
			}
		}

		// up-right
		lNeighbourItr = fetch( aEta + 4, aPhi - 4 );
		// If lNeighbourItr exists
		if ( lNeighbourItr != mInputCollection->end(  ) )
		{
			// Compare the energies and prune if the lNeighbourItr has higher Et
			if ( lJetItr->E(  ) < lNeighbourItr->E(  ) )
			{
				lFilteredJet.removeConstituent( 4, 0 );
				lCentralFlag = false;
			}
		}

		//std::cout << "original E = " << lJetItr->E(  ) << " original size = " << lJetItr->getConstituents().size() << "\tnew E = " << lFilteredJet.E(  ) << " new size = " << lFilteredJet.getConstituents().size() << std::endl;
		// Check if the jet is over threshold
		if ( lFilteredJet.E(  ) >= mCaloTriggerSetup->minJetET(  ) )
		{
			lFilteredJet.setCentral( lCentralFlag );

			int lIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
			std::pair < int, int >lEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lIndex );
			mOutputCollection->insert( lEtaPhi.first, lEtaPhi.second, lFilteredJet );
		}
	}

}

DEFINE_EDM_PLUGIN( edm::MakerPluginFactory, edm::WorkerMaker < L1CaloJetFilter >, "L1CaloJetFilter" );
DEFINE_FWK_PSET_DESC_FILLER( L1CaloJetFilter );
