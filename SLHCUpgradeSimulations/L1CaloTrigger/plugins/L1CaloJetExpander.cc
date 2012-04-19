
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include "SimDataFormats/SLHC/interface/L1CaloJetFwd.h"


class L1CaloJetExpander:
public L1CaloAlgoBase < l1slhc::L1CaloJetCollection, l1slhc::L1CaloJetCollection >
{
  public:
	L1CaloJetExpander( const edm::ParameterSet & );
	 ~L1CaloJetExpander(  );

//	void initialize( );

	void algorithm( const int &, const int & );

  private:
	std::vector< std::pair< int, int > > mAddrLUT;
};

L1CaloJetExpander::L1CaloJetExpander( const edm::ParameterSet & aConfig ):
L1CaloAlgoBase < l1slhc::L1CaloJetCollection, l1slhc::L1CaloJetCollection > ( aConfig )
{
	// mPhiOffset = 0;
	mEtaOffset = -3;
	mPhiIncrement = 4;
	mEtaIncrement = 4;

	mAddrLUT.resize(8);

	mAddrLUT[0] = std::make_pair( -4 , -4 );
	mAddrLUT[1] = std::make_pair( -4 , 0 );
	mAddrLUT[2] = std::make_pair( -4 , +4 );

	mAddrLUT[3] = std::make_pair( 0 , -4 );
	mAddrLUT[4] = std::make_pair( 0 , +4 );

	mAddrLUT[5] = std::make_pair( +4 , -4 );
	mAddrLUT[6] = std::make_pair( +4 , 0 );
	mAddrLUT[7] = std::make_pair( +4 , +4 );
}

L1CaloJetExpander::~L1CaloJetExpander(  )
{
}

/* 
void L1CaloJetExpander::initialize( )
{
}
*/

void L1CaloJetExpander::algorithm( const int &aEta, const int &aPhi )
{
	l1slhc::L1CaloJetCollection::const_iterator lJetItr = fetch( aEta, aPhi );
	if ( lJetItr != mInputCollection->end(  ) )
	{
		if ( lJetItr->central(  ) )
		{
			// jet is central
			// so should always be added to the event
			// L1CaloJet lExpandedJet( *lJetItr );

			int lIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
			std::pair < int, int >lEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lIndex );
			l1slhc::L1CaloJetCollection::iterator lItr = mOutputCollection->find( lEtaPhi.first, lEtaPhi.second );
			if ( lItr == mOutputCollection->end(  ) )
			{
				// not already in the collection, but should be, so add a copy of it
				mOutputCollection->insert( lEtaPhi.first, lEtaPhi.second, *lJetItr );
			}

		}
		else
		{

			// not central so
			// loop and find the highest central neighbor
			// and give it the constituents
			l1slhc::L1CaloJetCollection::const_iterator lHighestCentralNeighbourItr = mInputCollection->end(  );
			int highestCentralNeighbourE = -1000;

			for ( std::vector< std::pair< int, int > >::iterator lAddrLUTIt = mAddrLUT.begin(); lAddrLUTIt != mAddrLUT.end(); ++lAddrLUTIt  )
			{
				l1slhc::L1CaloJetCollection::const_iterator lNeighbourItr = fetch( aEta + (lAddrLUTIt->first), aPhi + (lAddrLUTIt->second) );
				if ( lNeighbourItr != mInputCollection->end(  ) )
				{
					if ( lNeighbourItr->central(  ) )
					{
						if ( lNeighbourItr->E(  ) > highestCentralNeighbourE )
						{
							highestCentralNeighbourE = lNeighbourItr->E(  );
							lHighestCentralNeighbourItr = lNeighbourItr;
						}
					}
				}
			}

			if ( lHighestCentralNeighbourItr != mInputCollection->end(  ) )
			{
				// If the highestCentralNeighbour is not already in the output collection, then add it to the collection
//				int lIndex = mCaloTriggerSetup->getBin( lHighestCentralNeighbourItr->iEta(  ), lHighestCentralNeighbourItr->iPhi(  ) );
//				std::pair < int, int >lEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lIndex );
//				l1slhc::L1CaloJetCollection::iterator lItr = mOutputCollection->find( lEtaPhi.first, lEtaPhi.second );
				l1slhc::L1CaloJetCollection::iterator lItr = mOutputCollection->find( lHighestCentralNeighbourItr->iEta(  ), lHighestCentralNeighbourItr->iPhi(  ) );
				if ( lItr == mOutputCollection->end(  ) )
				{
					lItr = mOutputCollection->insert( lHighestCentralNeighbourItr->iEta(  ), lHighestCentralNeighbourItr->iPhi(  ), *lHighestCentralNeighbourItr );
				}

				// Now the highestCentralNeighbour is definately the output collection already

				for ( l1slhc::L1CaloRegionRefVector::const_iterator lConstituentItr = lJetItr->getConstituents(  ).begin(  ); lConstituentItr != lJetItr->getConstituents(  ).end(  ); ++lConstituentItr )
				{
					lItr->addConstituent( *lConstituentItr );
				}

			    lItr->setP4( math::PtEtaPhiMLorentzVector( double( lItr->E() )/2. , lItr->p4().eta() , lItr->p4().phi() , 0.0 ) );


			}
		}
	}


}

DEFINE_EDM_PLUGIN( edm::MakerPluginFactory, edm::WorkerMaker < L1CaloJetExpander >, "L1CaloJetExpander" );
DEFINE_FWK_PSET_DESC_FILLER( L1CaloJetExpander );
