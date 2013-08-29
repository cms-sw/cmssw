#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloRegion.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloRegionFwd.h"



class L1CaloRegionProducer:public L1CaloAlgoBase < l1slhc::L1CaloTowerCollection, l1slhc::L1CaloRegionCollection >
{
  public:
	L1CaloRegionProducer( const edm::ParameterSet & );
	 ~L1CaloRegionProducer(  );

	// void initialize( );

	void algorithm( const int &, const int & );

  private:

};







L1CaloRegionProducer::L1CaloRegionProducer( const edm::ParameterSet & aConfig ):
L1CaloAlgoBase < l1slhc::L1CaloTowerCollection, l1slhc::L1CaloRegionCollection > ( aConfig )
{
mPhiOffset = -3;
mEtaOffset = -3;
mPhiIncrement = 4;
mEtaIncrement = 4;
}

L1CaloRegionProducer::~L1CaloRegionProducer(  )
{
}

/* 
void L1CaloRegionProducer::initialize( )
{
}
*/

void L1CaloRegionProducer::algorithm( const int &aEta, const int &aPhi )
{
	int lRegionEnergy = 0;
	for ( int lTowerEta = aEta; lTowerEta < aEta + 4; ++lTowerEta )
	{
		for ( int lTowerPhi = aPhi; lTowerPhi < aPhi + 4; ++lTowerPhi )
		{
			l1slhc::L1CaloTowerCollection::const_iterator lTowerItr = fetch( lTowerEta, lTowerPhi );
			if ( lTowerItr != mInputCollection->end(  ) )
			{
				// this is no longer done in filling map, so do it here
				if ( lTowerItr->E(  ) > 0 )
					lRegionEnergy += lTowerItr->E(  ) + lTowerItr->H(  );
			}

		}
	}


	if ( lRegionEnergy > 0 )
	{
		int lRegionIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
		std::pair < int, int >lClusterEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lRegionIndex );

		l1slhc::L1CaloRegion lCaloRegion( lClusterEtaPhi.first, lClusterEtaPhi.second, lRegionEnergy );
		mOutputCollection->insert( lClusterEtaPhi.first, lClusterEtaPhi.second, lCaloRegion );
	}

}




DEFINE_EDM_PLUGIN( edm::MakerPluginFactory, edm::WorkerMaker < L1CaloRegionProducer >, "L1CaloRegionProducer" );
DEFINE_FWK_PSET_DESC_FILLER( L1CaloRegionProducer );
