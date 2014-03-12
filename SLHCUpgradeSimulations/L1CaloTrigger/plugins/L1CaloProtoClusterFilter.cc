
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeed.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeedFwd.h"

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/TriggerTowerGeometry.h"

#include "SimDataFormats/SLHC/interface/L1TowerNav.h"


class L1CaloProtoClusterFilter:public L1CaloAlgoBase < l1slhc::L1CaloClusterWithSeedCollection , l1slhc::L1CaloClusterWithSeedCollection  > 
{
  public:
	L1CaloProtoClusterFilter( const edm::ParameterSet & );
	 ~L1CaloProtoClusterFilter(  );

//	void initialize(  );

	void algorithm( const int &, const int & );


};

L1CaloProtoClusterFilter::L1CaloProtoClusterFilter( const edm::ParameterSet & aConfig ):
L1CaloAlgoBase < l1slhc::L1CaloClusterWithSeedCollection , l1slhc::L1CaloClusterWithSeedCollection > ( aConfig )
{
}

L1CaloProtoClusterFilter::~L1CaloProtoClusterFilter(  )
{
}

/*
void L1CaloProtoClusterFilter::initialize(  )
{
}
*/

void L1CaloProtoClusterFilter::algorithm( const int &aEta, const int &aPhi )
{
    // Look if there is a cluster here
    l1slhc::L1CaloClusterWithSeedCollection::const_iterator lClusterItr = fetch( aEta, aPhi );
    if ( lClusterItr != mInputCollection->end(  ) )
    {

        l1slhc::L1CaloClusterWithSeed lFilteredCluster( *lClusterItr );
        bool lFiltered = false;
        // looking at neighboring clusters
        for ( int lClusterEta = aEta-1; lClusterEta <= aEta + 1; ++lClusterEta )
        {
            for ( int lClusterPhi = aPhi-1; lClusterPhi <= aPhi + 1; ++lClusterPhi )
            {
                if(lClusterEta==aEta && lClusterPhi==aPhi)
                {
                    continue;
                }
                l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighborItr = fetch( lClusterEta, lClusterPhi );
                if ( lNeighborItr != mInputCollection->end(  ) )
                {
                    // first look at seed E
                    if(lNeighborItr->seedEmEt() > lClusterItr->seedEmEt())
                    {
                        lFiltered = true;
                        break;
                    }
                    // if equal seed E, then look at total cluster E
                    if(lNeighborItr->seedEmEt() == lClusterItr->seedEmEt() && lNeighborItr->EmEt() > lClusterItr->EmEt())
                    {
                        lFiltered = true;
                        break;
                    }
                    // if equal seed E + equal total E, favor clusters with larger eta and larger phi
                    if( lClusterEta-aEta==1 || (lClusterEta-aEta==0 && lClusterPhi-aPhi==1) )
                    {
                        if(lNeighborItr->seedEmEt() == lClusterItr->seedEmEt() &&
                                lNeighborItr->EmEt() == lClusterItr->EmEt())
                        {
                            lFiltered = true;
                                break;
                        }
                    }
                }
            }
            if(lFiltered) 
            {
                break;
            }
        }


        if ( !lFiltered )
        {
            int lIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
            std::pair < int, int >lEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lIndex );
            mOutputCollection->insert( lEtaPhi.first , lEtaPhi.second , lFilteredCluster );
        }
    }
}



DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<L1CaloProtoClusterFilter>,"L1CaloProtoClusterFilter");
DEFINE_FWK_PSET_DESC_FILLER(L1CaloProtoClusterFilter);

