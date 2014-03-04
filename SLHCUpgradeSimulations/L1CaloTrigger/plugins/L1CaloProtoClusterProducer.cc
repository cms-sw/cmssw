#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeed.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeedFwd.h"

#include "SimDataFormats/SLHC/interface/L1TowerNav.h"



class L1CaloProtoClusterProducer:public L1CaloAlgoBase < l1slhc::L1CaloTowerCollection, l1slhc::L1CaloClusterWithSeedCollection >
{
  public:
    L1CaloProtoClusterProducer( const edm::ParameterSet & );
    ~L1CaloProtoClusterProducer(  );

    void initialize(  );

    void algorithm( const int &, const int & );

  private:
    int mSeedingThreshold, mClusteringThreshold;
    int mHadThreshold;

};







L1CaloProtoClusterProducer::L1CaloProtoClusterProducer( const edm::ParameterSet & aConfig ):
  L1CaloAlgoBase < l1slhc::L1CaloTowerCollection, l1slhc::L1CaloClusterWithSeedCollection > ( aConfig )
{
}


L1CaloProtoClusterProducer::~L1CaloProtoClusterProducer(  )
{
}


void L1CaloProtoClusterProducer::initialize(  )
{
  // thresholds hard-coded for the moment
  mSeedingThreshold = 4; // these are ECAL thresholds
  mClusteringThreshold = 2;

  mHadThreshold = 2; // HCAL tower threshold for H/E calculation
}


void L1CaloProtoClusterProducer::algorithm( const int &aEta, const int &aPhi )
{
    int lClusterIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
    std::pair < int, int >lClusterEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lClusterIndex );
    l1slhc::L1CaloTowerCollection::const_iterator lSeedItr = fetch( aEta, aPhi );
    if(lSeedItr->E() < mSeedingThreshold)
        return;

    l1slhc::L1CaloTowerRef lSeed( mInputCollection, lSeedItr - mInputCollection->begin(  ) );
    l1slhc::L1CaloClusterWithSeed lCaloCluster( lSeed, mHadThreshold );// ieta, iphi, FG are set here


    // Building 3x3 cluster
    for ( int lTowerEta = aEta-1; lTowerEta <= aEta + 1; ++lTowerEta )
    {
        for ( int lTowerPhi = aPhi-1; lTowerPhi <= aPhi + 1; ++lTowerPhi )
        {
            // The seed tower is already taken into account. Not added as a constituent
            if(lTowerEta==aEta && lTowerPhi==aPhi)
            {
                continue;
            }

            l1slhc::L1CaloTowerCollection::const_iterator lTowerItr = fetch( lTowerEta, lTowerPhi );
            if ( lTowerItr != mInputCollection->end(  ) )
            {
                int lTowerE = lTowerItr->E();
                int lTowerH = lTowerItr->H();
                if( lTowerE >= mClusteringThreshold || lTowerH >= mHadThreshold)
                {
                    l1slhc::L1CaloTowerRef lRef( mInputCollection, lTowerItr - mInputCollection->begin(  ) );
                    lCaloCluster.addConstituent( lRef );
                }
            }

        }
    }
    // add phi extensions as friends for later e/g extended clusters production (at this stage their energies are not added to the cluster energy)
    l1slhc::L1CaloTowerCollection::const_iterator lTowerPhiPlusItr = fetch( aEta, aPhi+2);
    if(lTowerPhiPlusItr != mInputCollection->end() && lTowerPhiPlusItr->E()>=mClusteringThreshold)
    {
        l1slhc::L1CaloTowerRef lRef( mInputCollection, lTowerPhiPlusItr - mInputCollection->begin() );
        lCaloCluster.addFriend( lRef );
    }
    l1slhc::L1CaloTowerCollection::const_iterator lTowerPhiMinusItr = fetch( aEta, aPhi-2);
    if(lTowerPhiMinusItr != mInputCollection->end() && lTowerPhiMinusItr->E()>=mClusteringThreshold)
    {
        l1slhc::L1CaloTowerRef lRef( mInputCollection, lTowerPhiMinusItr - mInputCollection->begin() );
        lCaloCluster.addFriend( lRef );
    }


    mOutputCollection->insert( lClusterEtaPhi.first, lClusterEtaPhi.second , lCaloCluster );

}







DEFINE_EDM_PLUGIN( edm::MakerPluginFactory, edm::WorkerMaker < L1CaloProtoClusterProducer >, "L1CaloProtoClusterProducer" );
DEFINE_FWK_PSET_DESC_FILLER( L1CaloProtoClusterProducer );
