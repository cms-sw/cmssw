#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"



class L1CaloClusterProducer:public L1CaloAlgoBase < l1slhc::L1CaloTowerCollection, l1slhc::L1CaloClusterCollection >
{
  public:
    L1CaloClusterProducer( const edm::ParameterSet & );
    ~L1CaloClusterProducer(  );

    void initialize(  );

    void algorithm( const int &, const int & );

  private:
    int mElectronThrA, mElectronThrB;

};







L1CaloClusterProducer::L1CaloClusterProducer( const edm::ParameterSet & aConfig ):
  L1CaloAlgoBase < l1slhc::L1CaloTowerCollection, l1slhc::L1CaloClusterCollection > ( aConfig )
{
}


L1CaloClusterProducer::~L1CaloClusterProducer(  )
{
}


void L1CaloClusterProducer::initialize(  )
{
  // these values are constant within an event, so do not recalculate them every time...
  mElectronThrA = int ( double ( mCaloTriggerSetup->electronThr( 2 ) ) / 10. );
  mElectronThrB = mCaloTriggerSetup->electronThr( 0 ) + ( mElectronThrA * mCaloTriggerSetup->electronThr( 1 ) );
}


void L1CaloClusterProducer::algorithm( const int &aEta, const int &aPhi )
{
  int lClusterIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
  std::pair < int, int >lClusterEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lClusterIndex );

  l1slhc::L1CaloCluster lCaloCluster( lClusterEtaPhi.first, lClusterEtaPhi.second );

  bool lFineGrain = false;
  int lClusterEcalE = 0;
  int lClusterTotalE = 0;
  int lLeadTower = 0;
  int lSecondTower = 0;

  for ( int lTowerEta = aEta; lTowerEta <= aEta + 1; ++lTowerEta )
  {
    for ( int lTowerPhi = aPhi; lTowerPhi <= aPhi + 1; ++lTowerPhi )
    {

      // ---------- new way ------------
      l1slhc::L1CaloTowerCollection::const_iterator lTowerItr = fetch( lTowerEta, lTowerPhi );
      if ( lTowerItr != mInputCollection->end(  ) )
      {
        int lTowerTotalE = lTowerItr->E(  ) + lTowerItr->H(  );
        // this is no longer done in filling map, so do it here
        if ( lTowerTotalE > 0 )
        {
          // Skip over fine grain bit calculation if desired
          if ( mCaloTriggerSetup->fineGrainPass(  ) == 1 )
          {
            lFineGrain = false;
          }
          else
          {
            lFineGrain = lFineGrain || lTowerItr->EcalFG(  );
          }

          lClusterTotalE += lTowerTotalE;
          lClusterEcalE += lTowerItr->E(  );

          l1slhc::L1CaloTowerRef lRef( mInputCollection, lTowerItr - mInputCollection->begin(  ) );
          lCaloCluster.addConstituent( lRef );

          if ( lTowerTotalE > lLeadTower )
	    lSecondTower = lLeadTower;
            lLeadTower = lTowerTotalE;
        }
	else if(lTowerTotalE > lSecondTower && lTowerTotalE < lLeadTower){
	  lSecondTower = lTowerTotalE;
	}
      }
      // ---------- new way ------------

    }
  }

  // we only register the Cluster into the event if this condition is satisfied....
  if ( lCaloCluster.E(  ) > 0 )
  {
    // Calculate Electron Cut and Save it in the Cluster
    int lElectronValue = ( int )( 100. * ( ( double )lClusterEcalE ) / ( ( double )lClusterTotalE ) );
    lCaloCluster.setEGammaValue( lElectronValue );

    lCaloCluster.setLeadTower( lLeadTower >= mCaloTriggerSetup->seedTowerThr(  ) );

    lCaloCluster.setLeadTowerE( lLeadTower+lSecondTower );

    // Electron Bit Decision
    bool lLowPtElectron = lCaloCluster.E(  ) <= mCaloTriggerSetup->electronThr( 1 ) && lElectronValue > mCaloTriggerSetup->electronThr( 0 );

    bool lHighPtElectron = lCaloCluster.E(  ) > mCaloTriggerSetup->electronThr( 1 ) && lElectronValue > ( mElectronThrB - ( mElectronThrA * lCaloCluster.E(  ) ) );

    lCaloCluster.setEGamma( lLowPtElectron || lHighPtElectron );

    // FineGrain bit
    lCaloCluster.setFg( lFineGrain );

    mOutputCollection->insert( lClusterEtaPhi.first, lClusterEtaPhi.second , lCaloCluster );
  }

}







DEFINE_EDM_PLUGIN( edm::MakerPluginFactory, edm::WorkerMaker < L1CaloClusterProducer >, "L1CaloClusterProducer" );
DEFINE_FWK_PSET_DESC_FILLER( L1CaloClusterProducer );
