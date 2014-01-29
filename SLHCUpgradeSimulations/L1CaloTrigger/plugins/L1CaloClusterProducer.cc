#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"

//SHarper changes:
//1) FG veto is now just the leading EmEt tower, no longer the OR of the 2x2. On second thoughts, I need to understand this a bit better to know if the OR of the 2x2 is the right thing to do or not. But I've made the change for now
//2) renamed poorly named variables
//3) the EmEt/(totEt) cut relax threshold now works of EmEt otherwise it turns into a jet trigger.... 



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

  //SHarper: as soon I as figure out what these varibles actually are, I'm renaming them...
  mElectronThrA = int ( double ( mCaloTriggerSetup->electronThr( 2 ) ) / 10. ); //SHarper: extremely odd..
  mElectronThrB = mCaloTriggerSetup->electronThr( 0 ) + ( mElectronThrA * mCaloTriggerSetup->electronThr( 1 ) );

  // std::cout <<"mElectronThrA "<<mElectronThrA<<" mElectronThrB "<<mElectronThrB<<std::endl;
}


void L1CaloClusterProducer::algorithm( const int &aEta, const int &aPhi )
{
  int lClusterIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
  std::pair < int, int >lClusterEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lClusterIndex );

  l1slhc::L1CaloCluster lCaloCluster( lClusterEtaPhi.first, lClusterEtaPhi.second );

  bool lFineGrain = false;
  int lClusterEmEt = 0;
  int lClusterTotalEt = 0;
  int lLeadTowerTotalEt = 0;
  int lSecondTowerTotalEt = 0;

  int lHighestTowerEmEt =0;

  for ( int lTowerEta = aEta; lTowerEta <= aEta + 1; ++lTowerEta ){
    for ( int lTowerPhi = aPhi; lTowerPhi <= aPhi + 1; ++lTowerPhi ){
      
      l1slhc::L1CaloTowerCollection::const_iterator lTowerItr = fetch( lTowerEta, lTowerPhi );
      if ( lTowerItr != mInputCollection->end(  ) ){
        int lTowerTotalEt = lTowerItr->E(  ) + lTowerItr->H(  );
	int lTowerEmEt  = lTowerItr->E();
	   
	lClusterTotalEt += lTowerTotalEt; 
        // this is no longer done in filling map, so do it here
        if ( lTowerTotalEt > 0 ){ //so this really just decides if its added the caloCluster as  constituent, currently we only add towers with em energy to the cluster, but we could relax this requirement, however still wouldnt contribute to anything other than marking it as existing
      
          lClusterEmEt += lTowerEmEt;
	  
          l1slhc::L1CaloTowerRef lRef( mInputCollection, lTowerItr - mInputCollection->begin(  ) );
          lCaloCluster.addConstituent( lRef );
	  
          if(lTowerTotalEt > lLeadTowerTotalEt ) {
	    lSecondTowerTotalEt = lLeadTowerTotalEt;
	    lLeadTowerTotalEt = lTowerTotalEt;
          } else if(lTowerTotalEt > lSecondTowerTotalEt && lTowerTotalEt < lLeadTowerTotalEt){
	    lSecondTowerTotalEt = lTowerTotalEt;
	  }
	  
	  if(lTowerEmEt>lHighestTowerEmEt){
	    lHighestTowerEmEt = lTowerEmEt;
	    if ( mCaloTriggerSetup->fineGrainPass(  ) != 1 ) lFineGrain = lTowerItr->EcalFG(  ); //SHarper, changed FG to be just the highest EmEt rather than OR of the 2x2
	  }
	}//end lTowerHadEt>0 check
	
      }//end lTowerItr is valid check
      
      
    }//end phi loop
  }//end eta loop

  // we only register the Cluster into the event if this condition is satisfied....
  if ( lCaloCluster.Et(  ) > 0 ) {
    // Calculate Electron Cut and Save it in the Cluster
    int lElectronValue = ( int )( 100. * ( ( double )lClusterEmEt ) / ( ( double )lClusterTotalEt ) );
    lCaloCluster.setEGammaValue( lElectronValue );

    //  if(lCaloCluster.iEta()==-5) std::cout <<"ieta "<<lCaloCluster.iEta()<<" iphi "<<lCaloCluster.iPhi()<<" lClusterEmEt "<<lClusterEmEt<<" l1ClusterTotalEt "<<lClusterTotalEt<<" ele val "<<lElectronValue<<std::endl;

    lCaloCluster.setLeadTower( lLeadTowerTotalEt >= mCaloTriggerSetup->seedTowerThr(  ) );

    lCaloCluster.setLeadTowerE( lLeadTowerTotalEt+lSecondTowerTotalEt ); //SH: this is not clear to me at all why leadTowerE has leadTower + second tower Et

    // Electron Bit Decision
    bool lLowPtElectron = lCaloCluster.EmEt(  ) <= mCaloTriggerSetup->electronThr( 1 ) && lElectronValue > mCaloTriggerSetup->electronThr( 0 );

    bool lHighPtElectron = lCaloCluster.EmEt(  ) > mCaloTriggerSetup->electronThr( 1 ) && lElectronValue > ( mElectronThrB - ( mElectronThrA * lCaloCluster.EmEt(  ) ) );

    lCaloCluster.setEGamma( lLowPtElectron || lHighPtElectron );

    // FineGrain bit
    lCaloCluster.setFg( lFineGrain );

    mOutputCollection->insert( lClusterEtaPhi.first, lClusterEtaPhi.second , lCaloCluster );
  }

}







DEFINE_EDM_PLUGIN( edm::MakerPluginFactory, edm::WorkerMaker < L1CaloClusterProducer >, "L1CaloClusterProducer" );
DEFINE_FWK_PSET_DESC_FILLER( L1CaloClusterProducer );
