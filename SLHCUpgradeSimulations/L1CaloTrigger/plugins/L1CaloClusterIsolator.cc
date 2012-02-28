#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"


class L1CaloClusterIsolator:public L1CaloAlgoBase < l1slhc::L1CaloClusterCollection , l1slhc::L1CaloClusterCollection >
{
  public:
    L1CaloClusterIsolator( const edm::ParameterSet & );
    ~L1CaloClusterIsolator(  );

    void algorithm( const int &, const int & );
    //	void initialize(  );

  private:
    bool isoLookupTable( const int& clusters, const int& aCoeffA, const int& aCoeffB, const int& E );

};







L1CaloClusterIsolator::L1CaloClusterIsolator( const edm::ParameterSet & aConfig ):
  L1CaloAlgoBase < l1slhc::L1CaloClusterCollection , l1slhc::L1CaloClusterCollection > ( aConfig )
{
  // mPhiOffset = 0; 
  mEtaOffset = -1;
  //mPhiIncrement = 1; 
  //mEtaIncrement = 1;
}


L1CaloClusterIsolator::~L1CaloClusterIsolator(  )
{
}

/*
   void L1CaloClusterIsolator::initialize(  )
   {
   }
   */

void L1CaloClusterIsolator::algorithm( const int &aEta, const int &aPhi )
{

  // Look if there is a cluster here and if the cluster is central (not pruned)

  l1slhc::L1CaloClusterCollection::const_iterator lClusterItr = fetch( aEta, aPhi );
  if ( lClusterItr != mInputCollection->end(  ) )
  {

    if ( lClusterItr->isCentral(  ) )
    {
      int lEgammaClusterCount= 0;
      int lTauClusterCount = 0;
      l1slhc::L1CaloCluster lIsolatedCluster( *lClusterItr );

      // There is a cluster here:Calculate isoDeposits
      for ( int lPhi = aPhi - mCaloTriggerSetup->nIsoTowers(  ); lPhi <= aPhi + mCaloTriggerSetup->nIsoTowers(  ) + 1; ++lPhi )
      {
        for ( int lEta = aEta - mCaloTriggerSetup->nIsoTowers(  ); lEta <= aEta + mCaloTriggerSetup->nIsoTowers(  ) + 1; ++lEta )
        {
          if ( !( lEta == aEta && lPhi == aPhi ) )
          {
            // If neighbor exists
            l1slhc::L1CaloClusterCollection::const_iterator lNeighbourItr = fetch( lEta, lPhi );
            if ( lNeighbourItr != mInputCollection->end(  ) )
            {
              if ( lNeighbourItr->E(  ) >= mCaloTriggerSetup->isoThr(0) )
              {
                lEgammaClusterCount++;
              }
              if ( lNeighbourItr->E(  ) >= mCaloTriggerSetup->isoThr(1) )
              {
                lTauClusterCount++;
              }
            }
          }
        }
      }


      lIsolatedCluster.setIsoClusters( lEgammaClusterCount, lTauClusterCount );


      // Calculate Bits Tau isolation / electron Isolation
      if ( isoLookupTable( lEgammaClusterCount, mCaloTriggerSetup->isolationE(0), mCaloTriggerSetup->isolationE(1), lIsolatedCluster.E(  ) ) )
      {
        lIsolatedCluster.setIsoEG( true );
      }

      // Add the LUT inputs 

      if ( isoLookupTable( lTauClusterCount, mCaloTriggerSetup->isolationT(0), mCaloTriggerSetup->isolationT(1), lIsolatedCluster.E(  ) ) )
      {
        lIsolatedCluster.setIsoTau( true );
      }
      mOutputCollection->insert( lIsolatedCluster.iEta(  ) , lIsolatedCluster.iPhi() , lIsolatedCluster );
    }
  }


}



bool L1CaloClusterIsolator::isoLookupTable( const int& aClusters, const int& aCoeffA, const int& aCoeffB, const int& aE )	// takes as input the # Clusters the isolation coefficients
{
  if( aE < 0 ) return false;
  if( aE >= 160 ) return true;

  int lRegime = (16*(aE/16))+8;

  int lThresh = aCoeffA + int( double( aCoeffB * lRegime ) / 1000. );

  return ( aClusters <= lThresh );
}


DEFINE_EDM_PLUGIN(edm::MakerPluginFactory,edm::WorkerMaker<L1CaloClusterIsolator>,"L1CaloClusterIsolator");
DEFINE_FWK_PSET_DESC_FILLER(L1CaloClusterIsolator);
