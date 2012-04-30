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
        bool isoLookupTable( const int& clusters, const int& aCoeffA, const int& E );
  bool isoLookupTable( const int& aConeEnergy, const int& aTwoHighTowers, const int& aCoeffA, const int& aE );
        int ItrLookUp(const int& lPhi, const int& nTowers);
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

			  if ( lClusterItr->isCentral(  ) ) ///is central--> Not pruned at all, aE is also declared here
				{
					int lEgammaClusterCount= 0;
					int lTauClusterCount = 0;
					int lEgammaConeEnergy = 0;
					int lTauConeEnergy = 0;

					l1slhc::L1CaloCluster lIsolatedCluster( *lClusterItr );
					for ( int lPhi = aPhi - mCaloTriggerSetup->nIsoTowers(  ); lPhi <= aPhi + mCaloTriggerSetup->nIsoTowers(  ); ++lPhi ) //phi -n iso towers, aph ==central cluster
					{
					  for ( int lEta = aEta - ItrLookUp(abs(lPhi-aPhi),mCaloTriggerSetup->nIsoTowers()); lEta <= aEta + ItrLookUp(abs(lPhi-aPhi),mCaloTriggerSetup->nIsoTowers())+1; ++lEta )
						{
						  if ( !( lEta == aEta && lPhi == aPhi ) ) //requires that the clusters are not the central
							{
								l1slhc::L1CaloClusterCollection::const_iterator lNeighbourItr = fetch( lEta, lPhi );
								if ( lNeighbourItr != mInputCollection->end(  ) )//if it found a cluster
								  {
								    lEgammaConeEnergy = lEgammaConeEnergy + lNeighbourItr->E( );
								    if ( lNeighbourItr->E(  ) >= mCaloTriggerSetup->isoThr(0) )
								      {
									lEgammaClusterCount++; ///count the number of clusters above threshold
								      }

								    if ( lNeighbourItr->E(  ) >= mCaloTriggerSetup->isoThr(1) )//leakage outside 2x2 are different for taus
								      {
									lTauConeEnergy = lTauConeEnergy + lNeighbourItr->E( );
									lTauClusterCount++;
								      }
								  }
							}
						  else if( ( lEta == aEta && lPhi == aPhi )  ){
						    l1slhc::L1CaloClusterCollection::const_iterator lNeighbourItr = fetch( lEta, lPhi );

						  }
						}
					}


					lIsolatedCluster.setIsoClusters( lEgammaClusterCount, lTauClusterCount );
					lIsolatedCluster.setIsoEnergy( lEgammaConeEnergy , lTauConeEnergy );

					// Calculate Bits Tau isolation / electron Isolation
					if ( isoLookupTable( lEgammaConeEnergy, lIsolatedCluster.LeadTowerE(), mCaloTriggerSetup->isolationE(0), lIsolatedCluster.E(  ) ) )
					{
					 
					    lIsolatedCluster.setIsoEG( true );
						
					}

					// Add the LUT inputs 
					if ( isoLookupTable( lTauConeEnergy, mCaloTriggerSetup->isolationT(0), lIsolatedCluster.E(  ) ) )
					{
						lIsolatedCluster.setIsoTau( true );
					}
					mOutputCollection->insert( lIsolatedCluster.iEta(  ) , lIsolatedCluster.iPhi() , lIsolatedCluster );
				}
			}


}


//change look up table to be cone/pt < something

int L1CaloClusterIsolator::ItrLookUp(const int& lPhi, const int& nTowers)
{
  int itrVal = (nTowers/2)+abs(lPhi-nTowers);

    if(itrVal < nTowers){
      return itrVal;
      }
    else
      {return nTowers;}
}

bool L1CaloClusterIsolator::isoLookupTable( const int& aConeEnergy, const int& aCoeffA, const int& aE )	// takes as input the # Clusters the isolation coefficients
{
  	if( aE < 0 ) return false;

	if( aE >= 100 ) return true;

	int cut = 0;
	if( int( double( aConeEnergy*100 )/ double(aE) ) <30){
	cut = aE-aConeEnergy;

	//printf("aE-aClusters=%i aE=%i coneE/aE= %f\n",cut,aE,double( aConeEnergy*100 )/ double(aE));

	return (cut > 20);
	}
	else
	  return false;
}


bool L1CaloClusterIsolator::isoLookupTable( const int& aConeEnergy, const int& aTwoHighTowers, const int& aCoeffA, const int& aE )	// takes as input the # Clusters the isolation coefficients/
{
  printf("Goes to Isolation Algo aE = %i\n",aE);
	int cut = 0;
	int TotalaConeEnergy = aConeEnergy + aE - aTwoHighTowers; //define total cone energy
	//printf("TotalConeE: %i, coneE:%i, aE=%i, 2hightowers:%i\n",aConeEnergy + aE - aTwoHighTowers,aConeEnergy,aE,aTwoHighTowers);
  	if( aE < 0 ) return false;    //less than zero then false
	if( aE >= 100 ) return true;  //>= 100 then true (translates to >=50GeV for electrons) return: Isolated!!

	//printf("rel iso: %f, cut:%i\n",double ( TotalaConeEnergy*100 )/ double(aE),aCoeffA );

	if( double ( TotalaConeEnergy*100 )/ double(aE)  <  aCoeffA){ //change from int to double, multiply by 100: apply relative isolation here ConeE/Electron Energy <0.2
	  //printf("fail1\n");
	  cut = aE-TotalaConeEnergy;                      // define "cut" here == central cluster energy-TotalConeEnergy
	  //if(cut<32)
	    //printf("fail2\b");
	}
	return (cut > 32); // if central cluster relative Iso < 0.20 && Central cluster - coneEnergy >32 then return: Isolated!!



}


DEFINE_EDM_PLUGIN(edm::MakerPluginFactory,edm::WorkerMaker<L1CaloClusterIsolator>,"L1CaloClusterIsolator");
DEFINE_FWK_PSET_DESC_FILLER(L1CaloClusterIsolator);
