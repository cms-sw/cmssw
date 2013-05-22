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
  bool isoLookupTable( const int& aConeE, const int& LeadTowerE, const int& SecondTowerE, const int& aCoeffB, const int& aE, const int& ThirdTowerE, const int& FourthTowerE, const int& Ring1E, const int& Ring2E, const int& Ring3E, const int& Ring4E, const int& Ring5E, const int& Ring6E );	// takes as input the # Clusters the isolation coefficients/
  //bool isoLookupTable( const int& aConeE, const int& LeadTowerE, const int& SecondTowerE, const int& aCoeffB, const int& aE );	// takes as input the # Clusters the isolation coefficients/
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
					int lRing1=0;
					int lRing2=0;
					int lRing3=0;
					int lRing4=0;
					int lRing5=0;
					int lRing6=0;
					//printf("Lead Tower Energy: %i\n",lClusterItr->LeadTowerE( ));
					l1slhc::L1CaloCluster lIsolatedCluster( *lClusterItr );
					for ( int lPhi = aPhi - mCaloTriggerSetup->nIsoTowers(  ); lPhi <= aPhi + mCaloTriggerSetup->nIsoTowers(  ); ++lPhi ) //phi -n iso towers, aph ==central cluster
					{
					  for ( int lEta = aEta - ItrLookUp(abs(lPhi-aPhi),mCaloTriggerSetup->nIsoTowers()); lEta <= aEta + ItrLookUp(abs(lPhi-aPhi),mCaloTriggerSetup->nIsoTowers())+1; ++lEta )
						{
						  if ( !( lEta == aEta && lPhi == aPhi ) ) //requires that the clusters are not the central
							{
								// If neighbor exists
								l1slhc::L1CaloClusterCollection::const_iterator lNeighbourItr = fetch( lEta, lPhi );
								if ( lNeighbourItr != mInputCollection->end(  ) )//if it found a cluster
								  {
								    lEgammaConeEnergy = lEgammaConeEnergy + lNeighbourItr->Et( );

								    if ( lNeighbourItr->Et(  ) >= mCaloTriggerSetup->isoThr(0) )
								      {
									lEgammaClusterCount++; ///count the number of clusters above threshold
									//printf("above thresh neighbor E: %i \n",lNeighbourItr->Et( ));
								      }

								    if ( lNeighbourItr->Et(  ) >= mCaloTriggerSetup->isoThr(1) )//leakage outside 2x2 are different for taus
								      {
									lTauConeEnergy = lTauConeEnergy + lNeighbourItr->Et( );
									lTauClusterCount++;
								      }

								    if((abs(lEta-aEta)+abs(lPhi-aPhi) ==1))
								      {lRing1 = lRing1 + lNeighbourItr->Et( );}
								    else if((abs(lEta-aEta)+abs(lPhi-aPhi) ==2)||(abs(lEta-aEta)==2&&abs(lPhi-aPhi)==1)||(abs(lEta-aEta)==1&&abs(lPhi-aPhi)==2))
								      {lRing2 = lRing2 + lNeighbourItr->Et( );}
								    else if((abs(lEta-aEta)==3||abs(lPhi-aPhi)==3)&&!(abs(lEta-aEta)==3&&abs(lPhi-aPhi)==3))
								      {lRing3 = lRing3 + lNeighbourItr->Et( );}
								    else
								      {lRing4 = lRing4 + lNeighbourItr->Et( );}

								  }

							}
						}
					}


					lIsolatedCluster.setIsoClusters( lEgammaClusterCount, lTauClusterCount );
					lIsolatedCluster.setIsoEnergy( lEgammaConeEnergy , lTauConeEnergy );
					lIsolatedCluster.setRing1E( lRing1 );
					lIsolatedCluster.setRing2E( lRing2 );
					lIsolatedCluster.setRing3E( lRing3 );
					lIsolatedCluster.setRing4E( lRing4 );

					// Calculate Bits Tau isolation / electron Isolation
					if ( isoLookupTable( lEgammaConeEnergy, lIsolatedCluster.LeadTowerE(), lIsolatedCluster.SecondTowerE(), mCaloTriggerSetup->isolationE(0), lIsolatedCluster.Et(  ), lIsolatedCluster.ThirdTowerE(), lIsolatedCluster.FourthTowerE(),lIsolatedCluster.Ring1E(),lIsolatedCluster.Ring2E(),lIsolatedCluster.Ring3E(),lIsolatedCluster.Ring4E(),lRing5, lRing6 ) )
					{
					  //printf("PassedLevel1Isolation\n");
					    lIsolatedCluster.setIsoEG( true );
						
					}

					// Add the LUT inputs 
					if ( isoLookupTable( lTauConeEnergy, mCaloTriggerSetup->isolationT(0), lIsolatedCluster.Et(  ) ) )
					  //if ( isoLookupTable( lTauClusterCount, mCaloTriggerSetup->isolationT(0), mCaloTriggerSetup->isolationT(1), lIsolatedCluster.E(  ) ) )
					{
						lIsolatedCluster.setIsoTau( true );
					}
					mOutputCollection->insert( lIsolatedCluster.iEta(  ) , lIsolatedCluster.iPhi() , lIsolatedCluster );
				}
			}


}


int L1CaloClusterIsolator::ItrLookUp(const int& lPhi, const int& nTowers)
{
  int itrVal = (nTowers/2)+abs(lPhi-nTowers);

    if(itrVal < nTowers)
      {return itrVal;
      }
    else
      {return nTowers;}
}

bool L1CaloClusterIsolator::isoLookupTable( const int& aClusters, const int& aCoeffA, const int& aE )	// takes as input the # Clusters the isolation coefficients
{
  	if( aE < 0 ) return false;
	int cut = 0;
	if( int( double( aClusters*100 )/ double(aE) ) <30){
	cut = aE-aClusters;
	//printf("aE-aClusters=%i aE=%i coneE/aE= %f\n",cut,aE,double( aClusters*100 )/ double(aE));
	return (cut > 50);
	}
	else
	  return false;

}


bool L1CaloClusterIsolator::isoLookupTable( const int& aConeE, const int& LeadTowerE, const int& SecondTowerE, const int& aCoeffB, const int& aE, const int& ThirdTowerE, const int& FourthTowerE, const int& Ring1E, const int& Ring2E, const int& Ring3E, const int& Ring4E,const int& Ring5E, const int& Ring6E )	// takes as input the # Clusters the isolation coefficients/
{

  	if( aE < 0 ) return false;
	//if( aE >= 100 ) return true;

	int cut = 10;
	double absCut = double(FourthTowerE*5+ThirdTowerE*2+Ring1E*10+Ring2E)/double(LeadTowerE+SecondTowerE);

	//printf("4thT:%i 3rdT:%i Ring1:%i LeadT:%i 2ndT:%i absCut: %f\n",FourthTowerE,ThirdTowerE,Ring1E,LeadTowerE,SecondTowerE,absCut);
	//int TotalaClusters = aConeE + aE - LeadTowerE-SecondTowerE;
	//printf("coneEnergy:%i aE:%i aTwoHighestTowers:%i LeadtowerE: %i SecondTowerE: %i\n",aConeE,aE,LeadTowerE+SecondTowerE,LeadTowerE,SecondTowerE);
       	//if(TotalaClusters == aE){
	//  TotalaClusters = aClusters;
      	//}
	//printf("aE-aClusters=%i aE=%i coneE/aE= %f\n",aE-TotalaClusters,aE,double( TotalaClusters*100 )/ double(aE));

	if(  absCut*100  <20){//relative isolation less that .2
	  cut = Ring2E*+Ring3E+Ring4E+Ring5E+Ring6E;
	}

	return (cut < 20);
}

DEFINE_EDM_PLUGIN(edm::MakerPluginFactory,edm::WorkerMaker<L1CaloClusterIsolator>,"L1CaloClusterIsolator");
DEFINE_FWK_PSET_DESC_FILLER(L1CaloClusterIsolator);
