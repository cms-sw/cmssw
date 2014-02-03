
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeed.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeedFwd.h"

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/TriggerTowerGeometry.h"

#include "SimDataFormats/SLHC/interface/L1TowerNav.h"


class L1CaloExtendedEgammaClusterProducer:public L1CaloAlgoBase < l1slhc::L1CaloClusterWithSeedCollection , l1slhc::L1CaloClusterWithSeedCollection  > 
{
  public:
	L1CaloExtendedEgammaClusterProducer( const edm::ParameterSet & );
	 ~L1CaloExtendedEgammaClusterProducer(  );

//	void initialize(  );

	void algorithm( const int &, const int & );
    void calculateClusterFourVector(l1slhc::L1CaloClusterWithSeed & cluster);


};

L1CaloExtendedEgammaClusterProducer::L1CaloExtendedEgammaClusterProducer( const edm::ParameterSet & aConfig ):
L1CaloAlgoBase < l1slhc::L1CaloClusterWithSeedCollection , l1slhc::L1CaloClusterWithSeedCollection > ( aConfig )
{
}

L1CaloExtendedEgammaClusterProducer::~L1CaloExtendedEgammaClusterProducer(  )
{
}

/*
void L1CaloExtendedEgammaClusterProducer::initialize(  )
{
}
*/

void L1CaloExtendedEgammaClusterProducer::algorithm( const int &aEta, const int &aPhi )
{

    // Look if there is a cluster here
    l1slhc::L1CaloClusterWithSeedCollection::const_iterator lClusterItr = fetch( aEta, aPhi );
    if ( lClusterItr != mInputCollection->end(  ) )
    {
        l1slhc::L1CaloClusterWithSeed lExtendedEgammaCluster( *lClusterItr );
        //int posPhiPlus  = lExtendedEgammaCluster.hasConstituent(0, 1);
        //int posPhiMinus = lExtendedEgammaCluster.hasConstituent(0, -1);
        int EPhiPlus  = lExtendedEgammaCluster.constituentEmEt(0,1);
        int EPhiMinus = lExtendedEgammaCluster.constituentEmEt(0,-1);

        // Try to extend the cluster starting from non-zero phi+/-1 towers
        if(EPhiPlus>0)
        {
            int posExtPlus = lExtendedEgammaCluster.hasFriend(0, 2);
            if(posExtPlus!=-1)
            {
                l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor1Itr = fetch( aEta-1, aPhi+2 );
                l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor2Itr = fetch( aEta, aPhi+2 );
                l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor3Itr = fetch( aEta+1, aPhi+2 );
                l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor4Itr = fetch( aEta-1, aPhi+3 );
                l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor5Itr = fetch( aEta, aPhi+3 );
                l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor6Itr = fetch( aEta+1, aPhi+3 );
                if(lNeighbor1Itr==mInputCollection->end() &&
                        lNeighbor2Itr==mInputCollection->end() &&
                        lNeighbor3Itr==mInputCollection->end() &&
                        lNeighbor4Itr==mInputCollection->end() &&
                        lNeighbor5Itr==mInputCollection->end() &&
                        lNeighbor6Itr==mInputCollection->end()
                        )
                {
                    l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor7Itr = fetch( aEta, aPhi+4 );
                    if(lNeighbor7Itr==mInputCollection->end() ||  // no neighbor cluster
                            lNeighbor7Itr->constituentEmEt(0,-1)==0 ||
                             lExtendedEgammaCluster.EmEt()>lNeighbor7Itr->EmEt() // larger cluster E (favor cluster with larger phi)
                      ) 
                    {
                        lExtendedEgammaCluster.addConstituent( lExtendedEgammaCluster.getFriend(posExtPlus) );
                        lExtendedEgammaCluster.removeFriend(0, 2); 
                    }
                }
            }
        }
        if(EPhiMinus>0)
        {
            int posExtMinus = lExtendedEgammaCluster.hasFriend(0, -2);
            if(posExtMinus!=-1)
            {
                l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor1Itr = fetch( aEta-1, aPhi-2 );
                l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor2Itr = fetch( aEta, aPhi-2 );
                l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor3Itr = fetch( aEta+1, aPhi-2 );
                l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor4Itr = fetch( aEta-1, aPhi-3 );
                l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor5Itr = fetch( aEta, aPhi-3 );
                l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor6Itr = fetch( aEta+1, aPhi-3 );
                if(lNeighbor1Itr==mInputCollection->end() &&
                        lNeighbor2Itr==mInputCollection->end() &&
                        lNeighbor3Itr==mInputCollection->end() &&
                        lNeighbor4Itr==mInputCollection->end() &&
                        lNeighbor5Itr==mInputCollection->end() &&
                        lNeighbor6Itr==mInputCollection->end()
                        )
                {

                    l1slhc::L1CaloClusterWithSeedCollection::const_iterator lNeighbor7Itr = fetch( aEta, aPhi-4 );
                    if(lNeighbor7Itr==mInputCollection->end() ||  // no neighbor cluster
                            lNeighbor7Itr->constituentEmEt(0,1)==0 ||
                            lExtendedEgammaCluster.EmEt()>=lNeighbor7Itr->EmEt() // larger or equal cluster E (favor cluster with larger phi)
                      ) 
                    {
                        lExtendedEgammaCluster.addConstituent( lExtendedEgammaCluster.getFriend(posExtMinus) );
                        lExtendedEgammaCluster.removeFriend(0, -2); 
                    }
                }
            }
        }

        calculateClusterFourVector(lExtendedEgammaCluster);

        int lIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
        std::pair < int, int >lEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lIndex );
        mOutputCollection->insert( lEtaPhi.first , lEtaPhi.second , lExtendedEgammaCluster );
    }
}

void L1CaloExtendedEgammaClusterProducer::calculateClusterFourVector( l1slhc::L1CaloClusterWithSeed & cluster )
{
	int etaBit = 0;
	int phiBit = 0;


	// get et
	double et = double(cluster.EmEt())/2.;

	TriggerTowerGeometry geo;

	double eta = 0;
	double phi = 0;

	int pos = -1;

    // Find eta position
	// eta+ sum
    int etaPlusSum = 0;
	pos = cluster.hasConstituent( 1, 0 );
	if ( pos != -1 )
		etaPlusSum += cluster.constituentEmEt(1, 0);
	pos = cluster.hasConstituent( 1, 1 );
	if ( pos != -1 )
		etaPlusSum += cluster.constituentEmEt(1, 1);
	pos = cluster.hasConstituent( 1, -1 );
	if ( pos != -1 )
		etaPlusSum += cluster.constituentEmEt(1, -1);

	// eta- sum
    int etaMinusSum = 0;
	pos = cluster.hasConstituent( -1, 0 );
	if ( pos != -1 )
		etaMinusSum += cluster.constituentEmEt(-1, 0);
	pos = cluster.hasConstituent( -1, 1 );
	if ( pos != -1 )
		etaMinusSum += cluster.constituentEmEt(-1, 1);
	pos = cluster.hasConstituent( -1, -1 );
	if ( pos != -1 )
		etaMinusSum += cluster.constituentEmEt(-1, -1);

    if(etaPlusSum>etaMinusSum)
    {
        etaBit = 1;
    }
    else if(etaPlusSum<etaMinusSum)
    {
        etaBit = -1;
    }
    else // equality
    {
        etaBit = 0;
    }
    
    // Find phi position
	// phi+ sum
    int phiPlusSum = 0;
	pos = cluster.hasConstituent( 0, 1 );
	if ( pos != -1 )
		phiPlusSum += cluster.constituentEmEt(0, 1);
	pos = cluster.hasConstituent( 1, 1 );
	if ( pos != -1 )
		phiPlusSum += cluster.constituentEmEt(1, 1);
	pos = cluster.hasConstituent( -1, 1 );
	if ( pos != -1 )
		phiPlusSum += cluster.constituentEmEt(-1, 1);
	pos = cluster.hasConstituent( 0, 2 );
	if ( pos != -1 )
		phiPlusSum += cluster.constituentEmEt(0, 2);

	// phi- sum
    int phiMinusSum = 0;
	pos = cluster.hasConstituent( 0, -1 );
	if ( pos != -1 )
		phiMinusSum += cluster.constituentEmEt(0, -1);
	pos = cluster.hasConstituent( 1, -1 );
	if ( pos != -1 )
		phiMinusSum += cluster.constituentEmEt(1, -1);
	pos = cluster.hasConstituent( -1, -1 );
	if ( pos != -1 )
		phiMinusSum += cluster.constituentEmEt(-1, -1);
	pos = cluster.hasConstituent( 0, -2 );
	if ( pos != -1 )
		phiMinusSum += cluster.constituentEmEt(0, -2);

    if(phiPlusSum>phiMinusSum)
    {
        phiBit = 1;
    }
    else if(phiPlusSum<phiMinusSum)
    {
        phiBit = -1;
    }
    else // equality
    {
        phiBit = 0;
    }


    // Set physical eta,phi
	if( etaBit==1 )
	{
		eta = geo.eta( cluster.iEta() ) + ( geo.towerEtaSize( cluster.iEta() ) * 0.25 ); // center + 1/4;
	}
	else if( etaBit==-1 )
	{
		eta = geo.eta( cluster.iEta() ) - ( geo.towerEtaSize( cluster.iEta() ) * 0.25 ); // center - 1/4;
	}
	else // etaBit==0
	{
		eta = geo.eta( cluster.iEta() ); // center;
	}

	if( phiBit==1 )
	{
		phi = geo.phi( cluster.iPhi() ) + ( geo.towerPhiSize( cluster.iPhi() ) * 0.25 ); // center + 1/4;
	}
	else if( phiBit==-1 )
	{
		phi = geo.phi( cluster.iPhi() ) - ( geo.towerPhiSize( cluster.iPhi() ) * 0.25 ); // center - 1/4;
	}
	else // phiBit==0
	{
		phi = geo.phi( cluster.iPhi() ); // center;
	}


	math::PtEtaPhiMLorentzVector v( et, eta, phi, 0. );

	cluster.setPosBits( etaBit, phiBit );
	cluster.setLorentzVector( v );
}



DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<L1CaloExtendedEgammaClusterProducer>,"L1CaloExtendedEgammaClusterProducer");
DEFINE_FWK_PSET_DESC_FILLER(L1CaloExtendedEgammaClusterProducer);

