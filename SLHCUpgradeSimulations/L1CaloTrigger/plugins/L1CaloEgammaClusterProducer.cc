
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeed.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeedFwd.h"

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/TriggerTowerGeometry.h"


class L1CaloEgammaClusterProducer:public L1CaloAlgoBase < l1slhc::L1CaloClusterWithSeedCollection , l1slhc::L1CaloClusterWithSeedCollection  > 
{
    public:
        L1CaloEgammaClusterProducer( const edm::ParameterSet & );
        ~L1CaloEgammaClusterProducer(  );

        //	void initialize(  );

        void algorithm( const int &, const int & );

    private:
        void trimCluster(l1slhc::L1CaloClusterWithSeed& egammaCluster);
        void extendCluster(l1slhc::L1CaloClusterWithSeed& egammaCluster, int aEta, int aPhi);
        void calculateClusterFourVector(l1slhc::L1CaloClusterWithSeed & cluster);



};

L1CaloEgammaClusterProducer::L1CaloEgammaClusterProducer( const edm::ParameterSet & aConfig ):
L1CaloAlgoBase < l1slhc::L1CaloClusterWithSeedCollection , l1slhc::L1CaloClusterWithSeedCollection > ( aConfig )
{
}

L1CaloEgammaClusterProducer::~L1CaloEgammaClusterProducer(  )
{
}

/*
void L1CaloEgammaClusterProducer::initialize(  )
{
}
*/

void L1CaloEgammaClusterProducer::algorithm( const int &aEta, const int &aPhi )
{

    // Look if there is a cluster here
    l1slhc::L1CaloClusterWithSeedCollection::const_iterator lClusterItr = fetch( aEta, aPhi );
    if ( lClusterItr != mInputCollection->end(  ) )
    {
        // Check if the proto-cluster has been flagged as e/g
        if( lClusterItr->isEGamma() )
        {
            l1slhc::L1CaloClusterWithSeed lEgammaCluster( *lClusterItr );

            trimCluster(lEgammaCluster);
            extendCluster(lEgammaCluster, aEta, aPhi);
            calculateClusterFourVector(lEgammaCluster);

            int lIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
            std::pair < int, int >lEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lIndex );
            mOutputCollection->insert( lEtaPhi.first , lEtaPhi.second , lEgammaCluster );
        }
    }
}

void L1CaloEgammaClusterProducer::trimCluster( l1slhc::L1CaloClusterWithSeed& egammaCluster )
{
    int posEtaPlus  = egammaCluster.hasConstituent(1, 0);
    int posEtaMinus = egammaCluster.hasConstituent(-1, 0);

    int posEtaPlusPhiPlus   = egammaCluster.hasConstituent(1, 1);
    int posEtaMinusPhiPlus  = egammaCluster.hasConstituent(-1, 1);
    int posEtaMinusPhiMinus = egammaCluster.hasConstituent(-1, -1);
    int posEtaPlusPhiMinus  = egammaCluster.hasConstituent(1, -1);

    int EetaPlus  = (posEtaPlus!=-1  ? egammaCluster.constituentEmEt(1, 0) : 0);
    int EetaMinus = (posEtaMinus!=-1 ? egammaCluster.constituentEmEt(-1, 0) : 0);

    int EetaPlusPhiPlus   = (posEtaPlusPhiPlus!=-1    ? egammaCluster.constituentEmEt(1, 1) : 0);
    int EetaMinusPhiPlus  = (posEtaMinusPhiPlus!=-1   ? egammaCluster.constituentEmEt(-1, 1) : 0);
    int EetaMinusPhiMinus = (posEtaMinusPhiMinus!=-1  ? egammaCluster.constituentEmEt(-1, -1) : 0);
    int EetaPlusPhiMinus  = (posEtaPlusPhiMinus!=-1   ? egammaCluster.constituentEmEt(1, -1) : 0);

    bool keepEtaPlus = false;
    bool keepEtaMinus = false;

    // First choose to remove or keep towers at eta +/- 1
    if(EetaPlus>EetaMinus) // -> remove eta-1
    {
        keepEtaPlus = true;
    }
    else if(EetaPlus<EetaMinus) // -> remove eta+1
    {
        keepEtaMinus = true;
    }
    else // eta+1 = eta-1 -> look at the corners of the 3x3 proto-cluster
    {
        int EtotEtaPlus  = EetaPlusPhiPlus + EetaPlusPhiMinus;
        int EtotEtaMinus = EetaMinusPhiPlus + EetaMinusPhiMinus;
        if(EtotEtaPlus>EtotEtaMinus) // -> remove eta-1
        {
            keepEtaPlus = true;
        }
        else if(EtotEtaPlus<EtotEtaMinus) // -> remove eta+1
        {
            keepEtaMinus = true;
        }
        else // -> keep both eta+1 and eta-1
        {
            keepEtaPlus = true;
            keepEtaMinus = true;
        }
    }
    if(!keepEtaPlus)
    {
        egammaCluster.removeConstituent(1, -1);
        egammaCluster.removeConstituent(1, 0);
        egammaCluster.removeConstituent(1, 1);
        egammaCluster.setTrimmedPlus();
    }
    if(!keepEtaMinus)
    {
        egammaCluster.removeConstituent(-1, -1);
        egammaCluster.removeConstituent(-1, 0);
        egammaCluster.removeConstituent(-1, 1);
        egammaCluster.setTrimmedMinus();
    }
    // check if the towers in the corners are kept or not (they have to be adjacent to a non-zero tower)
    if(keepEtaPlus)
    {
        if(egammaCluster.hasConstituent(0, 1)==-1 && egammaCluster.hasConstituent(1, 0)==-1)
        {
            egammaCluster.removeConstituent(1, 1);
        }
        if(egammaCluster.hasConstituent(0, -1)==-1 && egammaCluster.hasConstituent(1, 0)==-1)
        {
            egammaCluster.removeConstituent(1, -1);
        }
    }
    if(keepEtaMinus)
    {
        if(egammaCluster.hasConstituent(0, 1)==-1 && egammaCluster.hasConstituent(-1, 0)==-1)
        {
            egammaCluster.removeConstituent(-1, 1);
        }
        if(egammaCluster.hasConstituent(0, -1)==-1 && egammaCluster.hasConstituent(-1, 0)==-1)
        {
            egammaCluster.removeConstituent(-1, -1);
        }
    }   
}

void L1CaloEgammaClusterProducer::extendCluster( l1slhc::L1CaloClusterWithSeed& egammaCluster, int aEta, int aPhi )
{
    //int posPhiPlus  = egammaCluster.hasConstituent(0, 1);
    //int posPhiMinus = egammaCluster.hasConstituent(0, -1);
    int EPhiPlus  = egammaCluster.constituentEmEt(0,1);
    int EPhiMinus = egammaCluster.constituentEmEt(0,-1);

    // Try to extend the cluster starting from non-zero phi+/-1 towers
    if(EPhiPlus>0)
    {
        int posExtPlus = egammaCluster.hasFriend(0, 2);
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
                        egammaCluster.EmEt()>lNeighbor7Itr->EmEt() // larger cluster E (favor cluster with larger phi)
                  ) 
                {
                    egammaCluster.addConstituent( egammaCluster.getFriend(posExtPlus) );
                    egammaCluster.removeFriend(0, 2); 
                }
            }
        }
    }
    if(EPhiMinus>0)
    {
        int posExtMinus = egammaCluster.hasFriend(0, -2);
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
                        egammaCluster.EmEt()>=lNeighbor7Itr->EmEt() // larger or equal cluster E (favor cluster with larger phi)
                  ) 
                {
                    egammaCluster.addConstituent( egammaCluster.getFriend(posExtMinus) );
                    egammaCluster.removeFriend(0, -2); 
                }
            }
        }
    }
}

void L1CaloEgammaClusterProducer::calculateClusterFourVector( l1slhc::L1CaloClusterWithSeed & cluster )
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

DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<L1CaloEgammaClusterProducer>,"L1CaloEgammaClusterProducer");
DEFINE_FWK_PSET_DESC_FILLER(L1CaloEgammaClusterProducer);

