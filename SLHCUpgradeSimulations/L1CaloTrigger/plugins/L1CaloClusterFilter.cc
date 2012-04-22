
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/TriggerTowerGeometry.h"


class L1CaloClusterFilter:public L1CaloAlgoBase < l1slhc::L1CaloClusterCollection , l1slhc::L1CaloClusterCollection  > 
{
  public:
	L1CaloClusterFilter( const edm::ParameterSet & );
	 ~L1CaloClusterFilter(  );

//	void initialize(  );

	void algorithm( const int &, const int & );

  private:
	std::pair < int, int > calculateClusterPosition( l1slhc::L1CaloCluster & cluster );

};

L1CaloClusterFilter::L1CaloClusterFilter( const edm::ParameterSet & aConfig ):
L1CaloAlgoBase < l1slhc::L1CaloClusterCollection , l1slhc::L1CaloClusterCollection > ( aConfig )
{
// mPhiOffset = 0; 
mEtaOffset = -1;
//mPhiIncrement = 1; 
//mEtaIncrement = 1;
}

L1CaloClusterFilter::~L1CaloClusterFilter(  )
{
}

/*
void L1CaloClusterFilter::initialize(  )
{
}
*/

void L1CaloClusterFilter::algorithm( const int &aEta, const int &aPhi )
{

			// Look if there is a cluster here
			l1slhc::L1CaloClusterCollection::const_iterator lClusterItr = fetch( aEta, aPhi );
			if ( lClusterItr != mInputCollection->end(  ) )
			{

				l1slhc::L1CaloCluster lFilteredCluster( *lClusterItr );
				// Set lCentralFlag bit
				bool lCentralFlag = true;


				// right
				l1slhc::L1CaloClusterCollection::const_iterator lNeighbourItr = fetch( aEta + 1, aPhi );
				if ( lNeighbourItr != mInputCollection->end(  ) )
				{
					if ( lClusterItr->E(  ) <= lNeighbourItr->E(  ) )
					{
						lFilteredCluster.removeConstituent( 1, 0 );
						lFilteredCluster.removeConstituent( 1, 1 );
						lCentralFlag = false;
					}
				}


				// right-down
				lNeighbourItr = fetch( aEta + 1, aPhi + 1 );
				if ( lNeighbourItr != mInputCollection->end(  ) )
				{
					if ( lClusterItr->E(  ) <= lNeighbourItr->E(  ) )
					{
						lFilteredCluster.removeConstituent( 1, 1 );
						lCentralFlag = false;
					}
				}



				// down
				lNeighbourItr = fetch( aEta, aPhi + 1 );
				if ( lNeighbourItr != mInputCollection->end(  ) )
				{
					if ( lClusterItr->E(  ) <= lNeighbourItr->E(  ) )
					{
						lFilteredCluster.removeConstituent( 0, 1 );
						lFilteredCluster.removeConstituent( 1, 1 );
						lCentralFlag = false;
					}
				}


				// down-left
				lNeighbourItr = fetch( aEta - 1, aPhi + 1 );
				if ( lNeighbourItr != mInputCollection->end(  ) )
				{
					if ( lClusterItr->E(  ) <= lNeighbourItr->E(  ) )
					{
						lFilteredCluster.removeConstituent( 0, 1 );
						lCentralFlag = false;
					}
				}


				// left
				lNeighbourItr = fetch( aEta - 1, aPhi );
				if ( lNeighbourItr != mInputCollection->end(  ) )
				{
					if ( lClusterItr->E(  ) < lNeighbourItr->E(  ) )
					{
						lFilteredCluster.removeConstituent( 0, 0 );
						lFilteredCluster.removeConstituent( 0, 1 );
						lCentralFlag = false;
					}
				}


				// left-up
				lNeighbourItr = fetch( aEta - 1, aPhi - 1 );
				if ( lNeighbourItr != mInputCollection->end(  ) )
				{
					if ( lClusterItr->E(  ) < lNeighbourItr->E(  ) )
					{
						lFilteredCluster.removeConstituent( 0, 0 );
						lCentralFlag = false;
					}
				}


				// up
				lNeighbourItr = fetch( aEta, aPhi - 1 );
				if ( lNeighbourItr != mInputCollection->end(  ) )
				{
					if ( lClusterItr->E(  ) < lNeighbourItr->E(  ) )
					{
						lFilteredCluster.removeConstituent( 0, 0 );
						lFilteredCluster.removeConstituent( 1, 0 );
						lCentralFlag = false;
					}
				}


				// up-right
				lNeighbourItr = fetch( aEta + 1, aPhi - 1 );
				if ( lNeighbourItr != mInputCollection->end(  ) )
				{
					if ( lClusterItr->E(  ) < lNeighbourItr->E(  ) )
					{

						lFilteredCluster.removeConstituent( 1, 0 );
						lCentralFlag = false;
					}
				}



				// Check if the cluster is over threshold
				if ( lFilteredCluster.E(  ) >= mCaloTriggerSetup->clusterThr(  ) )
				{
					calculateClusterPosition( lFilteredCluster );
					lFilteredCluster.setCentral( lCentralFlag );

					int lIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
					std::pair < int, int >lEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lIndex );
					mOutputCollection->insert( lEtaPhi.first , lEtaPhi.second , lFilteredCluster );
				}
			}
}







std::pair < int, int > L1CaloClusterFilter::calculateClusterPosition( l1slhc::L1CaloCluster & cluster )
{
	int etaBit = 0;
	int phiBit = 0;


	// get et
	double et = double( cluster.E(  ) / 2 );

	TriggerTowerGeometry geo;

	double eta = 0;
	double phi = 0;
	double etaW = 0;
	double phiW = 0;

	int pos = -1;

	// eta sum;
	pos = cluster.hasConstituent( 0, 0 );
	if ( pos != -1 )
		etaW -= cluster.getConstituent( pos )->E(  ) + cluster.getConstituent( pos )->H(  );
	pos = cluster.hasConstituent( 0, 1 );
	if ( pos != -1 )
		etaW -= cluster.getConstituent( pos )->E(  ) + cluster.getConstituent( pos )->H(  );
	pos = cluster.hasConstituent( 1, 0 );
	if ( pos != -1 )
		etaW += cluster.getConstituent( pos )->E(  ) + cluster.getConstituent( pos )->H(  );
	pos = cluster.hasConstituent( 1, 1 );
	if ( pos != -1 )
		etaW += cluster.getConstituent( pos )->E(  ) + cluster.getConstituent( pos )->H(  );
	etaW = ( etaW / cluster.E(  ) ) + 1;


	if ( etaW < 0.5 )
	{
		eta = geo.eta( cluster.iEta(  ) ) + ( geo.towerEtaSize( cluster.iEta(  ) ) * 0.125 ); // * 1 / 8;
		etaBit = 0;
	}
	else if ( etaW < 1.0 )
	{
		eta = geo.eta( cluster.iEta(  ) ) + ( geo.towerEtaSize( cluster.iEta(  ) ) * 0.375 ); // * 3 / 8;
		etaBit = 1;
	}
	else if ( etaW < 1.5 )
	{
		eta = geo.eta( cluster.iEta(  ) ) + ( geo.towerEtaSize( cluster.iEta(  ) ) * 0.625 ); // * 5 / 8;
		etaBit = 2;
	}
	else if ( etaW < 2.0 )
	{
		eta = geo.eta( cluster.iEta(  ) ) + ( geo.towerEtaSize( cluster.iEta(  ) ) * 0.875 ); // * 7 / 8;
		etaBit = 3;
	}


	pos = cluster.hasConstituent( 0, 0 );
	if ( pos != -1 )
		phiW -= cluster.getConstituent( pos )->E(  ) + cluster.getConstituent( pos )->H(  );

	pos = cluster.hasConstituent( 1, 0 );
	if ( pos != -1 )
		phiW -= cluster.getConstituent( pos )->E(  ) + cluster.getConstituent( pos )->H(  );

	pos = cluster.hasConstituent( 0, 1 );
	if ( pos != -1 )
		phiW += cluster.getConstituent( pos )->E(  ) + cluster.getConstituent( pos )->H(  );

	pos = cluster.hasConstituent( 1, 1 );
	if ( pos != -1 )
		phi += cluster.getConstituent( pos )->E(  ) + cluster.getConstituent( pos )->H(  );

	phiW = ( phiW / cluster.E(  ) ) + 1;

	if ( phiW < 0.5 )
	{
		phi = geo.phi( cluster.iPhi(  ) ) + ( geo.towerPhiSize( cluster.iPhi(  ) ) * 0.125 ); // * 1 / 8;
		phiBit = 0;
	}
	else if ( phiW < 1.0 )
	{
		phi = geo.phi( cluster.iPhi(  ) ) + ( geo.towerPhiSize( cluster.iPhi(  ) ) * 0.375 ); // * 3 / 8;
		phiBit = 1;
	}
	else if ( phiW < 1.5 )
	{
		phi = geo.phi( cluster.iPhi(  ) ) + ( geo.towerPhiSize( cluster.iPhi(  ) ) * 0.625 ); // * 5 / 8;
		phiBit = 2;
	}
	else if ( phiW < 2.0 )
	{
		phi = geo.phi( cluster.iPhi(  ) ) + ( geo.towerPhiSize( cluster.iPhi(  ) ) * 0.875 ); // * 7 / 8;
		phiBit = 3;
	}

	std::pair < int, int >p = std::make_pair( etaBit, phiBit );

	math::PtEtaPhiMLorentzVector v( et, eta, phi, 0. );

	cluster.setPosBits( etaBit, phiBit );
	cluster.setLorentzVector( v );
	return p;
}



DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<L1CaloClusterFilter>,"L1CaloClusterFilter");
DEFINE_FWK_PSET_DESC_FILLER(L1CaloClusterFilter);

