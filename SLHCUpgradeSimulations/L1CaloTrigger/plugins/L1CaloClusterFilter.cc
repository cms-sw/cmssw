
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/TriggerTowerGeometry.h"

//SHarper changes:
//1) fixed all the indenting, the functions were halfway across the screen!
//2) removed all unsafe bracketless if statements, either put it on the same line as the if statement or put brackets around it. There was a bug in L1CaloClusterProducer due to this practise so its not me just being picky...
//3) updated to reflect that cluster et is only the emet, hcal doesnt contribute to anything other than id/isolation cuts
//4) fixed the conversion back to real Et in GeV which was throwing away the LSB 
class L1CaloClusterFilter:public L1CaloAlgoBase < l1slhc::L1CaloClusterCollection , l1slhc::L1CaloClusterCollection  > 
{
public:
  L1CaloClusterFilter( const edm::ParameterSet & );
  ~L1CaloClusterFilter(  );
  
  //	void initialize(  );
  
  void algorithm( const int &, const int & );
  
private:
  std::pair < int, int > calculateClusterPosition( l1slhc::L1CaloCluster & cluster );
  int etMode_; //whether we consider Et = Em+Had (taus) or Et = Em only (electrons), has an impact on cluster position + whether isCentral

  
};

L1CaloClusterFilter::L1CaloClusterFilter( const edm::ParameterSet & aConfig ):
  L1CaloAlgoBase < l1slhc::L1CaloClusterCollection , l1slhc::L1CaloClusterCollection > ( aConfig )
{
  // mPhiOffset = 0; 
  mEtaOffset = -1;
  //mPhiIncrement = 1; 
  //mEtaIncrement = 1;
  etMode_ = aConfig.getParameter<int>("etMode");
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
  if ( lClusterItr != mInputCollection->end(  ) ){
    
    l1slhc::L1CaloCluster lFilteredCluster( *lClusterItr );
    // Set lCentralFlag bit
    bool lCentralFlag = true;
    
    
    // right
    l1slhc::L1CaloClusterCollection::const_iterator lNeighbourItr = fetch( aEta + 1, aPhi );
    if ( lNeighbourItr != mInputCollection->end(  ) ){
      if ( lClusterItr->Et(etMode_) <= lNeighbourItr->Et(etMode_) ){
	lFilteredCluster.removeConstituent( 1, 0 );
	lFilteredCluster.removeConstituent( 1, 1 );
	lCentralFlag = false;
      }
    }
      
      
    // right-down
    lNeighbourItr = fetch( aEta + 1, aPhi + 1 );
    if ( lNeighbourItr != mInputCollection->end(  ) ){
      if ( lClusterItr->Et(etMode_) <= lNeighbourItr->Et(etMode_) ){
	lFilteredCluster.removeConstituent( 1, 1 );
	lCentralFlag = false;
      }
    }
         
    // down
    lNeighbourItr = fetch( aEta, aPhi + 1 );
    if ( lNeighbourItr != mInputCollection->end(  ) ){
      if ( lClusterItr->Et(etMode_) <= lNeighbourItr->Et(etMode_) ){
	lFilteredCluster.removeConstituent( 0, 1 );
	lFilteredCluster.removeConstituent( 1, 1 );
	lCentralFlag = false;
      }
    }
    
      
      // down-left
    lNeighbourItr = fetch( aEta - 1, aPhi + 1 );
    if ( lNeighbourItr != mInputCollection->end(  ) ){
      if ( lClusterItr->Et(etMode_) <= lNeighbourItr->Et(etMode_) ){
	lFilteredCluster.removeConstituent( 0, 1 );
	lCentralFlag = false;
      }
    }
      
      
    // left
    lNeighbourItr = fetch( aEta - 1, aPhi );
    if ( lNeighbourItr != mInputCollection->end(  ) ){
      if ( lClusterItr->Et(etMode_) < lNeighbourItr->Et(etMode_) ){
	lFilteredCluster.removeConstituent( 0, 0 );
	lFilteredCluster.removeConstituent( 0, 1 );
	lCentralFlag = false;
      }
    }
    
    
    // left-up
    lNeighbourItr = fetch( aEta - 1, aPhi - 1 );
    if ( lNeighbourItr != mInputCollection->end(  ) ){
      if ( lClusterItr->Et(etMode_) < lNeighbourItr->Et(etMode_) ){
	lFilteredCluster.removeConstituent( 0, 0 );
	lCentralFlag = false;
      }
    }
      
      
    // up
    lNeighbourItr = fetch( aEta, aPhi - 1 );
    if ( lNeighbourItr != mInputCollection->end(  ) ){
      if ( lClusterItr->Et(etMode_) < lNeighbourItr->Et(etMode_) ){
	lFilteredCluster.removeConstituent( 0, 0 );
	lFilteredCluster.removeConstituent( 1, 0 );
	lCentralFlag = false;
      }
    }

      
    // up-right
    lNeighbourItr = fetch( aEta + 1, aPhi - 1 );
    if ( lNeighbourItr != mInputCollection->end(  ) ){
      if ( lClusterItr->Et(etMode_) < lNeighbourItr->Et(etMode_) ){	
	lFilteredCluster.removeConstituent( 1, 0 );
	lCentralFlag = false;
      }
    }
    
      
      
      // Check if the cluster is over threshold
    if ( lFilteredCluster.Et(etMode_) >= mCaloTriggerSetup->clusterThr(  ) ){
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
  double et = double( cluster.Et(etMode_) / 2. ); //SH: added a "." after the 2 so we dont throw away the 0.5 GeV we've gone through hell and back to have

  TriggerTowerGeometry geo;
  
  double eta = 0;
  double phi = 0;
  double etaW = 0;
  double phiW = 0;
  
  for(int etaNr=0;etaNr<=1;etaNr++){
    for(int phiNr=0;phiNr<=1;phiNr++){
      int towerIndex = cluster.hasConstituent(etaNr,phiNr);
      int towerEmEt = towerIndex!=-1 ? cluster.getConstituent(towerIndex)->E() : 0;
      etaW +=(-1+etaNr*2)*towerEmEt;  //0 is centre of 2x2 cluster so same eta/phi as seed is -1 and diff eta/phi is +1
      phiW +=(-1+phiNr*2)*towerEmEt;
    }
  }
  
  etaW = ( etaW / cluster.Et(etMode_) ) + 1; //SH: why the +1, mysterious..., ah so it defines zero as the seed tower, ie the bottom left tower, range is therefore 0 to 2
  phiW = ( phiW / cluster.Et(etMode_) ) + 1;
    
  //SH: the algorithm defines 4 bins (0-0.5,0.5-1.0,1.0-1.5,1.5-2.0) in average position away from the bottom left tower. It returns the central position of these bins
  //leave for now but I'm not happy with this, I would like to be a little smarter. I dislike the property that it can not be the eta of the seed tower even if there are no other towers about
  //would prefer to properly round if possible
  
  if ( etaW < 0.5 ){
    eta = geo.eta( cluster.iEta(  ) ) + ( geo.towerEtaSize( cluster.iEta(  ) ) * 0.125 ); // * 1 / 8;
    etaBit = 0;
  }else if ( etaW < 1.0 ) {
    eta = geo.eta( cluster.iEta(  ) ) + ( geo.towerEtaSize( cluster.iEta(  ) ) * 0.375 ); // * 3 / 8;
      etaBit = 1;
  }else if ( etaW < 1.5 ){
    eta = geo.eta( cluster.iEta(  ) ) + ( geo.towerEtaSize( cluster.iEta(  ) ) * 0.625 ); // * 5 / 8;
    etaBit = 2;
  }else if ( etaW < 2.0 ){
    eta = geo.eta( cluster.iEta(  ) ) + ( geo.towerEtaSize( cluster.iEta(  ) ) * 0.875 ); // * 7 / 8;
    etaBit = 3;
  }

 
  if ( phiW < 0.5 ){
    phi = geo.phi( cluster.iPhi(  ) ) + ( geo.towerPhiSize( cluster.iPhi(  ) ) * 0.125 ); // * 1 / 8;
    phiBit = 0;
  }else if ( phiW < 1.0 ){
    phi = geo.phi( cluster.iPhi(  ) ) + ( geo.towerPhiSize( cluster.iPhi(  ) ) * 0.375 ); // * 3 / 8;
    phiBit = 1;
  }else if ( phiW < 1.5 ){
    phi = geo.phi( cluster.iPhi(  ) ) + ( geo.towerPhiSize( cluster.iPhi(  ) ) * 0.625 ); // * 5 / 8;
    phiBit = 2;
  }else if ( phiW < 2.0 ){
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

