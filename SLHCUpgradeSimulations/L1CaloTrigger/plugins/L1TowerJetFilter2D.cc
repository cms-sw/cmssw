
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1TowerJet.h"
#include "SimDataFormats/SLHC/interface/L1TowerJetFwd.h"

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/BitonicSort.hpp"

enum tComparisonDirection { eta , phi };

struct JetWrapper2D
{

  JetWrapper2D( ) :
      mJet( NULL ),
  mComparisonDirection( NULL )
  { /*....*/	}        //constructor


  JetWrapper2D( const l1slhc::L1TowerJet& aJet  ,  const  tComparisonDirection& aComparisonDirection ) :
      mJet( &aJet ),
  mComparisonDirection( &aComparisonDirection )
  { /*....*/	}        //constructor

  const l1slhc::L1TowerJet* mJet;
  const  tComparisonDirection* mComparisonDirection;
};


bool operator> ( JetWrapper2D& aA , JetWrapper2D& aB )
{
  if( aA.mJet == NULL ) return false;	//Jets that don't exist: require 128 or 64 spots
  if( aB.mJet == NULL ) return true;

  if ( aA.mJet->E() > aB.mJet->E() ) return true;
  if ( aA.mJet->E() < aB.mJet->E() ) return false;


  //those aA and aB with the same energy are all that remain
  if ( *(aA.mComparisonDirection) == phi ){
    return (  abs( aA.mJet-> AsymPhi() ) <= abs( aB.mJet->AsymPhi() ) );
  }else{
    return ( abs( aA.mJet-> AsymEta() )  <= abs(  aB.mJet->AsymEta() ) );	
  }	

}



class L1TowerJetFilter2D:public L1CaloAlgoBase < l1slhc::L1TowerJetCollection , l1slhc::L1TowerJetCollection > 
{
  public:
	L1TowerJetFilter2D( const edm::ParameterSet & );
	 ~L1TowerJetFilter2D(  );

//	void initialize(  );

	void algorithm( const int &, const int & );

   // int DO_ONCE;
  private:
        
    tComparisonDirection mComparisonDirection;

    int mNumOfOutputJets;
    


};

L1TowerJetFilter2D::L1TowerJetFilter2D( const edm::ParameterSet & aConfig ):
L1CaloAlgoBase < l1slhc::L1TowerJetCollection , l1slhc::L1TowerJetCollection > ( aConfig ),
mComparisonDirection( eta ),
mNumOfOutputJets( aConfig.getParameter<uint32_t>("NumOfOutputJets") )
{

  std::string lComparisonDirection = aConfig.getParameter<std::string>("ComparisonDirection");

  //do the comparison in upper case so config file can read "Eta", "eta", "ETA", "eTa", etc. and give the same result.
  std::transform( lComparisonDirection.begin() , lComparisonDirection.end() , lComparisonDirection.begin() , ::toupper ); 

  if( lComparisonDirection == "PHI" ){
    mComparisonDirection = phi;
  }
  //else comparison in eta but this is the default

}


L1TowerJetFilter2D::~L1TowerJetFilter2D(  )
{
}

/*
void L1TowerJetFilter2D::initialize(  )
{
}
*/


void L1TowerJetFilter2D::algorithm( const int &aEta, const int &aPhi )
{  
  //Only need an output collection produced once per event:
  if( aPhi != mCaloTriggerSetup->phiMin() )return;
  if( aEta != mCaloTriggerSetup->etaMin() )return;


//---------------------------------------------------------------------------------------------------
//  When we call fetch, we use the Wisconsin coordinate system
//  ie aEta, aPhi, lEta and lPhi need to be defined between mCaloTriggerSetup->phiMin() , mCaloTriggerSetup->phiMax(), etc.
// ie aEta 4->60, aPhi 4->75
//---------------------------------------------------------------------------------------------------

    //declare a vector of structs to pass to algorithm 
  
    //for just one value of eta/phi depending on mComparisonDirection
    std::vector< JetWrapper2D > lJetWrapper2DVector;
  
    //Want all of the input collection so we can veto properly.
    //Its now the same for both directions
  
    for(  int lPhi = mCaloTriggerSetup->phiMin() ; lPhi <= mCaloTriggerSetup->phiMax() ; ++lPhi ){
      for(  int lEta = mCaloTriggerSetup->etaMin() ; lEta <= mCaloTriggerSetup->etaMax() ; ++lEta ){
  
        l1slhc::L1TowerJetCollection::const_iterator lIt = fetch( lEta, lPhi);
  
        if( lIt != mInputCollection->end(  ) ){
          lJetWrapper2DVector.push_back( JetWrapper2D( *lIt , mComparisonDirection ) ); 		
        }
  
      }
    }
    lJetWrapper2DVector.resize(4096); //resize to nearest power of 2 to 72*56
  
  
  //--------------------------------------------------------------------------------------------------
  //  When we call iEta() and iPhi() on the Jets, we use the sensible integer coordinate system
  //  ie Eta runs from -28-28 with no 0, and Phi runs from 1-72
  // This system is used everywhere beyond this point
  //--------------------------------------------------------------------------------------------------
  
    //std::cout << "Sorting Jets produced by " << sourceName() << std::endl;
  
  //  for( std::vector<JetWrapper2D>::iterator lIt =lJetWrapper2DVector.begin(); lIt != lJetWrapper2DVector.end(); ++lIt){
  //    if( (*lIt).mJet )	{
        //std::cout << "Before sort, (eta, phi) = " << (*lIt).mJet->iEta() << " " << (*lIt).mJet->iPhi() <<"  energy " << (*lIt).mJet->E() << " and asym = " << (*lIt).mJet->AsymPhi() <<std::endl;	
  //    }
   // }
  
    // sort jets around eta/phi by energy
    std::vector<JetWrapper2D>::iterator lStart( lJetWrapper2DVector.begin() );
    std::vector<JetWrapper2D>::iterator lEnd( lJetWrapper2DVector.end() );	
    BitonicSort< JetWrapper2D >( down , lStart , lEnd );
  
    //Filter the jets with vetoes
    std::deque< std::pair<int, int> > lVetos; 
    int lCounter(0);
    for( std::vector<JetWrapper2D>::iterator lIt =lJetWrapper2DVector.begin(); lIt != lJetWrapper2DVector.end(); ++lIt){
 
      if( (*lIt).mJet ){ //if jet exists	
        int lJetsize =  (*lIt).mJet->JetSize() ;
        bool lVetoed( false );
  
        for( std::deque< std::pair<int, int> >::iterator lIt2 = lVetos.begin() ; lIt2 != lVetos.end() ; ++ lIt2 ){
  
          if ( (*lIt).mJet->iEta() == (*lIt2).first && (*lIt).mJet->iPhi()  == (*lIt2).second ){ 
            //if jet is already vetoed break
            lVetoed = true;
            break;
          }
        }
  
        if( !lVetoed ){	//if jet not vetoed then add to collection and create vetoes around it

//            std::cout <<" Added jet to the output collection = (" << (*lIt).mJet->iEta() << " , " << (*lIt).mJet->iPhi() <<"), energy = " << (*lIt).mJet->E() << " and asym = " << (*lIt).mJet->AsymPhi() <<" it "<< lCounter <<std::endl;	
          
          mOutputCollection->insert( (*lIt).mJet->iEta() , (*lIt).mJet->iPhi() , *((*lIt).mJet)  );
          lCounter++;
          for( int i = -lJetsize +1 ; i != lJetsize ; ++i ){
  
            int lPhi( (*lIt).mJet->iPhi() + i );
            if( lPhi >  72 ) lPhi -=  72;
            if( lPhi <  1 ) lPhi +=  72;
            std::pair<int,int> veto;
            veto.second = lPhi;
            //for each phi, want to eradicate all the etas along it
            for( int j = -lJetsize +1 ; j != lJetsize ; ++j ){
              int lEta( (*lIt).mJet->iEta() + j );
  
              // no eta=0 in this coordinate system: need to allow for this
              if(lEta == 0 && j<0) lEta = (*lIt).mJet->iEta() - lJetsize;
              if(lEta == 0 && j>0) lEta = (*lIt).mJet->iEta() + lJetsize;
  
              veto.first = lEta;
              lVetos.push_back( veto );
            }
          }
          if( lCounter > mNumOfOutputJets ) break;  //only N jets per ring/strip
        }
      }//jet exists
    }

  
}
 
DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<L1TowerJetFilter2D>,"L1TowerJetFilter2D");
DEFINE_FWK_PSET_DESC_FILLER(L1TowerJetFilter2D);

