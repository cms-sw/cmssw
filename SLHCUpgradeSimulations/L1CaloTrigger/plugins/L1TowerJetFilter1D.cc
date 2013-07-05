
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1TowerJet.h"
#include "SimDataFormats/SLHC/interface/L1TowerJetFwd.h"

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/BitonicSort.hpp"

#define DEBUG std::cout<< __FILE__ << " : " << __LINE__ << std::endl;

enum tComparisonDirection { eta , phi };

struct JetWrapper
{

  JetWrapper( ) :
    mJet( NULL ),
    mComparisonDirection( NULL )
    { /*....*/	}        //constructor


  JetWrapper( const l1slhc::L1TowerJet& aJet  ,  const  tComparisonDirection& aComparisonDirection ) :
    mJet( &aJet ),
    mComparisonDirection( &aComparisonDirection )
    { /*....*/	}        //constructor

  const l1slhc::L1TowerJet* mJet;
  const  tComparisonDirection* mComparisonDirection;
};


bool operator> ( JetWrapper& aA , JetWrapper& aB )
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


class L1TowerJetFilter1D:public L1CaloAlgoBase < l1slhc::L1TowerJetCollection , l1slhc::L1TowerJetCollection > 
{
  public:
     L1TowerJetFilter1D( const edm::ParameterSet & );
     ~L1TowerJetFilter1D(  );

//	void initialize(  );

     void algorithm( const int &, const int & );
        

  private:
        
    tComparisonDirection mComparisonDirection;

    int mNumOfOutputJets;

};


L1TowerJetFilter1D::L1TowerJetFilter1D( const edm::ParameterSet & aConfig ):
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


L1TowerJetFilter1D::~L1TowerJetFilter1D(  )
{
}


//void L1TowerJetFilter1D::initialize(  )
//{
//}

void L1TowerJetFilter1D::algorithm( const int &aEta, const int &aPhi )
{
  if ( mComparisonDirection == phi && aPhi != mCaloTriggerSetup->phiMin() ) return;
  else if ( mComparisonDirection == eta && aEta != mCaloTriggerSetup->etaMin() ) return;


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//  When we call fetch, we use the Wisconsin coordinate system
//  ie aEta, aPhi, lEta and lPhi need to be defined between mCaloTriggerSetup->phiMin() , mCaloTriggerSetup->phiMax(), etc.
// ie aEta 4->60, aPhi 4->75
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  //declare a vector of structs to pass to algorithm 
  std::vector< JetWrapper > lJetWrapperVector;
  if ( mComparisonDirection == phi ){
    for(  int lPhi = mCaloTriggerSetup->phiMin() ; lPhi <= mCaloTriggerSetup->phiMax() ; ++lPhi ){
      l1slhc::L1TowerJetCollection::const_iterator lIt = fetch( aEta, lPhi);
      if( lIt != mInputCollection->end(  ) ){
        lJetWrapperVector.push_back( JetWrapper( *lIt , mComparisonDirection ) ); 		
      }
    }
    lJetWrapperVector.resize(128);    //algorithm requires 2^N inputs
  }else{ 	// eta	
    for(  int lEta = mCaloTriggerSetup->etaMin() ; lEta <= mCaloTriggerSetup->etaMax() ; ++lEta ){
      l1slhc::L1TowerJetCollection::const_iterator lIt = fetch( lEta, aPhi);
      if( lIt != mInputCollection->end(  ) ){
        lJetWrapperVector.push_back( JetWrapper( *lIt , mComparisonDirection ) );
      }
    }
    lJetWrapperVector.resize(64);     //algorithm requires 2^N inputs
  }


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//  When we call iEta() and iPhi() on the Jets, we use the sensible integer coordinate system
//  ie Eta runs from -28-28 with no 0, and Phi runs from 1-72
// This system is used everywhere beyond this point
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//  std::cout << "Sorting Jets produced by " << sourceName() << std::endl;  
//   for( std::vector<JetWrapper>::iterator lIt =lJetWrapperVector.begin(); lIt != lJetWrapperVector.end(); ++lIt){
//     if( (*lIt).mJet )	{
//       //  std::cout << "Before sort, (eta, phi) = " << (*lIt).mJet->iEta() << " " << (*lIt).mJet->iPhi() <<"  energy " << (*lIt).mJet->E() << " and asym = " << (*lIt).mJet->AsymPhi() <<std::endl;	
//     }
//   }

  // sort jets around eta/phi by energy
  std::vector<JetWrapper>::iterator lStart( lJetWrapperVector.begin() );
  std::vector<JetWrapper>::iterator lEnd( lJetWrapperVector.end() );	
  BitonicSort< JetWrapper >( down , lStart , lEnd );

  //Veto overlapping jets in 1D
  std::deque<int> lVetos;
  int lCounter(0);

  for( std::vector<JetWrapper>::iterator lIt =lJetWrapperVector.begin(); lIt != lJetWrapperVector.end(); ++lIt){

    if( (*lIt).mJet ){ //if jet exists	
      int lJetsize =  (*lIt).mJet->JetSize() ;

      if( mComparisonDirection == phi ){ //phi
        bool lVetoed( false );
        for( std::deque< int >::iterator lIt2 = lVetos.begin() ; lIt2 != lVetos.end() ; ++ lIt2 ){

          if ( (*lIt).mJet->iPhi()  == (*lIt2) ){ //if jet is already vetoed break
            lVetoed = true;
            break;
          }
        }

        if( !lVetoed ){	//if jet not vetoed then add to collection and create vetoes around it

          mOutputCollection->insert( (*lIt).mJet->iEta() , (*lIt).mJet->iPhi() , *((*lIt).mJet)  );
          lCounter++;
          for( int i = -lJetsize +1 ; i != lJetsize ; ++i ){

            int lPhi( (*lIt).mJet->iPhi() + i );
            if( lPhi >  72 ) lPhi -=  72;
            if( lPhi <  1 ) lPhi +=  72;
            lVetos.push_back( lPhi);

          }
          if( lCounter > mNumOfOutputJets ) break;  //only N jets per ring/strip
        }

      }else{ //eta

        bool lVetoed( false );
        for( std::deque<int>::iterator lIt2 = lVetos.begin() ; lIt2 != lVetos.end() ; ++lIt2 ){

          if(  (*lIt).mJet->iEta()  == (*lIt2) ){
                  lVetoed = true;
                  break;
          }
        }

        if( !lVetoed ){	

          mOutputCollection->insert( (*lIt).mJet->iEta() , (*lIt).mJet->iPhi() , *((*lIt).mJet)  );
          ++lCounter;

          for( int i = -lJetsize +1 ; i != lJetsize  ; ++i ){
            int lEta( (*lIt).mJet->iEta() + i );
            // no eta=0 in this coordinate system: need to allow for this
            if(lEta == 0 && i<0) lEta = (*lIt).mJet->iEta() - lJetsize;
            if(lEta == 0 && i>0) lEta = (*lIt).mJet->iEta() + lJetsize;
            lVetos.push_back( lEta);
          }
          if( lCounter > mNumOfOutputJets ) break;  //only N jets per ring/strip
        }
      }
    }
  }

  
}

DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<L1TowerJetFilter1D>,"L1TowerJetFilter1D");
DEFINE_FWK_PSET_DESC_FILLER(L1TowerJetFilter1D);


