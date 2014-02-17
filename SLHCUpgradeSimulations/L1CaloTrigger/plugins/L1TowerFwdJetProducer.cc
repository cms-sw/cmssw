// Original Author:  Robyn Elizabeth Lucas,510 1-002,+41227673823,
// Modifications  :  Mark Baber Imperial College, London

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1TowerJet.h"
#include "SimDataFormats/SLHC/interface/L1TowerJetFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"

#include <algorithm> 
#include <string> 
#include <vector>


class L1TowerFwdJetProducer:public L1CaloAlgoBase < l1slhc::L1CaloTowerCollection, l1slhc::L1TowerJetCollection > 
{
  public:
	L1TowerFwdJetProducer( const edm::ParameterSet & );
	 ~L1TowerFwdJetProducer(  );

//	void initialize(  );

	void algorithm( const int &, const int & );

  private:	
        void calculateFwdJetPosition( l1slhc::L1TowerJet & lJet );
	//some helpful members
	int mJetDiameter;
        l1slhc::L1TowerJet::tJetShape mJetShape;

        std::vector< std::pair< int , int > > mHFJetShapeMap;
};

L1TowerFwdJetProducer::L1TowerFwdJetProducer( const edm::ParameterSet & aConfig ):
L1CaloAlgoBase < l1slhc::L1CaloTowerCollection, l1slhc::L1TowerJetCollection > ( aConfig )

{
	mJetDiameter = aConfig.getParameter<unsigned>("JetDiameter");
	mPhiOffset = 0;
	//mEtaOffset = -mJetDiameter;
	mEtaOffset = 0;
	// mPhiIncrement = 1;
	// mEtaIncrement = 1;

	mHFJetShapeMap.reserve(256);

	std::string lJetShape = aConfig.getParameter< std::string >("JetShape");
	std::transform( lJetShape.begin() , lJetShape.end() , lJetShape.begin() , ::toupper ); //do the comparison in upper case so config

        std::cout<<" Creating HF jet map." <<std::endl;
 	//Create the HF jet shape map: a square 2x2 jet
	for( int x = 0 ; x != mJetDiameter/4 ; ++x ){
		for( int y = 0 ; y != mJetDiameter ; ++y ){
			mHFJetShapeMap.push_back( std::make_pair( x , y ) );
		}
	}


}

L1TowerFwdJetProducer::~L1TowerFwdJetProducer(  )
{
}

/*
void L1TowerFwdJetProducer::initialize(  )
{
}
*/

void L1TowerFwdJetProducer::algorithm( const int &aEta, const int &aPhi )
{

  int lTowerIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
  std::pair < int, int > lTowerEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lTowerIndex );

  l1slhc::L1TowerJet lJet( mJetDiameter, mJetShape , mHFJetShapeMap , lTowerEtaPhi.first , lTowerEtaPhi.second  );


  if(aEta>=60 || aEta<4){
    for ( std::vector< std::pair< int , int > >::const_iterator lHFJetShapeMapIt = mHFJetShapeMap.begin() ; lHFJetShapeMapIt != mHFJetShapeMap.end() ; ++lHFJetShapeMapIt )
    {
    int lPhi = aPhi+(lHFJetShapeMapIt->second);
    if ( lPhi > mCaloTriggerSetup->phiMax(  ) ) lPhi -= 72;
      l1slhc::L1CaloTowerCollection::const_iterator lTowerItr = fetch( aEta+(lHFJetShapeMapIt->first) , lPhi );
// 	std::cout<<"Fetching "<<aEta<<","<<lPhi<<" the tower "<< aEta+(lHFJetShapeMapIt->first) <<" , "<<lPhi<<" has energy "<<lTowerItr->E()<<std::endl;

      if ( lTowerItr != mInputCollection->end(  ) )
      {
        l1slhc::L1CaloTowerRef lRef( mInputCollection, lTowerItr - mInputCollection->begin(  ) );
        lJet.addConstituent( lRef );
      }
    }
  } 
  




  if ( lJet.E(  ) > 0 )
  {
    calculateFwdJetPosition( lJet );
    mOutputCollection->insert( lTowerEtaPhi.first, lTowerEtaPhi.second, lJet );
//     std::cout<<" jet "<< aEta<<" , "<< aPhi <<" has energy "<<lJet.E()<<std::endl;

  }

}



void L1TowerFwdJetProducer::calculateFwdJetPosition( l1slhc::L1TowerJet & lJet )
{

  double eta;
  //double halfTowerOffset = 0.0435;

  double JetSize = double(lJet.JetSize()) / 2.0;

  if(  abs( lJet.iEta() ) == 29 ) eta = 3.5;
  if(  abs( lJet.iEta() ) == 30 ) eta = 4.0;
  if(  abs( lJet.iEta() ) == 31 ) eta = 4.5;
  if(  abs( lJet.iEta() ) == 32 ) eta = 5.0;  

  if(lJet.iEta()<0) eta = -eta;

  double phi = ( ( lJet.iPhi(  ) + JetSize ) * 0.0873 );
  //Need this because 72*0.087 != 2pi: else get uneven phi dist
  phi -= 0.0873;
  double pi=(72*0.0873)/2;
  if(phi>pi) phi-=2*pi; 
  double Et = double( lJet.E(  ) ) / 2.;

  lJet.setP4( math::PtEtaPhiMLorentzVector( Et, eta, phi, 0. ) );

} 


DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<L1TowerFwdJetProducer>,"L1TowerFwdJetProducer");
DEFINE_FWK_PSET_DESC_FILLER(L1TowerFwdJetProducer);

