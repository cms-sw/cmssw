
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1TowerJet.h"
#include "SimDataFormats/SLHC/interface/L1TowerJetFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"

#include <algorithm> 
#include <string> 
#include <vector>
 

class L1TowerJetProducer:
public L1CaloAlgoBase < l1slhc::L1CaloTowerCollection, l1slhc::L1TowerJetCollection >
{
  public:
	L1TowerJetProducer( const edm::ParameterSet & );
	 ~L1TowerJetProducer(  );

	// void initialize( );

	void algorithm( const int &, const int & );

  private:
	void calculateJetPosition( l1slhc::L1TowerJet & lJet );
	//some helpful members
	int mJetDiameter;
	l1slhc::L1TowerJet::tJetShape mJetShape;

	std::vector< std::pair< int , int > > mJetShapeMap;

};

L1TowerJetProducer::L1TowerJetProducer( const edm::ParameterSet & aConfig ):L1CaloAlgoBase < l1slhc::L1CaloTowerCollection, l1slhc::L1TowerJetCollection > ( aConfig )
{
	mJetDiameter = aConfig.getParameter<unsigned>("JetDiameter");
	mPhiOffset = 0;

    //note: this mEtaOffset works when read in setupHF.xml in SLHCUpgradeSimulations/L1CaloTrigger/python/SLHCCaloTrigger_cfi.py
    //must be used with aEta>4 in algorithm() function below
//	mEtaOffset = -(mJetDiameter+4);

    //if use with setup.xml where don't read in HF wires use
    mEtaOffset = -(mJetDiameter);	
    //and don't need aEta>4 condition

    // mPhiIncrement = 1;
	// mEtaIncrement = 1;

	mJetShapeMap.reserve(256); //jets will never be 16x16 but it is a nice round number

	std::string lJetShape = aConfig.getParameter< std::string >("JetShape");
	std::transform( lJetShape.begin() , lJetShape.end() , lJetShape.begin() , ::toupper ); //do the comparison in upper case so config file can read "Circle", "circle", "CIRCLE", "cIrClE", etc. and give the same result.

	std::cout << "Creating JetShapeMap:" << std::endl;

	if ( lJetShape == "CIRCLE" ){
		mJetShape = l1slhc::L1TowerJet::circle;


		double lCentre( (mJetDiameter-1) / 2.0 );
		double lDelta;

		std::vector<double> lDeltaSquare;
		for( int i = 0 ; i != mJetDiameter ; ++i ){
			lDelta = double(i) - lCentre;
			lDeltaSquare.push_back( lDelta*lDelta );
		}

		double lDeltaRSquare;
		double lDeltaRSquareMax( (mJetDiameter*mJetDiameter) / 4.0 );

		for( int x = 0 ; x != mJetDiameter ; ++x ){
			for( int y = 0 ; y != mJetDiameter ; ++y ){
				lDeltaRSquare = lDeltaSquare[x] + lDeltaSquare[y];
				if( lDeltaRSquare <= lDeltaRSquareMax ){
					mJetShapeMap.push_back( std::make_pair( x , y ) );
					//std::cout << "#" << std::flush;
				}else{
					//std::cout << "-" << std::flush;
				}
			}
			//std::cout << std::endl;
		}

	}else{

		mJetShape = l1slhc::L1TowerJet::square;

		for( int x = 0 ; x != mJetDiameter ; ++x ){
			for( int y = 0 ; y != mJetDiameter ; ++y ){
				mJetShapeMap.push_back( std::make_pair( x , y ) );
			}
		}
	}

	std::cout << "JetShapeMap includes " << mJetShapeMap.size() << " towers." << std::endl;
    std::cout<<" Eta offset is "<< mEtaOffset << std::endl;
}

L1TowerJetProducer::~L1TowerJetProducer(  )
{
}

/* 
   void L1TowerJetProducer::initialize( ) { }
*/

void L1TowerJetProducer::algorithm( const int &aEta, const int &aPhi )
{
//  if(aEta>4){   
      
  int lTowerIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
  std::pair < int, int > lTowerEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lTowerIndex );
  
  l1slhc::L1TowerJet lJet( mJetDiameter, mJetShape , mJetShapeMap.size() , lTowerEtaPhi.first , lTowerEtaPhi.second  );

  
  for ( std::vector< std::pair< int , int > >::const_iterator lJetShapeMapIt = mJetShapeMap.begin() ; lJetShapeMapIt != mJetShapeMap.end() ; ++lJetShapeMapIt )
  {
    int lPhi = aPhi+(lJetShapeMapIt->second);
    if ( lPhi > mCaloTriggerSetup->phiMax(  ) ) lPhi -= 72; //mCaloTriggerSetup->phiMax(  );
  
    l1slhc::L1CaloTowerCollection::const_iterator lTowerItr = fetch( aEta+(lJetShapeMapIt->first) , lPhi );

    if ( lTowerItr != mInputCollection->end(  ) )
    {
      l1slhc::L1CaloTowerRef lRef( mInputCollection, lTowerItr - mInputCollection->begin(  ) );
      lJet.addConstituent( lRef );

	lJet.CalcWeightediEta();
    lJet.CalcWeightediPhi();
    }
  }
  
  if ( lJet.E(  ) > 0 )
  {
    calculateJetPosition( lJet );
    
    lJet.calculateWeightedEta();lJet.calculateWeightedPhi();
    mOutputCollection->insert( lTowerEtaPhi.first, lTowerEtaPhi.second, lJet );

  }
//}
}



void L1TowerJetProducer::calculateJetPosition( l1slhc::L1TowerJet & lJet )
{

  double tmpeta(9999);
  double halfTowerOffset = 0.0435;
  const double endcapEta[8] = { 0.09, 0.1, 0.113, 0.129, 0.15, 0.178, 0.15, 0.35 };

  double JetSize = double(lJet.JetSize()) / 2.0;
  //std::cout<<" jet ieta: "<< lJet.iEta(  );
  int abs_eta = lJet.iEta(  ) + int(JetSize) ;
  //std::cout<<" centre of jet: "<<abs_eta ;
  //account for the fact there is no zero create a zero
  if( abs_eta>=0 && lJet.iEta(  )<0 ) abs_eta += 1;
 
  //std::cout<<" account for no zero: "<<abs_eta<<std::endl;
  abs_eta = fabs( abs_eta);//fabs( lJet.iEta(  ) + int(JetSize) );


  if ( abs_eta < 21 )
  {
    tmpeta = ( abs_eta * 0.0870 - halfTowerOffset);
    
    if( lJet.JetSize() % 2 == 1 )      tmpeta += halfTowerOffset;

  }
  else
  {
    abs_eta -= 21;

    tmpeta = 1.74;

    for ( int i = 0; i != int(abs_eta); ++i )
    {
      tmpeta += endcapEta[i];
    }


    if( lJet.JetSize() % 2 == 0 )   tmpeta += endcapEta[abs_eta] / 2.; 
    else      tmpeta += endcapEta[abs_eta];

  }

  if(( lJet.iEta(  ) + int(JetSize)  )<0) tmpeta = (-1)*tmpeta;
  //if (lJet.iEta()>0) tmpeta-=0.087;

  // std::cout<<"jet ieta: "<<lJet.iEta()<<" jet centre "<< abs_eta <<" to eta "<<tmpeta<<std::endl;

  double phi = ( ( lJet.iPhi(  ) + JetSize ) * 0.087 );
  //Need this because 72*0.087 != 2pi: else get uneven phi dist
  phi -= 0.087;
  double pi=(72*0.087)/2;
  if(phi>pi) phi-=2*pi; 
  
  double Et = double( lJet.E(  ) ) / 2.;

  lJet.setP4( math::PtEtaPhiMLorentzVector( Et, tmpeta, phi, 0. ) );

} 



DEFINE_EDM_PLUGIN( edm::MakerPluginFactory, edm::WorkerMaker < L1TowerJetProducer >, "L1TowerJetProducer" );
DEFINE_FWK_PSET_DESC_FILLER( L1TowerJetProducer );
