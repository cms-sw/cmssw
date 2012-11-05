#include "SimDataFormats/SLHC/interface/L1TowerJet.h"
#include <stdlib.h>

namespace l1slhc
{
  
  L1TowerJet::L1TowerJet( ):
  mIeta( 0 ), 
  mIphi( 0 ), 

  mE( 0 ), 
  mCentral( true ),
  mAsymEta(0),
  mAsymPhi(0),
  mWeightedIeta( 0 ),
  mWeightedIphi( 0 ),
  mJetSize( 12 ),
  mJetShapeType( square ),
  mJetArea( 144 )
  {
  }
  
  L1TowerJet::L1TowerJet( const int& aJetSize, const L1TowerJet::tJetShape& aJetShapeType , const int& aJetArea ):
  mIeta( 0 ), 
  mIphi( 0 ), 

  mE( 0 ), 
  mCentral( true ),
  mAsymEta(0),
  mAsymPhi(0),
  mWeightedIeta( 0 ),
  mWeightedIphi( 0 ),

  mJetSize( aJetSize ),
  mJetShapeType( aJetShapeType ),
  mJetArea( aJetArea )
  {
  }
  
  L1TowerJet::L1TowerJet( const int& aJetSize, const L1TowerJet::tJetShape& aJetShapeType , const int& aJetArea , const int &iEta, const int &iPhi  ):
  mIeta( iEta ), 
  mIphi( iPhi ), 

  mE( 0 ), 
  mCentral( true ),
  mAsymEta(0),
  mAsymPhi(0),

  mWeightedIeta( 0 ),
  mWeightedIphi( 0 ),

  mJetSize( aJetSize ),
  mJetShapeType( aJetShapeType ),
  mJetArea( aJetArea )
  {
  }
  
  L1TowerJet::~L1TowerJet(  )
  {
  }
  
  
  
  
  void L1TowerJet::setP4( const math::PtEtaPhiMLorentzVector & p4 )
  {
          mP4 = p4;
  }
  
  void L1TowerJet::setCentral( const bool & central )
  {
          mCentral = central;
  }
/*
  void L1TowerJet::setE( const int &E )
  {
          mE = E;
  }
*/


  const int &L1TowerJet::iEta(  ) const
  {
          return mIeta;
  }

  const int &L1TowerJet::iPhi(  ) const
  {
          return mIphi;
  }
  

  

   const double &L1TowerJet::iWeightedEta( ) const 
  {

    return mWeightedIeta;

  }

   const double &L1TowerJet::iWeightedPhi( ) const 
  {

    return mWeightedIphi;

  }

   const double &L1TowerJet::WeightedEta( ) const 
  {

    return mWeightedEta;

  }

   const double &L1TowerJet::WeightedPhi( ) const 
  {

    return mWeightedPhi;

  }




  const int &L1TowerJet::E(  ) const
  {
          return mE;
  }
  
  const int& L1TowerJet::AsymEta(  ) const
  {
          return mAsymEta;
  }
  
  const int& L1TowerJet::AsymPhi(  ) const
  {
          return mAsymPhi;
  }


  const bool & L1TowerJet::central(  ) const
  {
          return mCentral;
  }

  const math::PtEtaPhiMLorentzVector & L1TowerJet::p4(  ) const
  {
          return mP4;
  }
  
  const int& L1TowerJet::JetSize(  ) const
  {
          return mJetSize;
  }
  
  const L1TowerJet::tJetShape& L1TowerJet::JetShape(  ) const
  {
          return mJetShapeType;
}


/*
  double L1TowerJet::EcalVariance(  ) const
  {
    double lMean(0.0);
    double lMeanSq(0.0);
  
    for ( L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ){
      lMean += (**lConstituentIt).E();
      lMeanSq += ((**lConstituentIt).E() * (**lConstituentIt).E());
    }
  
    lMean /= mConstituents.size();	
    lMeanSq /= mConstituents.size();	
  
    return lMeanSq - (lMean*lMean);
  }
  
  
  double L1TowerJet::HcalVariance(  ) const
  {
    double lMean(0.0);
    double lMeanSq(0.0);
    
    for ( L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ){
      lMean += (**lConstituentIt).H();
      lMeanSq += ((**lConstituentIt).H() * (**lConstituentIt).H());
    }
    
    lMean /= mConstituents.size();	
    lMeanSq /= mConstituents.size();	
    
    return lMeanSq - (lMean*lMean);
  }


  double L1TowerJet::EnergyVariance(  ) const
  {
    double lMean( double(mE) / double(mConstituents.size()) );
    double lMeanSq(0.0);
    
    double lTower;
    for ( L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ){
      lTower = (**lConstituentIt).E() + (**lConstituentIt).H();
      lMeanSq += ( lTower * lTower );
    }
    
    lMeanSq /= mConstituents.size();	
  
    return lMeanSq - (lMean*lMean);
  }
*/


  double L1TowerJet::EcalMAD() const
  {
    std::deque< int > lEnergy;
    for ( L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ){
      lEnergy.push_back( (**lConstituentIt).E() );
    }
    lEnergy.resize( mJetArea , 0 );
    return MAD( lEnergy );
  
  }
  
  double L1TowerJet::HcalMAD() const
  {
    std::deque< int > lEnergy;
    for ( L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ){
      lEnergy.push_back( (**lConstituentIt).H() );
    }
    lEnergy.resize( mJetArea , 0 );
    return MAD( lEnergy );
  
  }
  

  void L1TowerJet::CalcWeightediEta() 
  {
    int etaSumEt(0); 
    int sumEt (0);

    for ( L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ){
      etaSumEt += ( (**lConstituentIt).E() + (**lConstituentIt).H() ) * ( (**lConstituentIt).iEta() );
      sumEt += ( (**lConstituentIt).E() + (**lConstituentIt).H() ) ;
    }
//	std::cout<<" eta* energy = "<<etaSumEt<<" sum energy: "<<sumEt<<std::endl;
    mWeightedIeta = etaSumEt/sumEt ; 
//	std::cout<<"weighted eta "<<mWeightedIeta<<std::endl;
  }


  void L1TowerJet::calculateWeightedEta()
  {
    double halfTowerOffset = 0.0435;
    double WeightedEta(9999);
    double JetSize = double(mJetSize) / 2.0;
  //   std::cout<<" input ieta "<<mWeightedIeta<<std::endl;
    int abs_eta =mWeightedIeta+ int(JetSize);
//	std::cout<<abs_eta<<" absolute eta "<< std::endl;
   if ( abs_eta < 21 )
  {

    WeightedEta = ( abs_eta * 0.0870 );
//	std::cout<<" barrel: weigted eta "<<WeightedEta<<std::endl;
    if( mJetSize % 2 == 1 ){
     WeightedEta += halfTowerOffset;
    }

  }
  else
  {
    const double endcapEta[8] = { 0.09, 0.1, 0.113, 0.129, 0.15, 0.178, 0.15, 0.35 };
    abs_eta -= 21;

    WeightedEta = 1.74;

    for ( int i = 0; i != abs_eta; ++i )
    {
      WeightedEta += endcapEta[i];
    }

    if( mJetSize % 2 == 0 ){
      WeightedEta += endcapEta[abs_eta] / 2.;
    }else{
      WeightedEta += endcapEta[abs_eta];
    }
//std::cout<<"endcap eta "<<WeightedEta<<std::endl;
  }


//std::cout<<"weighted eta "<<WeightedEta<<std::endl;

 if(mWeightedIeta>0) WeightedEta-=0.087;

  mWeightedEta = WeightedEta;

  //std::cout<<" iweighted eta -> weighted eta "<<mWeightedEta<<std::endl;
  }





  void L1TowerJet::CalcWeightediPhi() 
  {
    int phiSumEt(0); 
    int sumEt (0);

    for ( L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ){
    //  std::cout<<"iPhi: "<<(**lConstituentIt).iPhi() <<" energy: "<<(**lConstituentIt).E() + (**lConstituentIt).H() <<std::endl;
      phiSumEt += ( (**lConstituentIt).E() + (**lConstituentIt).H() ) * ( (**lConstituentIt).iPhi() );
      sumEt += ( (**lConstituentIt).E() + (**lConstituentIt).H() ) ;
    }
//	std::cout<<"phi sum et: "<<phiSumEt<<"sum Et: "<<sumEt<<std::endl;
    mWeightedIphi = phiSumEt/sumEt ; 
    //std::cout<<" weighted phi: "<<mWeightedIphi<<std::endl;
  }

void L1TowerJet::calculateWeightedPhi( )
{
  double JetSize = double(mJetSize) / 2.0;
  double WeightedPhi = ( ( mWeightedIphi + JetSize ) * 0.087 );
  //Need this because 72*0.087 != 2pi: else get uneven phi dist
  WeightedPhi -= 0.087;
  double pi=(72*0.087)/2;
  if(WeightedPhi > pi) WeightedPhi -=2*pi;
  mWeightedPhi=WeightedPhi;

}






  double L1TowerJet::EnergyMAD() const
  {
    std::deque< int > lEnergy;
    for ( L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ){
      lEnergy.push_back( (**lConstituentIt).E() + (**lConstituentIt).H() );
    }
    lEnergy.resize( mJetArea , 0 );
    return MAD( lEnergy );
  }



  double L1TowerJet::MAD( std::deque<int>& aDataSet ) const
  {
    std::sort( aDataSet.begin() , aDataSet.end() );
    
    std::size_t lDataSetSize( aDataSet.size() );
    
    double lMedian(0);
    if( lDataSetSize%2 == 0 ){
      lMedian = double ( aDataSet[ (lDataSetSize/2) - 1 ] + aDataSet[ lDataSetSize/2 ] ) / 2.0 ;
    }else{
      lMedian = double( aDataSet[ (lDataSetSize-1)/2 ] );
    }
    
    
    std::deque< double > lMedianSubtractedDataSet;
    for ( std::deque< int >::const_iterator lIt = aDataSet.begin() ; lIt != aDataSet.end(); ++lIt ){
      lMedianSubtractedDataSet.push_back( fabs( double(*lIt) - lMedian ) );
    }
    
    std::sort( lMedianSubtractedDataSet.begin() , lMedianSubtractedDataSet.end() );
    
    if( lDataSetSize%2 == 0 ){
      return double ( lMedianSubtractedDataSet[ (lDataSetSize/2) - 1 ] + lMedianSubtractedDataSet[ lDataSetSize/2 ] ) / 2.0 ;
    }else{
      return double( lMedianSubtractedDataSet[ (lDataSetSize-1)/2 ] );
    }
  
  }


  void L1TowerJet::addConstituent( const L1CaloTowerRef & Tower )
  { 
  
    int lHalfJetSize( mJetSize >> 1 );
    int lTowerEnergy( Tower->E(  ) + Tower->H(  ) );
  
    mE += lTowerEnergy;
    mConstituents.push_back( Tower );
  
     //std::cout<<"JET: ("<<iEta()<<","<<iPhi()<<std::endl;
     //std::cout<<"TOWER  = ( " <<Tower->iEta()<< " , " <<Tower->iPhi()<<" ), energy = " <<Tower->E()+Tower->H()<<std::endl;
  
    //when add a tower, also add the asymmetry terms
    //positive asym: top RH corner of jet
  
    if( mJetSize % 2 == 0 ){ //even jet size
    
    //if(Tower->iEta(  ) == iEta() ) mAsymEta += 0; //do nothing 
      if( Tower->iEta(  ) >= (iEta() + lHalfJetSize) ) {
        mAsymEta +=  lTowerEnergy;
      }		
      else{ /*if( Tower->iEta(  ) <  iEta() + lHalfJetSize )*/ 
        mAsymEta -= lTowerEnergy;
  
      }
  
    //if(  Tower->iPhi(  ) == iPhi() ) mAsymEta += 0; //do nothing
      if( Tower->iPhi(  ) < (iPhi() + lHalfJetSize) ){
        mAsymPhi += lTowerEnergy;
  
      }
      else{ /*if( Tower->iPhi(  ) > iPhi() + lHalfJetSize )*/  
        mAsymPhi -= lTowerEnergy;
  
      }
  
    }else{ //odd jet size: miss out central towers
      
      if( Tower->iEta(  ) ==  (lHalfJetSize + iEta()) ) {
        mAsymEta += 0; //do nothing
      }
      else if( Tower->iEta(  ) > (iEta() + lHalfJetSize) ) {
        mAsymEta +=  lTowerEnergy;
  
      }else /*if( Tower->iEta(  ) <  iEta() + lHalfJetSize )*/ {
        mAsymEta -= lTowerEnergy;
  
      }
    // else it is in the middle so does not contribute to the asymmetry
  
      if( Tower->iPhi(  ) == (lHalfJetSize + iPhi()) ) {
        mAsymEta += 0; //do nothing
  
      }
      else if( Tower->iPhi(  ) < (iPhi() + lHalfJetSize) ) {
        mAsymPhi += lTowerEnergy;
  
      }else /*if( Tower->iPhi(  ) > iPhi() + lHalfJetSize )*/  {
        mAsymPhi -= lTowerEnergy;
  
      }
  // else it is in the middle so does not contribute to the asymmetry

    }
  } 


  void L1TowerJet::removeConstituent( const int &eta, const int &phi )
  {
    L1CaloTowerRefVector::iterator lConstituent = getConstituent( eta, phi );
    if ( lConstituent != mConstituents.end() )
    {
      int lHalfJetSize( mJetSize >> 1 );
      int lTowerEnergy( (**lConstituent).E(  ) + (**lConstituent).H(  ) );
  
      mE -= lTowerEnergy;
      mConstituents.erase( lConstituent );
  
      
      
      if( mJetSize % 2 == 0 ){ //even jet size
  
        if( eta >= (iEta() + lHalfJetSize) ) {
          mAsymEta -=  lTowerEnergy;
        }		
        else{ /*if( Tower->iEta(  ) <  iEta() + lHalfJetSize )*/ 
          mAsymEta += lTowerEnergy;
        }
  
  
        if( phi < (iPhi() + lHalfJetSize) ){
          mAsymPhi -= lTowerEnergy;
  
        }else{ /*if( Tower->iPhi(  ) > iPhi() + lHalfJetSize )*/  
          mAsymPhi += lTowerEnergy;
        }
  
  
      }else{ //odd jet size: miss out central towers
    
        if( eta ==  (lHalfJetSize + iEta()) ) {
          mAsymEta += 0; //do nothing
        }
        else if( eta > (iEta() + lHalfJetSize) ) {
          mAsymEta -=  lTowerEnergy;
  
        }else /*if( Tower->iEta(  ) <  iEta() + lHalfJetSize )*/ {
          mAsymEta += lTowerEnergy;
  
        }
  
        if( phi == (lHalfJetSize + iPhi()) ) {
          mAsymEta -= 0; //do nothing
  
        }
        else if( phi < (iPhi() + lHalfJetSize) ) {
          mAsymPhi -= lTowerEnergy;
  
        }else /*if( Tower->iPhi(  ) > iPhi() + lHalfJetSize )*/  {
          mAsymPhi += lTowerEnergy;
  
        }
      }
    }
  }


  const L1CaloTowerRefVector & L1TowerJet::getConstituents(  ) const
  {
    return mConstituents;
  }
  
  
  L1CaloTowerRefVector::iterator L1TowerJet::getConstituent( const int &eta, const int &phi )
  {
    for ( L1CaloTowerRefVector::iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt )
      if ( (**lConstituentIt).iEta(  ) == eta + mIeta && (**lConstituentIt).iPhi(  ) == phi + mIphi )
        return lConstituentIt;
  
    return mConstituents.end();
  }

}




namespace std
{
	bool operator<( const l1slhc::L1TowerJet & aLeft, const l1slhc::L1TowerJet & aRight )
	{
		if ( aLeft.E(  ) == aRight.E(  ) )
		{
			// for two objects with equal energy, favour the more central one
			return ( abs( aLeft.iEta(  ) ) > abs( aRight.iEta(  ) ) );
		}
		else
		{
			return ( aLeft.E(  ) < aRight.E(  ) );
		}
	}
}


// pretty print
std::ostream & operator<<( std::ostream & aStream, const l1slhc::L1TowerJet & aL1TowerJet )
{
	aStream << "L1TowerJet" 
		<< " iEta=" << aL1TowerJet.iEta(  ) 
		<< " iPhi=" << aL1TowerJet.iPhi(  ) 
		<< "\n with constituents:\n";
	for ( l1slhc::L1CaloTowerRefVector::const_iterator i = aL1TowerJet.getConstituents(  ).begin(  ); i < aL1TowerJet.getConstituents(  ).end(  ); ++i )
		aStream << "  iEta=" << ( **i ).iEta(  ) 
			<< " iPhi=" << ( **i ).iPhi(  ) 
			<< " ET=" << ( **i ).E(  ) 
			<< "\n";
	return aStream;
}
