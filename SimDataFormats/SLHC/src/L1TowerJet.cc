#include "SimDataFormats/SLHC/interface/L1TowerJet.h"
#include <stdlib.h>


namespace l1slhc
{

  // Initialising the static tower geometry member variable
  TriggerTowerGeometry L1TowerJet::mTowerGeo = TriggerTowerGeometry();
  
  const double L1TowerJet::PI = 3.14159265359;

  
  L1TowerJet::L1TowerJet( ):
    mIeta( 0 ), 
    mIphi( 0 ), 

    mE2GeV( 0 ), 
    mE( 0 ), 
    mCentral( true ),
    mAsymEta(0),
    mAsymPhi(0),
    /*    mWeightedIeta( 0 ),
	  mWeightedIphi( 0 ),*/
    mJetCenterEta( 0 ),
    mJetCenterPhi( 0 ),

    mJetSize( 12 ),
    mJetShapeType( square ),
    mJetArea( 144 ),
    mJetRealArea( mJetArea * 0.087 * 0.087 )
  {
  }
  
  L1TowerJet::L1TowerJet( const int& aJetSize, const L1TowerJet::tJetShape& aJetShapeType , const int& aJetArea ):

    mIeta( 0 ), 
    mIphi( 0 ), 

    mE2GeV( 0 ), 
    mE( 0 ), 
    mCentral( true ),
    mAsymEta(0),
    mAsymPhi(0),
    /*    mWeightedIeta( 0 ),
	  mWeightedIphi( 0 ),*/
    mJetCenterEta( 0 ),
    mJetCenterPhi( 0 ),

    mJetSize( aJetSize ),
    mJetShapeType( aJetShapeType ),
    mJetArea( aJetArea ),
    mJetRealArea( mJetArea * 0.087 * 0.087 ) 
  { // The above mJetRealArea calculation assumes the towers in the jet all occupy the region |eta| < 1.74
  }
  

  L1TowerJet::L1TowerJet( const int& aJetSize, const L1TowerJet::tJetShape& aJetShapeType , 
			  const std::vector< std::pair< int, int > >& aJetShapeMap, const int &iEta, const int &iPhi  ):

    mIeta( iEta ), 
    mIphi( iPhi ), 

    mE2GeV( 0 ), 
    mE( 0 ), 
    mCentral( true ),
    mAsymEta(0),
    mAsymPhi(0),
    /*    mWeightedIeta( 0 ),
	  mWeightedIphi( 0 ),*/
    mJetCenterEta( 0 ),
    mJetCenterPhi( 0 ),

    mJetSize( aJetSize ),
    mJetShapeType( aJetShapeType ),
    mJetArea( aJetShapeMap.size() ),
    mJetRealArea( 0 )
  {
    
    // **************************************************
    // *           Calculate the real jet area          *
    // **************************************************

     // Check if jet fully contained within |eta| < 1.74, in which case all TTs have the same area
    if ( (iEta + mJetSize < 22) && (iEta > -21) ){

      // real jet area = Number of TTs * deltaPhi * deltaEta
      mJetRealArea = mJetArea * 0.087 * 0.087;
    }
    else{

      // calculate the real jet area, accounting for the change in trigger tower eta width
      for ( std::vector< std::pair< int , int > >::const_iterator lJetShapeMapIt = aJetShapeMap.begin() ;
	    lJetShapeMapIt != aJetShapeMap.end() ; ++lJetShapeMapIt ){

	// Get the iEta of the TT
	int TTiEta    = iEta + lJetShapeMapIt->first;
	double TTarea = mTowerGeo.towerEtaSize( TTiEta ) * 0.087;
	mJetRealArea += TTarea; 


	// Calculate the geometric center of the jet
	//calculateJetCenter();
      
      }

    }

  }
  
  L1TowerJet::~L1TowerJet(  )
  {
  }
  


  // ********************************************************************************
  // *                                   Setters                                    *
  // ********************************************************************************
  
  void L1TowerJet::setP4( const math::PtEtaPhiMLorentzVector & p4 )
  {
    mP4 = p4;
  }
  
  void L1TowerJet::setPt( const double & pT )
  {
    mP4.SetPt(pT);
  }
  

  void L1TowerJet::setCentral( const bool & central )
  {
    mCentral = central;
  }
  

  // ********************************************************************************
  // *                                   Getters                                    *
  // ********************************************************************************

  // iEta, eta of the top-left TT indexing the tower jet
  const int &L1TowerJet::iEta(  ) const
  {
    return mIeta;
  }

  // iPhi, phi of the top-left TT indexing the tower jet
  const int &L1TowerJet::iPhi(  ) const
  {
    return mIphi;
  }

  // Return Pt of jet
  const double L1TowerJet::Pt( ) const
  {
    return  mP4.Pt();
  }

  // Eta of tower jet geometric center
  const double L1TowerJet::Eta( ) const
  {
    return mJetCenterEta;
  }

  // Phi of tower jet geometric center
  const double L1TowerJet::Phi( ) const
  {
    return mJetCenterPhi;
  }

  // Energy weighted eta
  const double L1TowerJet::WeightedEta( ) const 
  {
    // The weighted eta information is stored within the four-vector
    return mP4.Eta();
    //    return mWeightedEta;
  }

  // Energy weighted phi 
  const double L1TowerJet::WeightedPhi( ) const 
  {
    // The weighted phi information is stored within the four-vector
    return mP4.Phi();
    // return mWeightedPhi;
  }

  // Total TT energy enclosed by tower jet in GeV (Corrected to account for energy being stored in multiples of 2 GeV)
  const double &L1TowerJet::E(  ) const
  { // True value in GeV
    return mE;
  }
  
  const double &L1TowerJet::Centrality(  ) const
  {
    return mCentrality;
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

  // The jet diameter in TTs
  const int& L1TowerJet::JetSize(  ) const
  {
    return mJetSize;
  }
  
  const L1TowerJet::tJetShape& L1TowerJet::JetShape(  ) const
  {
    return mJetShapeType;
  }

  // The area of the jet in TTs
  const int& L1TowerJet::JetArea(  ) const
  {
    return mJetArea;
  }

  // The true eta*phi area of the jet
  const double& L1TowerJet::JetRealArea( ) const
  {
    return mJetRealArea;
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


  // ********************************************************************************
  // *                           Median absolute deviations                         *
  // ********************************************************************************

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
    if( (lDataSetSize % 2) == 0 ){
      lMedian = double( aDataSet[ (lDataSetSize/2) - 1 ] + aDataSet[ lDataSetSize/2 ] ) / 2.0 ;
    }else{
      lMedian = double( aDataSet[ (lDataSetSize-1)/2 ] );
    }
    
    
    std::deque< double > lMedianSubtractedDataSet;
    for ( std::deque< int >::const_iterator lIt = aDataSet.begin() ; lIt != aDataSet.end(); ++lIt ){
      lMedianSubtractedDataSet.push_back( fabs( double(*lIt) - lMedian ) );
    }
    
    std::sort( lMedianSubtractedDataSet.begin() , lMedianSubtractedDataSet.end() );
    
    if( (lDataSetSize % 2) == 0 ){
      return double ( lMedianSubtractedDataSet[ (lDataSetSize/2) - 1 ] + lMedianSubtractedDataSet[ lDataSetSize/2 ] ) / 2.0 ;
    }else{
      return double( lMedianSubtractedDataSet[ (lDataSetSize-1)/2 ] );
    }
  
  }




  // ********************************************************************************
  // *                           Jet eta and phi variables                          *
  // ********************************************************************************






  // **********************************************************************
  // *                     Jet iEta and iPhi variables                    *
  // **********************************************************************



  // ~ Why is this not following the getter naming convention? ~ 
  // ~ Weighted iEta assumes all trigger towers have the same width, the result will be different from ~
  // ~ discretising the weighted Eta.                                                                   ~
  /*
  void L1TowerJet::CalcWeightediEta() 
  {
    double etaSumEt(0); 
    double sumEt (0);

    for ( L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ){
      etaSumEt += ( (**lConstituentIt).E() + (**lConstituentIt).H() ) * ( (**lConstituentIt).iEta() );
      sumEt += ( (**lConstituentIt).E() + (**lConstituentIt).H() ) ;
    }
    //	std::cout<<" eta* energy = "<<etaSumEt<<" sum energy: "<<sumEt<<std::endl;
    
    //discretize weighted Ieta: run from |1-28| (no HF)
    mWeightedIeta = etaSumEt/sumEt ; 

    //discrete-ize and account for the fact there is no zero
    int discrete_iEta(999);
    //add 0.5 so that we get rounding up as well as down

    if( mWeightedIeta>=0 ) discrete_iEta=int(mWeightedIeta+0.5);
    else  discrete_iEta=int(mWeightedIeta-0.5); 
   
    //account for the fact there is no 0
    if(mWeightedIeta>=0 && mWeightedIeta<1)      discrete_iEta=1;
  
    if(mWeightedIeta<0 && mWeightedIeta>(-1))    discrete_iEta=-1;

    //std::cout<<"weighted ieta: "<<mWeightedIeta <<" discrete ieta: "<< discrete_iEta<<std::endl;
    mWeightedIeta = double(discrete_iEta);

  }
  */
  /*
  void L1TowerJet::CalcWeightediPhi() 
  {
    double phiSumEt(0); 
    double sumEt (0);

    for ( L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ){
      double tower_iPhi =(**lConstituentIt).iPhi(); 
      if( iPhi() >= (72-JetSize()) ) { //constituents may go over edge, iPhi>66 for 8x8 jet
        if( tower_iPhi  < (72 - JetSize()) ){//if constituent tower is over edge, ie. iPhi>1 
	  tower_iPhi+=72;  //iPhi=1 -> iPhi=73
	}
      }
      //calculate weighted phi using corrected iPhi value 
      phiSumEt += ( (**lConstituentIt).E() + (**lConstituentIt).H() ) * (tower_iPhi  );

      sumEt += ( (**lConstituentIt).E() + (**lConstituentIt).H() ) ;
    }
    //	std::cout<<"phi sum et: "<<phiSumEt<<"sum Et: "<<sumEt<<std::endl;
    mWeightedIphi = phiSumEt/sumEt ; 

    // take weighted Iphi back to 1-72 range
    if(mWeightedIphi>72) mWeightedIphi-=72;


  }
  */




  // **********************************************************************
  // *                      Jet eta and phi variables                     *
  // **********************************************************************


  // Calculate the eta and phi of the  geometric center of the jet
  void L1TowerJet::calculateJetCenter(){

    // determine the lowest and highest TTs enclosed by the jet shape (assumes square bounding box)
    int lowIeta  = mIeta;
    int lowIphi  = mIphi;
    int highIeta = lowIeta + (mJetSize - 1);
    int highIphi = lowIphi + (mJetSize - 1);

    // Extract the true eta of these TTs
    double lowEta  = mTowerGeo.eta(lowIeta);
    double lowPhi  = mTowerGeo.phi(lowIphi);
    double highEta = mTowerGeo.eta(highIeta);
    double highPhi = mTowerGeo.phi(highIphi);

    // Correct for phi wrap around, if jet map exceeds phi calorimeter range
    if ( (lowIphi + (mJetSize - 1) ) > 72 ){

      // Current tower has wrapped around
      if ( highIphi < lowIphi ){
	highPhi += 2*PI;
      }

    }

    // Determine the geometric jet center
    mJetCenterEta = (highEta + lowEta)/2.0;
    mJetCenterPhi = (highPhi + lowPhi)/2.0;

    // Constrain jets to the range [-Pi,Pi]
    if (mJetCenterPhi > PI)
      mJetCenterPhi -= 2*PI;
    

  }



  // Calculate the energy weighted eta and phi center of the jet. 
  //     Defined:  Sum ( TT_Eta * TT_Et ) / Sum ( TT_Et ), etc
  void L1TowerJet::calculateWeightedJetCenter()
  {

    // Eta, phi and Et of the TT
    double ttEta(0), ttPhi(0), ttEt(0);
    // Sums of eta*Et phi*Et and Et
    double etaEtSum(0), phiEtSum(0), etSum(0);
    
    // Iterate through the TTs in the jet map
    for (L1CaloTowerRefVector::const_iterator lTT = mConstituents.begin() ; lTT != mConstituents.end(); ++lTT ) {

      // Extract the eta, phi and Et of the TT
      ttEta = mTowerGeo.eta( (**lTT).iEta() );
      ttPhi = mTowerGeo.phi( (**lTT).iPhi() );
      ttEt  = 0.5*((**lTT).E() + (**lTT).H());

      
      // Correct for phi wrap around, if jet map exceeds phi calorimeter range
      if ( (mIphi + (mJetSize - 1) ) > 72 ){

	// Current tower has wrapped around
	if ( (**lTT).iPhi() < mIphi ){
	    ttPhi += 2*PI;
	}

      }

      // Calculate the weighted eta, weighted phi and Et sums
      etaEtSum += ttEta*ttEt;
      phiEtSum += ttPhi*ttEt;
      etSum    += ttEt;

    }
    
    // Calculate the weighted eta and phi
    
    double lWeightedEta = etaEtSum/etSum;
    double lWeightedPhi = phiEtSum/etSum;

    // Restrict phi to [-pi,pi] range
    if(lWeightedPhi > PI){
      lWeightedPhi -= 2*PI;
    }

    // Store the weighted eta and phi
    mP4.SetEta(lWeightedEta);
    mP4.SetPhi(lWeightedPhi);


  }


    



  // Calculate the energy weighted eta and phi center of the jet. 
  //     Defined:  Sum ( TT_Eta * TT_Et ) / Sum ( TT_Et ), etc
  void L1TowerJet::calculateCentrality()
  {



    // ********************************************************************
    // *                Calculate the centrality parameter                *
    // ********************************************************************
    //
    // Measure of the deltaR between the jet window centre and the centre of energy 
    // of the constituent energy deposits.
    //
    // Definition:
    // ~~~~~~~~~~
    // 

    // Eta, phi and Et of the TT and delta eta, phi and R between the jet centre and the constituent energy deposit
    double ttEta(0), ttPhi(0), ttEt(0), deltaEta(0), deltaPhi(0), deltaR(0);
    // Sums of deltaR*Et and Et
    double deltaREtSum(0), etSum(0);
    // Jet mask center (eta, phi)
    double jetCenterPhi = Phi();
    double jetCenterEta = Eta();
    int   jetCenteriPhi = (mIphi + mJetSize/2);

    // Correct for the jet mask center wrap around
    if (jetCenteriPhi > 72){
      jetCenterPhi += 2*PI;
    }


    // Iterate through the TTs in the jet map
    for (L1CaloTowerRefVector::const_iterator lTT = mConstituents.begin() ; lTT != mConstituents.end(); ++lTT ) {

      // Extract the eta, phi and Et of the TT
      ttEta = mTowerGeo.eta( (**lTT).iEta() );
      ttPhi = mTowerGeo.phi( (**lTT).iPhi() );
      ttEt  = 0.5*((**lTT).E() + (**lTT).H());
      
      // Correct for phi wrap around, if jet map exceeds phi calorimeter range
      if ( (mIphi + (mJetSize - 1) ) > 72 ){

	// Current tower has wrapped around
	if ( (**lTT).iPhi() < mIphi ){
	  ttPhi += 2*PI;
	}

      }

      // Unwrap the [-Pi,Pi] range
      if (jetCenterPhi < 0)
	jetCenterPhi += 2*PI;

    
      // Calculate deltaR between energy deposit and jet centre
      deltaEta = jetCenterEta - ttEta;
      deltaPhi = jetCenterPhi - ttPhi;
      deltaR   = sqrt( deltaEta*deltaEta + deltaPhi*deltaPhi );

      // Calculate the weighted deltaR*Et and Et sums
      deltaREtSum += deltaR*ttEt;
      etSum       += ttEt;

      /*      
      // DEBUGGING Eta = 0, Phi = 0
	std::cout << "----------------------------------------------------------\n" 
		  << "JET : iEta = " << mIeta + mJetSize/2 << "\tiPhi = " << jetCenteriPhi << "\tEta = " << jetCenterEta << "\tPhi = " << jetCenterPhi << "\n"
		  << "TT  : iEta = " << (**lTT).iEta()     << "\tiPhi = " << (**lTT).iPhi()<< "\tEta = " << ttEta        << "\tPhi = " << ttPhi        << "\tE = " 
		  << ttEt << "\n"
		  << "DeltaEta = " << deltaEta << "\tDeltaPhi = " << deltaPhi << "\tRi = " << sqrt(deltaEta*deltaEta + deltaPhi*deltaPhi)<< "\n";
      */
      
    }

    
    
    // Calculate the centrality of the jet energy deposits
    mCentrality =  deltaREtSum/etSum;
    //    std::cout << "Centrality = " << mCentrality << "\n==========================================================\n\n";

  }




  //
  // Below code is incorrrect. Should keep the calculation in eta and phi space and then transform to
  // i-eta and i-phi space to avoid discretisation issues and changing eta widths of TTs.

  // --------------------------------------------------
  // modified version to calculate weighted eta by first converting iEta to eta, than weighting
  /*
  void L1TowerJet::calculateWeightedEta()
  {
    
    const double endcapEta[8] = { 0.09, 0.1, 0.113, 0.129, 0.15, 0.178, 0.15, 0.35 };
    
    double etaSumEt(0); 
    double sumEt(0);

    double tmpEta(9999);
    double abs_eta(9999);

    for (L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ) {

      abs_eta = fabs((**lConstituentIt).iEta());
      if (abs_eta < 21) tmpEta = (0.087*abs_eta - 0.0435);
      else {
	abs_eta -= 21;
	tmpEta = 1.74;
	
	for (int i=0; i!=int(abs_eta); ++i) tmpEta += endcapEta[i];
      
	// ~ Why are we treating these differently??? ~
	if (mJetSize % 2 == 0) tmpEta += endcapEta[int(abs_eta)] / 2.;
	else tmpEta += endcapEta[int(abs_eta)];
      }
      if ((**lConstituentIt).iEta()<0) tmpEta = (-1)*tmpEta;

      etaSumEt += ( (**lConstituentIt).E() + (**lConstituentIt).H() ) * ( tmpEta );
      sumEt += ( (**lConstituentIt).E() + (**lConstituentIt).H() ) ;
    }
    
    mWeightedEta = etaSumEt/sumEt ; 

  }
  */
  /*
  void L1TowerJet::calculateWeightedPhi( )
  {
    //  double JetSize = double(mJetSize) / 2.0;
    double WeightedPhi = ( ( mWeightedIphi-0.5 ) * 0.087 );
    //Need this because 72*0.087 != 2pi: else get uneven phi dist
    double pi=(72*0.087)/2;
    if(WeightedPhi > pi) WeightedPhi -=2*pi;
    mWeightedPhi=WeightedPhi;
    //  std::cout<<"weighted IPhi: "<<mWeightedIphi<<" weighted phi: "<<WeightedPhi<<std::endl;
  }
  */

  /*
  // old version of calculating weighted eta using as input weighted iEta (this does not give the right result)
  void L1TowerJet::calculateWeightedEta()
  {
  double WeightedEta(9999);
  double abs_eta =fabs(mWeightedIeta);
  if ( abs_eta < 21 )
  {
  //with discrete iEta this first option should always be the case
  if( abs_eta >=1) WeightedEta = (0.087*abs_eta - 0.0435);
  else WeightedEta = 0.0435*abs_eta ;

  }
  else
  {
  const double endcapEta[8] = { 0.09, 0.1, 0.113, 0.129, 0.15, 0.178, 0.15, 0.35 };
  abs_eta -= 21;

  WeightedEta = 1.74;

  for ( int i = 0; i != int(abs_eta); ++i )
  {
  WeightedEta += endcapEta[i];
  }
  if( mJetSize % 2 == 0 ){
  WeightedEta += endcapEta[int(abs_eta)] / 2.;
  }else{
  WeightedEta += endcapEta[int(abs_eta)];
  }
  }
  if(mWeightedIeta<0) WeightedEta=-WeightedEta;

  mWeightedEta = WeightedEta;
  }
  */








  void L1TowerJet::addConstituent( const L1CaloTowerRef & Tower )
  { 
  
    int lHalfJetSizeEta( mJetSize >> 1 );
    int lHalfJetSizePhi( mJetSize >> 1 );
    // Tower energy in 2 GeV units -> 1 unit = 2 GeV
    int lTowerEnergy( Tower->E(  ) + Tower->H(  ) );

    //slightly different sizes for HF jets
    if( abs( iEta() ) > 28 ){
      //???? Not sure what this is doing ????
      lHalfJetSizeEta = 1; //ie mJetSize/4 as in HF jets 2 in eta
    }
  
    // Store the energy in 2GeV units
    mE2GeV += lTowerEnergy;
    mE     += 0.5*lTowerEnergy;
    mConstituents.push_back( Tower );


//     std::cout << Tower->iEta() << "\t" << Tower->iPhi() << "\t"
// 	      << Tower->E()    << "\t" << Tower->H()    << "\t" << "\n";


    // ********************************************************************
    // *                Calculate the asymmetry parameters                *
    // ********************************************************************
    //
    // Currently the choice of definition of these parameters may not be optimal, as the parameters
    // favour symmetric jets rather than jets with central energy deposits. There are also problems
    // with degeneracy in the sorting stage, making the algorithm dependent on the sorting algorithm.
    //
    // Definition:
    // ~~~~~~~~~~
    // Positive asymmetry = For TT energy deposits 'above' jet center
    // Negative asymmetry = For TT energy deposits 'below' jet center

    // TT iPhi without calorimeter wrapping i.e. iPhi can exceed 71
    int ToweriPhi = Tower->iPhi();
    
    // Check whether the edge of the jet mask wraps around the calorimeter
    if ( iPhi() > (72 - mJetSize) ){ 
      if ( ToweriPhi < mJetSize ){
	// Undo the wrapping of the calorimeter
	ToweriPhi += 72;
      }
    }

    // ********************************************************************
    // *                          Even jet size                           *
    // ********************************************************************

    if ( (mJetSize % 2) == 0 ){
    
      // Eta asymmetry
      if( Tower->iEta() >= (iEta() + lHalfJetSizeEta) ) {
        mAsymEta += lTowerEnergy; 
      }		
      else{
        mAsymEta -= lTowerEnergy; 
      }
  
      // Phi asymmetry
      if( ToweriPhi < (iPhi() + lHalfJetSizePhi) ){
        mAsymPhi += lTowerEnergy;
      }
      else{ 
        mAsymPhi -= lTowerEnergy;
      }
  
    }

    // ********************************************************************
    // *                          Odd jet size                            *
    // ********************************************************************

    else{ //odd jet size: miss out central towers
      
      if( Tower->iEta() ==  (iEta() + lHalfJetSizeEta) ) {
        mAsymEta += 0; // It is in the middle so does not contribute to the asymmetry
      }
      else if( Tower->iEta() > (iEta() + lHalfJetSizeEta) ) {
        mAsymEta +=  lTowerEnergy;
      }
      else{
        mAsymEta -= lTowerEnergy;
      }
      
  
      if( ToweriPhi == (iPhi() + lHalfJetSizePhi) ) {
        mAsymPhi += 0; // It is in the middle so does not contribute to the asymmetry
      }
      else if( ToweriPhi < (iPhi() + lHalfJetSizePhi) ) {
        mAsymPhi += lTowerEnergy;
      }
      else{
        mAsymPhi -= lTowerEnergy;
      }
    }

  } 


  void L1TowerJet::removeConstituent( const int &eta, const int &phi )
  {
    L1CaloTowerRefVector::iterator lConstituent = getConstituent( eta, phi );
    if ( lConstituent != mConstituents.end() ){

	int lHalfJetSizeEta( mJetSize >> 1 );
	int lHalfJetSizePhi( mJetSize >> 1 );
	int lTowerEnergy( (**lConstituent).E(  ) + (**lConstituent).H(  ) );

	mE2GeV -= lTowerEnergy;
	mE     -= 0.5*lTowerEnergy;
	mConstituents.erase( lConstituent );
    
	if( abs( iEta() ) > 28 ){
	  lHalfJetSizeEta = 1; //ie mJetSize/4 as in HF jets 2 in eta
	}

	int ToweriPhi = phi;
	//if iPhi at the edge of the calo wrap round in phi
	if( iPhi() > (72 - mJetSize) ) { 
	  if (ToweriPhi < mJetSize){
	    ToweriPhi += 72;
	  }
	}
        
      
	if( mJetSize % 2 == 0 ){ //even jet size
  
	  if( eta >= (iEta() + lHalfJetSizeEta) ) {
	    mAsymEta -=  lTowerEnergy;
	  }		
	  else{ /*if( Tower->iEta(  ) <  iEta() + lHalfJetSize )*/ 
	    mAsymEta += lTowerEnergy;
	  }
  
  
	  if( ToweriPhi < (iPhi() + lHalfJetSizePhi) ){
	    mAsymPhi -= lTowerEnergy;
  
	  }else{ /*if( Tower->iPhi(  ) > iPhi() + lHalfJetSize )*/  
	    mAsymPhi += lTowerEnergy;
	  }
  
  
	}else{ //odd jet size: miss out central towers
    
	  if( eta ==  (lHalfJetSizeEta + iEta()) ) {
	    mAsymEta += 0; //do nothing
	  }
	  else if( eta > (iEta() + lHalfJetSizeEta) ) {
	    mAsymEta -=  lTowerEnergy;
  
	  }else /*if( Tower->iEta(  ) <  iEta() + lHalfJetSize )*/ {
	    mAsymEta += lTowerEnergy;
  
	  }
  
	  if( ToweriPhi == (lHalfJetSizePhi + iPhi()) ) {
	    mAsymEta -= 0; //do nothing
  
	  }
	  else if( ToweriPhi < (iPhi() + lHalfJetSizePhi) ) {
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



namespace std{

  // Overloaded jet rank comparison operator: First order by Et and, in the case of degenerate energy, rank by centrality
  bool operator<( const l1slhc::L1TowerJet & aLeft, const l1slhc::L1TowerJet & aRight ){

//     std::cout << "Pt1 = "   << aLeft.Pt()         << "\tPt2 = "        << aRight.Pt()
// 	      << "Cent1 = " << aLeft.Centrality() << "\tCent2 = " << aRight.Centrality();

    // Degenerate energies
    if ( aLeft.E() == aRight.E() ){
      if ( aLeft.Centrality() == aRight.Centrality() ){
	//	std::cout << "ERROR: Degeneracy in the secondary sort parameter.\n\nOffending jets:\n";
	//		  << aLeft << "\n" << aRight << "\n\n";
	std::cout << "\t1<2 = " << false << "\n";
	return false; // Current solution, pick the jet in the list that is left most. i.e. lowest iEta, iPhi
	// 	throw cms::Exception("Sort degeneracy")
	// 	  << "Please debug me! :(\n";
      }
      else{
	// Order by the lowest centrality
	std::cout << "\t1<2 = " << ( aLeft.Centrality() > aRight.Centrality() ) << "\n";
        return ( aLeft.Centrality() > aRight.Centrality() );
      }
    }
    else{
      std::cout << "\t1<2 = " << ( aLeft.E() < aRight.E() ) << "\n";
      return ( aLeft.E() < aRight.E() );
    }
  }


  // NOTE: Added to be compatible with the bitonic sort routine which is currently also utilised in Asymmetry filtering
  // Overloaded jet rank comparison operator: First order by Et and, in the case of degenerate energy, rank by centrality
  bool operator>( const l1slhc::L1TowerJet & aLeft, const l1slhc::L1TowerJet & aRight ){
 
    // THIS DOESN'T WORK FOR SOME REASON....
    //    return ( !(aLeft < aRight) );


    //<TEMP>

//     std::cout << "Pt1 = "   << aLeft.Pt()         << "\tPt2 = "        << aRight.Pt()
// 	      << "Cent1 = " << aLeft.Centrality() << "\tCent2 = " << aRight.Centrality();

    // Degenerate energies
    if ( aLeft.E() == aRight.E() ){
      if ( aLeft.Centrality() == aRight.Centrality() ){
	//      std::cout << "ERROR: Degeneracy in the secondary sort parameter.\n\nOffending jets:\n";
	//                << aLeft << "\n" << aRight << "\n\n";
	std::cout << "\t1>2 = " << true << "\n";
	return true; // Current solution, pick the jet in the list that is left most. i.e. lowest iEta, iPhi
	//      throw cms::Exception("Sort degeneracy")
	//        << "Please debug me! :(\n";
      }
      else{
	// Order by the lowest centrality
	std::cout << "\t1>2 = " << ( aLeft.Centrality() < aRight.Centrality() ) << "\n";
	return ( aLeft.Centrality() < aRight.Centrality() );
      }
    }
    else{
      std::cout << "\t1>2 = " << ( aLeft.E() > aRight.E() ) << "\n";
      return ( aLeft.E() > aRight.E() );
    }
    //<TEMP>

  }
   
}


// pretty print
std::ostream & operator<<( std::ostream & aStream, const l1slhc::L1TowerJet & aL1TowerJet )
{
  aStream << "L1TowerJet" 
	  << " iEta=" << aL1TowerJet.iEta(  ) 
	  << " \tiPhi=" << aL1TowerJet.iPhi(  ) 
	  << "\n with constituents:\n";
  for ( l1slhc::L1CaloTowerRefVector::const_iterator i = aL1TowerJet.getConstituents(  ).begin(  ); i < aL1TowerJet.getConstituents(  ).end(  ); ++i ){
    aStream << " \tiEta = " << ( **i ).iEta(  ) 
	    << " \tiPhi = " << ( **i ).iPhi(  ) 
	    << " \tET = "   << ( **i ).E(  ) 
	    << "\n";
  }
  return aStream;
}
