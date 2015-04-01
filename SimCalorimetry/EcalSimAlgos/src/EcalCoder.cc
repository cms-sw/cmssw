#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

#include <iostream>

//#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"


EcalCoder::EcalCoder( bool                  addNoise     , 
		      bool                  PreMix1      ,
		      EcalCoder::Noisifier* ebCorrNoise0 ,
		      EcalCoder::Noisifier* eeCorrNoise0 ,
		      EcalCoder::Noisifier* ebCorrNoise1 ,
		      EcalCoder::Noisifier* eeCorrNoise1 ,
		      EcalCoder::Noisifier* ebCorrNoise2 ,
		      EcalCoder::Noisifier* eeCorrNoise2   ) :
   m_peds        (           0 ) ,
   m_gainRatios  (           0 ) ,
   m_intercals   (           0 ) ,
   m_maxEneEB    (      1668.3 ) , // 4095(MAXADC)*12(gain 2)*0.035(GeVtoADC)*0.97
   m_maxEneEE    (      2859.9 ) , // 4095(MAXADC)*12(gain 2)*0.060(GeVtoADC)*0.97
   m_addNoise    ( addNoise    ) ,
   m_PreMix1     ( PreMix1     ) 
   {
   m_ebCorrNoise[0] = ebCorrNoise0 ;
   assert( 0 != m_ebCorrNoise[0] ) ;
   m_eeCorrNoise[0] = eeCorrNoise0 ;
   m_ebCorrNoise[1] = ebCorrNoise1 ;
   m_eeCorrNoise[1] = eeCorrNoise1 ;
   m_ebCorrNoise[2] = ebCorrNoise2 ;
   m_eeCorrNoise[2] = eeCorrNoise2 ;
}  

EcalCoder::~EcalCoder()
{
}

void 
EcalCoder::setFullScaleEnergy( double EBscale ,
			       double EEscale   )
{
   m_maxEneEB = EBscale ;
   m_maxEneEE = EEscale ;
}


void  
EcalCoder::setPedestals( const EcalPedestals* pedestals ) 
{
   m_peds = pedestals ;
}

void  
EcalCoder::setGainRatios( const EcalGainRatios* gainRatios ) 
{
   m_gainRatios = gainRatios ; 
}

void 
EcalCoder::setIntercalibConstants( const EcalIntercalibConstantsMC* ical ) 
{
   m_intercals = ical ;
}

double 
EcalCoder::fullScaleEnergy( const DetId & detId ) const 
{
   return detId.subdetId() == EcalBarrel ? m_maxEneEB : m_maxEneEE ;
}

void 
EcalCoder::analogToDigital( CLHEP::HepRandomEngine* engine,
                            const EcalSamples& clf ,
			    EcalDataFrame&     df    ) const 
{
   df.setSize( clf.size() ) ;
   encode( clf, df, engine );
/*   std::cout<<"   **Id=" ;
   if( CaloGenericDetId( clf.id() ).isEB() )
   {
      std::cout<<EBDetId( clf.id() ) ;
   }
   else
   {
      std::cout<<EEDetId( clf.id() ) ;
   }
   std::cout<<", size="<<df.size();
   for( int i ( 0 ) ; i != df.size() ; ++i )
   {
      std::cout<<", "<<df[i];
   }
   std::cout<<std::endl ;*/
}

void 
EcalCoder::encode( const EcalSamples& ecalSamples , 
		   EcalDataFrame&     df,
                   CLHEP::HepRandomEngine* engine ) const
{
   assert( 0 != m_peds ) ;

   const unsigned int csize ( ecalSamples.size() ) ;

  
   DetId detId = ecalSamples.id();             
   double Emax = fullScaleEnergy(detId);       

   //....initialisation
   if ( ecalSamples[5] > 0. ) LogDebug("EcalCoder") << "Input caloSample" << "\n" << ecalSamples;
  
   double LSB[NGAINS+1];
   double pedestals[NGAINS+1];
   double widths[NGAINS+1];
   double gains[NGAINS+1];
   int    maxADC[NGAINS+1];
   double trueRMS[NGAINS+1];

   double icalconst = 1. ;
   findIntercalibConstant( detId, icalconst );

   for( unsigned int igain ( 0 ); igain <= NGAINS ; ++igain ) 
   {
      // fill in the pedestal and width
      findPedestal( detId ,
		    igain , 
		    pedestals[igain] , 
		    widths[igain]      ) ;

      if( 0 < igain ) 
	 trueRMS[igain] = std::sqrt( widths[igain]*widths[igain] - 1./12. ) ;

      // set nominal value first
      findGains( detId , 
		 gains  );               

      LSB[igain] = 0.;
      if ( igain > 0 ) LSB[igain]= Emax/(MAXADC*gains[igain]);
      maxADC[igain] = ADCGAINSWITCH;                   // saturation at 4080 for middle and high gains x6 & x12
      if ( igain == NGAINS ) maxADC[igain] = MAXADC;   // saturation at 4095 for low gain x1 
   }

   CaloSamples noiseframe[] = { CaloSamples( detId , csize ) ,
				CaloSamples( detId , csize ) ,
				CaloSamples( detId , csize )   } ;

   const Noisifier* noisy[3] = { ( 0 == m_eeCorrNoise[0]          ||
				   EcalBarrel == detId.subdetId()    ?
				   m_ebCorrNoise[0] :
				   m_eeCorrNoise[0]                  ) ,
				 ( EcalBarrel == detId.subdetId() ?
				   m_ebCorrNoise[1] :
				   m_eeCorrNoise[1]                  ) ,
				 ( EcalBarrel == detId.subdetId() ?
				   m_ebCorrNoise[2] :
				   m_eeCorrNoise[2]                  )   } ;

   if( m_addNoise )
   {
     noisy[0]->noisify( noiseframe[0], engine ) ; // high gain
      if( 0 == noisy[1] ) noisy[0]->noisify( noiseframe[1] ,
                                             engine,
					     &noisy[0]->vecgau() ) ; // med 
      if( 0 == noisy[2] ) noisy[0]->noisify( noiseframe[2] ,
                                             engine,
					     &noisy[0]->vecgau() ) ; // low
   }


   //   std::cout << " intercal, LSBs, gains " << icalconst << " " << LSB[0] << " " << LSB[1] << " " << gains[0] << " " << gains[1] << " " << Emax <<  std::endl;

   int wait = 0 ;
   int gainId = 0 ;
   bool isSaturated = 0;

   for( unsigned int i ( 0 ) ; i != csize ; ++i )
   {    
      bool done ( false ) ;
      int adc = MAXADC ;
      if( 0 == wait ) gainId = 1 ;

      // see which gain bin it fits in
      int igain ( gainId - 1 ) ;
      do
      {
	 ++igain ;

//	 if( igain != gainId ) std::cout <<"$$$$ Gain switch from " << gainId
//					 <<" to "<< igain << std::endl ;

	 if( 1 != igain                    &&   // not high gain
	     m_addNoise                    &&   // want to add noise
	     0 != noisy[igain-1]           &&   // exists
	     noiseframe[igain-1].isBlank()    ) // not already done
	 {
	    noisy[igain-1]->noisify( noiseframe[igain-1] ,
                                     engine,
				     &noisy[0]->vecgau()   ) ;
	    //std::cout<<"....noisifying gain level = "<<igain<<std::endl ;
	 }
	
	 double signal;

	 if(!m_PreMix1) {

	   // noiseframe filled with zeros if !m_addNoise
	   const double asignal ( pedestals[igain] +
			       ecalSamples[i] /( LSB[igain]*icalconst ) +
			       trueRMS[igain]*noiseframe[igain-1][i]      ) ;
	   signal = asignal;
	 }
	 else {  // Any changes made here must be reverse-engineered in EcalSignalGenerator!

           if( igain == 1) {
             const double asignal ( ecalSamples[i]*1000. );  // save low level info                   
             signal = asignal;
           }
           else if( igain == 2) {
             const double asignal ( ecalSamples[i]/( LSB[1]*icalconst ));
             signal = asignal;
           }
           else if( igain == 3) {   // bet that no pileup hit has an energy over Emax/2             
             const double asignal ( ecalSamples[i]/( LSB[2]*icalconst ) );
             signal = asignal;
           }
	   else { //not sure we ever get here at gain=0, but hit wil be saturated anyway
	     const double asignal ( ecalSamples[i]/( LSB[3]*icalconst ) ); // just calculate something
             signal = asignal;
	   }
	   // old version
	   //const double asignal ( // no pedestals for pre-mixing
	   //			 ecalSamples[i] /( LSB[igain]*icalconst ) );
	   //signal = asignal;
	 }

	 //	 std::cout << " " << ecalSamples[i] << " " << noiseframe[igain-1][i] << std::endl;


	 const int isignal ( signal ) ;
	 const int tmpadc ( signal - (double)isignal < 0.5 ?
			    isignal : isignal + 1 ) ;
	 // LogDebug("EcalCoder") << "DetId " << detId.rawId() << " gain " << igain << " caloSample " 
	 //                       <<  ecalSamples[i] << " pededstal " << pedestals[igain] 
	 //                       << " noise " << widths[igain] << " conversion factor " << LSB[igain] 
	 //                       << " result (ped,tmpadc)= " << ped << " " << tmpadc;
         
	 if( tmpadc <= maxADC[igain] ) 
	 {
	    adc = tmpadc;
	    done = true ;
	 }
      }
      while( !done       &&
	     igain < 3    ) ;

      if (igain == 1 ) 
      {
         wait = 0 ;
         gainId = igain ;
      }
      else 
      {
         if (igain == gainId) --wait ;
         else 
	 {
	    wait = 5 ;
	    gainId = igain ;
	 }
      }


      // change the gain for saturation
      int storeGainId = gainId;
      if ( gainId == 3 && adc == MAXADC ) 
      {
	 storeGainId = 0;
	 isSaturated = true;
      }
      // LogDebug("EcalCoder") << " Writing out frame " << i << " ADC = " << adc << " gainId = " << gainId << " storeGainId = " << storeGainId ; 
     
      df.setSample( i, EcalMGPASample( adc, storeGainId ) );   
   }
   // handle saturation properly according to IN-2007/056
   // N.B. - FIXME 
   // still missing the possibility for a signal to pass the saturation threshold
   // again within the 5 subsequent samples (higher order effect).

   if ( isSaturated ) 
   {
      for (unsigned int i = 0 ; i < ecalSamples.size() ; ++i) 
      {
	 if ( df.sample(i).gainId() == 0 ) 
	 {
	    unsigned int hyst = i+1+2;
	    for ( unsigned int j = i+1; j < hyst && j < ecalSamples.size(); ++j ) 
	    {
	       df.setSample(j, EcalMGPASample(MAXADC, 0));   
	    }
	 }
      }
   }
}

void 
EcalCoder::findPedestal( const DetId & detId  , 
			 int           gainId , 
			 double&       ped    , 
			 double&       width     ) const
{
   /*
     EcalPedestalsMapIterator mapItr 
     = m_peds->m_pedestals.find(detId.rawId());
     // should I care if it doesn't get found?
     if(mapItr == m_peds->m_pedestals.end()) {
     edm::LogError("EcalCoder") << "Could not find pedestal for " << detId.rawId() << " among the " << m_peds->m_pedestals.size();
     } else {
     EcalPedestals::Item item = mapItr->second;
   */
   
   /*   
	EcalPedestals::Item const & item = (*m_peds)(detId);
	ped = item.mean(gainId);
	width = item.rms(gainId);
   */

   EcalPedestalsMap::const_iterator itped = m_peds->getMap().find( detId );
   ped   = (*itped).mean(gainId);
   width = (*itped).rms(gainId);
  
   if ( (detId.subdetId() != EcalBarrel) && (detId.subdetId() != EcalEndcap) ) 
   { 
      edm::LogError("EcalCoder") << "Could not find pedestal for " << detId.rawId() << " among the " << m_peds->getMap().size();
   } 

  /*
    switch(gainId) {
    case 0:
      ped = 0.;
      width = 0.;
    case 1:
      ped = item.mean_x12;
      width = item.rms_x12;
      break;
    case 2:
      ped = item.mean_x6;
      width = item.rms_x6;
      break;
    case 3:
      ped = item.mean_x1;
      width = item.rms_x1;
      break;
    default:
      edm::LogError("EcalCoder") << "Bad Pedestal " << gainId;
      break;
    }
  */

   LogDebug("EcalCoder") << "Pedestals for " << detId.rawId() << " gain range " << gainId << " : \n" << "Mean = " << ped << " rms = " << width;
}


void 
EcalCoder::findGains( const DetId & detId , 
		      double Gains[]        ) const
{
  /*
    EcalGainRatios::EcalGainRatioMap::const_iterator grit=m_gainRatios->getMap().find(detId.rawId());
    EcalMGPAGainRatio mgpa;
    if( grit!=m_gainRatios->getMap().end() ){
    mgpa = grit->second;
    Gains[0] = 0.;
    Gains[3] = 1.;
    Gains[2] = mgpa.gain6Over1() ;
    Gains[1] = Gains[2]*(mgpa.gain12Over6()) ;
    LogDebug("EcalCoder") << "Gains for " << detId.rawId() << "\n" << " 1 = " << Gains[1] << "\n" << " 2 = " << Gains[2] << "\n" << " 3 = " << Gains[3];
    } else {
    edm::LogError("EcalCoder") << "Could not find gain ratios for " << detId.rawId() << " among the " << m_gainRatios->getMap().size();
    }
  */

   EcalGainRatioMap::const_iterator grit = m_gainRatios->getMap().find( detId );
   Gains[0] = 0.;
   Gains[3] = 1.;
   Gains[2] = (*grit).gain6Over1();
   Gains[1] = Gains[2]*( (*grit).gain12Over6() );   
   
  
   if ( (detId.subdetId() != EcalBarrel) && (detId.subdetId() != EcalEndcap) ) 
   { 
      edm::LogError("EcalCoder") << "Could not find gain ratios for " << detId.rawId() << " among the " << m_gainRatios->getMap().size();
   }   
  
}

void 
EcalCoder::findIntercalibConstant( const DetId& detId, 
				   double&      icalconst ) const
{
   EcalIntercalibConstantMC thisconst = 1.;
   // find intercalib constant for this xtal
   const EcalIntercalibConstantMCMap &icalMap = m_intercals->getMap();
   EcalIntercalibConstantMCMap::const_iterator icalit = icalMap.find(detId);
   if( icalit!=icalMap.end() )
   {
      thisconst = (*icalit);
      if ( icalconst == 0. ) { thisconst = 1.; }
   } 
   else
   {
      edm::LogError("EcalCoder") << "No intercalib const found for xtal " << detId.rawId() << "! something wrong with EcalIntercalibConstants in your DB? ";
   }
   icalconst = thisconst;
}
