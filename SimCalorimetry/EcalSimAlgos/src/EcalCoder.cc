#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
//#include "CLHEP/Random/RandGaussQ.h"
#include <iostream>

EcalCoder::EcalCoder( bool                                 addNoise    , 
		      CorrelatedNoisifier<EcalCorrMatrix>* ebCorrNoise ,
		      CorrelatedNoisifier<EcalCorrMatrix>* eeCorrNoise     ) :
   m_peds        (           0 ) ,
   m_gainRatios  (           0 ) ,
   m_intercals   (           0 ) ,
   m_maxEneEB    (      1668.3 ) , // 4095(MAXADC)*12(gain 2)*0.035(GeVtoADC)*0.97
   m_maxEneEE    (      2859.9 ) , // 4095(MAXADC)*12(gain 2)*0.060(GeVtoADC)*0.97
   m_addNoise    ( addNoise    ) ,
   m_ebCorrNoise ( ebCorrNoise ) ,
   m_eeCorrNoise ( eeCorrNoise ) 
{
   assert( 0 != m_ebCorrNoise ) ;
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
EcalCoder::analogToDigital( const CaloSamples& clf , 
			    EcalDataFrame&     df    ) const 
{
   df.setSize( clf.size() ) ;
   encode( clf, df );
}

void 
EcalCoder::encode( const CaloSamples& caloSamples , 
		   EcalDataFrame&     df            ) const
{
   assert(m_peds != 0);
  
   DetId detId = caloSamples.id();             
   double Emax = fullScaleEnergy(detId);       

   //....initialisation
   if ( caloSamples[5] > 0. ) LogDebug("EcalCoder") << "Input caloSample" << "\n" << caloSamples;
  
   double LSB[NGAINS+1];
   double pedestals[NGAINS+1];
   double widths[NGAINS+1];
   double gains[NGAINS+1];
   double threeSigmaADCNoise[NGAINS+1];
   int    maxADC[NGAINS+1];

   double icalconst = 1. ;
   findIntercalibConstant( detId, icalconst );

   for( unsigned int igain ( 0 ); igain <= NGAINS ; ++igain ) 
   {
      // fill in the pedestal and width
      findPedestal( detId ,
		    igain , 
		    pedestals[igain] , 
		    widths[igain]      ) ;

      // set nominal value first
      findGains( detId , 
		 gains  );               

      LSB[igain] = 0.;
      if ( igain > 0 ) LSB[igain]= Emax/(MAXADC*gains[igain]);
      threeSigmaADCNoise[igain] = 0.;
      if ( igain > 0 ) threeSigmaADCNoise[igain] = widths[igain] * 3.;
      maxADC[igain] = ADCGAINSWITCH;                   // saturation at 4080 for middle and high gains x6 & x12
      if ( igain == NGAINS ) maxADC[igain] = MAXADC;   // saturation at 4095 for low gain x1 
   }

   CaloSamples noiseframe ( detId , 
			    caloSamples.size() ) ;        

   if( m_addNoise ) 
   { 
      if( 0 == m_eeCorrNoise             ||
	  EcalBarrel == detId.subdetId()     )
      {
	 m_ebCorrNoise->noisify( noiseframe ) ;
      }
      else
      {
	 m_eeCorrNoise->noisify( noiseframe ) ;
      }
      LogDebug("EcalCoder") << "Normalized correlated noise calo frame = " << noiseframe;
   }

   int wait = 0 ;
   int gainId = 0 ;
   bool isSaturated = 0;
   for( unsigned int i ( 0 ) ; i < (unsigned int) caloSamples.size() ; ++i )
   {    
      int adc = MAXADC;
      if (wait == 0) gainId = 1;

      // see which gain bin it fits in
      int igain = gainId-1 ;
      while (igain != 3) 
      {
	 ++igain;

	 double ped = pedestals[igain];
	 double signal = ped + caloSamples[i] / LSB[igain] / icalconst;
	 
	 // see if it's close enough to the boundary that we have to throw noise
	 if( m_addNoise && (signal <= maxADC[igain]+threeSigmaADCNoise[igain]) ) 
	 {
	    // width is the actual final noise, subtract the additional one from the trivial quantization
	    double trueRMS = std::sqrt(widths[igain]*widths[igain]-1./12.);
	    ped = ped + trueRMS*noiseframe[i];
	    signal = ped + caloSamples[i] / LSB[igain] / icalconst;
	 }
	 int tmpadc = (signal-(int)signal <= 0.5 ? (int)signal : (int)signal + 1);
	 // LogDebug("EcalCoder") << "DetId " << detId.rawId() << " gain " << igain << " caloSample " 
	 //                       <<  caloSamples[i] << " pededstal " << pedestals[igain] 
	 //                       << " noise " << widths[igain] << " conversion factor " << LSB[igain] 
	 //                       << " result (ped,tmpadc)= " << ped << " " << tmpadc;
         
	 if(tmpadc <= maxADC[igain] ) 
	 {
	    adc = tmpadc;
	    break ;
	 }
      }
     
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
      for (int i = 0 ; i < caloSamples.size() ; ++i) 
      {
	 if ( df.sample(i).gainId() == 0 ) 
	 {
	    int hyst = i+1+2;
	    for ( int j = i+1; j < hyst && j < caloSamples.size(); ++j ) 
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
