#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CLHEP/Random/RandGaussQ.h"
#include <iostream>


EcalCoder::EcalCoder(bool addNoise)
:  thePedestals(0),
   addNoise_(addNoise)

{
  theGains[0] = 1.;
  theGains[1] = 6.;
  theGains[2] = 12.;
  // 0.2% gain variation
  theGainErrors[0] = 0.002;
  theGainErrors[1] = 0.002;
  theGainErrors[2] = 0.002;
}


double EcalCoder::fullScaleEnergy(const DetId & detId) const 
{
 //PG Emax = x MeV/ADC * 4095 ADC * 12(gain) / 1000 MeV/GeV
 //PG for the Barrel
  if (detId.subdetId() == 1) //PG for the Barrel
  return 1719.9 ;  //PG assuming 35 MeV/ADC
//  return 1818.18 ; //PG assuming 37 MeV/ADC
  else //PG for the Endcap
  return 2948.40 ; //PG assuming 60 MeV/ADC
}


void EcalCoder::digitalToAnalog(const EBDataFrame& df, CaloSamples& lf) const {
  for(int i = 0; i < df.size(); ++i) {
    lf[i] = decode(df[i], df.id());
  }
}  


void EcalCoder::digitalToAnalog(const EEDataFrame& df, CaloSamples& lf) const {
  for(int i = 0; i < df.size(); ++i) {
    lf[i] = decode(df[i], df.id());
  }
}


void EcalCoder::analogToDigital(const CaloSamples& clf, EBDataFrame& df) const {
  std::vector<EcalMGPASample> mgpaSamples = encode(clf);

  df.setSize(clf.size());
  for(int i = 0; i < df.size(); ++i) {
    df.setSample(i, mgpaSamples[i]);
  }

}


void EcalCoder::analogToDigital(const CaloSamples& clf, EEDataFrame& df) const {
  std::vector<EcalMGPASample> mgpaSamples = encode(clf);

  df.setSize(clf.size());
  for(int i = 0; i < df.size(); ++i) {
    df.setSample(i, mgpaSamples[i]);
  }
}



std::vector<EcalMGPASample>  
EcalCoder::encode(const CaloSamples& timeframe) const
{
  assert(thePedestals != 0);
  std::vector<EcalMGPASample> results;

  DetId detId = timeframe.id();
  double Emax = fullScaleEnergy(detId);
  //....initialisation

  double LSB[NGAINS];
  double pedestals[NGAINS];
  double widths[NGAINS];
  double gains[NGAINS];
  double threeSigmaADCNoise[NGAINS];
  for(int igain = 0; igain < NGAINS; ++igain) {
    // fill in the pedestal and width
    findPedestal(detId, igain, pedestals[igain], widths[igain]);
    // set nominal value first
    gains[igain] = theGains[igain];
    if(addNoise_) {
      gains[igain] *= RandGauss::shoot(1., theGainErrors[igain]);
    }
    LSB[igain]= Emax/(MAXADC*gains[igain]);
    threeSigmaADCNoise[igain] = widths[igain]/LSB[igain] * 3.;
  }


  int wait = 0 ;
  int gainId = NGAINS - 1 ;
  for (int i = 0 ; i < timeframe.size() ; ++i)
  {    
     int adc = MAXADC;
     if (wait == 0) gainId = NGAINS - 1;

     // see which gain bin it fits in
     int igain = gainId + 1 ;
     while (igain != 0) {
       --igain;

       double ped = + pedestals[igain];
       int tmpadc = int (ped + timeframe[i] / LSB[igain]) ;

       // see if it's close enough to the boundary that we have to throw noise
       if(addNoise_ && (tmpadc <= MAXADC+threeSigmaADCNoise[igain]) ) {
          ped = RandGauss::shoot(ped, widths[igain]);
          tmpadc = int (ped + timeframe[i] / LSB[igain]) ;
       }
         
       if(tmpadc <= MAXADC ) {
         adc = tmpadc;
         break ;
       }
     }
     
     if (igain == NGAINS - 1) 
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

     results.push_back(EcalMGPASample(adc, gainId));
  }
  return results;
}

double EcalCoder::decode(const EcalMGPASample & sample, const DetId & id) const
{
  double Emax = fullScaleEnergy(id); 
  int gainNumber  = sample.gainId();
  assert( gainNumber >=0 && gainNumber <=2);
  double LSB = Emax/(MAXADC*theGains[gainNumber]) ;
  double pedestal = 0.;
  double width = 0.;
  findPedestal(id, gainNumber, pedestal, width);
  // we shift by LSB/2 to be centered
  return LSB * (sample.adc() + 0.5 - pedestal) ;
}


void EcalCoder::findPedestal(const DetId & detId, int gainId, 
                             double & ped, double & width) const
{
  EcalPedestalsMapIterator mapItr 
    = thePedestals->m_pedestals.find(detId.rawId());
  // should I care if it doesn't get found?
  if(mapItr == thePedestals->m_pedestals.end()) {
    std::cerr << "Could not find pedestal for " << detId.rawId() << " among the " << thePedestals->m_pedestals.size() << std::endl;
  } else {
    EcalPedestals::Item item = mapItr->second;
    switch(gainId) {
    case 0:
      ped = item.mean_x1;
      width = item.rms_x1;
      break;
    case 1:
      ped = item.mean_x6;
      width = item.rms_x6;
      break;
    case 2:
      ped = item.mean_x12;
      width = item.rms_x12;
      break;
    default:
      std::cerr << "Bad Pedestal " << gainId << std::endl;
      break;
    }
  }
}
