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
  theGainErrors[0] = 0.;
  theGainErrors[1] = 0.;
  theGainErrors[2] = 0.;
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
  double threeSigmaADCNoise[NGAINS];
  for(int igain = 0; igain < NGAINS; ++igain) {
    LSB[igain]= Emax/(MAXINT*theGains[igain]);
    // fill in the pedestal and width
    double width = 0.;
    findPedestal(detId, igain, pedestals[igain], widths[igain]);
    if(addNoise_) {
      threeSigmaADCNoise[igain] = widths[igain]/LSB[igain] * 3.;
    }
  }


  int wait = 0 ;
  int gainId = NGAINS - 1 ;
  for (int i = 0 ; i < timeframe.size() ; ++i)
  {    
     int adc = MAXINT;
     if (wait == 0) gainId = NGAINS - 1;

     // see which gain bin it fits in
     int igain = gainId + 1 ;
     while (igain != 0) {
       --igain;

       double ped = + pedestals[igain];
       int tmpadc = int((timeframe[i]+ped)/LSB[igain]);

       // see if it's close enough to the boundary that we have to throw noise
       if(addNoise_ && (tmpadc <= MAXINT+threeSigmaADCNoise[igain]) ) {
          ped = RandGauss::shoot(ped, widths[igain]);
          tmpadc = int((timeframe[i]+ped)/LSB[igain]);
       }
         
       if(tmpadc <= MAXINT ) {
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
  double LSB = Emax/(MAXINT*theGains[gainNumber]) ;
  double pedestal = 0.;
  double width = 0.;
  findPedestal(id, gainNumber, pedestal, width);
  // we shift by LSB/2 to be centered
  return LSB*(sample.adc() + 0.5) - pedestal;
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
