#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFgvbEE.h>
#include <DataFormats/EcalDigi/interface/EEDataFrame.h>
#include <CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

EcalFenixStripFgvbEE::EcalFenixStripFgvbEE()
{
}

EcalFenixStripFgvbEE::~EcalFenixStripFgvbEE(){
}

void EcalFenixStripFgvbEE::process( std::vector<std::vector<int> > &linout ,std::vector<int> & output)
{
  unsigned int maskFgvb[]={1,2,4,8,0x10};

  //  int threshold_fg = (*params_)[6];
  //  int lut_fg = (*params_)[7];

  int threshold_fg = fgparams_->threshold ;
  int lut_fg = fgparams_->lut ;

  
  std::vector<int> indexLut(output.size());

  for (unsigned int i=0;i<output.size();i++) {
    output[i]=0;
    indexLut[i]=0;
    for (unsigned int ixtal=0;ixtal<linout.size();ixtal++) {
      int adc=linout[ixtal][i];
      int res = (((adc & 0xffff) > threshold_fg) || ((adc & 0x30000) != 0x0)) ? 1 : 0;
      //int res= ((adc>threshold_fg) || ((adc & 0x30000) > 0x0)) ? 1 : 0;
      //indexLut[i]=indexLut[i] | (res <<ixtal & maskFgvb[ixtal]);
      indexLut[i] = indexLut[i] | (res << ixtal);
    }
    int mask = 1<<(indexLut[i]);
    output[i]= ((lut_fg & mask) == 0x0) ? 0 : 1;
    if(i > 0) output[i-1] = output[i]; // Delay one clock
  }
  return;
}  

void EcalFenixStripFgvbEE::setParameters(uint32_t id,const EcalTPGFineGrainStripEE * ecaltpgFgStripEE)
{
  id_ = id;
  const EcalTPGFineGrainStripEEMap &fgmap = ecaltpgFgStripEE -> getMap();
  EcalTPGFineGrainStripEEMapIterator it=fgmap.find(id);
  if (it!=fgmap.end()) fgparams_=&(*it).second;
  else edm::LogWarning("EcalTPG")<<" could not find EcalTPGFineGrainStripEEMap entry for "<<id;
   
    

}
