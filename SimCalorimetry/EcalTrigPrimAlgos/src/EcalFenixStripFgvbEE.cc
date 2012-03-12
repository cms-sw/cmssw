#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFgvbEE.h>
#include <DataFormats/EcalDigi/interface/EEDataFrame.h>
#include <CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
      int res= (adc>threshold_fg) ? 1 : 0;
      indexLut[i]=indexLut[i] | (res <<ixtal & maskFgvb[ixtal]);
    }
    int mask = 1<<indexLut[i];
    output[i]= lut_fg & mask;
    if (output[i]>0) output[i]=1;
  }
  return;
}  

void EcalFenixStripFgvbEE::setParameters(uint32_t id,const EcalTPGFineGrainStripEE * ecaltpgFgStripEE)
{
  const EcalTPGFineGrainStripEEMap &fgmap = ecaltpgFgStripEE -> getMap();
  EcalTPGFineGrainStripEEMapIterator it=fgmap.find(id);
  if (it!=fgmap.end()) fgparams_=&(*it).second;
  else edm::LogWarning("EcalTPG")<<" could not find EcalTPGFineGrainStripEEMap entry for "<<id;
   
    

}
