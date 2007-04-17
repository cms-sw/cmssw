#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/DBInterface.h>

EcalFenixTcpFormat::EcalFenixTcpFormat(DBInterface * db)
  : db_(db)
{
}
 
EcalFenixTcpFormat::~EcalFenixTcpFormat() {
}

 
void EcalFenixTcpFormat::process(std::vector<int> &Et, std::vector<int> &fgvb, std::vector<EcalTriggerPrimitiveSample> & out){
    
  for (unsigned int i=0; i<Et.size();++i) {
    Et[i]=Et[i]>>2;
    if (Et[i]>0xfff) Et[i]=0xfff ;
    int lut_out = lut_[Et[i]] ;
    Et[i] = lut_out & 0xff ;
    int ttFlag = (lut_out & 0x700) >> 8 ;
    out.push_back(EcalTriggerPrimitiveSample( Et[i],fgvb[i],ttFlag)); 
    }
}

void EcalFenixTcpFormat::setParameters(int SM, int towerInSM) 
{
  lut_ = db_->getTowerParameters(SM, towerInSM) ;
}

