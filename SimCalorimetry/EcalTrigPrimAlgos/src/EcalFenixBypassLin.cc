#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixBypassLin.h>
//#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <iostream>
#include <vector>

class EBDataFrame;

  EcalFenixBypassLin::EcalFenixBypassLin(){
  }
  EcalFenixBypassLin::~EcalFenixBypassLin(){
  }
  void EcalFenixBypassLin::process(std::vector<int> &stripin,std::vector<int> &out)
  {
    //    return stripin;
    out=stripin;
    return;
  }


