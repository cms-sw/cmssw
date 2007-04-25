#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixBypassLin.h>
//#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <iostream>
#include <vector>

class EBDataFrame;

  EcalFenixBypassLin::EcalFenixBypassLin(){
  }
  EcalFenixBypassLin::~EcalFenixBypassLin(){
  }
//   EBDataFrame EcalFenixBypassLin::process(EBDataFrame &df)
//   {
//     return df;
//   }

  std::vector<int> EcalFenixBypassLin::process(std::vector<int> stripin)
  {
    return stripin;
  }


