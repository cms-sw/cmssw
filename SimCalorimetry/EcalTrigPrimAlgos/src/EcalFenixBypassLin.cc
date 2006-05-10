using namespace std; 
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixBypassLin.h>
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <iostream>

class EBDataFrame;

namespace tpg {

  EcalFenixBypassLin::EcalFenixBypassLin(){
  }
  EcalFenixBypassLin::~EcalFenixBypassLin(){
  }
  EBDataFrame EcalFenixBypassLin::process(EBDataFrame df)
  {
    return df;
  }

  std::vector<int> EcalFenixBypassLin::process(std::vector<int> stripin)
  {
    return stripin;
  }

  // global type definitions for class implementation in source file defined by Tag entries in ArgoUML
  // Result: typedef <typedef_global_source> <tag_value>;

} /* End of namespace tpg */

