#ifndef FENIX_CHIP_H
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVAdder.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVAdder.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFgvb.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVAmplitudeFilter.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVPeakFinder.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFormatter.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFormatter.h>

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;

class FenixChip {
  /* {src_lang=Cpp}*/



 public:
  EcalVLinearizer Linearizer;
  /* {transient=false, volatile=false}*/

  EcalVAdder adder;
  /* {transient=false, volatile=false}*/

  null veto_bit;
  /* {transient=false, volatile=false}*/

  int ;
  /* {transient=false, volatile=false}*/

  null formatter;
  /* {transient=false, volatile=false}*/
};


#endif
