#ifndef ECAL_FENIX_CHIP_H
#define ECAL_FENIX_CHIP_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVAdder.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFgvb.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFormatter.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVAdder.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFormatter.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFgvb.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVAmplitudeFilter.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVPeakFinder.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFormatter.h>
#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <stdio.h>
#include <vector>

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;
  /** 
     \class EcalFenixChip

     \brief abstract base class for all Fenix chips (barrel +endcap, strip and tcp)
  */

class EcalFenixChip {


 public:
  //  virtual void process(std::vector<EBDataFrame> &,EcalTriggerPrimitiveSample&) {;}  //version strip
  //  virtual void process(std::vector<EcalTriggerPrimitiveSample>&,EcalTriggerPrimitiveSample&) {;};  //version tcp

 protected:
  enum {nCrystalsPerStrip_ = 5};
  enum {nStripsPerTower_ = 5};

  //  EcalVLinearizer *linearizer_[nCrystalsPerStrip_];

  EcalVAdder *adder_;

  EcalVFormatter *formatter_;

  EcalVFgvb *fgvb_;

};

#endif
