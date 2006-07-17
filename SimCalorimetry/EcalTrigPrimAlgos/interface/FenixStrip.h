#ifndef FENIX_STRIP_H
#define FENIX_STRIP_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/FenixChip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixAmplitudeFilter.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormat.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixPeakFinder.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixFgvb5.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtStrip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtStrip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixFgvb5.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixAmplitudeFilter.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixPeakFinder.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormat.h>

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;

class FenixStrip : FenixChip {
  /* {src_lang=Cpp}*/



 private:
  EcalFenixLinearizer *linearizer[ nCrystalsPerStrip_];
  /* {transient=false, volatile=false, pointer=true}*/

  EcalFenixEtStrip *adder;
  /* {transient=false, volatile=false, pointer=true}*/

  EcalFenixFgvb5 *veto_bit[ nCrystalsPerStrip_ ];
  /* {transient=false, volatile=false, pointer=true}*/

  EcalFenixAmplitudeFilter *amplitude_filter;
  /* {transient=false, volatile=false, pointer=true}*/

  EcalFenixPeakFinder *peak_finder;
  /* {transient=false, volatile=false, pointer=true}*/

  EcalFenixStripFormat *fenix_format;
  /* {transient=false, volatile=false, pointer=true}*/


 public:
  EcalFenixLinearizer myEcalFenixLinearizer;

  EcalFenixAmplitudeFilter myEcalFenixAmplitudeFilter;

  EcalFenixStripFormat myEcalFenixStripFormat;

  EcalFenixPeakFinder myEcalFenixPeakFinder;

  EcalFenixFgvb5 myEcalFenixFgvb5;

  EcalFenixEtStrip myEcalFenixEtStrip;
};

#endif
