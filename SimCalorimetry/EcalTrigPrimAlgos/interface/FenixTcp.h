#ifndef FENIX_TCP_H
#define FENIX_TCP_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/FenixChip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixBypassLin.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixMaxof2.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixFgvbEB.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixFgvbEE.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtTot.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixBypassLin.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtTot.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixMaxof2.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixFgvb.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;

class FenixTcp : FenixChip {
  /* {src_lang=Cpp}*/



 public:
  EcalFenixFgvb fgvb;
  /* {transient=false, volatile=false}*/

  EcalFenixTcpFormat format;
  /* {transient=false, volatile=false}*/

 private:
  EcalFenixBypassLin bypassLin[ nStripsPerTower_];
  /* {transient=false, volatile=false}*/

  EcalFenixEtTot etTot;
  /* {transient=false, volatile=false}*/

  EcalFenixMaxof2 maxOf2;
  /* {transient=false, volatile=false}*/


 public:
  EcalFenixBypassLin myEcalFenixBypassLin;

  EcalFenixMaxof2 myEcalFenixMaxof2;

  EcalFenixTcpFormat myEcalFenixTcpFormat;

  EcalFenixFgvbEB myEcalFenixFgvbEB;

  EcalFenixFgvbEE myEcalFenixFgvbEE;

  EcalFenixEtTot myEcalFenixEtTot;
};

#endif
