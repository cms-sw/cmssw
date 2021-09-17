#ifndef SimDataFormats_GeneratorProducts_ExternalGeneratorEventInfo_h
#define SimDataFormats_GeneratorProducts_ExternalGeneratorEventInfo_h

/** \class ExternalGeneratorEventInfo
 *
 * This class is an internal detail of the ExternalGeneratorFilter. It is the type
 *  used to transfer from the external process to ExternalGeneratorFilter the 
 *  event information.
 */
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DataFormats/Common/interface/RandomNumberGeneratorState.h"

struct ExternalGeneratorEventInfo {
  edm::HepMCProduct hepmc_;
  GenEventInfoProduct eventInfo_;
  edm::RandomNumberGeneratorState randomState_;
  bool keepEvent_;
};

#endif
