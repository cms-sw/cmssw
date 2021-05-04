#ifndef SimDataFormats_GeneratorProducts_ExternalGeneratorLumiInfo_h
#define SimDataFormats_GeneratorProducts_ExternalGeneratorLumiInfo_h

/** \class ExternalGeneratorLumiInfo
 *
 * This class is an internal detail of the ExternalGeneratorFilter. It is the type
 *  used to transfer from the external process to ExternalGeneratorFilter the 
 *  begin LuminosityBlock information.
 */
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"
#include "DataFormats/Common/interface/RandomNumberGeneratorState.h"

struct ExternalGeneratorLumiInfo {
  GenLumiInfoHeader header_;
  edm::RandomNumberGeneratorState randomState_;
};

#endif
