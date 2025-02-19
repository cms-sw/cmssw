/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/02/22 13:12:04 $
 *  $Revision: 1.4 $
 *  \author N. Amapane, R. Bellan - INFN Torino
 */

#include "SimMuon/DTDigitizer/src/DTDigiSyncFromTable.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/DTGeometry/interface/DTLayer.h"

DTDigiSyncFromTable::DTDigiSyncFromTable(const edm::ParameterSet& pSet){}

DTDigiSyncFromTable::~DTDigiSyncFromTable(){}

// Delays to be added to digi times during digitization, in ns.
double DTDigiSyncFromTable::digitizerOffset(const DTWireId * id, const DTLayer* layer) const {
  
  double result = 0;

  // ...

  return result;
}

// Offset to obtain "raw" TDCs for the L1 emulator from digis.
double DTDigiSyncFromTable::emulatorOffset(const DTWireId * id) const {

  double result = 0;

  // ...

  return result;
}
