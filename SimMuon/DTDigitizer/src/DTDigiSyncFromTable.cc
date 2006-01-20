/*
 *  See header file for a description of this class.
 *
 *  $Date: 2005/12/14 11:58:00 $
 *  $Revision: 1.1 $
 *  \author N. Amapane, R. Bellan - INFN Torino
 */

#include "SimMuon/DTDigitizer/src/DTDigiSyncFromTable.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/DTSimAlgo/interface/DTGeomDetUnit.h"

DTDigiSyncFromTable::DTDigiSyncFromTable(const edm::ParameterSet& pSet){}

DTDigiSyncFromTable::~DTDigiSyncFromTable(){}

// Delays to be added to digi times during digitization, in ns.
double DTDigiSyncFromTable::digitizerOffset(const DTWireId * id, const DTGeomDetUnit* layer) const {
  
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
