/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/01/20 15:49:00 $
 *  $Revision: 1.2 $
 *  \author N. Amapane, R. Bellan - INFN Torino
 */

#include "SimMuon/DTDigitizer/src/DTDigiSyncTOFCorr.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "Geometry/DTSimAlgo/interface/DTLayer.h"

using namespace std;

DTDigiSyncTOFCorr::DTDigiSyncTOFCorr(const edm::ParameterSet& pSet){
  
  theOffset = pSet.getParameter<double>("offset"); //500ns
  corrType = pSet.getParameter<int>("TOFCorrection"); //1
}


DTDigiSyncTOFCorr::~DTDigiSyncTOFCorr(){}


// Delays to be added to digi times during digitization, in ns.
double DTDigiSyncTOFCorr::digitizerOffset(const DTWireId * id, const DTLayer* layer) const {

  double offset = theOffset;
  const double cSpeed = 29.9792458; // cm/ns

  if (corrType==1) {
    // Subtraction of assumed TOF, per CHAMBER
    //FIXME: implement chamberId() in MuBarWireId!
    /*
    MuBarChamberId chId(id->wheel(),id->station(),id->sector());
    MuBarChamber* chamber = theMuMap.getChamber(chId);
    double flightL = chamber->position().mag();
    */
    double flightL = layer->position().mag(); //FIXME

    offset -= flightL/cSpeed;
    
  } else if (corrType==2) {
    // Subtraction of assumed TOF, per WIRE 
    // (legacy mode, cf. previous versions)
    //FIXME: implement layerId() in MuBarWireId!
    /*
    MuBarLayerId lId(id->wheel(),id->station(),id->sector(),id->superlayer(),id->layer());
    MuBarLayer* layer = theMuMap.getLayer(lId);
    double flightL = layer->toGlobal(layer->getWire(*id)->positionInLayer()).mag();
    */
    double flightL = layer->position().mag(); //FIXME

    offset -= flightL/cSpeed;

  } else if (corrType!=0){
    cout << "ERROR: SimMuon:DTDigitizer:DTDigiSyncTOFCorr:TOFCorrection = " << corrType
	 << "is not defined " << endl; 
  }

  return offset;
}


// Offset to obtain "raw" TDCs for the L1 emulator from digis.
double DTDigiSyncTOFCorr::emulatorOffset(const DTWireId * id) const {
  return theOffset;
}
