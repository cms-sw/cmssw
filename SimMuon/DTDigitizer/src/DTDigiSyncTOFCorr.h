#ifndef SimMuon_DTDigitizer_DTDigiSyncTOFCorr_H
#define SimMuon_DTDigitizer_DTDigiSyncTOFCorr_H

/** \class DTDigiSyncTOFCorr
 *  Digi offset computed as:           <br>
 *  t0 = Tcommon - aTOF                <br><br>
 *  
 *  where Tcommon is a fixed offset defined in                    <br>
 *  DTDigiSyncTOFCorr:offset (in ORCA the default was = 500 ns)       <br><br>
 *
 *  and aTOF is set according to MuBarDigiSyncTOFCorr:TOFCorrection: <br> 
 *  0: no TOF correction (aTOF=0)                                         <br>
 *  1: aTOF = the TOF of an infinite-momentum particle travelling from the
 *     nominal IP to the 3D center of the chamber                         <br>
 *  2: ditto, but for a particle travelling to the 3D center of the wire. 
 *     (This mode is avaliable for comparison with older data which were  
 *     produced in this way)
 *
 *  $Date: 2005/12/08 11:30:50 $
 *  $Revision: 1.0 $
 *  \author N. Amapane, R. Bellan - INFN Torino
 */

#include "SimMuon/DTDigitizer/interface/DTDigiSyncBase.h"

class DTDetId;
class DTGeomDetUnit;
namespace edm{class ParameterSet;}

class DTDigiSyncTOFCorr : public DTDigiSyncBase {
public:
  /// Constructor
  DTDigiSyncTOFCorr(const edm::ParameterSet&);

  /// Destructor
  virtual ~DTDigiSyncTOFCorr();

  /// Delays to be added to digi times during digitization, in ns.
  virtual double digitizerOffset(const DTDetId * id, const DTGeomDetUnit* layer=0) const;

  /// Offset to obtain "raw" TDCs for the L1 emulator from digis.
  virtual double emulatorOffset(const DTDetId * id) const;

private:
  double theOffset;
  int corrType;
};
#endif

