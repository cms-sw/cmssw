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
 *  3: aTOF = the TOF of an infinite-momentum particle travelling from the
 *     nominal IP to the 3D center of the SL. Use this, unless you 
 *     really know what you are doing.
 *
 *  $Date: 2010/07/16 14:44:04 $
 *  $Revision: 1.4 $
 *  \author N. Amapane, R. Bellan - INFN Torino
 */

#include "SimMuon/DTDigitizer/interface/DTDigiSyncBase.h"

class DTWireId;
class DTLayer;
namespace edm{class ParameterSet;}

class DTDigiSyncTOFCorr : public DTDigiSyncBase {
public:
  /// Constructor
  DTDigiSyncTOFCorr(const edm::ParameterSet&);

  /// Destructor
  virtual ~DTDigiSyncTOFCorr();

  /// Delays to be added to digi times during digitization, in ns.
  virtual double digitizerOffset(const DTWireId * id, const DTLayer* layer=0) const;

  /// Offset to obtain "raw" TDCs for the L1 emulator from digis.
  virtual double emulatorOffset(const DTWireId * id) const;

private:
  double theOffset;
  int corrType;
};
#endif

