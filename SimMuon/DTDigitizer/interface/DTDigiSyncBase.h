#ifndef SimMuon_DTDigitizer_DTDigiSyncBase_H
#define SimMuon_DTDigitizer_DTDigiSyncBase_H

/** \class DTDigiSyncBase
 *  Base class to define the offsets for digis.
 *
 *  $Date: 2006/01/25 11:09:31 $
 *  $Revision: 1.3 $
 *  \author N. Amapane, G. Cerminara, R. Bellan - INFN Torino
 */

class DTWireId;
class DTLayer;

class DTDigiSyncBase {
public:

  /// Constructor
  DTDigiSyncBase(){};

  /// Destructor
  virtual ~DTDigiSyncBase(){};

  /// Delays to be added to digi times during digitization, in ns.
  virtual double digitizerOffset(const DTWireId * id, const DTLayer* layer) const = 0;

  /// Offset to obtain "raw" TDCs for the L1 emulator from digis.
  virtual double emulatorOffset(const DTWireId * id) const = 0;

};
#endif

