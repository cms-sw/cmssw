#ifndef SimMuon_DTDigitizer_DTDigiSyncBase_H
#define SimMuon_DTDigitizer_DTDigiSyncBase_H

/** \class DTDigiSyncBase
 *  Base class to define the offsets for digis.
 *
 *  $Date: 2005/12/08 11:30:16 $
 *  $Revision: 1.0 $
 *  \author N. Amapane, G. Cerminara, R. Bellan - INFN Torino
 */

class DTDetId;
class DTGeomDetUnit;

class DTDigiSyncBase {
public:

  /// Constructor
  DTDigiSyncBase(){};

  /// Destructor
  virtual ~DTDigiSyncBase(){};

  /// Delays to be added to digi times during digitization, in ns.
  virtual double digitizerOffset(const DTDetId * id, const DTGeomDetUnit* layer) const = 0;

  /// Offset to obtain "raw" TDCs for the L1 emulator from digis.
  virtual double emulatorOffset(const DTDetId * id) const = 0;

};
#endif

