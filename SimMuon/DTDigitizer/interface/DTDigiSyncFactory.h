#ifndef SimMuon_DTDigitizer_DTDigiSyncFactory_H
#define SimMuon_DTDigitizer_DTDigiSyncFactory_H

/** \class DTDigiSyncFactory
 *  Factory of digi syncronizers for digi building.
 *  The concrete instances of DTDigiSyncBase selected by the card
 *  Muon:MuBarDigiSyncFactory:Sync            FIXME
 *  are accessed via ComponentFactoryByName
 *
 *
 *  $Date: 2005/02/22 16:51:16 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino
 */

#include <PluginManager/PluginFactory.h>
#include "SimMuon/DTDigitizer/interface/DTDigiSyncBase.h"

namespace edm {class ParameterSet;}

class DTDigiSyncFactory : public seal::PluginFactory<DTDigiSyncBase*(const edm::ParameterSet&)>{
 public:

  /// Constructor
  DTDigiSyncFactory();

  /// Destructor
  virtual ~DTDigiSyncFactory(){};

  /// to get the pointer to the pointer to the dt synchronizer. 
  static DTDigiSyncFactory* get (void);

 private:
  static DTDigiSyncFactory theDTDigiSyncFactory;
  std::string theSyncType;
  
};
#endif

