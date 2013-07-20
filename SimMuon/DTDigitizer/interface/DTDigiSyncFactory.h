#ifndef SimMuon_DTDigitizer_DTDigiSyncFactory_H
#define SimMuon_DTDigitizer_DTDigiSyncFactory_H

/** \class DTDigiSyncFactory
 *  Factory of digi syncronizers for digi building.
 *  The concrete instances of DTDigiSyncBase selected by the card
 *  Muon:MuBarDigiSyncFactory:Sync            FIXME
 *  are accessed via ComponentFactoryByName
 *
 *
 *  $Date: 2007/04/17 22:47:09 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "SimMuon/DTDigitizer/interface/DTDigiSyncBase.h"

namespace edm {class ParameterSet;}

typedef edmplugin::PluginFactory<DTDigiSyncBase *(const edm::ParameterSet &)> DTDigiSyncFactory;
  
#endif

