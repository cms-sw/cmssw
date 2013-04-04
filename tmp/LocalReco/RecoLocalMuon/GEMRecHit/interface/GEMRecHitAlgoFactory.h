#ifndef RecoLocalMuon_GEMRecHitAlgoFactory_H
#define RecoLocalMuon_GEMRecHitAlgoFactory_H

/** \class GEMRecHitAlgoFactory
 *  Factory of seal plugins for 1D RecHit reconstruction algorithms.
 *  The plugins are concrete implementations of GEMRecHitBaseAlgo base class.
 *
 *  $Date: 2007/04/17 22:46:52 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitBaseAlgo.h"

typedef edmplugin::PluginFactory<GEMRecHitBaseAlgo *(const edm::ParameterSet &)> GEMRecHitAlgoFactory;
#endif




