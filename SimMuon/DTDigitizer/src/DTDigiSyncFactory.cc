/*
 *  See header file for a description of this class.
 *
 *  $Date: 2005/12/08 11:38:17 $
 *  $Revision: 1.0 $
 *  \author R. Bellan - INFN Torino
 */

#include "SimMuon/DTDigitizer/interface/DTDigiSyncFactory.h"
#include "SimMuon/DTDigitizer/interface/DTDigiSyncBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace std;

DTDigiSyncFactory DTDigiSyncFactory::theDTDigiSyncFactory;

DTDigiSyncFactory::DTDigiSyncFactory() :
  seal::PluginFactory<DTDigiSyncBase*(const edm::ParameterSet& pSet)>("DTDigiSyncFactory"){}

// Return a pointer to the syncronizer
DTDigiSyncFactory* DTDigiSyncFactory::get() {
  return &theDTDigiSyncFactory;
}
