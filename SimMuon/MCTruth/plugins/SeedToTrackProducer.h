#ifndef SimMuon_MCTruth_SeedToTrackProducer_h
#define SimMuon_MCTruth_SeedToTrackProducer_h

/** \class SeedToTrackProducer
 *  
 *  SeedToTrackProducerBase class specialized for Phase 1 
 *  Muon seeds
 * 
 *  \author Luca Ferragina (INFN BO), 2024
 */

#include "SimMuon/MCTruth/plugins/SeedToTrackProducerBase.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

typedef SeedToTrackProducerBase<TrajectorySeedCollection> SeedToTrackProducer;

#endif
