#ifndef SimMuon_MCTruth_Phase2SeedToTrackProducer_h
#define SimMuon_MCTruth_Phase2SeedToTrackProducer_h

/** \class SeedToTrackProducer
 *  
 *  SeedToTrackProducerBase class specialized for Phase 2 
 *  Muon seeds
 * 
 *  \author Luca Ferragina (INFN BO), 2024
 */

#include "SimMuon/MCTruth/plugins/SeedToTrackProducerBase.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"

typedef SeedToTrackProducerBase<L2MuonTrajectorySeedCollection> Phase2SeedToTrackProducer;

#endif
