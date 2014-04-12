#ifndef TauAnalysis_MCEmbeddingTools_MuonCaloCleanerAllCrossed_h
#define TauAnalysis_MCEmbeddingTools_MuonCaloCleanerAllCrossed_h

/** \class MuonCaloCleanerAllCrossed
 *
 * Produce collections of calorimeter recHits
 * from which energy deposits of muons are subtracted.
 *
 * This module attributes to the muon all energy 
 * deposited in calorimeter cells crossed by the muon.
 * 
 * WARNING: this module is to be used only for calorimeter types
 *          for which the assumption that no other particle deposited
 *          energy close to the muon track is a good approximation
 *         (e.g. HO)
 * 
 * \author Tomasz Maciej Frueboes;
 *         Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: MuonCaloCleanerAllCrossed.h,v 1.2 2012/11/25 15:43:12 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <DataFormats/MuonReco/interface/Muon.h>

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

class MuonCaloCleanerAllCrossed : public edm::EDProducer 
{
 public:
  explicit MuonCaloCleanerAllCrossed(const edm::ParameterSet&);
  ~MuonCaloCleanerAllCrossed();

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  typedef std::map<uint32_t, float> detIdToFloatMap;
  void fillEnergyDepositMap(edm::Event&, const edm::EventSetup&, const reco::Candidate*, detIdToFloatMap&);
      
  edm::InputTag srcSelectedMuons_;
  edm::InputTag srcESrecHits_;

  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters trackAssociatorParameters_;
};

#endif
