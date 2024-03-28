/** \class CaloCleaner
 *
 * Clean collections of calorimeter recHits
 * (detectors supported at the moment: EB/EE, HB/HE and HO)
 *
 * \author Tomasz Maciej Frueboes;
 *         Christian Veelken, LLR
 *
 *
 *
 *  Clean Up from STefan Wayand, KIT
 */
#ifndef TauAnalysis_MCEmbeddingTools_CaloCleaner_H
#define TauAnalysis_MCEmbeddingTools_CaloCleaner_H

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include <iostream>
#include <map>
#include <string>

template <typename T>
class CaloCleaner : public edm::stream::EDProducer<> {
public:
  explicit CaloCleaner(const edm::ParameterSet &);
  ~CaloCleaner() override;

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  typedef edm::SortedCollection<T> RecHitCollection;

  const edm::EDGetTokenT<edm::View<pat::Muon>> mu_input_;

  std::map<std::string, edm::EDGetTokenT<RecHitCollection>> inputs_;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;

  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters parameters_;

  bool is_preshower_;
  void fill_correction_map(TrackDetMatchInfo *, std::map<uint32_t, float> *);
};
#endif