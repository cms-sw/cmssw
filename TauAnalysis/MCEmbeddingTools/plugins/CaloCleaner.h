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

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include <string>
#include <iostream>
#include <map>

template <typename T>
class CaloCleaner : public edm::stream::EDProducer<> {
public:
  explicit CaloCleaner(const edm::ParameterSet&);
  ~CaloCleaner() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  typedef edm::SortedCollection<T> RecHitCollection;

  const edm::EDGetTokenT<edm::View<pat::Muon> > mu_input_;

  std::map<std::string, edm::EDGetTokenT<RecHitCollection> > inputs_;

  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters parameters_;

  bool is_preshower_;
  void fill_correction_map(TrackDetMatchInfo*, std::map<uint32_t, float>*);
};

template <typename T>
CaloCleaner<T>::CaloCleaner(const edm::ParameterSet& iConfig)
    : mu_input_(consumes<edm::View<pat::Muon> >(iConfig.getParameter<edm::InputTag>("MuonCollection"))) {
  std::vector<edm::InputTag> inCollections = iConfig.getParameter<std::vector<edm::InputTag> >("oldCollection");
  for (const auto& inCollection : inCollections) {
    inputs_[inCollection.instance()] = consumes<RecHitCollection>(inCollection);
    produces<RecHitCollection>(inCollection.instance());
  }

  is_preshower_ = iConfig.getUntrackedParameter<bool>("is_preshower", false);
  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  edm::ConsumesCollector iC = consumesCollector();
  parameters_.loadParameters(parameters, iC);
  //trackAssociator_.useDefaultPropagator();
}

template <typename T>
CaloCleaner<T>::~CaloCleaner() {
  // nothing to be done yet...
}

template <typename T>
void CaloCleaner<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<Propagator> propagator;
  iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", propagator);
  trackAssociator_.setPropagator(propagator.product());

  edm::Handle<edm::View<pat::Muon> > muonHandle;
  iEvent.getByToken(mu_input_, muonHandle);
  edm::View<pat::Muon> muons = *muonHandle;

  std::map<uint32_t, float> correction_map;

  // Fill the correction map
  for (edm::View<pat::Muon>::const_iterator iMuon = muons.begin(); iMuon != muons.end(); ++iMuon) {
    // get the basic informaiton like fill reco mouon does
    //     RecoMuon/MuonIdentification/plugins/MuonIdProducer.cc
    const reco::Track* track = nullptr;
    if (iMuon->track().isNonnull())
      track = iMuon->track().get();
    else if (iMuon->standAloneMuon().isNonnull())
      track = iMuon->standAloneMuon().get();
    else
      throw cms::Exception("FatalError")
          << "Failed to fill muon id information for a muon with undefined references to tracks";
    TrackDetMatchInfo info =
        trackAssociator_.associate(iEvent, iSetup, *track, parameters_, TrackDetectorAssociator::Any);
    fill_correction_map(&info, &correction_map);
  }

  // Copy the old collection and correct if necessary
  for (auto input_ : inputs_) {
    std::unique_ptr<RecHitCollection> recHitCollection_output(new RecHitCollection());
    edm::Handle<RecHitCollection> recHitCollection;
    // iEvent.getByToken(input_.second[0], recHitCollection);
    iEvent.getByToken(input_.second, recHitCollection);
    for (typename RecHitCollection::const_iterator recHit = recHitCollection->begin();
         recHit != recHitCollection->end();
         ++recHit) {
      if (correction_map[recHit->detid().rawId()] > 0) {
        float new_energy = recHit->energy() - correction_map[recHit->detid().rawId()];
        if (new_energy <= 0)
          continue;  // Do not save empty Hits
        T newRecHit(*recHit);
        newRecHit.setEnergy(new_energy);
        recHitCollection_output->push_back(newRecHit);
      } else {
        recHitCollection_output->push_back(*recHit);
      }
      /* For the inveted collection   
     if (correction_map[recHit->detid().rawId()] > 0){
	float new_energy =   correction_map[recHit->detid().rawId()];
	if (new_energy < 0) new_energy =0;
	T newRecHit(*recHit);
	newRecHit.setEnergy(new_energy); 
	recHitCollection_output->push_back(newRecHit);
      }*/
    }
    // Save the new collection
    recHitCollection_output->sort();
    iEvent.put(std::move(recHitCollection_output), input_.first);
  }
}
#endif
