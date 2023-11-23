/** \class MuonDetCleaner
 *
 * Clean collections of hits in muon detectors (CSC, DT and RPC)
 * for original Zmumu event and "embedded" simulated tau decay products
 * 
 * \author Christian Veelken, LLR
 *
 * 
 *
 * 
 *
 * Clean Up from STefan Wayand, KIT
 * 
 */
#ifndef TauAnalysis_MCEmbeddingTools_MuonDetCleaner_H
#define TauAnalysis_MCEmbeddingTools_MuonDetCleaner_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Transition.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

#include <string>
#include <vector>
#include <map>

template <typename T1, typename T2>
class MuonDetCleaner : public edm::stream::EDProducer<> {
public:
  explicit MuonDetCleaner(const edm::ParameterSet&);
  ~MuonDetCleaner() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  typedef edm::RangeMap<T1, edm::OwnVector<T2> > RecHitCollection;
  void fillVetoHits(const TrackingRecHit&, std::vector<uint32_t>*);

  uint32_t getRawDetId(const T2&);
  bool checkrecHit(const TrackingRecHit&);

  const edm::EDGetTokenT<edm::View<pat::Muon> > mu_input_;

  std::map<std::string, edm::EDGetTokenT<RecHitCollection> > inputs_;

  TrackAssociatorParameters parameters_;
  TrackDetectorAssociator trackAssociator_;
  edm::EDGetTokenT<DTDigiCollection> m_dtDigisToken;
  edm::EDGetTokenT<CSCStripDigiCollection> m_cscDigisToken;
  edm::Handle<DTDigiCollection> m_dtDigis;
  edm::Handle<CSCStripDigiCollection> m_cscDigis;

  edm::ESHandle<DTGeometry> m_dtGeometry;
  edm::ESHandle<CSCGeometry> m_cscGeometry;
  double m_digiMaxDistanceX;
};

template <typename T1, typename T2>
MuonDetCleaner<T1, T2>::MuonDetCleaner(const edm::ParameterSet& iConfig):
    mu_input_(consumes<edm::View<pat::Muon> >(iConfig.getParameter<edm::InputTag>("MuonCollection")),
    m_dtDigisToken(consumes<DTDigiCollection>(iConfig.getParameter<edm::InputTag>("dtDigiCollectionLabel"))),
    m_cscDigisToken(consumes<CSCStripDigiCollection>(iConfig.getParameter<edm::InputTag>("cscDigiCollectionLabel"))),
    m_digiMaxDistanceX(iConfig.getParameter<double>("digiMaxDistanceX"))) {
  std::vector<edm::InputTag> inCollections = iConfig.getParameter<std::vector<edm::InputTag> >("oldCollection");
  for (const auto& inCollection : inCollections) {
    inputs_[inCollection.instance()] = consumes<RecHitCollection>(inCollection);
    produces<RecHitCollection>(inCollection.instance());
  }

  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  edm::ConsumesCollector iC = consumesCollector();
  parameters_.loadParameters(parameters, iC);
}

template <typename T1, typename T2>
MuonDetCleaner<T1, T2>::~MuonDetCleaner() {
  // nothing to be done yet...
}

template <typename T1, typename T2>
void MuonDetCleaner<T1, T2>::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  std::map<T1, std::vector<T2> > recHits_output;  // This data format is easyer to handle
  std::vector<uint32_t> vetoHits;

  // First fill the veto RecHits colletion with the Hits from the input muons
  edm::Handle<edm::View<pat::Muon> > muonHandle;
  iEvent.getByToken(mu_input_, muonHandle);
  edm::View<pat::Muon> muons = *muonHandle;
  for (edm::View<pat::Muon>::const_iterator iMuon = muons.begin(); iMuon != muons.end(); ++iMuon) {
    const reco::Track* track = nullptr;
    if (iMuon->isGlobalMuon())
      track = iMuon->outerTrack().get();
    else if (iMuon->isStandAloneMuon())
      track = iMuon->outerTrack().get();
    else if (iMuon->isRPCMuon())
      track = iMuon->innerTrack().get();  // To add, try to access the rpc track
    else if (iMuon->isTrackerMuon())
      track = iMuon->innerTrack().get();
    else {
      edm::LogError("TauEmbedding") << "The imput muon: " << (*iMuon)
                                    << " must be either global or does or be tracker muon";
      assert(0);
    }

    for (trackingRecHit_iterator hitIt = track->recHitsBegin(); hitIt != track->recHitsEnd(); ++hitIt) {
      const TrackingRecHit& murechit = **hitIt;  // Base class for all rechits
      if (!(murechit).isValid())
        continue;
      if (!checkrecHit(murechit))
        continue;                         // Check if the hit belongs to a specifc detector section
      fillVetoHits(murechit, &vetoHits);  // Go back to the very basic rechits
    }

	sort(vetoHits.begin(), vetoHits.end());
	vetoHits.erase(unique( vetoHits.begin(), vetoHits.end() ), vetoHits.end());
	iEvent.getByToken(m_dtDigisToken, m_dtDigis);
	iEvent.getByToken(m_cscDigisToken, m_cscDigis);
	iSetup.get<MuonGeometryRecord>().get(m_dtGeometry);
	iSetup.get<MuonGeometryRecord>().get(m_cscGeometry);
	edm::ESHandle<Propagator> propagator;
	iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", propagator);
	trackAssociator_.setPropagator(propagator.product());
	TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, *track, parameters_, TrackDetectorAssociator::Any);

  // inspired from Muon Identification algorithm: https://github.com/cms-sw/cmssw/blob/3b943c0dbbdf4494cd66064a5a147301f38af295/RecoMuon/MuonIdentification/plugins/MuonIdProducer.cc#L911
	for (const auto &chamber : info.chambers)
	{
		if (chamber.id.subdetId() == MuonSubdetId::RPC) //&& rpcHitHandle_.isValid())
			continue;                                   // Skip RPC chambers, they are taken care of below)

		reco::MuonChamberMatch matchedChamber;

		const auto &lErr = chamber.tState.localError();
		const auto &lPos = chamber.tState.localPosition();
		const auto &lDir = chamber.tState.localDirection();
		const auto &localError = lErr.positionError();

		matchedChamber.x = lPos.x();
		matchedChamber.y = lPos.y();
		matchedChamber.xErr = sqrt(localError.xx());
		matchedChamber.yErr = sqrt(localError.yy());
		matchedChamber.dXdZ = lDir.z() != 0 ? lDir.x() / lDir.z() : 9999;
		matchedChamber.dYdZ = lDir.z() != 0 ? lDir.y() / lDir.z() : 9999;

		// DANGEROUS - compiler cannot guaranty parameters ordering
		AlgebraicSymMatrix55 trajectoryCovMatrix = lErr.matrix();
		matchedChamber.dXdZErr = trajectoryCovMatrix(1, 1) > 0 ? sqrt(trajectoryCovMatrix(1, 1)) : 0;
		matchedChamber.dYdZErr = trajectoryCovMatrix(2, 2) > 0 ? sqrt(trajectoryCovMatrix(2, 2)) : 0;

		matchedChamber.edgeX = chamber.localDistanceX;
		matchedChamber.edgeY = chamber.localDistanceY;

		matchedChamber.id = chamber.id;

		// DT chamber
		if (matchedChamber.detector() == MuonSubdetId::DT)
		{
			double xTrack = matchedChamber.x;

			for (int sl = 1; sl <= DTChamberId::maxSuperLayerId; sl += 2)
			{
				for (int layer = 1; layer <= DTChamberId::maxLayerId; ++layer)
				{
					const DTLayerId layerId(DTChamberId(matchedChamber.id.rawId()), sl, layer);
					auto range = m_dtDigis->get(layerId);

					for (auto digiIt = range.first; digiIt != range.second; ++digiIt)
					{
						const auto topo = m_dtGeometry->layer(layerId)->specificTopology();
						double xWire = topo.wirePosition((*digiIt).wire());
						double dX = std::abs(xWire - xTrack);

						if (dX < m_digiMaxDistanceX)
						{
							vetoHits.push_back(matchedChamber.id.rawId());
						}
					}
				}
			}
		}

		else if (matchedChamber.detector() == MuonSubdetId::CSC)
		{
			double xTrack = matchedChamber.x;
			double yTrack = matchedChamber.y;

			for (int iLayer = 1; iLayer <= CSCDetId::maxLayerId(); ++iLayer)
			{
				const CSCDetId chId(matchedChamber.id.rawId());
				const CSCDetId layerId(chId.endcap(), chId.station(), chId.ring(), chId.chamber(), iLayer);
				auto range = m_cscDigis->get(layerId);

				for (auto digiIt = range.first; digiIt != range.second; ++digiIt)
				{
					std::vector<int> adcVals = digiIt->getADCCounts();
					bool hasFired = false;
					float pedestal = 0.5 * (float)(adcVals[0] + adcVals[1]);
					float threshold = 13.3;
					float diff = 0.;
					for (const auto &adcVal : adcVals)
					{
						diff = (float)adcVal - pedestal;
						if (diff > threshold)
						{
							hasFired = true;
							break;
						}
					}

					if (!hasFired)
						continue;

					const CSCLayerGeometry *layerGeom = m_cscGeometry->layer(layerId)->geometry();
					Float_t xStrip = layerGeom->xOfStrip(digiIt->getStrip(), yTrack);
					float dX = std::abs(xStrip - xTrack);

					if (dX < m_digiMaxDistanceX)
					{
						vetoHits.push_back(matchedChamber.id.rawId());
					}
				}
			}
		}
	}

	// std::cout << "END CUSTOM MUON CLEANING" << std::endl;

	//-----------------
  }
}

  sort( vetoHits.begin(), vetoHits.end() );
  vetoHits.erase( unique( vetoHits.begin(), vetoHits.end() ), vetoHits.end() );

  // Now this can also handle different instance
  for (auto input_ : inputs_) {
    // Second read in the RecHit Colltection which is to be replaced, without the vetoRecHits
    typedef edm::Handle<RecHitCollection> RecHitCollectionHandle;
    RecHitCollectionHandle RecHitinput;
    iEvent.getByToken(input_.second, RecHitinput);
    for (typename RecHitCollection::const_iterator recHit = RecHitinput->begin(); recHit != RecHitinput->end();
         ++recHit) {  // loop over the basic rec hit collection (DT CSC or RPC)
      //if (find(vetoHits.begin(),vetoHits.end(),getRawDetId(*recHit)) == vetoHits.end()) continue; // For the invertec selcetion
      if (find(vetoHits.begin(), vetoHits.end(), getRawDetId(*recHit)) != vetoHits.end())
        continue;  // If the hit is not in the
      T1 detId(getRawDetId(*recHit));
      recHits_output[detId].push_back(*recHit);
    }

    // Last step savet the output in the CMSSW Data Format
    std::unique_ptr<RecHitCollection> output(new RecHitCollection());
    for (typename std::map<T1, std::vector<T2> >::const_iterator recHit = recHits_output.begin();
         recHit != recHits_output.end();
         ++recHit) {
      output->put(recHit->first, recHit->second.begin(), recHit->second.end());
    }
    output->post_insert();
    iEvent.put(std::move(output), input_.first);
  }
}

template <typename T1, typename T2>
void MuonDetCleaner<T1, T2>::fillVetoHits(const TrackingRecHit& rh, std::vector<uint32_t>* HitsList) {
  std::vector<const TrackingRecHit*> rh_components = rh.recHits();
  if (rh_components.empty()) {
    HitsList->push_back(rh.rawId());
  } else {
    for (std::vector<const TrackingRecHit*>::const_iterator rh_component = rh_components.begin();
         rh_component != rh_components.end();
         ++rh_component) {
      fillVetoHits(**rh_component, HitsList);
    }
  }
}

#endif
