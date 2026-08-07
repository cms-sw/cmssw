#include <alpaka/alpaka.hpp>

#include <numeric>

#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "CAHitNtupletGenerator.h"

// #define GPU_DEBUG
// #define NTRACKS_DEBUG
// #define DUPLICATE_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelTracksSoAMerger : public global::EDProducer<> {
    using Algo = CAHitMaskingAndMerger;

  public:
    explicit PixelTracksSoAMerger(const edm::ParameterSet& iConfig);
    ~PixelTracksSoAMerger() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    pixelTrack::Quality const minQuality_;
    double const matchFraction_;

    std::vector<device::EDGetToken<reco::TracksSoACollection>> inputTkSoATokenV_;
    std::vector<edm::InputTag> inputTkSoATagV_;

    const device::EDPutToken<reco::TracksSoACollection> outputTkSoAToken_;

    Algo deviceAlgo_;
  };

  PixelTracksSoAMerger::PixelTracksSoAMerger(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
        matchFraction_(iConfig.getParameter<double>("matchFraction")),
        inputTkSoATagV_(iConfig.getParameter<std::vector<edm::InputTag>>("inputTkSoAs")),
        outputTkSoAToken_(produces()) {
    for (const auto& it : inputTkSoATagV_) {
      inputTkSoATokenV_.push_back(consumes(it));
    }
    if (minQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
    }
    if (minQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " not supported";
    }
  }

  void PixelTracksSoAMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<std::vector<edm::InputTag>>(
        "inputTkSoAs", {edm::InputTag("pixelTracksHighPtAlpaka"), edm::InputTag("pixelTracksLowPtAlpaka")});
    desc.add<std::string>("minQuality", "highPurity");
    desc.add<double>("matchFraction", 0.0);

    descriptions.addWithDefaultLabel(desc);
  }

  namespace {
    // This utility unrolls the SoA columns (tuples) at compile time, calling the provided functor 'f'
    // once for each element. The index is passed as a std::integral_constant so it
    // is available at compile time.
    template <typename F, std::size_t... Is>
    void unrollColumns(F&& f, std::index_sequence<Is...>) {
      (f(std::integral_constant<std::size_t, Is>{}), ...);
    }
    // User-facing wrapper to deduce the size of the tuple and create the index sequence
    // Usage: mergeSoAColumns<NumberOfColumns>([&](auto columnIndex) { ... });
    template <std::size_t N, typename F>
    void mergeSoAColumns(F&& f) {
      unrollColumns(std::forward<F>(f), std::make_index_sequence<N>{});
    }
  }  // namespace

  void PixelTracksSoAMerger::produce(edm::StreamID streamID,
                                     device::Event& iEvent,
                                     const device::EventSetup& es) const {
    // get both Pixel and Tracker SoA collections
    auto queue = iEvent.queue();

    std::vector<const reco::TracksSoACollection*> inputTkSoAs;
    for (const auto& it : inputTkSoATokenV_) {
      auto const& aux = iEvent.get(it);
      inputTkSoAs.push_back(&aux);
    }

    // input SoA collections have the same layout
    // each of them is made up of two SoAs:
    // - one that contains the tracks
    // - one that contains the hits associated to the tracks
    // this code merges and copy them into a new SoA collection

    std::vector<int> nTks;
    std::vector<int> nHits;

    for (const auto& it : inputTkSoAs) {
      auto nTksAuxDev = it->view().tracks().metadata().size();
      auto nHitsAuxDev = it->view().trackHits().metadata().size();

      reco::TracksHost itHost(queue, nTksAuxDev, nHitsAuxDev);

      alpaka::memcpy(queue, itHost.buffer(), it->buffer());
      alpaka::wait(queue);

      int nTksAux = itHost.view().tracks().nTracks();

      nTks.push_back(nTksAux);

      int nHitsAux = 0;
      for (int i = 0; i < nTksAux; ++i) {
        nHitsAux += ::reco::nHits(itHost.view().tracks(), i);
      }

      nHits.push_back(nHitsAux);
    }

    std::vector<int> cumulNTks{0};
    std::vector<int> cumulNHits{0};

    int auxCumulNTks = 0;
    int auxCumulNHits = 0;

    for (int i = 0; i < int(nTks.size()); ++i) {
      auxCumulNTks = auxCumulNTks + nTks[i];
      auxCumulNHits = auxCumulNHits + nHits[i];
      cumulNTks.push_back(auxCumulNTks);
      cumulNHits.push_back(auxCumulNHits);
    }

    // the outputTemp is also a SoA collection with the same layout as the input ones
    auto outputTemp = reco::TracksSoACollection(
        queue, std::reduce(nTks.begin(), nTks.end()), std::reduce(nHits.begin(), nHits.end()));

#ifdef NTRACKS_DEBUG
    std::cout << "----------------- Merging Input Tracks -----------------\n";
    for (int i = 0; i < int(nTks.size()); ++i)
      std::cout << "Number of tracks input " << i + 1 << ": " << nTks[i] << '\n';
    std::cout << "Total number of tracks: " << outputTemp.view().tracks().metadata().size() << '\n';
    for (int i = 0; i < int(nHits.size()); ++i)
      std::cout << "Number of hits input " << i + 1 << ": " << nHits[i] << '\n';
    std::cout << "Total number of hits: " << outputTemp.view().trackHits().metadata().size() << '\n'
              << "---------------------------------------------------------------------\n";
#endif

    // start from the tracks SoA, use metarecords to loop over all the columns
    auto outView = outputTemp.view().tracks();

    // start a loop here over the input SoAs to be easier to access each object
    int nSoAsAux = 0;  // auxiliar index to correctly access nTks and nHits
    for (const auto& it : inputTkSoAs) {
      if (nTks[nSoAsAux] == 0) {
        nSoAsAux = nSoAsAux + 1;  // still need to increase the SoA position to correctly access the cumul vectors
        continue;
      }

      auto inpTkView = it->view().tracks();

      // auxiliar for correctly memcpy-ing eigen columns
      int nEigenAux = 5;

      // layout type (same for all views)
      using ViewType = decltype(outView);
      using LayoutType = typename ViewType::Metadata::TypeOf_Layout;

      // build descriptors (tuple of spans: one span for each column)
      auto outDesc = LayoutType::Descriptor(outView);
      auto inpTkDesc = LayoutType::ConstDescriptor(inpTkView);

      // merge all columns using a compile-time loop

      // number of columns (same for all hits SoAs)
      constexpr std::size_t N = std::tuple_size_v<decltype(outDesc.buff)>;
      mergeSoAColumns<N>([&](auto columnIndex) {
        auto& outCol = std::get<columnIndex>(outDesc.buff);
        const auto& inpTkCol = std::get<columnIndex>(inpTkDesc.buff);
        // distinguish between scalar and column types
        if constexpr (std::get<columnIndex>(outDesc.columnTypes) == cms::soa::SoAColumnType::scalar) {
          // scalar type, copy the value directly
          // for some reason this doesn't work on device, so it is done after the loop in the iterations finishes
          // alpaka::memcpy(queue,
          //                cms::alpakatools::make_device_view(queue, outCol.data(), 1),
          //                cms::alpakatools::make_device_view(queue, &nTotal, 1));

#ifdef GPU_DEBUG
          alpaka::wait(queue);
          std::cout << "Copied scalar with index " << columnIndex << '\n';
#endif
        } else if constexpr (std::get<columnIndex>(outDesc.columnTypes) == cms::soa::SoAColumnType::eigen) {
          for (int i = 0; i < nEigenAux; ++i) {
            // eigen column type, copy the whole column with the number of eigen elements
            alpaka::memcpy(
                queue,
                cms::alpakatools::make_device_view(
                    queue, outCol.data() + cumulNTks[nSoAsAux] + (i * (outCol.size() / nEigenAux)), nTks[nSoAsAux]),
                cms::alpakatools::make_device_view(
                    queue, inpTkCol.data() + (i * (inpTkCol.size() / nEigenAux)), nTks[nSoAsAux]));

#ifdef GPU_DEBUG
            alpaka::wait(queue);
            std::cout << "Copied eigen column with index " << columnIndex << ", " << i << '\n';
#endif
          }
          nEigenAux = nEigenAux + 10;
        } else {
          // column type, copy the whole column
          alpaka::memcpy(queue,
                         cms::alpakatools::make_device_view(queue, outCol.data() + cumulNTks[nSoAsAux], nTks[nSoAsAux]),
                         cms::alpakatools::make_device_view(queue, inpTkCol.data(), nTks[nSoAsAux]));
#ifdef GPU_DEBUG
          alpaka::wait(queue);
          std::cout << "Copied column with index " << columnIndex << '\n';
#endif
        }
      });

      // update outputTemp hitOffsets to take into account the previous SoAs
      deviceAlgo_.updateHitOffsets(
          cumulNTks[nSoAsAux], cumulNTks[nSoAsAux + 1], cumulNHits[nSoAsAux], outputTemp, queue);

      // copy track hits information
      alpaka::memcpy(queue,
                     cms::alpakatools::make_device_view(
                         queue, outputTemp.view().trackHits().id().data() + cumulNHits[nSoAsAux], nHits[nSoAsAux]),
                     cms::alpakatools::make_device_view(queue, it->view().trackHits().id().data(), nHits[nSoAsAux]));
      alpaka::memcpy(queue,
                     cms::alpakatools::make_device_view(
                         queue, outputTemp.view().trackHits().detId().data() + cumulNHits[nSoAsAux], nHits[nSoAsAux]),
                     cms::alpakatools::make_device_view(queue, it->view().trackHits().detId().data(), nHits[nSoAsAux]));
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Copied track hits\n";

#endif

      nSoAsAux = nSoAsAux + 1;
    }

    // correctly copy total of nTracks
    const int nTotal = cumulNTks[cumulNTks.size() - 1];
    alpaka::memcpy(queue,
                   cms::alpakatools::make_device_view(queue, outputTemp.view().tracks().nTracks()),
                   cms::alpakatools::make_host_view(nTotal));

#ifdef DUPLICATE_DEBUG

    // This block copies all input and output collections to host to print quantities and ensure that outN = inpN
    // "out1", "out2", ..., "outN" refer to the position of the input inside of the output collection, but there is only one output SoA
    // It only prints at most the first 10 tracks of each input collection, defined in std::min_element(nTks.begin(), nTks.end()))),10)
    // This is also useful to check for duplicate tracks in distinct input collections

    auto nTksOutDebug = outputTemp.view().tracks().metadata().size();
    auto nHitsOutDebug = outputTemp.view().trackHits().metadata().size();

    reco::TracksHost outputTempHost(queue, nTksOutDebug, nHitsOutDebug);

    alpaka::memcpy(queue, outputTempHost.buffer(), outputTemp.buffer());
    alpaka::wait(queue);

    auto outViewHost = outputTempHost.view().tracks();

    std::vector<reco::TracksHost> inputTkSoAsHost;

    for (int k = 0; k < int(cumulNTks.size()) - 1; ++k) {
      auto nTksInpDebug = inputTkSoAs[k]->view().tracks().metadata().size();
      auto nHitsInpDebug = inputTkSoAs[k]->view().trackHits().metadata().size();

      reco::TracksHost inputTempHost(queue, nTksInpDebug, nHitsInpDebug);

      alpaka::memcpy(queue, inputTempHost.buffer(), inputTkSoAs[k]->buffer());
      alpaka::wait(queue);

      inputTkSoAsHost.push_back(std::move(inputTempHost));
    }

    std::cout << "Number of tracks: " << outViewHost.nTracks() << std::endl;

    for (int i = 0; i < std::min(int(*(std::min_element(nTks.begin(), nTks.end()))), 10); ++i) {
      std::cout << "Number of tracks: " << outViewHost.nTracks() << std::endl;
      std::cout << "------------------------------------------------------------------------------------------"
                << std::endl;

      std::cout << "track quality (";
      for (int k = 1; k < int(cumulNTks.size()); ++k)
        std::cout << "inp" << k << " - out" << k << " -- ";
      std::cout << "): ";
      for (int k = 0; k < int(cumulNTks.size()) - 1; ++k)
        std::cout << pixelTrack::qualityName[int(inputTkSoAsHost[k].view().tracks()[i].quality())] << " - "
                  << pixelTrack::qualityName[int(outViewHost[i + cumulNTks[k]].quality())] << " -- ";
      std::cout << std::endl;
      std::cout << "------------------------------------------------------------------------------------------"
                << std::endl;

      std::cout << "track pt (";
      for (int k = 1; k < int(cumulNTks.size()); ++k)
        std::cout << "inp" << k << " - out" << k << " -- ";
      std::cout << "): ";
      for (int k = 0; k < int(cumulNTks.size()) - 1; ++k)
        std::cout << inputTkSoAsHost[k].view().tracks()[i].pt() << " - " << outViewHost[i + cumulNTks[k]].pt()
                  << " -- ";
      std::cout << std::endl;
      std::cout << "------------------------------------------------------------------------------------------"
                << std::endl;

      std::cout << "track eta (";
      for (int k = 1; k < int(cumulNTks.size()); ++k)
        std::cout << "inp" << k << " - out" << k << " -- ";
      std::cout << "): ";
      for (int k = 0; k < int(cumulNTks.size()) - 1; ++k)
        std::cout << inputTkSoAsHost[k].view().tracks()[i].eta() << " - " << outViewHost[i + cumulNTks[k]].eta()
                  << " -- ";
      std::cout << std::endl;
      std::cout << "------------------------------------------------------------------------------------------"
                << std::endl;

      for (int j = 0; j < 5; ++j) {
        std::cout << "track state " << j << "(";
        for (int k = 1; k < int(cumulNTks.size()); ++k)
          std::cout << "inp" << k << " - out" << k << " -- ";
        std::cout << "): ";
        for (int k = 0; k < int(cumulNTks.size()) - 1; ++k)
          std::cout << inputTkSoAsHost[k].view().tracks()[i].state()(j) << " - "
                    << outViewHost[i + cumulNTks[k]].state()(j) << " -- ";
        std::cout << std::endl;
      }
      std::cout << "------------------------------------------------------------------------------------------"
                << std::endl;

      for (int j = 0; j < 15; ++j) {
        std::cout << "track covariance " << j << "(";
        for (int k = 1; k < int(cumulNTks.size()); ++k)
          std::cout << "inp" << k << " - out" << k << " -- ";
        std::cout << "): ";
        for (int k = 0; k < int(cumulNTks.size()) - 1; ++k)
          std::cout << inputTkSoAsHost[k].view().tracks()[i].covariance()(j) << " - "
                    << outViewHost[i + cumulNTks[k]].covariance()(j) << " -- ";
        std::cout << std::endl;
      }
      std::cout << "=========================================================================================="
                << std::endl;
    }
#endif

    // calculate the total number of tracks and hits
    int totTracks = std::reduce(nTks.begin(), nTks.end());
    int totHits = std::reduce(nHits.begin(), nHits.end());

    // emplace the merged SoA collection in the event
    iEvent.emplace(outputTkSoAToken_,
                   deviceAlgo_.makeFilteredTracks(totTracks, totHits, outputTemp, minQuality_, matchFraction_, queue));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelTracksSoAMerger);
