// Writes one line per event and digi collection: run, lumi, event, label, number of digis,
// and a 64-bit FNV-1a hash of all DetIds and samples. With dumpCells, it also writes one line per cell.
// Two jobs are compared by joining the lines on (run, lumi, event), not on the order of the events.

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace {
  constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
  constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;

  std::uint64_t fnv1a(std::uint64_t hash, std::uint32_t value) {
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= (value >> (8 * byte)) & 0xff;
      hash *= kFnvPrime;
    }
    return hash;
  }
}  // namespace

class HGCalDigiDigest : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalDigiDigest(edm::ParameterSet const& pset)
      : dumpCells_(pset.getUntrackedParameter<bool>("dumpCells")),
        output_(pset.getUntrackedParameter<std::string>("fileName")) {
    for (auto const& tag : pset.getParameter<std::vector<edm::InputTag>>("digis")) {
      labels_.push_back(tag.encode());
      tokens_.push_back(consumes<HGCalDigiCollection>(tag));
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<edm::InputTag>>("digis",
                                         {edm::InputTag("simHGCalUnsuppressedDigis", "EE"),
                                          edm::InputTag("simHGCalUnsuppressedDigis", "HEfront"),
                                          edm::InputTag("simHGCalUnsuppressedDigis", "HEback")});
    desc.addUntracked<std::string>("fileName", "hgcalDigiDigest.txt");
    desc.addUntracked<bool>("dumpCells", false);
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(edm::Event const& event, edm::EventSetup const&) override {
    for (std::size_t i = 0; i < tokens_.size(); ++i) {
      auto const& digis = event.get(tokens_[i]);
      std::uint64_t hash = kFnvOffset;
      for (auto const& frame : digis) {
        hash = fnv1a(hash, frame.id().rawId());
        if (dumpCells_) {
          output_ << "CELL " << event.id().run() << " " << event.luminosityBlock() << " " << event.id().event() << " "
                  << labels_[i] << " " << frame.id().rawId();
        }
        for (int sample = 0; sample < frame.size(); ++sample) {
          hash = fnv1a(hash, frame[sample].raw());
          if (dumpCells_) {
            output_ << " " << frame[sample].raw();
          }
        }
        if (dumpCells_) {
          output_ << "\n";
        }
      }
      output_ << "DIGEST " << event.id().run() << " " << event.luminosityBlock() << " " << event.id().event() << " "
              << labels_[i] << " " << digis.size() << " " << std::hex << hash << std::dec << "\n";
    }
  }

private:
  bool const dumpCells_;
  std::ofstream output_;
  std::vector<std::string> labels_;
  std::vector<edm::EDGetTokenT<HGCalDigiCollection>> tokens_;
};

DEFINE_FWK_MODULE(HGCalDigiDigest);
