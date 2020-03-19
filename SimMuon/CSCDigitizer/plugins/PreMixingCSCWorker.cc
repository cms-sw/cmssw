#include "FWCore/Framework/interface/ProducesCollector.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingMuonWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"

// Specialize put for CSC strip
template <>
void PreMixingMuonWorker<CSCStripDigiCollection>::put(edm::Event &iEvent) {
  auto merged = std::make_unique<CSCStripDigiCollection>();
  for (const auto &elem : *accumulated_) {
    // The layerId
    const CSCDetId &layerId = elem.first;

    // Get the iterators over the digis associated with this LayerId
    const CSCStripDigiCollection::Range &range = elem.second;

    std::vector<CSCStripDigi> NewDigiList;

    std::vector<int> StripList;
    std::vector<CSCStripDigiCollection::const_iterator> StripPointer;

    for (CSCStripDigiCollection::const_iterator dtdigi = range.first; dtdigi != range.second; ++dtdigi) {
      StripList.push_back((*dtdigi).getStrip());
      StripPointer.push_back(dtdigi);
    }

    int PrevStrip = -1;
    std::vector<int> DuplicateList;

    std::vector<CSCStripDigiCollection::const_iterator>::const_iterator StripPtr = StripPointer.begin();

    for (std::vector<int>::const_iterator istrip = StripList.begin(); istrip != StripList.end(); ++istrip) {
      const int CurrentStrip = *(istrip);

      if (CurrentStrip > PrevStrip) {
        PrevStrip = CurrentStrip;

        int dupl_count;
        dupl_count = std::count(StripList.begin(), StripList.end(), CurrentStrip);
        if (dupl_count > 1) {
          std::vector<int>::const_iterator duplicate = istrip;
          ++duplicate;
          std::vector<CSCStripDigiCollection::const_iterator>::const_iterator DuplPointer = StripPtr;
          ++DuplPointer;
          for (; duplicate != StripList.end(); ++duplicate) {
            if ((*duplicate) == CurrentStrip) {
              DuplicateList.push_back(CurrentStrip);

              std::vector<int> pileup_adc = (**DuplPointer).getADCCounts();
              std::vector<int> signal_adc = (**StripPtr).getADCCounts();

              std::vector<int>::const_iterator minplace;

              minplace = std::min_element(pileup_adc.begin(), pileup_adc.end());

              int minvalue = (*minplace);

              std::vector<int> new_adc;

              std::vector<int>::const_iterator newsig = signal_adc.begin();

              for (std::vector<int>::const_iterator ibin = pileup_adc.begin(); ibin != pileup_adc.end(); ++ibin) {
                new_adc.push_back((*newsig) + (*ibin) - minvalue);

                ++newsig;
              }

              CSCStripDigi newDigi(CurrentStrip, new_adc);
              NewDigiList.push_back(newDigi);
            }
            ++DuplPointer;
          }
        } else {
          NewDigiList.push_back(**StripPtr);
        }
      }                    // if strips monotonically increasing...  Haven't hit duplicates yet
      else {               // reached end of signal digis, or there was no overlap
        PrevStrip = 1000;  // now into pileup signals, stop looking forward for
                           // duplicates

        // check if this digi was in the duplicate list
        int check;
        check = std::count(DuplicateList.begin(), DuplicateList.end(), CurrentStrip);
        if (check == 0)
          NewDigiList.push_back(**StripPtr);
      }
      ++StripPtr;
    }

    CSCStripDigiCollection::Range stripRange(NewDigiList.begin(), NewDigiList.end());

    merged->put(stripRange, layerId);
  }

  iEvent.put(std::move(merged), collectionDM_);
  accumulated_.reset();
}

// CSC has three digi collections
class PreMixingCSCWorker : public PreMixingWorker {
public:
  PreMixingCSCWorker(const edm::ParameterSet &ps, edm::ProducesCollector producesCollector, edm::ConsumesCollector &&iC)
      : stripWorker_(ps.getParameter<edm::ParameterSet>("strip"), producesCollector, iC),
        wireWorker_(ps.getParameter<edm::ParameterSet>("wire"), producesCollector, iC),
        comparatorWorker_(ps.getParameter<edm::ParameterSet>("comparator"), producesCollector, iC) {}
  ~PreMixingCSCWorker() override = default;

  void initializeEvent(edm::Event const &iEvent, edm::EventSetup const &iSetup) override {}

  void addSignals(edm::Event const &iEvent, edm::EventSetup const &iSetup) override {
    stripWorker_.addSignals(iEvent, iSetup);
    wireWorker_.addSignals(iEvent, iSetup);
    comparatorWorker_.addSignals(iEvent, iSetup);
  }

  void addPileups(PileUpEventPrincipal const &pep, edm::EventSetup const &iSetup) override {
    stripWorker_.addPileups(pep, iSetup);
    wireWorker_.addPileups(pep, iSetup);
    comparatorWorker_.addPileups(pep, iSetup);
  }

  void put(edm::Event &iEvent,
           edm::EventSetup const &iSetup,
           std::vector<PileupSummaryInfo> const &ps,
           int bunchSpacing) override {
    stripWorker_.put(iEvent);
    wireWorker_.put(iEvent);
    comparatorWorker_.put(iEvent);
  }

private:
  PreMixingMuonWorker<CSCStripDigiCollection> stripWorker_;
  PreMixingMuonWorker<CSCWireDigiCollection> wireWorker_;
  PreMixingMuonWorker<CSCComparatorDigiCollection> comparatorWorker_;
};

DEFINE_PREMIXING_WORKER(PreMixingCSCWorker);
