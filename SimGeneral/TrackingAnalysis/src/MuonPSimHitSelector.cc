
#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "SimGeneral/TrackingAnalysis/interface/MuonPSimHitSelector.h"

// #include "SimTracker/Common/interface/SimHitSelectorFromDB.h"

void MuonPSimHitSelector::select(PSimHitCollection &selection,
                                 edm::Event const &event,
                                 edm::EventSetup const &setup) const {
  // Look for psimhit collection associated to the muon system
  PSimHitCollectionMap::const_iterator pSimHitCollections = pSimHitCollectionMap_.find("muon");

  // Check that there are psimhit collections defined for the tracker
  if (pSimHitCollections == pSimHitCollectionMap_.end())
    return;

  // Grab all the PSimHit from the different sencitive volumes
  edm::Handle<CrossingFrame<PSimHit>> cfPSimHits;
  std::vector<const CrossingFrame<PSimHit> *> cfPSimHitProductPointers;

  // Collect the product pointers to the different psimhit collection
  for (std::size_t i = 0; i < pSimHitCollections->second.size(); ++i) {
    event.getByLabel("mix", pSimHitCollections->second[i], cfPSimHits);
    cfPSimHitProductPointers.push_back(cfPSimHits.product());
  }

  // Create a mix collection from the different psimhit collections
  std::unique_ptr<MixCollection<PSimHit>> pSimHits(new MixCollection<PSimHit>(cfPSimHitProductPointers));

  // Get CSC Bad Chambers (ME4/2)
  edm::ESHandle<CSCBadChambers> cscBadChambers;
  setup.get<CSCBadChambersRcd>().get(cscBadChambers);

  // Select only psimhits from alive modules
  for (MixCollection<PSimHit>::MixItr pSimHit = pSimHits->begin(); pSimHit != pSimHits->end(); ++pSimHit) {
    DetId dId = DetId(pSimHit->detUnitId());

    if (dId.det() == DetId::Muon && dId.subdetId() == MuonSubdetId::CSC) {
      if (!cscBadChambers->isInBadChamber(CSCDetId(dId)))
        selection.push_back(*pSimHit);
    } else
      selection.push_back(*pSimHit);
  }
}
