

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "SimGeneral/TrackingAnalysis/interface/TrackerPSimHitSelector.h"

#include "SimTracker/Common/interface/SimHitSelectorFromDB.h"


void TrackerPSimHitSelector::select(PSimHitCollection & selection, edm::Event const & event, edm::EventSetup const & setup) const
{
    // Look for psimhit collection associated o the tracker
    PSimHitCollectionMap::const_iterator pSimHitCollections = pSimHitCollectionMap_.find("tracker");

    // Check that there are psimhit collections defined for the tracker
    if (pSimHitCollections == pSimHitCollectionMap_.end()) return;

    // Grab all the PSimHit from the different sencitive volumes
    edm::Handle<CrossingFrame<PSimHit> > cfPSimHits;
    std::vector<const CrossingFrame<PSimHit> *> cfPSimHitProductPointers;

    // Collect the product pointers to the different psimhit collection
    for (std::size_t i = 0; i < pSimHitCollections->second.size(); ++i)
    {
      event.getByLabel(mixLabel_, pSimHitCollections->second[i], cfPSimHits);
      cfPSimHitProductPointers.push_back(cfPSimHits.product());
    }

    // Create a mix collection from the different psimhit collections
    std::auto_ptr<MixCollection<PSimHit> > pSimHits(new MixCollection<PSimHit>(cfPSimHitProductPointers));

    // Setup the cabling mapping
    std::map<uint32_t, std::vector<int> > theDetIdList;
    edm::ESHandle<SiStripDetCabling> detCabling;
    setup.get<SiStripDetCablingRcd>().get( detCabling );
    detCabling->addConnected(theDetIdList);

    // Select only psimhits from alive modules
    std::vector<std::pair<const PSimHit*,int> > psimhits(SimHitSelectorFromDB().getSimHit(pSimHits, theDetIdList));

    // Add the selected psimhit to the main list
    for (std::size_t i = 0; i < psimhits.size(); ++i)
        selection.push_back( *(const_cast<PSimHit*>(psimhits[i].first)) );
}
