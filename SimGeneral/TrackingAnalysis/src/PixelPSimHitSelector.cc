
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "SimGeneral/TrackingAnalysis/interface/PixelPSimHitSelector.h"


void PixelPSimHitSelector::select(PSimHitCollection & selection, edm::Event const & event, edm::EventSetup const & setup) const
{
    // Look for psimhit collection associated o the tracker
    PSimHitCollectionMap::const_iterator pSimHitCollections = pSimHitCollectionMap_.find("pixel");

    // Check that there are psimhit collections defined for the tracker
    if (pSimHitCollections == pSimHitCollectionMap_.end()) return;

    // Grab all the PSimHit from the different sencitive volumes
    edm::Handle<CrossingFrame<PSimHit> > cfPSimHits;
    std::vector<const CrossingFrame<PSimHit> *> cfPSimHitProductPointers;

    // Collect the product pointers to the different psimhit collection
    for (std::size_t i = 0; i < pSimHitCollections->second.size(); ++i)
    {
        event.getByLabel("mix", pSimHitCollections->second[i], cfPSimHits);
        cfPSimHitProductPointers.push_back(cfPSimHits.product());
    }

    // Create a mix collection from the different psimhit collections
    std::auto_ptr<MixCollection<PSimHit> > pSimHits( new MixCollection<PSimHit>(cfPSimHitProductPointers) );

    // Accessing dead pixel modules from DB:
    edm::ESHandle<SiPixelQuality> siPixelBadModule;
    setup.get<SiPixelQualityRcd>().get(siPixelBadModule);

    // Reading the DB information
    std::vector<SiPixelQuality::disabledModuleType> badModules( siPixelBadModule->getBadComponentList() );
    SiPixelQuality pixelQuality(badModules);

    // Select only psimhits from alive modules
    for (MixCollection<PSimHit>::MixItr pSimHit = pSimHits->begin(); pSimHit != pSimHits->end(); ++pSimHit)
    {
        if ( !pixelQuality.IsModuleBad(pSimHit->detUnitId()) )
            selection.push_back(*pSimHit);
    }
}
