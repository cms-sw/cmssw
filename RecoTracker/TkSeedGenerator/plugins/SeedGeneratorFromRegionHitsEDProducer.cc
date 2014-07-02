#include "SeedGeneratorFromRegionHitsEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"
#include "RecoPixelVertexing/PixelTriplets/interface/QuadrupletSeedMerger.h"


SeedGeneratorFromRegionHitsEDProducer::SeedGeneratorFromRegionHitsEDProducer(
    const edm::ParameterSet& cfg) 
  : theRegionProducer(nullptr),
    theClusterCheck(cfg.getParameter<edm::ParameterSet>("ClusterCheckPSet"),consumesCollector()),
    theMerger_(nullptr)
{
  theSilentOnClusterCheck = cfg.getParameter<edm::ParameterSet>("ClusterCheckPSet").getUntrackedParameter<bool>("silentClusterCheck",false);

  moduleName = cfg.getParameter<std::string>("@module_label");

  edm::ParameterSet creatorPSet =
      cfg.getParameter<edm::ParameterSet>("SeedCreatorPSet");

  // seed merger & its settings
  edm::ConsumesCollector iC = consumesCollector();
  if ( cfg.exists("SeedMergerPSet")) {
    edm::ParameterSet mergerPSet = cfg.getParameter<edm::ParameterSet>( "SeedMergerPSet" );
    theMerger_.reset(new QuadrupletSeedMerger(mergerPSet.getParameter<edm::ParameterSet>( "layerList" ), creatorPSet, iC));
    theMerger_->setTTRHBuilderLabel( mergerPSet.getParameter<std::string>( "ttrhBuilderLabel" ) );
    theMerger_->setMergeTriplets( mergerPSet.getParameter<bool>( "mergeTriplets" ) );
    theMerger_->setAddRemainingTriplets( mergerPSet.getParameter<bool>( "addRemainingTriplets" ) );
  }

  edm::ParameterSet regfactoryPSet = 
      cfg.getParameter<edm::ParameterSet>("RegionFactoryPSet");
  std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  theRegionProducer.reset(TrackingRegionProducerFactory::get()->create(regfactoryName,regfactoryPSet, consumesCollector()));

  edm::ParameterSet hitsfactoryPSet =
      cfg.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string hitsfactoryName = hitsfactoryPSet.getParameter<std::string>("ComponentName");
  OrderedHitsGenerator*  hitsGenerator =
    OrderedHitsGeneratorFactory::get()->create( hitsfactoryName, hitsfactoryPSet, iC);

  edm::ParameterSet comparitorPSet =
      cfg.getParameter<edm::ParameterSet>("SeedComparitorPSet");
  std::string comparitorName = comparitorPSet.getParameter<std::string>("ComponentName");
  SeedComparitor * aComparitor = (comparitorName == "none") ?
      0 :  SeedComparitorFactory::get()->create( comparitorName, comparitorPSet, iC);

  std::string creatorName = creatorPSet.getParameter<std::string>("ComponentName");
  SeedCreator * aCreator = SeedCreatorFactory::get()->create( creatorName, creatorPSet);

  theGenerator.reset(new SeedGeneratorFromRegionHits(hitsGenerator, aComparitor, aCreator));

  produces<TrajectorySeedCollection>();
}

SeedGeneratorFromRegionHitsEDProducer::~SeedGeneratorFromRegionHitsEDProducer()
{
}

void SeedGeneratorFromRegionHitsEDProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  std::auto_ptr<TrajectorySeedCollection> triplets(new TrajectorySeedCollection());
  std::auto_ptr<TrajectorySeedCollection> quadruplets( new TrajectorySeedCollection() );

  //protection for big ass events...
  size_t clustsOrZero = theClusterCheck.tooManyClusters(ev);
  if (clustsOrZero){
    if (!theSilentOnClusterCheck)
	edm::LogError("TooManyClusters") << "Found too many clusters (" << clustsOrZero << "), bailing out.\n";
    ev.put(triplets);
    return ;
  }

  typedef std::vector<TrackingRegion* > Regions;
  typedef Regions::const_iterator IR;
  Regions regions = theRegionProducer->regions(ev,es);
  if (theMerger_)
    theMerger_->update(es);

  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) {
    const TrackingRegion & region = **ir;

    // make job
    theGenerator->run(*triplets, region, ev,es);
    // std::cout << "created seeds for " << moduleName << " " << triplets->size() << std::endl;


    // make quadruplets
    // (TODO: can partly be propagated to the merger)
    if ( theMerger_ ) {
      TrajectorySeedCollection const& tempQuads = theMerger_->mergeTriplets( *triplets, region, es); //@@
      for( TrajectorySeedCollection::const_iterator qIt = tempQuads.begin(); qIt < tempQuads.end(); ++qIt ) {
	quadruplets->push_back( *qIt );
      }
    }
  }

  // clear memory
  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) delete (*ir);

  // put to event
  if ( theMerger_)
    ev.put(quadruplets);
  else
    ev.put(triplets);
}
