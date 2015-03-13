#include "SeedGeneratorFromRegionHitsEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

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
  triplets->shrink_to_fit();
  quadruplets->shrink_to_fit();

  // clear memory
  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) delete (*ir);

  // put to event
  if ( theMerger_)
    ev.put(quadruplets);
  else
    ev.put(triplets);
}

void
SeedGeneratorFromRegionHitsEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;

    psd0.add<std::string>("ComponentName","GlobalRegionProducerFromBeamSpot");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<edm::InputTag>("beamSpot",edm::InputTag("offlineBeamSpot"));
      psd1.add<bool>("precise",true);
      psd1.add<double>("ptMin",0.9);
      psd1.add<double>("originRadius",0.2);
      psd1.add<double>("originHalfLength",21.2);

      psd0.add<edm::ParameterSetDescription>("RegionPSet",psd1);
    }
    desc.add<edm::ParameterSetDescription>("RegionFactoryPSet",psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("ComponentName","none");
    desc.add<edm::ParameterSetDescription>("SeedComparitorPSet",psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<edm::InputTag>("PixelClusterCollectionLabel",edm::InputTag("siPixelClusters"));
    psd0.add<edm::InputTag>("ClusterCollectionLabel",edm::InputTag("siStripClusters"));
    psd0.add<unsigned int>("MaxNumberOfPixelClusters",40000);
    psd0.add<unsigned int>("MaxNumberOfCosmicClusters",400000);
    psd0.add<std::string>("cut","strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)");
    psd0.add<bool>("doClusterCheck",true);
    desc.add<edm::ParameterSetDescription>("ClusterCheckPSet",psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<edm::InputTag>("SeedingLayers",edm::InputTag(""));
    psd0.add<std::string>("ComponentName","");
    psd0.add<unsigned int>("maxElement",1000000);
    desc.add<edm::ParameterSetDescription>("OrderedHitsFactoryPSet",psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("ComponentName","SeedFromConsecutiveHitsCreator");
    psd0.add<std::string>("SimpleMagneticField","ParabolicMf");
    psd0.add<std::string>("propagator","PropagatorWithMaterial");
    psd0.add<std::string>("TTRHBuilder","WithTrackAngle");
    psd0.add<double>("SeedMomentumForBOFF",5.0);
    psd0.add<double>("MinOneOverPtError",1.0);
    psd0.add<double>("OriginTransverseErrorMultiplier",1.0);
    desc.add<edm::ParameterSetDescription>("SeedCreatorPSet",psd0);
  }

  descriptions.add("seedGeneratorFromRegionHitsEDProducer",desc);
  descriptions.setComment("");
}
