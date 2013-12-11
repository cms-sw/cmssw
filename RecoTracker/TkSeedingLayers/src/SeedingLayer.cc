#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "HitExtractor.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

using namespace ctfseeding;
using namespace std;


class SeedingLayer::SeedingLayerImpl {
public:
  SeedingLayerImpl(
                const std::string & name, int seqNum,
                GeomDetEnumerators::SubDetector subdet,
                Side side,
                int layerId,
                const std::string& hitBuilderName,
                const HitExtractor * hitExtractor)
  : theName(name),
    theSeqNum(seqNum),
    theSubdet(subdet),
    theSide(side),
    theLayerId(layerId),
    theTTRHBuilderName(hitBuilderName),
    theHitExtractor(hitExtractor),
    theHasPredefinedHitErrors(false),thePredefinedHitErrorRZ(0.),thePredefinedHitErrorRPhi(0.) { }

  SeedingLayerImpl(
    const string & name, int seqNum,
    GeomDetEnumerators::SubDetector subdet,
    Side side,
    int layerId,
    const std::string& hitBuilderName,
    const HitExtractor * hitExtractor,
    float hitErrorRZ, float hitErrorRPhi)
  : theName(name), theSeqNum(seqNum),
    theSubdet(subdet),
    theSide(side),
    theLayerId(layerId),
    theTTRHBuilderName(hitBuilderName),
    theHitExtractor(hitExtractor),
    theHasPredefinedHitErrors(true),
    thePredefinedHitErrorRZ(hitErrorRZ), thePredefinedHitErrorRPhi(hitErrorRPhi) { }

  ~SeedingLayerImpl() { delete theHitExtractor; }

  SeedingLayer::Hits hits(const edm::Event& ev, 
			  const edm::EventSetup& es) const { return theHitExtractor->hits(hitBuilder(es),ev,es);  }

  std::string name() const { return theName; }

  int seqNum() const { return theSeqNum; }

  const DetLayer*  detLayer(const edm::EventSetup& es) const {
    edm::ESHandle<GeometricSearchTracker> tracker;
    es.get<TrackerRecoGeometryRecord>().get( tracker );

    const int index = theLayerId-1;
    if (theSubdet == GeomDetEnumerators::PixelBarrel) {
      return tracker->barrelLayers()[index];
    }
    else if (theSubdet == GeomDetEnumerators::PixelEndcap) {
      if (theSide == SeedingLayer::PosEndcap) {
        return tracker->posForwardLayers()[index];
      } else {
        return tracker->negForwardLayers()[index];
      }
    }
    else if (theSubdet == GeomDetEnumerators::TIB) {
      return tracker->tibLayers()[index];
    }
    else if (theSubdet == GeomDetEnumerators::TID) {
      if (theSide == SeedingLayer::PosEndcap) {
        return tracker->posTidLayers()[index];
      } else {
        return tracker->negTidLayers()[index];
      }
    }
    else if (theSubdet == GeomDetEnumerators::TOB) {
      return tracker->tobLayers()[index];
    }
    else if (theSubdet == GeomDetEnumerators::TEC) {
      if (theSide == SeedingLayer::PosEndcap) {
        return tracker->posTecLayers()[index];
      } else {
        return tracker->negTecLayers()[index];
      }
    }

    return nullptr;
  }
  const TransientTrackingRecHitBuilder * hitBuilder(const edm::EventSetup& es) const {
    edm::ESHandle<TransientTrackingRecHitBuilder> builder;
    es.get<TransientRecHitRecord>().get(theTTRHBuilderName, builder);
    return builder.product();
  }

  bool  hasPredefinedHitErrors() const { return theHasPredefinedHitErrors; }
  float predefinedHitErrorRZ() const { return thePredefinedHitErrorRZ; }
  float predefinedHitErrorRPhi() const { return thePredefinedHitErrorRPhi; }

private:
  SeedingLayerImpl(const SeedingLayerImpl &);

private:
  std::string theName;
  int theSeqNum;
  const GeomDetEnumerators::SubDetector theSubdet;
  const Side theSide;
  const int theLayerId;
  const std::string theTTRHBuilderName;
  const HitExtractor * theHitExtractor;
  bool theHasPredefinedHitErrors;
  float thePredefinedHitErrorRZ, thePredefinedHitErrorRPhi;
};




SeedingLayer::SeedingLayer( 
    const std::string & name, int seqNum,
    GeomDetEnumerators::SubDetector subdet,
    Side side,
    int layerId,
    const std::string& hitBuilderName,
    const HitExtractor * hitExtractor,
    bool usePredefinedErrors, float hitErrorRZ, float hitErrorRPhi)
{
  SeedingLayerImpl * l = usePredefinedErrors ? 
      new SeedingLayerImpl(name,seqNum,subdet,side,layerId,hitBuilderName,hitExtractor,hitErrorRZ,hitErrorRPhi)
    : new SeedingLayerImpl(name,seqNum,subdet,side,layerId,hitBuilderName,hitExtractor);
  theImpl = boost::shared_ptr<SeedingLayerImpl> (l);
}

std::string SeedingLayer::name() const
{
  return theImpl->name();
}

int SeedingLayer::seqNum() const
{
  return theImpl->seqNum();
}

const DetLayer*  SeedingLayer::detLayer(const edm::EventSetup& es) const
{
  return theImpl->detLayer(es);
}

const TransientTrackingRecHitBuilder * SeedingLayer::hitBuilder(const edm::EventSetup& es) const 
{
  return theImpl->hitBuilder(es);
}

SeedingLayer::Hits SeedingLayer::hits(const edm::Event& ev, const edm::EventSetup& es) const
{
  return  theImpl->hits(ev,es);
}

bool SeedingLayer::hasPredefinedHitErrors() const 
{
  return theImpl->hasPredefinedHitErrors();
}

float SeedingLayer::predefinedHitErrorRZ() const
{
  return theImpl->predefinedHitErrorRZ();
}

float SeedingLayer::predefinedHitErrorRPhi() const
{
  return theImpl->predefinedHitErrorRPhi();
}
