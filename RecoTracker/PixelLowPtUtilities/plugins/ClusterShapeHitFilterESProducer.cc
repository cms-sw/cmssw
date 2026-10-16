#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelClusterShapeLimitsRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelClusterShapeLimits.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class ClusterShapeHitFilterESProducer : public edm::ESProducer {
public:
  ClusterShapeHitFilterESProducer(const edm::ParameterSet&);

  typedef std::unique_ptr<ClusterShapeHitFilter> ReturnType;
  ReturnType produce(const ClusterShapeHitFilter::Record&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // Retrieve magnetic field
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  // Retrieve geometry
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geoToken_;
  // Retrieve topology
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topolToken_;
  // Retrieve pixel Lorentz
  edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleRcd> pixelToken_;
  // Retrieve strip Lorentz
  edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleDepRcd> stripToken_;
  // Retrieve pixel cluster shape limits
  edm::ESGetToken<SiPixelClusterShapeLimits, SiPixelClusterShapeLimitsRcd> limitsToken_;

  const float minGoodPixelCharge_, minGoodStripCharge_;
  const bool isPhase2_;
  const bool cutOnPixelCharge_, cutOnStripCharge_;
  const bool cutOnPixelShape_, cutOnStripShape_;
};

/*****************************************************************************/
ClusterShapeHitFilterESProducer::ClusterShapeHitFilterESProducer(const edm::ParameterSet& iConfig)
    : minGoodPixelCharge_(0),
      minGoodStripCharge_(clusterChargeCut(iConfig)),
      isPhase2_(iConfig.getParameter<bool>("isPhase2")),
      cutOnPixelCharge_(false),
      cutOnStripCharge_(minGoodStripCharge_ > 0),
      cutOnPixelShape_(iConfig.getParameter<bool>("doPixelShapeCut")),
      cutOnStripShape_(iConfig.getParameter<bool>("doStripShapeCut")) {
  std::string componentName = iConfig.getParameter<std::string>("ComponentName");

  edm::LogInfo("ClusterShapeHitFilterESProducer") << " with name: " << componentName;

  auto cc = setWhatProduced(this, componentName);
  fieldToken_ = cc.consumes();
  geoToken_ = cc.consumes();
  topolToken_ = cc.consumes();
  pixelToken_ = cc.consumes();
  limitsToken_ = cc.consumes(edm::ESInputTag("", iConfig.getParameter<std::string>("pixelShapeLimitsLabel")));
  if (!isPhase2_) {
    stripToken_ = cc.consumes();
  }
}

/*****************************************************************************/
ClusterShapeHitFilterESProducer::ReturnType ClusterShapeHitFilterESProducer::produce(
    const ClusterShapeHitFilter::Record& iRecord) {
  using namespace edm::es;

  const SiStripLorentzAngle* theSiStripLorentzAngle = nullptr;
  if (!isPhase2_) {
    theSiStripLorentzAngle = &iRecord.get(stripToken_);
  }

  // Produce the filter using the plugin factory
  ClusterShapeHitFilterESProducer::ReturnType aFilter =
      std::make_unique<ClusterShapeHitFilter>(&iRecord.get(geoToken_),
                                              &iRecord.get(topolToken_),
                                              &iRecord.get(fieldToken_),
                                              &iRecord.get(pixelToken_),
                                              theSiStripLorentzAngle,
                                              iRecord.get(limitsToken_));

  aFilter->setShapeCuts(cutOnPixelShape_, cutOnStripShape_);
  aFilter->setChargeCuts(cutOnPixelCharge_, minGoodPixelCharge_, cutOnStripCharge_, minGoodStripCharge_);
  return aFilter;
}

/*****************************************************************************/
void ClusterShapeHitFilterESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("pixelShapeLimitsLabel", "")
      ->setComment("label of the SiPixelClusterShapeLimits payload in SiPixelClusterShapeLimitsRcd");
  desc.add<std::string>("ComponentName", "");
  desc.add<bool>("isPhase2", false);
  desc.add<bool>("doPixelShapeCut", true);
  desc.add<bool>("doStripShapeCut", true);
  desc.add<edm::ParameterSetDescription>("clusterChargeCut", getConfigurationDescription4CCC(CCC::kNone));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(ClusterShapeHitFilterESProducer);
