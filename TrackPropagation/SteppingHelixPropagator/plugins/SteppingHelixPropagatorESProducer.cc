#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>

#include <string>
#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

class SteppingHelixPropagatorESProducer : public edm::ESProducer {
public:
  SteppingHelixPropagatorESProducer(const edm::ParameterSet& p);
  ~SteppingHelixPropagatorESProducer() override;
  std::unique_ptr<Propagator> produce(const TrackingComponentsRecord&);

private:
  const edm::ParameterSet pset_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> vbMagToken_;
  const bool setVBFPointer_;
};

using namespace edm;

SteppingHelixPropagatorESProducer::SteppingHelixPropagatorESProducer(const edm::ParameterSet& p)
    : pset_{p}, setVBFPointer_{pset_.getParameter<bool>("SetVBFPointer")} {
  std::string myname = p.getParameter<std::string>("ComponentName");
  auto c = setWhatProduced(this, myname);
  c.setConsumes(magToken_);
  if (setVBFPointer_) {
    c.setConsumes(vbMagToken_, edm::ESInputTag("", pset_.getParameter<std::string>("VBFName")));
  }
}

SteppingHelixPropagatorESProducer::~SteppingHelixPropagatorESProducer() {}

std::unique_ptr<Propagator> SteppingHelixPropagatorESProducer::produce(const TrackingComponentsRecord& iRecord) {
  //   if (_propagator){
  //     delete _propagator;
  //     _propagator = 0;
  //   }
  auto const& magfield = iRecord.get(magToken_);

  std::string pdir = pset_.getParameter<std::string>("PropagationDirection");

  PropagationDirection dir = alongMomentum;

  if (pdir == "oppositeToMomentum")
    dir = oppositeToMomentum;
  if (pdir == "alongMomentum")
    dir = alongMomentum;
  if (pdir == "anyDirection")
    dir = anyDirection;

  std::unique_ptr<SteppingHelixPropagator> shProp;
  shProp = std::make_unique<SteppingHelixPropagator>(&magfield, dir);

  bool useInTeslaFromMagField = pset_.getParameter<bool>("useInTeslaFromMagField");
  bool useMagVolumes = pset_.getParameter<bool>("useMagVolumes");

  // if useMagVolumes == true and an alternate VBF field is not specified with setVBFPointer_,
  // Force "useInTeslaFromMagField=true" for a B=0 VBF map.
  if (useMagVolumes == true && !useInTeslaFromMagField && !setVBFPointer_ && magfield.nominalValue() == 0) {
    const VolumeBasedMagneticField* vbfCPtr = dynamic_cast<const VolumeBasedMagneticField*>(&magfield);
    if (vbfCPtr == nullptr) {
      edm::LogWarning("SteppingHelixPropagator")
          << "Config specifies useMagVolumes==True but no VBF field available: SHP has no access to yoke material "
             "properties. Use setVBFPointer=true and VBFName cards to set a VBF field, otherwise set "
             "useMagVolumes==False."
          << std::endl;
    } else {
      edm::LogInfo("SteppingHelixPropagator")
          << "Config specifies useMagVolumes==true and VBF field available: Forcing useInTeslaFromMagField = True."
          << std::endl;
      useInTeslaFromMagField = true;
    }
  }

  if (setVBFPointer_) {
    auto const& vbfField = iRecord.get(vbMagToken_);
    const VolumeBasedMagneticField* vbfCPtr = dynamic_cast<const VolumeBasedMagneticField*>(&vbfField);
    shProp->setVBFPointer(vbfCPtr);
  }

  shProp->setUseInTeslaFromMagField(useInTeslaFromMagField);

  bool haveX0Corr = pset_.getParameter<bool>("ApplyRadX0Correction");
  shProp->applyRadX0Correction(haveX0Corr);

  bool assumeNoMaterial = pset_.getParameter<bool>("AssumeNoMaterial");
  shProp->setMaterialMode(assumeNoMaterial);

  bool noErrorPropagation = pset_.getParameter<bool>("NoErrorPropagation");
  shProp->setNoErrorPropagation(noErrorPropagation);

  bool debugMode = pset_.getParameter<bool>("debug");
  shProp->setDebug(debugMode);

  shProp->setUseMagVolumes(useMagVolumes);

  bool useMatVolumes = pset_.getParameter<bool>("useMatVolumes");
  shProp->setUseMatVolumes(useMatVolumes);

  bool useIsYokeFlag = pset_.getParameter<bool>("useIsYokeFlag");
  shProp->setUseIsYokeFlag(useIsYokeFlag);

  bool returnTangentPlane = pset_.getParameter<bool>("returnTangentPlane");
  shProp->setReturnTangentPlane(returnTangentPlane);

  bool sendLogWarning = pset_.getParameter<bool>("sendLogWarning");
  shProp->setSendLogWarning(sendLogWarning);

  bool useTuningForL2Speed = pset_.getParameter<bool>("useTuningForL2Speed");
  shProp->setUseTuningForL2Speed(useTuningForL2Speed);

  bool useECoffsets = pset_.getParameter<bool>("useEndcapShiftsInZ");
  if (useECoffsets) {
    double valPos = pset_.getParameter<double>("endcapShiftInZPos");
    double valNeg = pset_.getParameter<double>("endcapShiftInZNeg");
    shProp->setEndcapShiftsInZPosNeg(valPos, valNeg);
  }

  return shProp;
}

#include "FWCore/Utilities/interface/typelookup.h"

DEFINE_FWK_EVENTSETUP_MODULE(SteppingHelixPropagatorESProducer);
