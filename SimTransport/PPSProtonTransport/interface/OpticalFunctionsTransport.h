#ifndef OPTICALFUNCTION_TRANSPORT
#define OPTICALFUNCTION_TRANSPORT
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"
#include "CondFormats/PPSObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"
#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimTransport/PPSProtonTransport/interface/BaseProtonTransport.h"

#include <array>
#include <unordered_map>

#include <cmath>

class OpticalFunctionsTransport : public BaseProtonTransport {
public:
  OpticalFunctionsTransport(const edm::ParameterSet& ps);
  ~OpticalFunctionsTransport() override{};

  void process(const HepMC::GenEvent* ev, const edm::EventSetup& es, CLHEP::HepRandomEngine* engine) override;

private:
  bool transportProton(const HepMC::GenParticle*);

  std::string lhcInfoLabel_;
  std::string opticsLabel_;

  edm::ESHandle<LHCInfo> lhcInfo_;
  edm::ESHandle<CTPPSBeamParameters> beamParameters_;
  edm::ESHandle<LHCInterpolatedOpticalFunctionsSetCollection> opticalFunctions_;
  unsigned int optFunctionId45_;
  unsigned int optFunctionId56_;

  bool useEmpiricalApertures_;
  double empiricalAperture45_xi0_int_, empiricalAperture45_xi0_slp_, empiricalAperture45_a_int_,
      empiricalAperture45_a_slp_;
  double empiricalAperture56_xi0_int_, empiricalAperture56_xi0_slp_, empiricalAperture56_a_int_,
      empiricalAperture56_a_slp_;

  bool produceHitsRelativeToBeam_;
};
#endif
