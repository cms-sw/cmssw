#ifndef OPTICALFUNCTION_TRANSPORT
#define OPTICALFUNCTION_TRANSPORT
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/SimBeamSpotObjectsRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"
#include "CondFormats/PPSObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"
#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimTransport/PPSProtonTransport/interface/BaseProtonTransport.h"

class OpticalFunctionsTransport : public BaseProtonTransport {
public:
  OpticalFunctionsTransport(const edm::ParameterSet& ps, edm::ConsumesCollector iC);
  ~OpticalFunctionsTransport() override;

  void process(const HepMC::GenEvent* ev, const edm::EventSetup& es, CLHEP::HepRandomEngine* engine) override;

private:
  bool transportProton(const HepMC::GenParticle*);

  edm::ESGetToken<LHCInfo, LHCInfoRcd> lhcInfoToken_;
  edm::ESGetToken<CTPPSBeamParameters, CTPPSBeamParametersRcd> beamParametersToken_;
  edm::ESGetToken<LHCInterpolatedOpticalFunctionsSetCollection, CTPPSInterpolatedOpticsRcd> opticsToken_;
  edm::ESGetToken<SimBeamSpotObjects, SimBeamSpotObjectsRcd> beamspotToken_;

  const LHCInfo* lhcInfo_{nullptr};
  const CTPPSBeamParameters* beamParameters_{nullptr};
  const LHCInterpolatedOpticalFunctionsSetCollection* opticalFunctions_{nullptr};
  const SimBeamSpotObjects* beamspot_{nullptr};

  unsigned int optFunctionId45_{0};
  unsigned int optFunctionId56_{0};

  bool useEmpiricalApertures_;
  double empiricalAperture45_xi0_int_, empiricalAperture45_xi0_slp_, empiricalAperture45_a_int_,
      empiricalAperture45_a_slp_;
  double empiricalAperture56_xi0_int_, empiricalAperture56_xi0_slp_, empiricalAperture56_a_int_,
      empiricalAperture56_a_slp_;
};
#endif
