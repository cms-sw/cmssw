#include "SimG4Core/PhysicsLists/plugins/adept/CMSEmStandardPhysicsA.h"
#include "SimG4Core/Physics/interface/CMSG4TrackInterface.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <CLHEP/Units/SystemOfUnits.h>
#include "G4ParticleDefinition.hh"
#include "G4LossTableManager.hh"
#include "G4EmParameters.hh"
#include "G4EmBuilder.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"

#include "G4MscStepLimitType.hh"

#include "G4eMultipleScattering.hh"
#include "G4hMultipleScattering.hh"
#include "G4eCoulombScatteringModel.hh"
#include "G4CoulombScattering.hh"
#include "G4WentzelVIModel.hh"
#include "G4UrbanMscModel.hh"

#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"

#include "G4hIonisation.hh"
#include "G4ionIonisation.hh"

#include "G4ParticleTable.hh"
#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4GenericIon.hh"

#include "G4PhysicsListHelper.hh"
#include "G4BuilderType.hh"
#include "G4GammaGeneralProcess.hh"

#include "G4ProcessManager.hh"
#include "G4TransportationWithMsc.hh"

#include "G4RegionStore.hh"
#include "G4Region.hh"

#include <AdePT/g4integration/AdePTConfiguration.hh>
#include <AdePT/g4integration/AdePTTrackingManager.hh>

#include "G4HepEmConfig.hh"

#include <cstdlib>
#include <utility>

namespace {
  // In the ECAL, Phase2TrackFilter only plays Russian roulette with gammas whose generator particle is not a gamma or
  // an e+-. AdePT evaluates this when a track is handed over to the GPU, and its descendants inherit the decision.
  bool gammaRussianRouletteApplicable(G4Track const& track) {
    auto const* trackInfo = dynamic_cast<TrackInformation const*>(track.GetUserInformation());
    if (trackInfo == nullptr)
      return true;
    const int genParticlePID = trackInfo->genParticlePID();
    return genParticlePID != 22 && std::abs(genParticlePID) != 11;
  }

  // Configuration of the AdePT stepping action: CMSSW applies Phase2SteppingAction and its Phase2TrackFilter to the replayed
  // GPU steps and erases the GPU-born secondaries that the filter kills, so the GPU must apply the same kills. Their
  // final configuration is forwarded by RunManagerMT; settings that AdePT cannot mirror are rejected.
  CMSSteppingActionConfig makeAdePTSteppingActionConfig(const edm::ParameterSet& physics) {
    const auto stepConfig = physics.getUntrackedParameter<edm::ParameterSet>("SteppingAction");
    // The Phase-2 secondary filter takes its settings from the "StackingAction" PSet.
    const auto secondaryFilterConfig = physics.getUntrackedParameter<edm::ParameterSet>("StackingAction");
    auto reject = [](std::string const& what) {
      throw cms::Exception("Configuration") << "CMSEmStandardPhysicsA: AdePT does not support " << what;
    };
    // AdePT applies the same cuts to tracks (Phase2SteppingAction) and to new secondaries (Phase2TrackFilter)
    for (std::string const name :
         {"MaxTrackTime", "MaxTrackTimeForward", "MaxZCentralCMS", "CriticalEnergyForVacuum"}) {
      if (stepConfig.getParameter<double>(name) != secondaryFilterConfig.getParameter<double>(name))
        reject("different values of " + name + " in the SteppingAction and StackingAction");
    }
    using Names = std::vector<std::string>;
    if (stepConfig.getParameter<Names>("DeadRegions") != secondaryFilterConfig.getParameter<Names>("DeadRegions"))
      reject("different DeadRegions in the SteppingAction and StackingAction");
    if (!stepConfig.getParameter<Names>("MaxTimeNames").empty() ||
        !secondaryFilterConfig.getParameter<Names>("MaxTimeNames").empty())
      reject("region-dependent time cuts (MaxTimeNames)");
    if (!stepConfig.getParameter<Names>("EkinNames").empty() &&
        !stepConfig.getParameter<Names>("EkinParticles").empty())
      reject("volume-dependent energy cuts (EkinNames)");
    if (secondaryFilterConfig.getParameter<bool>("KillDeltaRay") &&
        !secondaryFilterConfig.getParameter<bool>("KillHeavy"))
      reject("KillDeltaRay");
    if (secondaryFilterConfig.exists("TestKillingOptions")) {
      const auto testKilling = secondaryFilterConfig.getParameter<edm::ParameterSet>("TestKillingOptions");
      if (testKilling.getParameter<bool>("KillInCalo") || testKilling.getParameter<bool>("KillInCaloEfH"))
        reject("TestKillingOptions");
    }

    CMSSteppingActionConfig config;
    config.params.maxTrackTime = stepConfig.getParameter<double>("MaxTrackTime") * CLHEP::ns;
    config.params.maxTrackTimeForward = stepConfig.getParameter<double>("MaxTrackTimeForward") * CLHEP::ns;
    config.params.maxZCentral = stepConfig.getParameter<double>("MaxZCentralCMS") * CLHEP::m;
    config.params.criticalEnergyForVacuum = stepConfig.getParameter<double>("CriticalEnergyForVacuum") * CLHEP::MeV;
    config.params.criticalDensity = stepConfig.getParameter<double>("CriticalDensity") * CLHEP::g / CLHEP::cm3;
    if (secondaryFilterConfig.getParameter<bool>("KillGamma"))
      config.params.gammaEnergyCut = secondaryFilterConfig.getParameter<double>("GammaThreshold") * CLHEP::MeV;
    config.deadRegions = stepConfig.getParameter<Names>("DeadRegions");
    // low-density regions as hardcoded in Phase2TrackFilter
    config.lowDensityRegions = {"BeamPipeOutside", "BeamPipeVacuum"};

    // Gamma Russian roulette in the regions of Phase2TrackFilter, which enables it only if the probability is below
    // one in the ECAL, HCAL, MuonIron or World region
    const double energyLimit = secondaryFilterConfig.getParameter<double>("RusRoGammaEnergyLimit") * CLHEP::MeV;
    const double ecal = secondaryFilterConfig.getParameter<double>("RusRoEcalGamma");
    const double hcal = secondaryFilterConfig.getParameter<double>("RusRoHcalGamma");
    const double muonIron = secondaryFilterConfig.getParameter<double>("RusRoMuonIronGamma");
    const double world = secondaryFilterConfig.getParameter<double>("RusRoWorldGamma");
    if (energyLimit > 0.0 && (ecal < 1.0 || hcal < 1.0 || muonIron < 1.0 || world < 1.0)) {
      config.params.gammaRussianRouletteEnergyLimit = energyLimit;
      const std::pair<std::string, double> probabilities[] = {
          {"EcalRegion", ecal},
          {"HcalRegion", hcal},
          {"MuonIron", muonIron},
          {"DefaultRegionForTheWorld", world},
          {"HGCalRegion", secondaryFilterConfig.getParameter<double>("RusRoHGcalGamma")},
          {"ZDCRegion", secondaryFilterConfig.getParameter<double>("RusRoZDCGamma")}};
      for (auto const& [region, probability] : probabilities) {
        if (probability < 1.0)
          config.gammaRussianRoulette.emplace_back(region, probability);
      }
      config.gammaRussianRouletteRestrictedRegions = {"EcalRegion"};
      config.gammaRussianRouletteApplicability = gammaRussianRouletteApplicable;
    }

    return config;
  }
}  // namespace

CMSEmStandardPhysicsA::CMSEmStandardPhysicsA(G4int ver, const edm::ParameterSet& p)
    : G4VPhysicsConstructor("CMSEmStandard_ema") {
  fAdePTConfiguration = new AdePTConfiguration();
  SetVerboseLevel(ver);
  // EM parameters specific for this EM physics configuration
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(ver);
  param->SetApplyCuts(true);
  param->SetStepFunction(0.8, 1 * CLHEP::mm);
  param->SetMscRangeFactor(0.2);
  param->SetMscStepLimitType(fMinimal);
  param->SetFluo(false);
  SetPhysicsType(bElectromagnetic);
  fRangeFactor = p.getParameter<double>("G4MscRangeFactor");
  fGeomFactor = p.getParameter<double>("G4MscGeomFactor");
  fSafetyFactor = p.getParameter<double>("G4MscSafetyFactor");
  fLambdaLimit = p.getParameter<double>("G4MscLambdaLimit") * CLHEP::mm;
  std::string msc = p.getParameter<std::string>("G4MscStepLimit");
  fStepLimitType = fUseSafety;
  if (msc == "UseSafetyPlus") {
    fStepLimitType = fUseSafetyPlus;
  }
  if (msc == "Minimal") {
    fStepLimitType = fMinimal;
  }
  fAdePTConfiguration->SetCMSSteppingActionConfig(makeAdePTSteppingActionConfig(p));
  double tcut = p.getParameter<double>("G4TrackingCut") * CLHEP::MeV;
  param->SetLowestElectronEnergy(tcut);
  param->SetLowestMuHadEnergy(tcut);
}

void CMSEmStandardPhysicsA::ConstructParticle() {
  // minimal set of particles for EM physics
  G4EmBuilder::ConstructMinimalEmSet();
}

void CMSEmStandardPhysicsA::ConstructProcess() {
  if (verboseLevel > 0) {
    int id = CMSG4TrackInterface::instance()->getThreadID();
    edm::LogVerbatim("PhysicsList") << "### " << GetPhysicsName() << " Construct EM Processes; EMA threadID=" << id;
  }

  // This EM builder takes default models of Geant4 10 EMV.
  // Multiple scattering by WentzelVI for all particles except:
  //   a) e+e- below 100 MeV for which the Urban model is used
  //   b) ions for which Urban model is used
  G4EmBuilder::PrepareEMPhysics();

  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();
  // processes used by several particles
  G4hMultipleScattering* hmsc = new G4hMultipleScattering("ionmsc");
  G4NuclearStopping* pnuc(nullptr);

  const G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("HcalRegion", false);
  const G4Region* bRegion = G4RegionStore::GetInstance()->GetRegion("HGCalRegion", false);

  // G4HepEm is Active
  if (verboseLevel > 0) {
    edm::LogVerbatim("PhysicsList") << "AdePT is active, registering AdePTTrackingManager";
  }

  // number of worker threads must be passed to AdePT
  fAdePTConfiguration->SetNumThreads(CurrentG4Track::numberOfWorkers());

  // Construct the AdePT tracking manager
  auto* hepEmTM = new AdePTTrackingManager(fAdePTConfiguration, verboseLevel);

  // now configure the G4HepEm config in the AdePT TM, as it will be used to define the physics
  G4HepEmConfig* config = hepEmTM->GetG4HepEmConfig();
  // First set global configuration parameters:
  // ------------------------------------------
  // The default MSC `RangeFactor`, `SafetyFactor`, `StepLimitType` parameters
  // as well as  `SetApplyCuts` and `SetLowestElectronEnergy` are taken from
  // `G4EmParameters` so we set only the step function parameters here.

  config->SetEnergyLossStepLimitFunctionParameters(0.8, 1.0 * CLHEP::mm);

  // Then set special configuration for some regions:
  // ------------------------------------------------
  if (nullptr != aRegion) {
    // HCal region
    const G4String& rname = aRegion->GetName();
    config->SetMinimalMSCStepLimit(fStepLimitType == fMinimal, rname);
    config->SetMSCRangeFactor(fRangeFactor, rname);
    config->SetMSCSafetyFactor(fSafetyFactor, rname);
  }

  if (nullptr != bRegion) {
    // HGCal region
    const G4String& rname = bRegion->GetName();
    config->SetMinimalMSCStepLimit(fStepLimitType == fMinimal, rname);
    config->SetMSCRangeFactor(fRangeFactor, rname);
    config->SetMSCSafetyFactor(fSafetyFactor, rname);

    config->SetWoodcockTrackingRegion(rname);
    config->SetWDTEnergyLimit(0.5 * CLHEP::MeV);
  }

  G4Electron::Electron()->SetTrackingManager(hepEmTM);
  G4Positron::Positron()->SetTrackingManager(hepEmTM);
  G4Gamma::Gamma()->SetTrackingManager(hepEmTM);

  // generic ion
  G4ParticleDefinition* particle = G4GenericIon::GenericIon();
  G4ionIonisation* ionIoni = new G4ionIonisation();
  ph->RegisterProcess(hmsc, particle);
  ph->RegisterProcess(ionIoni, particle);

  // muons, hadrons ions
  G4EmBuilder::ConstructCharged(hmsc, pnuc);
}
