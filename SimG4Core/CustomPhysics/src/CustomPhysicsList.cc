#include <memory>

#include "SimG4Core/CustomPhysics/interface/CustomPhysicsList.h"
#include "SimG4Core/CustomPhysics/interface/CustomParticleFactory.h"
#include "SimG4Core/CustomPhysics/interface/CustomParticle.h"
#include "SimG4Core/CustomPhysics/interface/DummyChargeFlipProcess.h"
#include "SimG4Core/CustomPhysics/interface/CustomProcessHelper.h"
#include "SimG4Core/CustomPhysics/interface/CustomPDGParser.h"
#include "SimG4Core/CustomPhysics/interface/RHadronPythiaDecayer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/CustomPhysics/interface/FullModelHadronicProcess.h"
#include "SimG4Core/CustomPhysics/interface/CMSDarkPairProductionProcess.h"
#include "SimG4Core/CustomPhysics/interface/CMSQGSPSIMPBuilder.h"
#include "SimG4Core/CustomPhysics/interface/CMSSIMPInelasticProcess.h"

#include "SimG4Core/CustomPhysics/interface/CMSSQLoopProcess.h"
#include "SimG4Core/CustomPhysics/interface/CMSSQLoopProcessDiscr.h"
#include "SimG4Core/CustomPhysics/interface/CMSSQNeutronAnnih.h"
#include "SimG4Core/CustomPhysics/interface/CMSSQInelasticCrossSection.h"

#include "G4hMultipleScattering.hh"
#include "G4hIonisation.hh"
#include "G4ProcessManager.hh"
#include "G4HadronicProcess.hh"
#include "G4AutoDelete.hh"
#include "G4Decay.hh"

using namespace CLHEP;

G4ThreadLocal CustomProcessHelper* CustomPhysicsList::myHelper = nullptr;

CustomPhysicsList::CustomPhysicsList(const std::string& name, const edm::ParameterSet& p, bool apinew)
    : G4VPhysicsConstructor(name) {
  myConfig = p;
  if (apinew) {
    dfactor = p.getParameter<double>("DarkMPFactor");
    fHadronicInteraction = p.getParameter<bool>("RhadronPhysics");
  } else {
    // this is left for backward compatibility
    dfactor = p.getParameter<double>("dark_factor");
    fHadronicInteraction = p.getParameter<bool>("rhadronPhysics");
  }
  edm::FileInPath fp = p.getParameter<edm::FileInPath>("particlesDef");
  particleDefFilePath = fp.fullPath();
  fParticleFactory = std::make_unique<CustomParticleFactory>();

  edm::LogVerbatim("SimG4CoreCustomPhysics") << "CustomPhysicsList: Path for custom particle definition file: \n"
                                             << particleDefFilePath << "\n"
                                             << "      dark_factor= " << dfactor;
}

void CustomPhysicsList::ConstructParticle() {
  edm::LogVerbatim("SimG4CoreCustomPhysics") << "===== CustomPhysicsList::ConstructParticle ";
  fParticleFactory.get()->loadCustomParticles(particleDefFilePath);
}

void CustomPhysicsList::ConstructProcess() {
  edm::LogVerbatim("SimG4CoreCustomPhysics") << "CustomPhysicsList: adding CustomPhysics processes "
                                             << "for the list of particles";

  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();
  bool extRHadronDecayerSet = false;
  G4Decay* pythiaDecayProcess = nullptr;

  for (auto particle : fParticleFactory.get()->getCustomParticles()) {
    if (particle->GetParticleType() == "simp") {
      G4ProcessManager* pmanager = particle->GetProcessManager();
      if (pmanager) {
        CMSSIMPInelasticProcess* simpInelPr = new CMSSIMPInelasticProcess();
        CMSQGSPSIMPBuilder* theQGSPSIMPB = new CMSQGSPSIMPBuilder();
        theQGSPSIMPB->Build(simpInelPr);
        pmanager->AddDiscreteProcess(simpInelPr);
      } else
        edm::LogVerbatim("CustomPhysics") << "   No pmanager";
    }

    CustomParticle* cp = dynamic_cast<CustomParticle*>(particle);
    if (cp) {
      G4ProcessManager* pmanager = particle->GetProcessManager();
      edm::LogVerbatim("SimG4CoreCustomPhysics")
          << "CustomPhysicsList: " << particle->GetParticleName() << "  PDGcode= " << particle->GetPDGEncoding()
          << "  Mass= " << particle->GetPDGMass() / GeV << " GeV.";
      if (pmanager) {
        if (particle->GetPDGCharge() != 0.0) {
          ph->RegisterProcess(new G4hMultipleScattering, particle);
          ph->RegisterProcess(new G4hIonisation, particle);
        }
        if (cp->GetCloud() && fHadronicInteraction &&
            (CustomPDGParser::s_isgluinoHadron(particle->GetPDGEncoding()) ||
             (CustomPDGParser::s_isstopHadron(particle->GetPDGEncoding())) ||
             (CustomPDGParser::s_issbottomHadron(particle->GetPDGEncoding())))) {
          edm::LogVerbatim("SimG4CoreCustomPhysics")
              << "CustomPhysicsList: " << particle->GetParticleName()
              << " CloudMass= " << cp->GetCloud()->GetPDGMass() / GeV
              << " GeV; SpectatorMass= " << cp->GetSpectator()->GetPDGMass() / GeV << " GeV.";

          if (nullptr == myHelper) {
            myHelper = new CustomProcessHelper(myConfig, fParticleFactory.get());
            G4AutoDelete::Register(myHelper);
          }
          pmanager->AddDiscreteProcess(new FullModelHadronicProcess(myHelper));
        }
        if ((particle->GetParticleType() == "rhadron" || particle->GetParticleType() == "mesonino" ||
             particle->GetParticleType() == "sbaryon") &&
            particle->GetPDGStable() == false) {
          if (!extRHadronDecayerSet) {
            // Set the pythia decayer for Rhadrons if they are unstable
            pythiaDecayProcess = new RHadronPythiaDecayer(myConfig);
            G4VExtDecayer* extDecayer = dynamic_cast<G4VExtDecayer*>(pythiaDecayProcess);
            // Set the external decayer to itself. Seems redundant but is necessary as far as I can tell. Without doing this, RHadronPythiaDecayer::ImportDecayProducts() will not be called.
            pythiaDecayProcess->SetExtDecayer(extDecayer);
            extRHadronDecayerSet = true;
          }
          // Remove native G4 decay process in favor of RHadronPythiaDecayer
          G4ProcessVector* fullProcessList = pmanager->GetProcessList();
          for (unsigned int i = 0; i < fullProcessList->size(); ++i) {
            G4VProcess* process = (*fullProcessList)[i];
            if (process->GetProcessType() == fDecay) {
              pmanager->RemoveProcess(process);
              pmanager->AddProcess(pythiaDecayProcess);
              pmanager->SetProcessOrdering(pythiaDecayProcess, idxPostStep);
            }
          }
        }
        if (particle->GetParticleType() == "darkpho") {
          CMSDarkPairProductionProcess* darkGamma = new CMSDarkPairProductionProcess(dfactor);
          pmanager->AddDiscreteProcess(darkGamma);
        }
        if (particle->GetParticleName() == "anti_sexaq") {
          // here the different sexaquark interactions get defined
          G4HadronicProcess* sqInelPr = new G4HadronicProcess();
          CMSSQNeutronAnnih* sqModel = new CMSSQNeutronAnnih(particle->GetPDGMass() / GeV);
          sqInelPr->RegisterMe(sqModel);
          CMSSQInelasticCrossSection* sqInelXS = new CMSSQInelasticCrossSection(particle->GetPDGMass() / GeV);
          sqInelPr->AddDataSet(sqInelXS);
          pmanager->AddDiscreteProcess(sqInelPr);
          // add also the looping needed to simulate flat interaction probability
          CMSSQLoopProcess* sqLoopPr = new CMSSQLoopProcess();
          pmanager->AddContinuousProcess(sqLoopPr);
          CMSSQLoopProcessDiscr* sqLoopPrDiscr = new CMSSQLoopProcessDiscr(particle->GetPDGMass() / GeV);
          pmanager->AddDiscreteProcess(sqLoopPrDiscr);
        } else if (particle->GetParticleName() == "sexaq") {
          edm::LogVerbatim("CustomPhysics") << "   No pmanager implemented for sexaq, only for anti_sexaq";
        }
      }
    }
  }
}
