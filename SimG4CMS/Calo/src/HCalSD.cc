///////////////////////////////////////////////////////////////////////////////
// File: HCalSD.cc
// Description: Sensitive Detector class for calorimeters
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HCalSD.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HcalDumpGeometry.h"
#include "SimG4CMS/Calo/interface/HFFibreFiducial.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"

#include "DD4hep/Filter.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>

//#define EDM_ML_DEBUG
//#define plotDebug
//#define printDebug

#ifdef plotDebug
#include <TH1F.h>
#endif

HCalSD::HCalSD(const std::string& name,
               const HcalDDDSimConstants* hcns,
               const HcalDDDRecConstants* hcnr,
               const HcalSimulationConstants* hscs,
               const HBHEDarkening* hbd,
               const HBHEDarkening* hed,
               const SensitiveDetectorCatalog& clg,
               edm::ParameterSet const& p,
               const SimTrackManager* manager)
    : CaloSD(name,
             clg,
             p,
             manager,
             (float)(p.getParameter<edm::ParameterSet>("HCalSD").getParameter<double>("TimeSliceUnit")),
             p.getParameter<edm::ParameterSet>("HCalSD").getParameter<bool>("IgnoreTrackID")),
      hcalConstants_(hcns),
      hcalSimConstants_(hscs),
      m_HBDarkening(hbd),
      m_HEDarkening(hed),
      isHF(false),
      weight_(1.0),
      depth_(1) {
  numberingFromDDD.reset(nullptr);
  numberingScheme.reset(nullptr);
  showerLibrary.reset(nullptr);
  hfshower.reset(nullptr);
  showerParam.reset(nullptr);
  showerPMT.reset(nullptr);
  showerBundle.reset(nullptr);
  m_HFDarkening.reset(nullptr);
  m_HcalTestNS.reset(nullptr);

  //static SimpleConfigurable<double> bk1(0.013, "HCalSD:BirkC1");
  //static SimpleConfigurable<double> bk2(0.0568,"HCalSD:BirkC2");
  //static SimpleConfigurable<double> bk3(1.75,  "HCalSD:BirkC3");
  // Values from NIM 80 (1970) 239-244: as implemented in Geant3

  bool dd4hep = p.getParameter<bool>("g4GeometryDD4hepSource");
  edm::ParameterSet m_HC = p.getParameter<edm::ParameterSet>("HCalSD");
  useBirk = m_HC.getParameter<bool>("UseBirkLaw");
  double bunit = (CLHEP::g / (CLHEP::MeV * CLHEP::cm2));
  birk1 = m_HC.getParameter<double>("BirkC1") * bunit;
  birk2 = m_HC.getParameter<double>("BirkC2");
  birk3 = m_HC.getParameter<double>("BirkC3");
  useShowerLibrary = m_HC.getParameter<bool>("UseShowerLibrary");
  useParam = m_HC.getParameter<bool>("UseParametrize");
  testNumber = m_HC.getParameter<bool>("TestNumberingScheme");
  neutralDensity = m_HC.getParameter<bool>("doNeutralDensityFilter");
  usePMTHit = m_HC.getParameter<bool>("UsePMTHits");
  betaThr = m_HC.getParameter<double>("BetaThreshold");
  eminHitHB = m_HC.getParameter<double>("EminHitHB") * MeV;
  eminHitHE = m_HC.getParameter<double>("EminHitHE") * MeV;
  eminHitHO = m_HC.getParameter<double>("EminHitHO") * MeV;
  eminHitHF = m_HC.getParameter<double>("EminHitHF") * MeV;
  useFibreBundle = m_HC.getParameter<bool>("UseFibreBundleHits");
  deliveredLumi = m_HC.getParameter<double>("DelivLuminosity");
  agingFlagHB = m_HC.getParameter<bool>("HBDarkening");
  agingFlagHE = m_HC.getParameter<bool>("HEDarkening");
  bool agingFlagHF = m_HC.getParameter<bool>("HFDarkening");
  useHF = m_HC.getUntrackedParameter<bool>("UseHF", true);
  bool forTBHC = m_HC.getUntrackedParameter<bool>("ForTBHCAL", false);
  bool forTBH2 = m_HC.getUntrackedParameter<bool>("ForTBH2", false);
  useLayerWt = m_HC.getUntrackedParameter<bool>("UseLayerWt", false);
  std::string file = m_HC.getUntrackedParameter<std::string>("WtFile", "None");
  testNS_ = m_HC.getUntrackedParameter<bool>("TestNS", false);
  edm::ParameterSet m_HF = p.getParameter<edm::ParameterSet>("HFShower");
  applyFidCut = m_HF.getParameter<bool>("ApplyFiducialCut");
  bool dumpGeom = m_HC.getUntrackedParameter<bool>("DumpGeometry", false);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "***************************************************"
                              << "\n"
                              << "* Constructing a HCalSD  with name " << name << "\n"
                              << "\n"
                              << "***************************************************";
#endif
  edm::LogVerbatim("HcalSim") << "HCalSD:: Use of HF code is set to " << useHF
                              << "\n         Use of shower parametrization set to " << useParam
                              << "\n         Use of shower library is set to " << useShowerLibrary
                              << "\n         Use PMT Hit is set to " << usePMTHit << " with beta Threshold " << betaThr
                              << "\n         USe of FibreBundle Hit set to " << useFibreBundle
                              << "\n         Use of Birks law is set to " << useBirk
                              << " with three constants kB = " << birk1 / bunit << ", C1 = " << birk2
                              << ", C2 = " << birk3;
  edm::LogVerbatim("HcalSim") << "HCalSD:: Suppression Flag " << suppressHeavy << " protons below " << kmaxProton
                              << " MeV,"
                              << " neutrons below " << kmaxNeutron << " MeV and"
                              << " ions below " << kmaxIon << " MeV\n"
                              << "         Threshold for storing hits in HB: " << eminHitHB << " HE: " << eminHitHE
                              << " HO: " << eminHitHO << " HF: " << eminHitHF << "\n"
                              << "Delivered luminosity for Darkening " << deliveredLumi << " Flag (HE) " << agingFlagHE
                              << " Flag (HB) " << agingFlagHB << " Flag (HF) " << agingFlagHF << "\n"
                              << "Application of Fiducial Cut " << applyFidCut
                              << "Flag for test number|neutral density filter " << testNumber << " " << neutralDensity;

  if (forTBHC) {
    useHF = false;
    matNames.emplace_back("Scintillator");
  } else {
    matNames = hcalSimConstants_->hcalsimpar()->hcalMaterialNames_;
  }

  HcalNumberingScheme* scheme;
  if (testNumber || forTBH2) {
    scheme = dynamic_cast<HcalNumberingScheme*>(new HcalTestNumberingScheme(forTBH2));
  } else {
    scheme = new HcalNumberingScheme();
  }
  setNumberingScheme(scheme);

  // always call getFromLibrary() method to identify HF region
  setParameterized(true);

  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  //  std::vector<G4LogicalVolume *>::const_iterator lvcite;
  const G4LogicalVolume* lv;
  std::string attribute, value;

  if (useHF) {
    if (useParam) {
      showerParam = std::make_unique<HFShowerParam>(name, hcalConstants_, hcalSimConstants_->hcalsimpar(), p);
    } else {
      if (useShowerLibrary) {
        showerLibrary = std::make_unique<HFShowerLibrary>(name, hcalConstants_, hcalSimConstants_->hcalsimpar(), p);
      }
      hfshower = std::make_unique<HFShower>(name, hcalConstants_, hcalSimConstants_->hcalsimpar(), p, 0);
    }

    // HF volume names
    hfNames = hcalSimConstants_->hcalsimpar()->hfNames_;
    const std::vector<int>& temp = hcalSimConstants_->hcalsimpar()->hfLevels_;
#ifdef EDM_ML_DEBUG
    std::stringstream ss0;
    ss0 << "HCalSD: Names to be tested for Volume = HF has " << hfNames.size() << " elements";
#endif
    int addlevel = dd4hep ? 1 : 0;
    for (unsigned int i = 0; i < hfNames.size(); ++i) {
      G4String namv(static_cast<std::string>(dd4hep::dd::noNamespace(hfNames[i])));
      lv = nullptr;
      for (auto lvol : *lvs) {
        if (dd4hep::dd::noNamespace(lvol->GetName()) == namv) {
          lv = lvol;
          break;
        }
      }
      hfLV.emplace_back(lv);
      hfLevels.emplace_back(temp[i] + addlevel);
#ifdef EDM_ML_DEBUG
      ss0 << "\n        HF[" << i << "] = " << namv << " LV " << lv << " at level " << (temp[i] + addlevel);
#endif
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << ss0.str();
#endif
    // HF Fibre volume names
    fibreNames = hcalSimConstants_->hcalsimpar()->hfFibreNames_;
    fillLogVolumeVector("HFFibre", fibreNames, fibreLV);
    const std::vector<std::string>& pmtNames = hcalSimConstants_->hcalsimpar()->hfPMTNames_;
    fillLogVolumeVector("HFPMT", pmtNames, pmtLV);
    const std::vector<std::string>& straightNames = hcalSimConstants_->hcalsimpar()->hfFibreStraightNames_;
    fillLogVolumeVector("HFFibreBundleStraight", straightNames, fibre1LV);
    const std::vector<std::string>& conicalNames = hcalSimConstants_->hcalsimpar()->hfFibreConicalNames_;
    fillLogVolumeVector("HFFibreBundleConical", conicalNames, fibre2LV);
  }

  //Material list for HB/HE/HO sensitive detectors
  const G4MaterialTable* matTab = G4Material::GetMaterialTable();
  std::vector<G4Material*>::const_iterator matite;
  for (auto const& namx : matNames) {
    const G4Material* mat = nullptr;
    for (matite = matTab->begin(); matite != matTab->end(); ++matite) {
      if (static_cast<std::string>(dd4hep::dd::noNamespace((*matite)->GetName())) == namx) {
        mat = (*matite);
        break;
      }
    }
    materials.emplace_back(mat);
  }
#ifdef EDM_ML_DEBUG
  std::stringstream ss1;
  for (unsigned int i = 0; i < matNames.size(); ++i) {
    if (i / 10 * 10 == i) {
      ss1 << "\n";
    }
    ss1 << "  " << matNames[i];
  }
  edm::LogVerbatim("HcalSim") << "HCalSD: Material names for HCAL: " << ss1.str();
#endif
  if (useLayerWt) {
    readWeightFromFile(file);
  }
  numberingFromDDD = std::make_unique<HcalNumberingFromDDD>(hcalConstants_);

  //Special Geometry parameters
  gpar = hcalConstants_->getGparHF();
#ifdef EDM_ML_DEBUG
  std::stringstream ss2;
  for (unsigned int ig = 0; ig < gpar.size(); ig++) {
    ss2 << "\n         gpar[" << ig << "] = " << gpar[ig] / cm << " cm";
  }
  edm::LogVerbatim("HcalSim") << "Maximum depth for HF " << hcalConstants_->getMaxDepth(2) << gpar.size()
                              << " gpar (cm)" << ss2.str();
#endif

  //Test Hcal Numbering Scheme
  if (testNS_)
    m_HcalTestNS = std::make_unique<HcalTestNS>(hcnr);

  for (int i = 0; i < 9; ++i) {
    hit_[i] = time_[i] = dist_[i] = nullptr;
  }
  hzvem = hzvhad = nullptr;

  if (agingFlagHF) {
    m_HFDarkening = std::make_unique<HFDarkening>(m_HC.getParameter<edm::ParameterSet>("HFDarkeningParameterBlock"));
  }
#ifdef plotDebug
  edm::Service<TFileService> tfile;

  if (tfile.isAvailable()) {
    static const char* const labels[] = {"HB",
                                         "HE",
                                         "HO",
                                         "HF Absorber",
                                         "HF PMT",
                                         "HF Absorber Long",
                                         "HF Absorber Short",
                                         "HF PMT Long",
                                         "HF PMT Short"};
    TFileDirectory hcDir = tfile->mkdir("ProfileFromHCalSD");
    char name[20], title[60];
    for (int i = 0; i < 9; ++i) {
      sprintf(title, "Hit energy in %s", labels[i]);
      sprintf(name, "HCalSDHit%d", i);
      hit_[i] = hcDir.make<TH1F>(name, title, 2000, 0., 2000.);
      sprintf(title, "Energy (MeV)");
      hit_[i]->GetXaxis()->SetTitle(title);
      hit_[i]->GetYaxis()->SetTitle("Hits");
      sprintf(title, "Time of the hit in %s", labels[i]);
      sprintf(name, "HCalSDTime%d", i);
      time_[i] = hcDir.make<TH1F>(name, title, 2000, 0., 2000.);
      sprintf(title, "Time (ns)");
      time_[i]->GetXaxis()->SetTitle(title);
      time_[i]->GetYaxis()->SetTitle("Hits");
      sprintf(title, "Longitudinal profile in %s", labels[i]);
      sprintf(name, "HCalSDDist%d", i);
      dist_[i] = hcDir.make<TH1F>(name, title, 2000, 0., 2000.);
      sprintf(title, "Distance (mm)");
      dist_[i]->GetXaxis()->SetTitle(title);
      dist_[i]->GetYaxis()->SetTitle("Hits");
    }
    if (useHF && (!useParam)) {
      hzvem = hcDir.make<TH1F>("hzvem", "Longitudinal Profile (EM Part)", 330, 0.0, 1650.0);
      hzvem->GetXaxis()->SetTitle("Longitudinal Profile (EM Part)");
      hzvhad = hcDir.make<TH1F>("hzvhad", "Longitudinal Profile (Had Part)", 330, 0.0, 1650.0);
      hzvhad->GetXaxis()->SetTitle("Longitudinal Profile (Hadronic Part)");
    }
  }
#endif
  if (dumpGeom) {
    std::unique_ptr<HcalNumberingFromDDD> hcn = std::make_unique<HcalNumberingFromDDD>(hcalConstants_);
    const auto& lvNames = clg.logicalNames(name);
    HcalDumpGeometry geom(lvNames, hcn.get(), testNumber, false);
    geom.update();
  }
}

void HCalSD::fillLogVolumeVector(const std::string& value,
                                 const std::vector<std::string>& lvnames,
                                 std::vector<const G4LogicalVolume*>& lvvec) {
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  const G4LogicalVolume* lv;
  std::stringstream ss3;
  ss3 << "HCalSD: " << lvnames.size() << " names to be tested for Volume <" << value << ">:";
  for (unsigned int i = 0; i < lvnames.size(); ++i) {
    G4String namv(static_cast<std::string>(dd4hep::dd::noNamespace(lvnames[i])));
    lv = nullptr;
    for (auto lvol : *lvs) {
      if (dd4hep::dd::noNamespace(lvol->GetName()) == namv) {
        lv = lvol;
        break;
      }
    }
    lvvec.emplace_back(lv);
    if (i / 10 * 10 == i) {
      ss3 << "\n";
    }
    ss3 << "  " << namv;
  }
  edm::LogVerbatim("HcalSim") << ss3.str();
}

bool HCalSD::getFromLibrary(const G4Step* aStep) {
  auto const track = aStep->GetTrack();
  depth_ = (aStep->GetPreStepPoint()->GetTouchable()->GetReplicaNumber(0)) % 10;
  weight_ = 1.0;
  bool kill(false);
  isHF = isItHF(aStep);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "GetFromLibrary: isHF " << isHF << " darken " << (m_HFDarkening != nullptr)
                              << " useParam " << useParam << " useShowerLibrary " << useShowerLibrary << " Muon? "
                              << G4TrackToParticleID::isMuon(track) << " electron? "
                              << G4TrackToParticleID::isGammaElectronPositron(track) << " Stable Hadron? "
                              << G4TrackToParticleID::isStableHadronIon(track);
#endif
  if (isHF) {
    if (m_HFDarkening) {
      G4ThreeVector hitPoint = aStep->GetPreStepPoint()->GetPosition();
      const double invcm = 1. / CLHEP::cm;
      double r = hitPoint.perp() * invcm;
      double z = std::abs(hitPoint.z()) * invcm;
      double dose_acquired = 0.;
      if (z >= HFDarkening::lowZLimit && z <= HFDarkening::upperZLimit) {
        unsigned int hfZLayer = (unsigned int)((z - HFDarkening::lowZLimit) / 5);
        if (hfZLayer >= HFDarkening::upperZLimit)
          hfZLayer = (HFDarkening::upperZLimit - 1);
        float normalized_lumi = m_HFDarkening->int_lumi(deliveredLumi);
        for (int i = hfZLayer; i != HFDarkening::numberOfZLayers; ++i) {
          dose_acquired = m_HFDarkening->dose(i, r);
          weight_ *= m_HFDarkening->degradation(normalized_lumi * dose_acquired);
        }
      }
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalSim") << "HCalSD::getFromLibrary: HFLumiDarkening at "
                                  << "r= " << r << ", z= " << z << " Dose= " << dose_acquired << " weight= " << weight_;
#endif
    }

    if (useParam) {
      getFromParam(aStep, kill);
#ifdef EDM_ML_DEBUG
      G4String nameVolume = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName();
      edm::LogVerbatim("HcalSim") << "HCalSD: " << getNumberOfHits() << " hits from parametrization in " << nameVolume
                                  << " for Track " << track->GetTrackID() << " ("
                                  << track->GetDefinition()->GetParticleName() << ")";
#endif
    } else if (useShowerLibrary && !G4TrackToParticleID::isMuon(track)) {
      if (G4TrackToParticleID::isGammaElectronPositron(track) || G4TrackToParticleID::isStableHadronIon(track)) {
#ifdef EDM_ML_DEBUG
        auto nameVolume = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName();
        edm::LogVerbatim("HcalSim") << "HCalSD: Starts shower library from " << nameVolume << " for Track "
                                    << track->GetTrackID() << " (" << track->GetDefinition()->GetParticleName() << ")";

#endif
        getFromHFLibrary(aStep, kill);
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HCalSD::getFromLibrary ID= " << track->GetTrackID() << " ("
                              << track->GetDefinition()->GetParticleName() << ") kill= " << kill
                              << " weight= " << weight_ << " depth= " << depth_ << " isHF: " << isHF;
#endif
  return kill;
}

double HCalSD::getEnergyDeposit(const G4Step* aStep) {
  double destep(0.0);
  auto const lv = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
  auto const theTrack = aStep->GetTrack();

  if (isHF) {
    if (useShowerLibrary && G4TrackToParticleID::isMuon(theTrack) && isItFibre(lv)) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalSim") << "HCalSD: Hit at Fibre in LV " << lv->GetName() << " for track "
                                  << aStep->GetTrack()->GetTrackID() << " ("
                                  << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
#endif
      hitForFibre(aStep);
    }
    return destep;
  }

  if (isItPMT(lv)) {
    if (usePMTHit && showerPMT) {
      getHitPMT(aStep);
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << "HCalSD: Hit from PMT parametrization in LV " << lv->GetName() << " for Track "
                                << aStep->GetTrack()->GetTrackID() << " ("
                                << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
#endif
    return destep;

  } else if (isItStraightBundle(lv)) {
    if (useFibreBundle && showerBundle) {
      getHitFibreBundle(aStep, false);
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << "HCalSD: Hit from straight FibreBundle in LV: " << lv->GetName() << " for track "
                                << aStep->GetTrack()->GetTrackID() << " ("
                                << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
#endif
    return destep;

  } else if (isItConicalBundle(lv)) {
    if (useFibreBundle && showerBundle) {
      getHitFibreBundle(aStep, true);
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << "HCalSD: Hit from conical FibreBundle PV: " << lv->GetName() << " for track "
                                << aStep->GetTrack()->GetTrackID() << " ("
                                << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
#endif
    return destep;
  }

  // normal hit
  destep = aStep->GetTotalEnergyDeposit();

  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  uint32_t detid = setDetUnitId(aStep);
  int det(0), ieta(0), phi(0), z(0), lay, depth(-1);
  if (testNumber) {
    HcalTestNumbering::unpackHcalIndex(detid, det, z, depth, ieta, phi, lay);
    if (z == 0) {
      z = -1;
    }
  } else {
    HcalDetId hcid(detid);
    det = hcid.subdetId();
    ieta = hcid.ietaAbs();
    phi = hcid.iphi();
    z = hcid.zside();
  }
  lay = (touch->GetReplicaNumber(0) / 10) % 100 + 1;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HCalSD: det: " << det << " ieta: " << ieta << " iphi: " << phi << " zside " << z
                              << "  lay: " << lay - 2;
#endif
  if (depth_ == 0 && (det == 1 || det == 2) && ((!testNumber) || neutralDensity))
    weight_ = hcalConstants_->getLayer0Wt(det, phi, z);
  if (useLayerWt) {
    G4ThreeVector hitPoint = aStep->GetPreStepPoint()->GetPosition();
    weight_ = layerWeight(det + 2, hitPoint, depth_, lay);
  }

  if (agingFlagHB && m_HBDarkening && det == 1) {
    double dweight = m_HBDarkening->degradation(deliveredLumi, ieta, lay);
    weight_ *= dweight;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << "HCalSD: HB Lumi: " << deliveredLumi << " coefficient = " << dweight
                                << " Weight= " << weight_;
#endif
  }

  if (agingFlagHE && m_HEDarkening && det == 2) {
    double dweight = m_HEDarkening->degradation(deliveredLumi, ieta, lay);
    weight_ *= dweight;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << "HCalSD: HB Lumi: " << deliveredLumi << " coefficient = " << dweight
                                << " Weight= " << weight_;
#endif
  }

  if (suppressHeavy) {
    TrackInformation* trkInfo = (TrackInformation*)(theTrack->GetUserInformation());
    if (trkInfo) {
      int pdg = theTrack->GetDefinition()->GetPDGEncoding();
      if (!(trkInfo->isPrimary())) {  // Only secondary particles
        double ke = theTrack->GetKineticEnergy();
        if (pdg / 1000000000 == 1 && (pdg / 10000) % 100 > 0 && (pdg / 10) % 100 > 0 && ke < kmaxIon)
          weight_ = 0;
        if ((pdg == 2212) && (ke < kmaxProton))
          weight_ = 0;
        if ((pdg == 2112) && (ke < kmaxNeutron))
          weight_ = 0;
      }
    }
  }
  double wt0(1.0);
  if (useBirk) {
    const G4Material* mat = aStep->GetPreStepPoint()->GetMaterial();
    if (isItScintillator(mat))
      wt0 = getAttenuation(aStep, birk1, birk2, birk3);
  }
  weight_ *= wt0;
  double wt1 = getResponseWt(theTrack);
  double wt2 = theTrack->GetWeight();
  double edep = weight_ * wt1 * destep;
  if (wt2 > 0.0) {
    edep *= wt2;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HCalSD: edep= " << edep << " Det: " << det + 2 << " depth= " << depth_
                              << " weight= " << weight_ << " wt0= " << wt0 << " wt1= " << wt1 << " wt2= " << wt2;
#endif
  return edep;
}

uint32_t HCalSD::setDetUnitId(const G4Step* aStep) {
  auto const prePoint = aStep->GetPreStepPoint();
  auto const touch = prePoint->GetTouchable();
  const G4ThreeVector& hitPoint = prePoint->GetPosition();

  int depth = (touch->GetReplicaNumber(0)) % 10 + 1;
  int lay = (touch->GetReplicaNumber(0) / 10) % 100 + 1;
  int det = (touch->GetReplicaNumber(1)) / 1000;

  uint32_t idx = setDetUnitId(det, hitPoint, depth, lay);
#ifdef EDM_ML_DEBUG
  if (depth == 1) {
    edm::LogVerbatim("HcalSim") << "HCalSD: Check for " << det << ":" << depth << ":" << lay << " ID " << std::hex
                                << idx << std::dec;
    int det0, z0, depth0, eta0, phi0, lay0(-1);
    if (testNumber) {
      HcalTestNumbering::unpackHcalIndex(idx, det0, z0, depth0, eta0, phi0, lay0);
    } else {
      HcalDetId hcid0(idx);
      det0 = hcid0.subdetId();
      eta0 = hcid0.ietaAbs();
      phi0 = hcid0.iphi();
      z0 = hcid0.zside();
      depth0 = hcid0.depth();
    }
    edm::LogVerbatim("HcalSim") << "HCalSD: det|z|depth|eta|phi|lay " << det0 << ":" << z0 << ":" << depth0 << ":"
                                << eta0 << ":" << phi0 << ":" << lay0;
    printVolume(touch);
  }
#endif
  return idx;
}

void HCalSD::setNumberingScheme(HcalNumberingScheme* scheme) {
  if (scheme != nullptr) {
    edm::LogVerbatim("HcalSim") << "HCalSD: updates numbering scheme for " << GetName();
    numberingScheme.reset(scheme);
  }
}

void HCalSD::update(const BeginOfJob* job) {}

void HCalSD::initRun() {}

bool HCalSD::filterHit(CaloG4Hit* aHit, double time) {
  double threshold = 0;
  DetId theId(aHit->getUnitID());
  switch (theId.subdetId()) {
    case HcalBarrel:
      threshold = eminHitHB;
      break;
    case HcalEndcap:
      threshold = eminHitHE;
      break;
    case HcalOuter:
      threshold = eminHitHO;
      break;
    case HcalForward:
      threshold = eminHitHF;
      break;
    default:
      break;
  }
  return ((time <= tmaxHit) && (aHit->getEnergyDeposit() > threshold));
}

uint32_t HCalSD::setDetUnitId(int det, const G4ThreeVector& pos, int depth, int lay = 1) {
  uint32_t id = 0;
  if (det == 0) {
#ifdef printDebug
    double eta = std::abs(pos.eta());
#endif
    if (std::abs(pos.z()) > maxZ_) {
      det = 5;
#ifdef printDebug
      if (eta < 2.868)
        ++detNull_[2];
#endif
    } else if (!(hcalConstants_->isHE())) {
      det = 3;
#ifdef printDebug
      ++detNull_[0];
#endif
    } else {
      double minR = minRoff_ + slopeHE_ * std::abs(pos.z());
      double maxR = maxRoff_ + slopeHE_ * std::abs(pos.z());
      det = ((pos.perp() > minR) && (pos.perp() < maxR)) ? 4 : 3;
#ifdef printDebug
      ++detNull_[det - 3];
#endif
    }
#ifdef printDEBUG
    edm::LogVerbatim("HcalSim") << "Position " << pos.perp() << ":" << std::abs(pos.z()) << " Limits "
                                << !(hcalConstants_->isHE()) << ":" << maxZ_ << " det " << det;
  } else {
    ++detNull_[3];
#endif
  }

  if (numberingFromDDD.get()) {
    //get the ID's as eta, phi, depth, ... indices
    HcalNumberingFromDDD::HcalID tmp =
        numberingFromDDD->unitID(det, math::XYZVectorD(pos.x(), pos.y(), pos.z()), depth, lay);
    id = setDetUnitId(tmp);
  }
  return id;
}

uint32_t HCalSD::setDetUnitId(HcalNumberingFromDDD::HcalID& tmp) {
  modifyDepth(tmp);
  uint32_t id = (numberingScheme.get()) ? numberingScheme->getUnitID(tmp) : 0;
  if ((!testNumber) && m_HcalTestNS.get()) {
    bool ok = m_HcalTestNS->compare(tmp, id);
    if (!ok)
      edm::LogWarning("HcalSim") << "Det ID from HCalSD " << HcalDetId(id) << " " << std::hex << id << std::dec
                                 << " does not match one from relabller for " << tmp.subdet << ":" << tmp.etaR << ":"
                                 << tmp.phi << ":" << tmp.phis << ":" << tmp.depth << ":" << tmp.lay << std::endl;
  }
  return id;
}

bool HCalSD::isItHF(const G4Step* aStep) {
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int levels = (touch->GetHistoryDepth()) + 1;
  for (unsigned int it = 0; it < hfNames.size(); ++it) {
    if (levels >= hfLevels[it]) {
      const G4LogicalVolume* lv = touch->GetVolume(levels - hfLevels[it])->GetLogicalVolume();
      if (lv == hfLV[it])
        return true;
    }
  }
  return false;
}

bool HCalSD::isItHF(const G4String& name) {
  for (const auto& nam : hfNames)
    if (name == static_cast<G4String>(nam)) {
      return true;
    }
  return false;
}

bool HCalSD::isItFibre(const G4LogicalVolume* lv) {
  for (auto lvol : fibreLV)
    if (lv == lvol) {
      return true;
    }
  return false;
}

bool HCalSD::isItFibre(const G4String& name) {
  for (const auto& nam : fibreNames)
    if (name == static_cast<G4String>(nam)) {
      return true;
    }
  return false;
}

bool HCalSD::isItPMT(const G4LogicalVolume* lv) {
  for (auto lvol : pmtLV)
    if (lv == lvol) {
      return true;
    }
  return false;
}

bool HCalSD::isItStraightBundle(const G4LogicalVolume* lv) {
  for (auto lvol : fibre1LV)
    if (lv == lvol) {
      return true;
    }
  return false;
}

bool HCalSD::isItConicalBundle(const G4LogicalVolume* lv) {
  for (auto lvol : fibre2LV)
    if (lv == lvol) {
      return true;
    }
  return false;
}

bool HCalSD::isItScintillator(const G4Material* mat) {
  for (auto amat : materials)
    if (amat == mat) {
      return true;
    }
  return false;
}

bool HCalSD::isItinFidVolume(const G4ThreeVector& hitPoint) {
  bool flag = true;
  if (applyFidCut) {
    int npmt = HFFibreFiducial::PMTNumber(hitPoint);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << "HCalSD::isItinFidVolume:#PMT= " << npmt << " for hit point " << hitPoint;
#endif
    if (npmt <= 0)
      flag = false;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HCalSD::isItinFidVolume: point " << hitPoint << " return flag " << flag;
#endif
  return flag;
}

void HCalSD::getFromHFLibrary(const G4Step* aStep, bool& isKilled) {
  std::vector<HFShowerLibrary::Hit> hits = showerLibrary->getHits(aStep, isKilled, weight_, false);
  if (!isKilled || hits.empty()) {
    return;
  }

  int primaryID = setTrackID(aStep);

  // Reset entry point for new primary
  resetForNewPrimary(aStep);

  auto const theTrack = aStep->GetTrack();
  int det = 5;

  if (G4TrackToParticleID::isGammaElectronPositron(theTrack)) {
    edepositEM = 1. * GeV;
    edepositHAD = 0.;
  } else {
    edepositEM = 0.;
    edepositHAD = 1. * GeV;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HCalSD::getFromLibrary " << hits.size() << " hits for " << GetName() << " of "
                              << primaryID << " with " << theTrack->GetDefinition()->GetParticleName() << " of "
                              << aStep->GetPreStepPoint()->GetKineticEnergy() / GeV << " GeV";
#endif
  for (unsigned int i = 0; i < hits.size(); ++i) {
    G4ThreeVector hitPoint = hits[i].position;
    if (isItinFidVolume(hitPoint)) {
      int depth = hits[i].depth;
      double time = hits[i].time;
      unsigned int unitID = setDetUnitId(det, hitPoint, depth);
      currentID.setID(unitID, time, primaryID, 0);
#ifdef plotDebug
      plotProfile(aStep, hitPoint, 1.0 * GeV, time, depth);
      bool emType = G4TrackToParticleID::isGammaElectronPositron(theTrack->GetDefinition()->GetPDGEncoding());
      plotHF(hitPoint, emType);
#endif
      processHit(aStep);
    }
  }
}

void HCalSD::hitForFibre(const G4Step* aStep) {  // if not ParamShower

  std::vector<HFShower::Hit> hits = hfshower->getHits(aStep, weight_);
  if (hits.empty()) {
    return;
  }

  auto const theTrack = aStep->GetTrack();
  int primaryID = setTrackID(aStep);
  int det = 5;

  if (G4TrackToParticleID::isGammaElectronPositron(theTrack)) {
    edepositEM = 1. * GeV;
    edepositHAD = 0.;
  } else {
    edepositEM = 0.;
    edepositHAD = 1. * GeV;
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HCalSD::hitForFibre " << hits.size() << " hits for " << GetName() << " of "
                              << primaryID << " with " << theTrack->GetDefinition()->GetParticleName() << " of "
                              << aStep->GetPreStepPoint()->GetKineticEnergy() / GeV << " GeV in detector type " << det;
#endif

  for (unsigned int i = 0; i < hits.size(); ++i) {
    G4ThreeVector hitPoint = hits[i].position;
    if (isItinFidVolume(hitPoint)) {
      int depth = hits[i].depth;
      double time = hits[i].time;
      unsigned int unitID = setDetUnitId(det, hitPoint, depth);
      currentID.setID(unitID, time, primaryID, 0);
#ifdef plotDebug
      plotProfile(aStep, hitPoint, edepositEM, time, depth);
      bool emType = (edepositEM > 0.) ? true : false;
      plotHF(hitPoint, emType);
#endif
      processHit(aStep);
    }
  }
}

void HCalSD::getFromParam(const G4Step* aStep, bool& isKilled) {
  std::vector<HFShowerParam::Hit> hits = showerParam->getHits(aStep, weight_, isKilled);
  if (!isKilled || hits.empty()) {
    return;
  }

  int primaryID = setTrackID(aStep);
  int det = 5;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HCalSD::getFromParam " << hits.size() << " hits for " << GetName() << " of "
                              << primaryID << " with " << aStep->GetTrack()->GetDefinition()->GetParticleName()
                              << " of " << aStep->GetPreStepPoint()->GetKineticEnergy() / GeV
                              << " GeV in detector type " << det;
#endif
  for (unsigned int i = 0; i < hits.size(); ++i) {
    G4ThreeVector hitPoint = hits[i].position;
    int depth = hits[i].depth;
    double time = hits[i].time;
    unsigned int unitID = setDetUnitId(det, hitPoint, depth);
    currentID.setID(unitID, time, primaryID, 0);
    edepositEM = hits[i].edep * GeV;
    edepositHAD = 0.;
#ifdef plotDebug
    plotProfile(aStep, hitPoint, edepositEM, time, depth);
#endif
    processHit(aStep);
  }
}

void HCalSD::getHitPMT(const G4Step* aStep) {
  auto const preStepPoint = aStep->GetPreStepPoint();
  auto const theTrack = aStep->GetTrack();
  double edep = showerPMT->getHits(aStep);

  if (edep >= 0.) {
    edep *= GeV;
    double etrack = preStepPoint->GetKineticEnergy();
    int primaryID = 0;
    if (etrack >= energyCut) {
      primaryID = theTrack->GetTrackID();
    } else {
      primaryID = theTrack->GetParentID();
      if (primaryID == 0)
        primaryID = theTrack->GetTrackID();
    }
    // Reset entry point for new primary
    resetForNewPrimary(aStep);
    //
    int det = static_cast<int>(HcalForward);
    const G4ThreeVector& hitPoint = preStepPoint->GetPosition();
    double rr = (hitPoint.x() * hitPoint.x() + hitPoint.y() * hitPoint.y());
    double phi = (rr == 0. ? 0. : atan2(hitPoint.y(), hitPoint.x()));
    double etaR = showerPMT->getRadius();
    int depth = 1;
    if (etaR < 0) {
      depth = 2;
      etaR = -etaR;
    }
    if (hitPoint.z() < 0)
      etaR = -etaR;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << "HCalSD::Hit for Detector " << det << " etaR " << etaR << " phi " << phi / deg
                                << " depth " << depth;
#endif
    double time = (aStep->GetPostStepPoint()->GetGlobalTime());
    uint32_t unitID = 0;
    if (numberingFromDDD) {
      HcalNumberingFromDDD::HcalID tmp = numberingFromDDD->unitID(det, etaR, phi, depth, 1);
      unitID = setDetUnitId(tmp);
    }
    currentID.setID(unitID, time, primaryID, 1);

    edepositHAD = aStep->GetTotalEnergyDeposit();
    edepositEM = -edepositHAD + edep;
#ifdef plotDebug
    plotProfile(aStep, hitPoint, edep, time, depth);
#endif
#ifdef EDM_ML_DEBUG
    double beta = preStepPoint->GetBeta();
    edm::LogVerbatim("HcalSim") << "HCalSD::getHitPMT 1 hit for " << GetName() << " of " << primaryID << " with "
                                << theTrack->GetDefinition()->GetParticleName() << " of "
                                << preStepPoint->GetKineticEnergy() / GeV << " GeV with velocity " << beta << " UnitID "
                                << std::hex << unitID << std::dec;
#endif
    processHit(aStep);
  }
}

void HCalSD::getHitFibreBundle(const G4Step* aStep, bool type) {
  auto const preStepPoint = aStep->GetPreStepPoint();
  auto const theTrack = aStep->GetTrack();
  double edep = showerBundle->getHits(aStep, type);

  if (edep >= 0.0) {
    edep *= GeV;
    double etrack = preStepPoint->GetKineticEnergy();
    int primaryID = 0;
    if (etrack >= energyCut) {
      primaryID = theTrack->GetTrackID();
    } else {
      primaryID = theTrack->GetParentID();
      if (primaryID == 0)
        primaryID = theTrack->GetTrackID();
    }
    // Reset entry point for new primary
    resetForNewPrimary(aStep);
    //
    int det = static_cast<int>(HcalForward);
    const G4ThreeVector& hitPoint = preStepPoint->GetPosition();
    double rr = hitPoint.x() * hitPoint.x() + hitPoint.y() * hitPoint.y();
    double phi = rr == 0. ? 0. : atan2(hitPoint.y(), hitPoint.x());
    double etaR = showerBundle->getRadius();
    int depth = 1;
    if (etaR < 0.) {
      depth = 2;
      etaR = -etaR;
    }
    if (hitPoint.z() < 0.)
      etaR = -etaR;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << "HCalSD::Hit for Detector " << det << " etaR " << etaR << " phi " << phi / deg
                                << " depth " << depth;
#endif
    double time = (aStep->GetPostStepPoint()->GetGlobalTime());
    uint32_t unitID = 0;
    if (numberingFromDDD) {
      HcalNumberingFromDDD::HcalID tmp = numberingFromDDD->unitID(det, etaR, phi, depth, 1);
      unitID = setDetUnitId(tmp);
    }
    if (type)
      currentID.setID(unitID, time, primaryID, 3);
    else
      currentID.setID(unitID, time, primaryID, 2);

    edepositHAD = aStep->GetTotalEnergyDeposit();
    edepositEM = -edepositHAD + edep;
#ifdef plotDebug
    plotProfile(aStep, hitPoint, edep, time, depth);
#endif
#ifdef EDM_ML_DEBUG
    double beta = preStepPoint->GetBeta();
    edm::LogVerbatim("HcalSim") << "HCalSD::getHitFibreBundle 1 hit for " << GetName() << " of " << primaryID
                                << " with " << theTrack->GetDefinition()->GetParticleName() << " of "
                                << preStepPoint->GetKineticEnergy() / GeV << " GeV with velocity " << beta << " UnitID "
                                << std::hex << unitID << std::dec;
#endif
    processHit(aStep);
  }  // non-zero energy deposit
}

void HCalSD::readWeightFromFile(const std::string& fName) {
  std::ifstream infile;
  int entry = 0;
  infile.open(fName.c_str(), std::ios::in);
  if (infile) {
    int det, zside, etaR, phi, lay;
    double wt;
    while (infile >> det >> zside >> etaR >> phi >> lay >> wt) {
      uint32_t id = HcalTestNumbering::packHcalIndex(det, zside, 1, etaR, phi, lay);
      layerWeights.insert(std::pair<uint32_t, double>(id, wt));
      ++entry;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalSim") << "HCalSD::readWeightFromFile:Entry " << entry << " ID " << std::hex << id
                                  << std::dec << " (" << det << "/" << zside << "/1/" << etaR << "/" << phi << "/"
                                  << lay << ") Weight " << wt;
#endif
    }
    infile.close();
  }
  edm::LogVerbatim("HcalSim") << "HCalSD::readWeightFromFile: reads " << entry << " weights from " << fName;
  if (entry <= 0)
    useLayerWt = false;
}

double HCalSD::layerWeight(int det, const G4ThreeVector& pos, int depth, int lay) {
  double wt = 1.;
  if (numberingFromDDD) {
    //get the ID's as eta, phi, depth, ... indices
    HcalNumberingFromDDD::HcalID tmp =
        numberingFromDDD->unitID(det, math::XYZVectorD(pos.x(), pos.y(), pos.z()), depth, lay);
    modifyDepth(tmp);
    uint32_t id = HcalTestNumbering::packHcalIndex(tmp.subdet, tmp.zside, 1, tmp.etaR, tmp.phis, tmp.lay);
    std::map<uint32_t, double>::const_iterator ite = layerWeights.find(id);
    if (ite != layerWeights.end())
      wt = ite->second;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << "HCalSD::layerWeight: ID " << std::hex << id << std::dec << " (" << tmp.subdet << "/"
                                << tmp.zside << "/1/" << tmp.etaR << "/" << tmp.phis << "/" << tmp.lay << ") Weight "
                                << wt;
#endif
  }
  return wt;
}

void HCalSD::plotProfile(const G4Step* aStep, const G4ThreeVector& global, double edep, double time, int id) {
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  static const unsigned int names = 10;
  static const G4String modName[names] = {
      "HEModule", "HVQF", "HBModule", "MBAT", "MBBT", "MBBTC", "MBBT_R1P", "MBBT_R1M", "MBBT_R1PX", "MBBT_R1MX"};
  G4ThreeVector local;
  bool found = false;
  double depth = -2000;
  int idx = 4;
  for (int n = 0; n < touch->GetHistoryDepth(); ++n) {
    G4String name(static_cast<std::string>(dd4hep::dd::noNamespace(touch->GetVolume(n)->GetName())));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << "plotProfile Depth " << n << " Name " << name;
#endif
    for (unsigned int ii = 0; ii < names; ++ii) {
      if (name == modName[ii]) {
        found = true;
        int dn = touch->GetHistoryDepth() - n;
        local = touch->GetHistory()->GetTransform(dn).TransformPoint(global);
        if (ii == 0) {
          depth = local.z() - 4006.5;
          idx = 1;
        } else if (ii == 1) {
          depth = local.z() + 825.0;
          idx = 3;
        } else if (ii == 2) {
          depth = local.x() - 1775.;
          idx = 0;
        } else {
          depth = local.y() + 15.;
          idx = 2;
        }
        break;
      }
    }
    if (found)
      break;
  }
  if (!found)
    depth = std::abs(global.z()) - 11500;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "plotProfile Found " << found << " Global " << global << " Local " << local
                              << " depth " << depth << " ID " << id << " EDEP " << edep << " Time " << time;
#endif
  if (hit_[idx] != nullptr)
    hit_[idx]->Fill(edep);
  if (time_[idx] != nullptr)
    time_[idx]->Fill(time, edep);
  if (dist_[idx] != nullptr)
    dist_[idx]->Fill(depth, edep);
  int jd = 2 * idx + id - 7;
  if (jd >= 0 && jd < 4) {
    jd += 5;
    if (hit_[jd] != nullptr)
      hit_[jd]->Fill(edep);
    if (time_[jd] != nullptr)
      time_[jd]->Fill(time, edep);
    if (dist_[jd] != nullptr)
      dist_[jd]->Fill(depth, edep);
  }
}

void HCalSD::plotHF(const G4ThreeVector& hitPoint, bool emType) {
  double zv = std::abs(hitPoint.z()) - gpar[4];
  if (emType) {
    if (hzvem != nullptr)
      hzvem->Fill(zv);
  } else {
    if (hzvhad != nullptr)
      hzvhad->Fill(zv);
  }
}

void HCalSD::modifyDepth(HcalNumberingFromDDD::HcalID& id) {
  if (id.subdet == 4) {
    int ieta = (id.zside == 0) ? -id.etaR : id.etaR;
    if (hcalConstants_->maxHFDepth(ieta, id.phis) > 2) {
      if (id.depth <= 2) {
        if (G4UniformRand() > 0.5)
          id.depth += 2;
      }
    }
  } else if ((id.subdet == 1 || id.subdet == 2) && testNumber) {
    id.depth = (depth_ == 0) ? 1 : 2;
  }
}

void HCalSD::initEvent(const BeginOfEvent*) {
#ifdef printDebug
  detNull_ = {0, 0, 0, 0};
#endif
}

void HCalSD::endEvent() {
#ifdef printDebug
  int sum = detNull_[0] + detNull_[1] + detNull_[2];
  if (sum > 0)
    edm::LogVerbatim("HcalSim") << "NullDets " << detNull_[0] << " " << detNull_[1] << " " << detNull_[2] << " "
                                << detNull_[3] << " " << (static_cast<float>(sum) / (sum + detNull_[3]));
#endif
}

void HCalSD::printVolume(const G4VTouchable* touch) const {
  if (touch) {
#ifdef EDM_ML_DEBUG
    int level = ((touch->GetHistoryDepth()) + 1);
    edm::LogVerbatim("CaloSimX") << "HCalSD::printVolume with " << level << " levels";
    static const std::string unknown("Unknown");
    //Get name and copy numbers
    for (int ii = 0; ii < level; ii++) {
      int i = level - ii - 1;
      G4VPhysicalVolume* pv = touch->GetVolume(i);
      G4String name = (pv != nullptr) ? pv->GetName() : unknown;
      G4int copyno = touch->GetReplicaNumber(i);
      edm::LogVerbatim("HcalSim") << "[" << ii << "] " << name << ":" << copyno;
    }
#endif
  }
}
