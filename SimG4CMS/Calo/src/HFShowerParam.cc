///////////////////////////////////////////////////////////////////////////////
// File: HFShowerParam.cc
// Description: Parametrized version of HF hits
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShowerParam.h"
#include "SimG4CMS/Calo/interface/HFFibreFiducial.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4NavigationHistory.hh"
#include "Randomize.hh"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>

//#define EDM_ML_DEBUG
//#define plotDebug
//#define mkdebug

HFShowerParam::HFShowerParam(const std::string& name,
                             const HcalDDDSimConstants* hcons,
                             const HcalSimulationParameters* hps,
                             edm::ParameterSet const& p)
    : hcalConstants_(hcons), fillHisto_(false) {
  edm::ParameterSet m_HF = p.getParameter<edm::ParameterSet>("HFShower");
  edm::ParameterSet m_HF2 = m_HF.getParameter<edm::ParameterSet>("HFShowerBlock");
  pePerGeV_ = m_HF.getParameter<double>("PEPerGeV");
  trackEM_ = m_HF.getParameter<bool>("TrackEM");
  bool useShowerLibrary = m_HF.getParameter<bool>("UseShowerLibrary");
  bool useGflash = m_HF.getParameter<bool>("UseHFGflash");
  edMin_ = m_HF.getParameter<double>("EminLibrary");
  ignoreTimeShift_ = m_HF2.getParameter<bool>("IgnoreTimeShift");
  onlyLong_ = m_HF2.getParameter<bool>("OnlyLong");
  ref_index_ = m_HF.getParameter<double>("RefIndex");
  double lambdaMean = m_HF.getParameter<double>("LambdaMean");
  aperture_ = cos(asin(m_HF.getParameter<double>("Aperture")));
  applyFidCut_ = m_HF.getParameter<bool>("ApplyFiducialCut");
  parametrizeLast_ = m_HF.getUntrackedParameter<bool>("ParametrizeLast", false);
  gpar_ = hcalConstants_->getGparHF();

  edm::LogVerbatim("HFShower") << "HFShowerParam::Use of shower library is set to " << useShowerLibrary
                               << " Use of Gflash is set to " << useGflash << " P.E. per GeV " << pePerGeV_
                               << ", ref. index of fibre " << ref_index_ << ", Track EM Flag " << trackEM_ << ", edMin "
                               << edMin_ << " GeV, use of Short fibre info in"
                               << " shower library set to " << !(onlyLong_)
                               << " ignore flag for time shift in fire is set to " << ignoreTimeShift_
                               << ", use of parametrization for last part set to " << parametrizeLast_
                               << ", Mean lambda " << lambdaMean << ", aperture (cutoff) " << aperture_
                               << ", Application of Fiducial Cut " << applyFidCut_;

#ifdef plotDebug
  edm::Service<TFileService> tfile;
  if (tfile.isAvailable()) {
    fillHisto_ = true;
    edm::LogVerbatim("HFShower") << "HFShowerParam::Save histos in directory "
                                 << "ProfileFromParam";
    TFileDirectory showerDir = tfile->mkdir("ProfileFromParam");
    hzvem_ = showerDir.make<TH1F>("hzvem", "Longitudinal Profile (EM Part);Number of PE", 330, 0.0, 1650.0);
    hzvhad_ = showerDir.make<TH1F>("hzvhad", "Longitudinal Profile (Had Part);Number of PE", 330, 0.0, 1650.0);
    em_2d_1_ = showerDir.make<TH2F>(
        "em_2d_1", "Lateral Profile vs. Shower Depth;cm;Events", 800, 800.0, 1600.0, 100, 50.0, 150.0);
    em_long_1_ =
        showerDir.make<TH1F>("em_long_1", "Longitudinal Profile;Radiation Length;Number of Spots", 800, 800.0, 1600.0);
    em_long_1_tuned_ = showerDir.make<TH1F>(
        "em_long_1_tuned", "Longitudinal Profile;Radiation Length;Number of Spots", 800, 800.0, 1600.0);
    em_lateral_1_ = showerDir.make<TH1F>("em_lateral_1", "Lateral Profile;cm;Events", 100, 50.0, 150.0);
    em_2d_2_ = showerDir.make<TH2F>(
        "em_2d_2", "Lateral Profile vs. Shower Depth;cm;Events", 800, 800.0, 1600.0, 100, 50.0, 150.0);
    em_long_2_ =
        showerDir.make<TH1F>("em_long_2", "Longitudinal Profile;Radiation Length;Number of Spots", 800, 800.0, 1600.0);
    em_lateral_2_ = showerDir.make<TH1F>("em_lateral_2", "Lateral Profile;cm;Events", 100, 50.0, 150.0);
    em_long_gflash_ = showerDir.make<TH1F>(
        "em_long_gflash", "Longitudinal Profile From GFlash;cm;Number of Spots", 800, 800.0, 1600.0);
    em_long_sl_ = showerDir.make<TH1F>(
        "em_long_sl", "Longitudinal Profile From Shower Library;cm;Number of Spots", 800, 800.0, 1600.0);
  } else {
    fillHisto_ = false;
    edm::LogVerbatim("HFShower") << "HFShowerParam::No file is available for saving histos so the "
                                 << "flag is set to false";
  }
#endif

  if (useShowerLibrary)
    showerLibrary_ = std::make_unique<HFShowerLibrary>(name, hcalConstants_, hps, p);
  else
    showerLibrary_.reset(nullptr);
  if (useGflash)
    gflash_ = std::make_unique<HFGflash>(p);
  else
    gflash_.reset(nullptr);
  fibre_ = std::make_unique<HFFibre>(name, hcalConstants_, hps, p);
  attLMeanInv_ = fibre_->attLength(lambdaMean);
  edm::LogVerbatim("HFShower") << "att. length used for (lambda=" << lambdaMean
                               << ") = " << 1 / (attLMeanInv_ * CLHEP::cm) << " cm";
}

HFShowerParam::~HFShowerParam() {}

std::vector<HFShowerParam::Hit> HFShowerParam::getHits(const G4Step* aStep, double weight, bool& isKilled) {
  auto const preStepPoint = aStep->GetPreStepPoint();
  auto const track = aStep->GetTrack();
  bool isEM = G4TrackToParticleID::isGammaElectronPositron(track);
  const G4ThreeVector& hitPoint = preStepPoint->GetPosition();
  double zv = std::abs(hitPoint.z()) - gpar_[4] - 0.5 * gpar_[1];
  G4ThreeVector localPoint = G4ThreeVector(hitPoint.x(), hitPoint.y(), zv);

  double pin = (preStepPoint->GetTotalEnergy()) / CLHEP::GeV;
  double zint = hitPoint.z();
  double zz = std::abs(zint) - gpar_[4];

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShowerParam: getHits " << track->GetDefinition()->GetParticleName()
                               << " of energy " << pin << " GeV Pos x,y,z = " << hitPoint.x() << "," << hitPoint.y()
                               << "," << zint << " (" << zz << "," << localPoint.z() << ", "
                               << (localPoint.z() + 0.5 * gpar_[1]) << ") Local " << localPoint;
#endif
  std::vector<HFShowerParam::Hit> hits;
  HFShowerParam::Hit hit;
  hit.position = hitPoint;

  // look for other charged particles
  bool other = false;
  double pBeta = track->GetDynamicParticle()->GetTotalMomentum() / track->GetDynamicParticle()->GetTotalEnergy();
  double dirz = (track->GetDynamicParticle()->GetMomentumDirection()).z();
  if (hitPoint.z() < 0)
    dirz *= -1.;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShowerParam: getHits Momentum "
                               << track->GetDynamicParticle()->GetMomentumDirection() << " HitPoint " << hitPoint
                               << " dirz " << dirz;
#endif
  if (!isEM && track->GetDefinition()->GetPDGCharge() != 0 && pBeta > (1 / ref_index_) &&
      aStep->GetTotalEnergyDeposit() > 0.) {
    other = true;
  }

  // take only e+-/gamma/or special particles
  if (isEM || other) {
    // Leave out the last part
    double edep = 0.;
    if ((!trackEM_) && ((zz < (gpar_[1] - gpar_[2])) || parametrizeLast_) && (!other)) {
      edep = pin;
      isKilled = true;
    } else if ((track->GetDefinition()->GetPDGCharge() != 0) && (pBeta > (1 / ref_index_)) && (dirz > aperture_)) {
      edep = (aStep->GetTotalEnergyDeposit()) / GeV;
    }
    std::string path = "ShowerLibrary";
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HFShower") << "HFShowerParam: getHits edep = " << edep << " weight " << weight << " final "
                                 << edep * weight << ", Kill = " << isKilled << ", pin = " << pin
                                 << ", edMin = " << edMin_ << " Other " << other;
#endif
    edep *= weight;
    if (edep > 0) {
      if ((showerLibrary_.get() || gflash_.get()) && isKilled && pin > edMin_ && (!other)) {
        if (showerLibrary_.get()) {
          std::vector<HFShowerLibrary::Hit> hitSL = showerLibrary_->getHits(aStep, isKilled, weight, onlyLong_);
          for (unsigned int i = 0; i < hitSL.size(); i++) {
            bool ok = true;
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HFShower") << "HFShowerParam: getHits applyFidCut = " << applyFidCut_;
#endif
            if (applyFidCut_) {  // @@ For showerlibrary no z-cut for Short (no z)
              int npmt = HFFibreFiducial::PMTNumber(hitSL[i].position);
              if (npmt <= 0)
                ok = false;
            }
            if (ok) {
              hit.position = hitSL[i].position;
              hit.depth = hitSL[i].depth;
              hit.time = hitSL[i].time;
              hit.edep = 1;
              hits.push_back(hit);
#ifdef plotDebug
              if (fillHisto_) {
                double zv = std::abs(hit.position.z()) - gpar_[4];
                hzvem_->Fill(zv);
                em_long_sl_->Fill(hit.position.z() / CLHEP::cm);
                double sq = sqrt(pow(hit.position.x() / CLHEP::cm, 2) + pow(hit.position.y() / CLHEP::cm, 2));
                double zp = hit.position.z() / CLHEP::cm;
                if (hit.depth == 1) {
                  em_2d_1_->Fill(zp, sq);
                  em_lateral_1_->Fill(sq);
                  em_long_1_->Fill(zp);
                } else if (hit.depth == 2) {
                  em_2d_2_->Fill(zp, sq);
                  em_lateral_2_->Fill(sq);
                  em_long_2_->Fill(zp);
                }
              }
#endif
#ifdef EDM_ML_DEBUG
              edm::LogVerbatim("HFShower")
                  << "HFShowerParam: Hit at depth " << hit.depth << " with edep " << hit.edep << " Time " << hit.time;
#endif
            }
          }
        } else {  // GFlash clusters with known z
          std::vector<HFGflash::Hit> hitSL = gflash_->gfParameterization(aStep, onlyLong_);
          for (unsigned int i = 0; i < hitSL.size(); ++i) {
            bool ok = true;
            G4ThreeVector pe_effect(hitSL[i].position.x(), hitSL[i].position.y(), hitSL[i].position.z());
            double zv = std::abs(pe_effect.z()) - gpar_[4];
            //depth
            int depth = 1;
            int npmt = 0;
            if (zv < 0. || zv > gpar_[1]) {
#ifdef mkdebug
              edm::LogVerbatim("HFShower") << "-#Zcut-HFShowerParam::getHits:z=" << zv << ",m=" << gpar_[1];
#endif
              ok = false;
            }
            if (ok && applyFidCut_) {
              npmt = HFFibreFiducial::PMTNumber(pe_effect);
#ifdef EDM_ML_DEBUG
              edm::LogVerbatim("HFShower") << "HFShowerParam::getHits:#PMT= " << npmt << ",z = " << zv;
#endif
              if (npmt <= 0) {
#ifdef EDM_ML_DEBUG
                edm::LogVerbatim("HFShower") << "-#PMT=0 cut-HFShowerParam::getHits: npmt = " << npmt;
#endif
                ok = false;
              } else if (npmt > 24) {  // a short fibre
                if (zv > gpar_[0]) {
                  depth = 2;
                } else {
#ifdef EDM_ML_DEBUG
                  edm::LogVerbatim("HFShower") << "-SHORT cut-HFShowerParam::getHits:zMin=" << gpar_[0];
#endif
                  ok = false;
                }
              }
#ifdef EDM_ML_DEBUG
              edm::LogVerbatim("HFShower")
                  << "HFShowerParam: npmt " << npmt << " zv " << std::abs(pe_effect.z()) << ":" << gpar_[4] << ":" << zv
                  << ":" << gpar_[0] << " ok " << ok << " depth " << depth;
#endif
            } else {
              if (G4UniformRand() > 0.5)
                depth = 2;
              if (depth == 2 && zv < gpar_[0])
                ok = false;
            }
            //attenuation
            double dist = fibre_->zShift(localPoint, depth, 0);  // distance to PMT
            double r1 = G4UniformRand();
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HFShower") << "HFShowerParam:Distance to PMT (" << npmt << ") " << dist
                                         << ", exclusion flag " << (r1 > exp(-attLMeanInv_ * zv));
#endif
            if (r1 > exp(-attLMeanInv_ * dist))
              ok = false;
            if (ok) {
              double r2 = G4UniformRand();
#ifdef EDM_ML_DEBUG
              edm::LogVerbatim("HFShower")
                  << "HFShowerParam:Extra exclusion " << r2 << ">" << weight << " " << (r2 > weight);
#endif
              if (r2 < weight) {
                double time = (ignoreTimeShift_) ? 0 : fibre_->tShift(localPoint, depth, 0);

                hit.position = hitSL[i].position;
                hit.depth = depth;
                hit.time = time + hitSL[i].time;
                hit.edep = 1;
                hits.push_back(hit);
#ifdef plotDebug
                if (fillHisto_) {
                  em_long_gflash_->Fill(pe_effect.z() / CLHEP::cm, hitSL[i].edep);
                  hzvem_->Fill(zv);
                  double sq = sqrt(pow(hit.position.x() / CLHEP::cm, 2) + pow(hit.position.y() / CLHEP::cm, 2));
                  double zp = hit.position.z() / CLHEP::cm;
                  if (hit.depth == 1) {
                    em_2d_1_->Fill(zp, sq);
                    em_lateral_1_->Fill(s);
                    em_long_1_->Fill(zp);
                  } else if (hit.depth == 2) {
                    em_2d_2_->Fill(zp, sq);
                    em_lateral_2_->Fill(sq);
                    em_long_2_->Fill(zp);
                  }
                }
#endif
#ifdef EDM_ML_DEBUG
                edm::LogVerbatim("HFShower")
                    << "HFShowerParam: Hit at depth " << hit.depth << " with edep " << hit.edep << " Time " << hit.time;
#endif
              }
            }
          }
        }
      } else {
        path = "Rest";
        edep *= pePerGeV_;
        double tSlice = (aStep->GetPostStepPoint()->GetGlobalTime());
        double time = (ignoreTimeShift_) ? 0 : fibre_->tShift(localPoint, 1, 0);  // remaining part
        bool ok = true;
        if (applyFidCut_) {  // @@ For showerlibrary no z-cut for Short (no z)
          int npmt = HFFibreFiducial::PMTNumber(hitPoint);
          if (npmt <= 0)
            ok = false;
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HFShower") << "HFShowerParam: getHits hitPoint " << hitPoint << " flag " << ok;
#endif
        if (ok) {
          hit.depth = 1;
          hit.time = tSlice + time;
          hit.edep = edep;
          hits.push_back(hit);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HFShower") << "HFShowerParam: Hit at depth 1 with edep " << edep << " Time " << tSlice
                                       << ":" << time << ":" << hit.time;
#endif
#ifdef plotDebug
          double zv = std::abs(hitPoint.z()) - gpar_[4];
          if (fillHisto_) {
            hzvhad_->Fill(zv);
          }
#endif
          if (zz >= gpar_[0]) {
            time = (ignoreTimeShift_) ? 0 : fibre_->tShift(localPoint, 2, 0);
            hit.depth = 2;
            hit.time = tSlice + time;
            hits.push_back(hit);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HFShower") << "HFShowerParam: Hit at depth 2 with edep " << edep << " Time " << tSlice
                                         << ":" << time << hit.time;
#endif
#ifdef plotDebug
            if (fillHisto_) {
              hzvhad_->Fill(zv);
            }
#endif
          }
        }
      }
#ifdef EDM_ML_DEBUG
      for (unsigned int ii = 0; ii < hits.size(); ++ii) {
        double zv = std::abs(hits[ii].position.z());
        if (zv > 12790)
          edm::LogVerbatim("HFShower") << "HFShowerParam: Abnormal hit along " << path << " in "
                                       << preStepPoint->GetPhysicalVolume()->GetLogicalVolume()->GetName() << " at "
                                       << hits[ii].position << " zz " << zv << " Edep " << edep << " due to "
                                       << track->GetDefinition()->GetParticleName() << " time " << hit.time;
      }
      edm::LogVerbatim("HFShower") << "HFShowerParam: getHits kill (" << isKilled << ") track " << track->GetTrackID()
                                   << " at " << hitPoint << " and deposit " << edep << " " << hits.size() << " times"
                                   << " ZZ " << zz << " " << gpar_[0];
#endif
    }
  }
  return hits;
}
