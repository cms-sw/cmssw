///////////////////////////////////////////////////////////////////////////////
// File: SimG4HGCalValidation.cc
// Description: Main analysis class for HGCal Validation of G4 Hits
///////////////////////////////////////////////////////////////////////////////

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"

// to retreive hits
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "SimDataFormats/ValidationFormats/interface/PHGCalValidInfo.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HGCalNumberingScheme.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4NavigationHistory.hh"
#include "G4TouchableHistory.hh"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <string>

class SimG4HGCalValidation : public SimProducer,
                             public Observer<const BeginOfJob*>,
                             public Observer<const BeginOfEvent*>,
                             public Observer<const G4Step*> {
public:
  SimG4HGCalValidation(const edm::ParameterSet& p);
  ~SimG4HGCalValidation() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  SimG4HGCalValidation(const SimG4HGCalValidation&);  // stop default
  const SimG4HGCalValidation& operator=(const SimG4HGCalValidation&);

  void init();

  // observer classes
  void update(const BeginOfJob* job) override;
  void update(const BeginOfEvent* evt) override;
  void update(const G4Step* step) override;

  // analysis related class
  void layerAnalysis(PHGCalValidInfo&);
  void clear();

private:
  //Keep reference to instantiate HcalNumberingFromDDD later
  HcalNumberingFromDDD* numberingFromDDD_;

  //HGCal numbering scheme
  std::vector<HGCNumberingScheme*> hgcNumbering_;
  std::vector<HGCalNumberingScheme*> hgcalNumbering_;

  // to read from parameter set
  std::vector<std::string> names_;
  std::vector<int> types_, detTypes_, subdet_;
  std::string labelLayer_;

  // parameters from geometry
  int levelT1_, levelT2_;

  // some private members for ananlysis
  unsigned int count_;
  int verbosity_;
  double edepEE_, edepHEF_, edepHEB_;
  std::vector<double> hgcEEedep_, hgcHEFedep_, hgcHEBedep_;
  std::vector<unsigned int> dets_, hgchitDets_, hgchitIndex_;
  std::vector<double> hgchitX_, hgchitY_, hgchitZ_;
};

SimG4HGCalValidation::SimG4HGCalValidation(const edm::ParameterSet& p)
    : numberingFromDDD_(nullptr), levelT1_(999), levelT2_(999), count_(0) {
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("SimG4HGCalValidation");
  names_ = m_Anal.getParameter<std::vector<std::string> >("Names");
  types_ = m_Anal.getParameter<std::vector<int> >("Types");
  detTypes_ = m_Anal.getParameter<std::vector<int> >("DetTypes");
  labelLayer_ = m_Anal.getParameter<std::string>("LabelLayerInfo");
  verbosity_ = m_Anal.getUntrackedParameter<int>("Verbosity", 0);

  produces<PHGCalValidInfo>(labelLayer_);

  if (verbosity_ > 0) {
    edm::LogVerbatim("ValidHGCal") << "HGCalTestAnalysis:: Initialised as "
                                   << "observer of begin events and of G4step "
                                   << "with Parameter values: \n\tLabel : " << labelLayer_ << " and with "
                                   << names_.size() << " detectors";
    for (unsigned int k = 0; k < names_.size(); ++k)
      edm::LogVerbatim("ValidHGCal") << " [" << k << "] " << names_[k] << " Type " << types_[k] << " DetType "
                                     << detTypes_[k];
  }
}

SimG4HGCalValidation::~SimG4HGCalValidation() {
  delete numberingFromDDD_;
  for (auto number : hgcNumbering_)
    delete number;
  for (auto number : hgcalNumbering_)
    delete number;
}

void SimG4HGCalValidation::produce(edm::Event& e, const edm::EventSetup&) {
  std::unique_ptr<PHGCalValidInfo> productLayer(new PHGCalValidInfo);
  layerAnalysis(*productLayer);
  e.put(std::move(productLayer), labelLayer_);
}

void SimG4HGCalValidation::update(const BeginOfJob* job) {
  const edm::EventSetup* es = (*job)();
  for (unsigned int type = 0; type < types_.size(); ++type) {
    int layers(0);
    int detType = detTypes_[type];
    G4String nameX = "HGCal";
    if (types_[type] <= 1) {
      if (types_[type] == 0) {
        subdet_.emplace_back(ForwardSubdetector::ForwardEmpty);
        if (detType == 0)
          dets_.emplace_back((int)(DetId::HGCalEE));
        else if (detType == 1)
          dets_.emplace_back((int)(DetId::HGCalHSi));
        else
          dets_.emplace_back((int)(DetId::HGCalHSc));
      } else {
        dets_.push_back((unsigned int)(DetId::Forward));
        if (detType == 0)
          subdet_.emplace_back((int)(ForwardSubdetector::HGCEE));
        else if (detType == 1)
          subdet_.emplace_back((int)(ForwardSubdetector::HGCHEF));
        else
          subdet_.emplace_back((int)(ForwardSubdetector::HGCHEB));
      }
      if (detType == 0)
        nameX = "HGCalEESensitive";
      else if (detType == 1)
        nameX = "HGCalHESiliconSensitive";
      else
        nameX = "HGCalHEScintillatorSensitive";
      edm::ESHandle<HGCalDDDConstants> hdc;
      es->get<IdealGeometryRecord>().get(nameX, hdc);
      if (hdc.isValid()) {
        levelT1_ = hdc->levelTop(0);
        levelT2_ = hdc->levelTop(1);
        if (hdc->tileTrapezoid()) {
          types_[type] = -1;
          hgcalNumbering_.emplace_back(new HGCalNumberingScheme(*hdc, (DetId::Detector)(dets_[type]), nameX));
        } else if (hdc->waferHexagon6()) {
          types_[type] = 1;
          hgcNumbering_.push_back(new HGCNumberingScheme(*hdc, nameX));
        } else {
          types_[type] = 0;
          hgcalNumbering_.emplace_back(new HGCalNumberingScheme(*hdc, (DetId::Detector)(dets_[type]), nameX));
        }
        layers = hdc->layers(false);
      } else {
        edm::LogError("ValidHGCal") << "Cannot find HGCalDDDConstants for " << nameX;
        throw cms::Exception("Unknown", "ValidHGCal") << "Cannot find HGCalDDDConstants for " << nameX << "\n";
      }
    } else {
      nameX = "HcalEndcap";
      dets_.push_back((unsigned int)(DetId::Hcal));
      subdet_.push_back((int)(HcalSubdetector::HcalEndcap));
      edm::ESHandle<HcalDDDSimConstants> hdc;
      es->get<HcalSimNumberingRecord>().get(hdc);
      if (hdc.isValid()) {
        numberingFromDDD_ = new HcalNumberingFromDDD(hdc.product());
        layers = 18;
      } else {
        edm::LogError("ValidHGCal") << "Cannot find HcalDDDSimConstant";
        throw cms::Exception("Unknown", "ValidHGCal") << "Cannot find HcalDDDSimConstant\n";
      }
    }
    if (detType == 0) {
      for (int i = 0; i < layers; ++i)
        hgcEEedep_.push_back(0);
    } else if (detType == 1) {
      for (int i = 0; i < layers; ++i)
        hgcHEFedep_.push_back(0);
    } else {
      for (int i = 0; i < layers; ++i)
        hgcHEBedep_.push_back(0);
    }
    if (verbosity_ > 0)
      edm::LogVerbatim("ValidHGCal") << "[" << type << "]: " << nameX << " det " << dets_[type] << " subdet "
                                     << subdet_[type] << " with " << layers << " layers";
  }
}

//=================================================================== per EVENT
void SimG4HGCalValidation::update(const BeginOfEvent* evt) {
  int iev = (*evt)()->GetEventID();
  if (verbosity_ > 0)
    edm::LogVerbatim("ValidHGCal") << "SimG4HGCalValidation: =====> Begin "
                                   << "event = " << iev;

  ++count_;
  edepEE_ = edepHEF_ = edepHEB_ = 0.;

  //HGCal variables
  for (unsigned int i = 0; i < hgcEEedep_.size(); i++)
    hgcEEedep_[i] = 0.;
  for (unsigned int i = 0; i < hgcHEFedep_.size(); i++)
    hgcHEFedep_[i] = 0.;
  for (unsigned int i = 0; i < hgcHEBedep_.size(); i++)
    hgcHEBedep_[i] = 0.;

  //Cache reset
  clear();
}

//=================================================================== each STEP
void SimG4HGCalValidation::update(const G4Step* aStep) {
  if (aStep != nullptr) {
    G4VPhysicalVolume* curPV = aStep->GetPreStepPoint()->GetPhysicalVolume();
    const G4String& name = curPV->GetName();
    G4VSensitiveDetector* curSD = aStep->GetPreStepPoint()->GetSensitiveDetector();
    if (verbosity_ > 1)
      edm::LogVerbatim("ValidHGCal") << "ValidHGCal::Step in " << name << " at "
                                     << aStep->GetPreStepPoint()->GetPosition();

    // Only for Sensitive detector
    if (curSD != nullptr) {
      int type(-1);
      for (unsigned int k = 0; k < names_.size(); ++k) {
        if (name.find(names_[k].c_str()) != std::string::npos) {
          type = k;
          break;
        }
      }
      int detType = (type >= 0) ? detTypes_[type] : -1;
      if (verbosity_ > 0)
        edm::LogVerbatim("ValidHGCal") << "ValidHGCal::In SD " << name << " type " << type << ":" << detType;

      // Right type of SD
      if (type >= 0) {
        //Get the 32-bit index of the hit cell
        const G4TouchableHistory* touchable =
            static_cast<const G4TouchableHistory*>(aStep->GetPreStepPoint()->GetTouchable());
        unsigned int index(0);
        int layer(0);
        G4ThreeVector hitPoint = aStep->GetPreStepPoint()->GetPosition();
        if (types_[type] <= 1) {
          // HGCal
          G4ThreeVector localpos = touchable->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
          float globalZ = touchable->GetTranslation(0).z();
          int iz(globalZ > 0 ? 1 : -1);
          int module(-1), cell(-1);
          if (types_[type] == 1) {
            if (touchable->GetHistoryDepth() == levelT1_) {
              layer = touchable->GetReplicaNumber(0);
            } else {
              layer = touchable->GetReplicaNumber(2);
              module = touchable->GetReplicaNumber(1);
              cell = touchable->GetReplicaNumber(0);
            }
            index =
                hgcNumbering_[type]->getUnitID((ForwardSubdetector)(subdet_[type]), layer, module, cell, iz, localpos);
          } else {
            if ((touchable->GetHistoryDepth() == levelT1_) || (touchable->GetHistoryDepth() == levelT2_)) {
              layer = touchable->GetReplicaNumber(0);
            } else {
              layer = touchable->GetReplicaNumber(3);
              module = touchable->GetReplicaNumber(2);
              cell = touchable->GetReplicaNumber(1);
            }
            double weight(0);
            index = hgcalNumbering_[type]->getUnitID(layer, module, cell, iz, hitPoint, weight);
          }
          if (verbosity_ > 1)
            edm::LogVerbatim("ValidHGCal")
                << "HGCal: " << name << " Layer " << layer << " Module " << module << " Cell " << cell;
        } else {
          // Hcal
          int depth = (touchable->GetReplicaNumber(0)) % 10 + 1;
          int lay = (touchable->GetReplicaNumber(0) / 10) % 100 + 1;
          int det = (touchable->GetReplicaNumber(1)) / 1000;
          HcalNumberingFromDDD::HcalID tmp =
              numberingFromDDD_->unitID(det, math::XYZVectorD(hitPoint.x(), hitPoint.y(), hitPoint.z()), depth, lay);
          index = HcalTestNumbering::packHcalIndex(tmp.subdet, tmp.zside, tmp.depth, tmp.etaR, tmp.phis, tmp.lay);
          layer = tmp.lay;
          if (verbosity_ > 1)
            edm::LogVerbatim("ValidHGCal")
                << "HCAL: " << det << ":" << depth << ":" << lay << " o/p " << tmp.subdet << ":" << tmp.zside << ":"
                << tmp.depth << ":" << tmp.etaR << ":" << tmp.phis << ":" << tmp.lay << " point " << hitPoint << " "
                << hitPoint.rho() << ":" << hitPoint.eta() << ":" << hitPoint.phi();
        }

        double edeposit = aStep->GetTotalEnergyDeposit();
        if (verbosity_ > 0)
          edm::LogVerbatim("ValidHGCal") << "Layer " << layer << " Index " << std::hex << index << std::dec << " Edep "
                                         << edeposit << " hit at " << hitPoint;
        if (detType == 0) {
          edepEE_ += edeposit;
          if (layer < (int)(hgcEEedep_.size()))
            hgcEEedep_[layer] += edeposit;
        } else if (detType == 1) {
          edepHEF_ += edeposit;
          if (layer < (int)(hgcHEFedep_.size()))
            hgcHEFedep_[layer] += edeposit;
        } else {
          edepHEB_ += edeposit;
          if (layer < (int)(hgcHEBedep_.size()))
            hgcHEBedep_[layer] += edeposit;
        }
        G4String nextVolume("XXX");
        if (aStep->GetTrack()->GetNextVolume() != nullptr)
          nextVolume = aStep->GetTrack()->GetNextVolume()->GetName();

        if (nextVolume.c_str() != name.c_str()) {  //save hit when it exits cell
          if (std::find(hgchitIndex_.begin(), hgchitIndex_.end(), index) == hgchitIndex_.end()) {
            hgchitDets_.push_back(dets_[type]);
            hgchitIndex_.push_back(index);
            hgchitX_.push_back(hitPoint.x());
            hgchitY_.push_back(hitPoint.y());
            hgchitZ_.push_back(hitPoint.z());
          }
        }
      }  // it is right type of SD
    }    // it is in a SD
  }      //if aStep!=NULL
}  //end update aStep

//================================================================ End of EVENT

void SimG4HGCalValidation::layerAnalysis(PHGCalValidInfo& product) {
  if (verbosity_ > 0)
    edm::LogVerbatim("ValidHGCal") << "\n ===>>> SimG4HGCalValidation: Energy "
                                   << "deposit\n at EE : " << std::setw(6) << edepEE_ / CLHEP::MeV
                                   << "\n at HEF: " << std::setw(6) << edepHEF_ / CLHEP::MeV
                                   << "\n at HEB: " << std::setw(6) << edepHEB_ / CLHEP::MeV;

  //Fill HGC Variables
  product.fillhgcHits(hgchitDets_, hgchitIndex_, hgchitX_, hgchitY_, hgchitZ_);
  product.fillhgcLayers(edepEE_, edepHEF_, edepHEB_, hgcEEedep_, hgcHEFedep_, hgcHEBedep_);
}

//---------------------------------------------------
void SimG4HGCalValidation::clear() {
  hgchitDets_.erase(hgchitDets_.begin(), hgchitDets_.end());
  hgchitIndex_.erase(hgchitIndex_.begin(), hgchitIndex_.end());
  hgchitX_.erase(hgchitX_.begin(), hgchitX_.end());
  hgchitY_.erase(hgchitY_.begin(), hgchitY_.end());
  hgchitZ_.erase(hgchitZ_.begin(), hgchitZ_.end());
}

DEFINE_SIMWATCHER(SimG4HGCalValidation);
