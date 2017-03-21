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
			     public Observer<const BeginOfJob *>, 
			     public Observer<const BeginOfEvent *>, 
			     public Observer<const G4Step *> {

public:
  SimG4HGCalValidation(const edm::ParameterSet &p);
  virtual ~SimG4HGCalValidation();

  void produce(edm::Event&, const edm::EventSetup&);

private:
  SimG4HGCalValidation(const SimG4HGCalValidation&); // stop default
  const SimG4HGCalValidation& operator=(const SimG4HGCalValidation&);

  void  init();

  // observer classes
  void update(const BeginOfJob * job);
  void update(const BeginOfEvent * evt);
  void update(const G4Step * step);

  // analysis related class
  void layerAnalysis(PHGCalValidInfo&);
  void clear();

private:
  //Keep reference to instantiate HcalNumberingFromDDD later
  HcalNumberingFromDDD *           numberingFromDDD_;

  //HGCal numbering scheme
  std::vector<HGCNumberingScheme*> hgcNumbering_;
  
  // to read from parameter set
  std::vector<std::string>  names_;
  std::vector<int>          types_, subdet_;
  std::string               labelLayer_;

  // some private members for ananlysis 
  unsigned int              count_;                  
  double                    edepEE_, edepHEF_, edepHEB_;
  std::vector<double>       hgcEEedep_, hgcHEFedep_, hgcHEBedep_;
  std::vector<unsigned int> dets_, hgchitDets_, hgchitIndex_;
  std::vector<double>	    hgchitX_, hgchitY_, hgchitZ_; 
};

SimG4HGCalValidation::SimG4HGCalValidation(const edm::ParameterSet &p): 
  numberingFromDDD_(0), count_(0) {

  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("SimG4HGCalValidation");
  names_        = m_Anal.getParameter<std::vector<std::string> >("Names");
  types_        = m_Anal.getParameter<std::vector<int> >("Types");
  labelLayer_   = m_Anal.getParameter<std::string>("LabelLayerInfo");

  produces<PHGCalValidInfo>(labelLayer_);

  edm::LogInfo("ValidHGCal") << "HGCalTestAnalysis:: Initialised as observer "
			     << "of begin events and of G4step with Parameter "
			     << "values: \n\tLabel : " << labelLayer_
			     << " and with " << names_.size() << " detectors"
			     << std::endl;
  for (unsigned int k=0; k<names_.size(); ++k)
    edm::LogInfo("ValidHGCal") << " [" << k << "] " << names_[k] << " Type "
			       << types_[k] << std::endl;
} 
   
SimG4HGCalValidation::~SimG4HGCalValidation() {
  if (numberingFromDDD_) delete numberingFromDDD_;
  for (unsigned int k=0; k<hgcNumbering_.size(); ++k)
    if (hgcNumbering_[k]) delete hgcNumbering_[k];
}

void SimG4HGCalValidation::produce(edm::Event& e, const edm::EventSetup&) {

  std::unique_ptr<PHGCalValidInfo> productLayer(new PHGCalValidInfo);
  layerAnalysis(*productLayer);
  e.put(std::move(productLayer),labelLayer_);
}

void SimG4HGCalValidation::update(const BeginOfJob * job) {

  const edm::EventSetup* es = (*job)();
  for (unsigned int type=0; type<types_.size(); ++type) {
    int layers(0);
    G4String nameX = "HGCal";
    if (types_[type] <= 1) {
      dets_.push_back((unsigned int)(DetId::Forward));
      if (type == 0) {
	subdet_.push_back((int)(ForwardSubdetector::HGCEE));
	nameX        = "HGCalEESensitive";
      } else if (type == 1) {
	subdet_.push_back((int)(ForwardSubdetector::HGCHEF));
	nameX        = "HGCalHESiliconSensitive";
      } else {
	subdet_.push_back((int)(ForwardSubdetector::HGCHEB));
	nameX        = "HGCalHEScintillatorSensitive";
      }
      edm::ESHandle<HGCalDDDConstants>    hdc;
      es->get<IdealGeometryRecord>().get(nameX,hdc);
      if (hdc.isValid()) {
	HGCalGeometryMode m_mode = hdc->geomMode();
	hgcNumbering_.push_back(new HGCNumberingScheme(*hdc,nameX));
	if (m_mode == HGCalGeometryMode::Square) types_[type] = 0;
	else                                     types_[type] = 1;
	layers = hdc->layers(false);
      } else {
	edm::LogError("ValidHGCal") << "Cannot find HGCalDDDConstants for "
				    << nameX;
	throw cms::Exception("Unknown", "ValidHGCal") << "Cannot find HGCalDDDConstants for " << nameX << "\n";
      }
    } else {
      nameX = "HcalEndcap";
      dets_.push_back((unsigned int)(DetId::Hcal));
      subdet_.push_back((int)(HcalSubdetector::HcalEndcap));
      edm::ESHandle<HcalDDDSimConstants>    hdc;
      es->get<HcalSimNumberingRecord>().get(hdc);
      if (hdc.isValid()) {
	HcalDDDSimConstants* hcalConstants = (HcalDDDSimConstants*)(&(*hdc));
	numberingFromDDD_ = new HcalNumberingFromDDD(hcalConstants);
	layers = 18;
      } else {
	edm::LogError("ValidHGCal") << "Cannot find HcalDDDSimConstant";
	throw cms::Exception("Unknown", "ValidHGCal") << "Cannot find HcalDDDSimConstant\n";
      }
    }
    if (type == 0) {
      for (int i=0; i<layers; ++i) hgcEEedep_.push_back(0);
    } else if (type == 1) {
      for (int i=0; i<layers; ++i) hgcHEFedep_.push_back(0);
    } else {
      for (int i=0; i<layers; ++i) hgcHEBedep_.push_back(0);
    }
    edm::LogInfo("ValidHGCal") << "[" << type << "]: " << nameX << " det "
			       << dets_[type] << " subdet " << subdet_[type]
			       << " with " << layers << " layers" << std::endl;
  }
}

//=================================================================== per EVENT
void SimG4HGCalValidation::update(const BeginOfEvent * evt) {
 
  int iev = (*evt)()->GetEventID();
  edm::LogInfo("ValidHGCal") << "SimG4HGCalValidation: =====> Begin event = "
			     << iev << std::endl;
  
  ++count_;
  edepEE_ = edepHEF_ = edepHEB_ = 0.;

  //HGCal variables 
  for (unsigned int i = 0; i<hgcEEedep_.size();  i++) hgcEEedep_[i]  = 0.;
  for (unsigned int i = 0; i<hgcHEFedep_.size(); i++) hgcHEFedep_[i] = 0.; 
  for (unsigned int i = 0; i<hgcHEBedep_.size(); i++) hgcHEBedep_[i] = 0.; 

  //Cache reset  
  clear();
}

//=================================================================== each STEP
void SimG4HGCalValidation::update(const G4Step * aStep) {

  if (aStep != NULL) {
    G4VPhysicalVolume* curPV  = aStep->GetPreStepPoint()->GetPhysicalVolume();
    G4VSensitiveDetector* curSD = aStep->GetPreStepPoint()->GetSensitiveDetector();

    // Only for Sensitive detector
    if (curSD != 0) {
      G4String name = curPV->GetName();
      int type(-1);
      for (unsigned int k=0; k<names_.size(); ++k) {
	if (name.find(names_[k].c_str()) != std::string::npos) {
	  type = k; break;
	}
      }
      edm::LogInfo("ValidHGCal") << "Step in " << name << " type " << type;
      // Right type of SD
      if (type >= 0) {
	//Get the 32-bit index of the hit cell
	G4TouchableHistory* touchable = (G4TouchableHistory*)aStep->GetPreStepPoint()->GetTouchable();
	unsigned int index(0);
	int          layer(0);
	G4ThreeVector hitPoint = aStep->GetPreStepPoint()->GetPosition();
	if (types_[type] <= 1) {
	  // HGCal
	  G4ThreeVector localpos = touchable->GetHistory()->GetTopTransform().TransformPoint(hitPoint);	  
	  float globalZ=touchable->GetTranslation(0).z();
	  int iz( globalZ>0 ? 1 : -1);
	  int module(0), cell(0);
	  if (types_[type] == 0) {
	    layer  = touchable->GetReplicaNumber(0);
	    module = touchable->GetReplicaNumber(1);
	  } else {
	    layer  = touchable->GetReplicaNumber(2);
	    module = touchable->GetReplicaNumber(1);
	    cell   = touchable->GetReplicaNumber(0);
	  }
	  index = hgcNumbering_[type]->getUnitID((ForwardSubdetector)(subdet_[type]), layer, module, cell, iz, localpos);
	} else {
	  // Hcal
	  int depth = (touchable->GetReplicaNumber(0))%10 + 1;
	  int lay   = (touchable->GetReplicaNumber(0)/10)%100 + 1;
	  int det   = (touchable->GetReplicaNumber(1))/1000;
	  HcalNumberingFromDDD::HcalID tmp = numberingFromDDD_->unitID(det, hitPoint, depth, lay);
	  index = HcalTestNumbering::packHcalIndex(tmp.subdet,tmp.zside,tmp.depth,tmp.etaR,tmp.phis,tmp.lay);
	  layer = tmp.lay;
	  edm::LogInfo("ValidHGCal") << "HCAL: " << det << ":" << depth << ":"
				     << lay << " o/p " << tmp.subdet << ":" 
				     << tmp.zside << ":" << tmp.depth << ":" 
				     << tmp.etaR << ":" << tmp.phis << ":" 
				     << tmp.lay << " point " << hitPoint << " "
				     << hitPoint.rho() << ":" << hitPoint.eta()
				     << ":" << hitPoint.phi();
	}

	double edeposit = aStep->GetTotalEnergyDeposit();
	edm::LogInfo("ValidHGCal") << "Layer " << layer << " Index " 
				   << std::hex << index << std::dec
				   << " Edep " << edeposit << " hit at "
				   << hitPoint;
	if (type == 0) {
	  edepEE_  += edeposit;
	  if (layer < (int)(hgcEEedep_.size()))  hgcEEedep_[layer]  += edeposit;
	} else if (type == 1) {
	  edepHEF_ += edeposit;
	  if (layer < (int)(hgcHEFedep_.size())) hgcHEFedep_[layer] += edeposit;
	} else {
	  edepHEB_ += edeposit;
	  if (layer < (int)(hgcHEBedep_.size())) hgcHEBedep_[layer] += edeposit;
	}
	G4String nextVolume("XXX");
	if (aStep->GetTrack()->GetNextVolume()!=0)
	  nextVolume = aStep->GetTrack()->GetNextVolume()->GetName();

	if (nextVolume.c_str()!=name.c_str()) { //save hit when it exits cell
	  if (std::find(hgchitIndex_.begin(),hgchitIndex_.end(),index) == hgchitIndex_.end()) {
	    hgchitDets_.push_back(dets_[type]);
	    hgchitIndex_.push_back(index);
	    hgchitX_.push_back(hitPoint.x());
	    hgchitY_.push_back(hitPoint.y()); 
	    hgchitZ_.push_back(hitPoint.z());
	  }
	}
      } // it is right type of SD
    } // it is in a SD
  }//if aStep!=NULL
}//end update aStep

//================================================================ End of EVENT

void SimG4HGCalValidation::layerAnalysis(PHGCalValidInfo& product) {

  edm::LogInfo("ValidHGCal") << "\n ===>>> SimG4HGCalValidation: Energy deposit"
			     << "\n at EE : " << std::setw(6) << edepEE_/MeV 
			     << "\n at HEF: " << std::setw(6) << edepHEF_/MeV 
			     << "\n at HEB: " << std::setw(6) << edepHEB_/MeV 
			     << "\n";
  
  
  //Fill HGC Variables
  product.fillhgcHits(hgchitDets_,hgchitIndex_, hgchitX_, hgchitY_, hgchitZ_);
  product.fillhgcLayers(edepEE_,edepHEF_,edepHEB_,hgcEEedep_, hgcHEFedep_,hgcHEBedep_);
}

//---------------------------------------------------
void SimG4HGCalValidation::clear() {

  hgchitDets_.erase(hgchitDets_.begin(),hgchitDets_.end()); 
  hgchitIndex_.erase(hgchitIndex_.begin(),hgchitIndex_.end()); 
  hgchitX_.erase(hgchitX_.begin(),hgchitX_.end()); 
  hgchitY_.erase(hgchitY_.begin(),hgchitY_.end()); 
  hgchitZ_.erase(hgchitZ_.begin(),hgchitZ_.end());   	
}

DEFINE_SIMWATCHER (SimG4HGCalValidation);
