#include "SimG4CMS/HGCalTestBeam/interface/AHCalSD.h"
#include "SimG4CMS/HGCalTestBeam/interface/AHCalDetId.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ParticleTable.hh"
#include "G4VProcess.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include <iomanip>
//#define EDM_ML_DEBUG

AHCalSD::AHCalSD(const std::string& name, const DDCompactView & cpv,
		 const SensitiveDetectorCatalog & clg,
		 edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager,
         (float)(p.getParameter<edm::ParameterSet>("AHCalSD").getParameter<double>("TimeSliceUnit")),
         p.getParameter<edm::ParameterSet>("AHCalSD").getParameter<bool>("IgnoreTrackID")) {

  edm::ParameterSet m_HC = p.getParameter<edm::ParameterSet>("AHCalSD");
  useBirk          = m_HC.getParameter<bool>("UseBirkLaw");
  birk1            = m_HC.getParameter<double>("BirkC1")*(CLHEP::g/(CLHEP::MeV*CLHEP::cm2));
  birk2            = m_HC.getParameter<double>("BirkC2");
  birk3            = m_HC.getParameter<double>("BirkC3");
  eminHit          = m_HC.getParameter<double>("EminHit")*CLHEP::MeV;

  edm::LogInfo("HcalSim") << "AHCalSD::  Use of Birks law is set to      " 
                          << useBirk << "  with three constants kB = "
                          << birk1 << ", C1 = " << birk2 << ", C2 = " << birk3
			  << "\nAHCalSD:: Threshold for storing hits: "
                          << eminHit << std::endl;
}

AHCalSD::~AHCalSD() { }

double AHCalSD::getEnergyDeposit(G4Step* aStep) {

  double destep = aStep->GetTotalEnergyDeposit();
  double wt2    = aStep->GetTrack()->GetWeight();
  double weight = (wt2 > 0.0) ? wt2 : 1.0;
#ifdef EDM_ML_DEBUG
  double weight0 = weight;
#endif
  if (useBirk) weight *= getAttenuation(aStep, birk1, birk2, birk3);
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HcalSim") << "AHCalSD: weight " << weight0 << " " << weight 
			  << std::endl;
#endif
  double edep = weight*destep;
  return edep;
}

uint32_t AHCalSD::setDetUnitId(const G4Step * aStep) { 

  G4StepPoint* preStepPoint = aStep->GetPreStepPoint(); 
  const G4VTouchable* touch = preStepPoint->GetTouchable();

  int depth = (touch->GetReplicaNumber(1));
  int incol = ((touch->GetReplicaNumber(0))%10);
  int inrow = ((touch->GetReplicaNumber(0))/10)%10;
  int jncol = ((touch->GetReplicaNumber(0))/100)%10;
  int jnrow = ((touch->GetReplicaNumber(0))/1000)%10;
  int col   = (jncol == 0) ? incol : -incol;
  int row   = (jnrow == 0) ? inrow : -inrow;
  uint32_t index = AHCalDetId(row,col,depth).rawId();
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HcalSim") << "AHCalSD: det = " << HcalOther 
                          << " depth = " << depth << " row = " << row 
                          << " column = " << col << " packed index = 0x" 
			  << std::hex << index << std::dec << std::endl;
  bool flag = unpackIndex(index, row, col, depth);
  edm::LogInfo("HcalSim") << "Results from unpacker for 0x" << std::hex 
			  << index << std::dec << " Flag " << flag << " Row " 
			  << row << " Col " << col << " Depth " << depth 
			  << std::endl;
#endif
  return index;
}

bool AHCalSD::unpackIndex(const uint32_t& idx, int& row, int& col, int& depth) {

  DetId gen(idx);
  HcalSubdetector subdet = (HcalSubdetector(gen.subdetId()));
  bool rcode = (gen.det()==DetId::Hcal && subdet!=HcalOther);
  row = col = depth = 0;
  if (rcode) {
    row   = AHCalDetId(idx).irow();
    col   = AHCalDetId(idx).icol();
    depth = AHCalDetId(idx).depth();
  }
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HcalSim") << "AHCalSD: packed index = 0x" << std::hex << idx 
			  << std::dec << " Row " << row << " Column " << col
			  << " Depth " << depth << " OK " << rcode << std::endl;
#endif
  return rcode;
}
    
bool AHCalSD::filterHit(CaloG4Hit* aHit, double time) {
  return ((time <= tmaxHit) && (aHit->getEnergyDeposit() > eminHit));
}
