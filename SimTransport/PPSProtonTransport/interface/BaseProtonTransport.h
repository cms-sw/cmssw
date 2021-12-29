#ifndef BASEPROTONTRANSPORT
#define BASEPROTONTRANSPORT
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HepMC/GenEvent.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/PPS/interface/PPSUnitConversion.h"
#include "SimDataFormats/Forward/interface/LHCTransportLink.h"
#include <vector>
#include <map>
#include <string>
#include "TLorentzVector.h"

namespace CLHEP {
  class HepRandomEngine;
}
class BaseProtonTransport {
public:
  BaseProtonTransport(const edm::ParameterSet& iConfig);
  virtual ~BaseProtonTransport();

  std::vector<LHCTransportLink>& getCorrespondenceMap() { return m_CorrespondenceMap; }
  virtual void process(const HepMC::GenEvent* ev, const edm::EventSetup& es, CLHEP::HepRandomEngine* engine) = 0;

  void clear();

  void addPartToHepMC(const HepMC::GenEvent*, HepMC::GenEvent*);

  void ApplyBeamCorrection(HepMC::GenParticle* p);
  void ApplyBeamCorrection(TLorentzVector& p);

  double beamEnergy() { return beamEnergy_; };
  double beamMomentum() { return beamMomentum_; };

protected:
  enum class TransportMode { HECTOR, TOTEM, OPTICALFUNCTIONS };
  TransportMode MODE;

  CLHEP::HepRandomEngine* engine_{nullptr};
  std::vector<LHCTransportLink> m_CorrespondenceMap;
  std::map<unsigned int, TLorentzVector> m_beamPart;
  std::map<unsigned int, double> m_xAtTrPoint;
  std::map<unsigned int, double> m_yAtTrPoint;

  bool verbosity_;
  bool bApplyZShift_;
  bool useBeamPositionFromLHCInfo_;
  bool produceHitsRelativeToBeam_;

  std::string beam1Filename_;
  std::string beam2Filename_;

  double fPPSRegionStart_45_;
  double fPPSRegionStart_56_;
  double fCrossingAngleX_45_;
  double fCrossingAngleX_56_;
  double fCrossingAngleY_45_;
  double fCrossingAngleY_56_;

  double beamMomentum_;
  double beamEnergy_;
  double etaCut_;
  double momentumCut_;

  double m_sigmaSTX;
  double m_sigmaSTY;
  double m_sigmaSX;
  double m_sigmaSY;
  double m_sig_E;
};
#endif
