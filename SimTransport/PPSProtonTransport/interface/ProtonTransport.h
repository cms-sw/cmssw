#ifndef PROTONTRANSPORT
#define PROTONTRANSPORT
#include "FWCore/Framework/interface/EventSetup.h"
#include "HepMC/GenEvent.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/PPS/interface/PPSUnitConversion.h"
#include "SimDataFormats/Forward/interface/LHCTransportLink.h"
#include <vector>
#include <map>
#include "TLorentzVector.h"

namespace CLHEP {
  class HepRandomEngine;
}
class ProtonTransport {
public:
  ProtonTransport();
  virtual ~ProtonTransport();
  std::vector<LHCTransportLink>& getCorrespondenceMap() { return m_CorrespondenceMap; }
  virtual void process(const HepMC::GenEvent* ev, const edm::EventSetup& es, CLHEP::HepRandomEngine* engine) = 0;
  void ApplyBeamCorrection(HepMC::GenParticle* p);
  void ApplyBeamCorrection(TLorentzVector& p);
  void addPartToHepMC(HepMC::GenEvent*);
  void clear();

protected:
  enum class TransportMode { HECTOR, TOTEM };
  TransportMode MODE;
  int NEvent;
  bool m_verbosity;
  CLHEP::HepRandomEngine* engine;

  bool bApplyZShift;
  double fPPSRegionStart_56;
  double fPPSRegionStart_45;
  double fCrossingAngle_45;
  double fCrossingAngle_56;

  std::vector<LHCTransportLink> m_CorrespondenceMap;
  std::map<unsigned int, TLorentzVector> m_beamPart;
  std::map<unsigned int, double> m_xAtTrPoint;
  std::map<unsigned int, double> m_yAtTrPoint;

  double m_sigmaSX;
  double m_sigmaSY;
  double m_sigmaSTX;
  double m_sigmaSTY;
  double m_sig_E;
  double fVtxMeanX;
  double fVtxMeanY;
  double fVtxMeanZ;
  double fBeamXatIP;
  double fBeamYatIP;
  double fBeamMomentum;
  double fBeamEnergy;
};
#endif
