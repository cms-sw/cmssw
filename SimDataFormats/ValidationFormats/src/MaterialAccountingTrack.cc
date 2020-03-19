#include <iostream>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "G4VPhysicalVolume.hh"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingTrack.h"

void MaterialAccountingTrack::enterDetector(const G4VPhysicalVolume* volume,
                                            const GlobalPoint& position,
                                            double cosTheta) {
  if (m_current_volume != nullptr) {
    // error: entering a volume while inside an other (or the same) one !
    if (m_current_volume != volume)
      std::cerr << "MaterialAccountingTrack::leaveDetector(...): ERROR: entering volume (" << volume
                << ") while inside volume (" << m_current_volume << ")" << std::endl;
    else
      std::cerr << "MaterialAccountingTrack::leaveDetector(...): ERROR: entering volume (" << volume << ") twice"
                << std::endl;
    m_detector.clear();
    return;
  }

  m_current_volume = volume;
  m_detector.m_position = position;
  m_detector.m_curvilinearIn = m_total.length();
  m_detector.m_cosThetaIn = cosTheta;
}

void MaterialAccountingTrack::leaveDetector(const G4VPhysicalVolume* volume, double cosTheta) {
  if (m_current_volume != volume) {
    // error: leaving the wrong (or no) volume !
    if (m_current_volume)
      std::cerr << "MaterialAccountingTrack::leaveDetector(...): ERROR: leaving volume (" << volume
                << ") while inside volume (" << m_current_volume << ")" << std::endl;
    else
      std::cerr << "MaterialAccountingTrack::leaveDetector(...): ERROR: leaving volume (" << volume
                << ") while not inside any volume" << std::endl;
    m_detector.clear();
    return;
  }

  m_current_volume = nullptr;
  m_detector.m_curvilinearOut = m_total.length();
  m_detector.m_cosThetaOut = cosTheta;
  m_detectors.push_back(m_detector);
  m_detector.clear();
}
