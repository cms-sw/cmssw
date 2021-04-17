#include <iostream>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingTrack.h"

void MaterialAccountingTrack::enterDetector(const GlobalPoint& position, double cosTheta) {
  m_detector.m_position = position;
  m_detector.m_curvilinearIn = m_total.length();
  m_detector.m_cosThetaIn = cosTheta;
}

void MaterialAccountingTrack::leaveDetector(double cosTheta) {
  m_detector.m_curvilinearOut = m_total.length();
  m_detector.m_cosThetaOut = cosTheta;
  m_detectors.push_back(m_detector);
  m_detector.clear();
}
