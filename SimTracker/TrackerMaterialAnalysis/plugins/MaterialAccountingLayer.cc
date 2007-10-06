#include <sstream>
#include <string>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingDetector.h"
#include "MaterialAccountingLayer.h"

MaterialAccountingLayer::MaterialAccountingLayer( const DetLayer & layer ) :
  m_layer( & layer ),
  m_z( layer.surface().zSpan() ),
  m_r( layer.surface().rSpan() ),
  m_accounting(),
  m_errors(),
  m_tracks( 0 ),
  m_counted( false )
{ }

bool MaterialAccountingLayer::addDetector( const MaterialAccountingDetector& detector ) {
  if (not inside(detector))
    return false;

  // multiple hits in the same layer (overlaps, etc.) from a single track still count as one for averaging,
  // since the energy deposits from the track have been already split between the different detectors
  m_buffer += detector.material();
  m_counted = true;

  return true;
}

void MaterialAccountingLayer::endOfTrack(void) {
  if (m_counted) {
    m_accounting += m_buffer;
    m_errors     += m_buffer * m_buffer;
    ++m_tracks;
  }
  m_counted = false;
  m_buffer  = MaterialAccountingStep();
}

std::string MaterialAccountingLayer::getName(void) const {
  // extract the subdetector name using the overloaded operator<<
  std::stringstream name;
  name << m_layer->subDetector();
  return name.str();
}
