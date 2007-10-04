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
  m_z(0, 0),
  m_r(0, 0),
  m_accounting(),
  m_errors(),
  m_tracks( 0 ),
  m_counted( false )
{
  // FIXME: find a better way to discover the R and Z ranges
  const GlobalPoint & position = layer.surface().position();
  const Bounds &      bounds   = layer.surface().bounds();
  if (const SimpleCylinderBounds * cylinder = dynamic_cast<const SimpleCylinderBounds*>(&bounds)) {
    m_z.first  = position.z() + cylinder->theZmin;
    m_z.second = position.z() + cylinder->theZmax;
    m_r.first  = cylinder->theRmin;
    m_r.second = cylinder->theRmax;
  } else
  if (const SimpleDiskBounds * disk = dynamic_cast<const SimpleDiskBounds*>(&bounds)) {
    m_z.first  = position.z() + disk->theZmin;
    m_z.second = position.z() + disk->theZmax;
    m_r.first  = disk->theRmin;
    m_r.second = disk->theRmax;
  }
}

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
