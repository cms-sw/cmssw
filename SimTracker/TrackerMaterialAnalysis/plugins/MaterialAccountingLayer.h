#ifndef MaterialAccountingLayer_h
#define MaterialAccountingLayer_h

#include <iostream>
#include <string>

#include <TH1F.h>
#include <TH2F.h>

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingDetector.h"

class DetLayer;

class MaterialAccountingLayer {
public:
  /// explicit constructor
  explicit MaterialAccountingLayer( const DetLayer & layer );

  /// destructor
  ~MaterialAccountingLayer( void );

private:
  /// stop default copy ctor
  MaterialAccountingLayer(const MaterialAccountingLayer & layer);

  /// stop default assignment operator
  MaterialAccountingLayer& operator=( const MaterialAccountingLayer & layer);

public:
  /// buffer material from a detector, if the detector is inside the DetLayer bounds
  bool addDetector( const MaterialAccountingDetector& detector );
 
  /// commit the buffer and reset the "already hit by this track" flag
  void endOfTrack( void );
  
  /// check if detector is inside this layer 
  bool inside( const MaterialAccountingDetector& detector ) const 
  {
    return m_layer->surface().bounds().inside( m_layer->surface().toLocal( detector.position() ) );
  }

  /// return the average normalized material accounting informations
  MaterialAccountingStep average(void) const 
  {
    return m_tracks ? m_accounting / m_tracks : MaterialAccountingStep();
  }

  /// return the average normalized layer thickness
  double averageLength(void) const 
  {
    return m_tracks ? m_accounting.length() / m_tracks : 0.;
  }
  
  /// return the average normalized number of radiation lengths
  double averageRadiationLengths(void) const 
  {
    return m_tracks ? m_accounting.radiationLengths() / m_tracks : 0.;
  }
  
  /// return the average normalized energy loss density factor for Bethe-Bloch
  double averageEnergyLoss(void) const 
  {
    return m_tracks ? m_accounting.energyLoss() / m_tracks : 0.;
  }
 
  /// return the sigma of the normalized layer thickness
  double sigmaLength(void) const 
  {
    return m_tracks ? std::sqrt(m_errors.length() / m_tracks - averageLength()*averageLength()) : 0.;
  }
  
  /// return the sigma of the normalized number of radiation lengths
  double sigmaRadiationLengths(void) const 
  {
    return m_tracks ? std::sqrt(m_errors.radiationLengths() / m_tracks - averageRadiationLengths()*averageRadiationLengths()) : 0.;
  }
  
  /// return the sigma of the normalized energy loss density factor for Bethe-Bloch
  double sigmaEnergyLoss(void) const 
  {
    return m_tracks ? std::sqrt(m_errors.energyLoss() / m_tracks - averageEnergyLoss()*averageEnergyLoss()) : 0.;
  }
 
  /// return the number of tracks that hit this layer 
  unsigned int tracks(void) const 
  {
    return m_tracks;
  }
 
  /// ( r_min, r_max ) 
  const std::pair<float, float> & getRangeR(void) const 
  {
    return m_r;
  }

  /// ( z_min, z_max ) 
  const std::pair<float, float> & getRangeZ(void) const {
    return m_z;
  }
 
  /// get the layer subdetector name (PixelBarrel, TOB, ...) 
  std::string getName(void) const;

  /// access the DetLayer
  const DetLayer * layer(void) const 
  {
    return m_layer;
  }

  const MediumProperties * material(void) const
  {
    return m_layer->surface().mediumProperties();
  }
  
  void savePlots(const std::string & name) const;
  
private:
  // m_layer is used access subdetector, R & Z ranges, and to check for inside-ness
  const DetLayer *                  m_layer;
  std::pair<float, float>           m_z;
  std::pair<float, float>           m_r;
  MaterialAccountingStep            m_accounting;
  MaterialAccountingStep            m_errors;
  unsigned int                      m_tracks;
  bool                              m_counted;
  mutable MaterialAccountingStep    m_buffer;

  // plots of material distribution, and material vs. eta/Z/R
  TH1F * m_dedx_spectrum;
  TH1F * m_dedx_vs_eta;
  TH1F * m_dedx_vs_z;
  TH1F * m_dedx_vs_r;
  TH1F * m_radlen_spectrum;
  TH1F * m_radlen_vs_eta;
  TH1F * m_radlen_vs_z;
  TH1F * m_radlen_vs_r;
};

#endif // MaterialAccountingLayer_h
