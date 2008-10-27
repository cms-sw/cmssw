#ifndef MaterialAccountingLayer_h
#define MaterialAccountingLayer_h

#include <iostream>
#include <string>
#include <stdexcept>

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingDetector.h"

class TH1;
class TH1F;
class TProfile;
class TFile;
class DetLayer;

class MaterialAccountingLayer {
public:
  /// explicit constructors
  explicit MaterialAccountingLayer( const DetLayer & layer,                       const std::string & name, bool symmetric = false );
  explicit MaterialAccountingLayer( const std::vector<const DetLayer *> & layers, const std::string & name, bool symmetric = false );
    
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
  
  /// check if detector is inside any part of this layer 
  bool inside( const MaterialAccountingDetector& detector ) const;

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
 
  /// get the layer name 
  const std::string & name(void) const
  {
    return m_name;
  }

  /// return the number of DetLayers associated to this layer
  unsigned int layers(void) const 
  {
    return m_layers.size();
  }

  /// access one DetLayer (defaults to the 1st)
  const DetLayer * layer(unsigned int i = 0) const 
  {
    checkLayer(i);
    return m_layers[i];
  }

  /// access one DetLayer material properties (defaults to the 1st DetLayer's)
  const MediumProperties * material(unsigned int i = 0) const
  {
    checkLayer(i);
    return m_layers[i]->surface().mediumProperties();
  }
  
  void savePlots(void);
 
private:
  void checkLayer(unsigned int i) const 
  {
    if (i >= m_layers.size())
      throw std::out_of_range("DetLayer index out of range");
  }

  void savePlot(TH1F * plot, const std::string & name);
  void savePlot(TProfile * plot, float average, const std::string & name);
  
  // m_layers allow to access subdetector type (not used), access η / Z / R ranges (broken), and to check for inside-ness
  std::vector<const DetLayer *> m_layers;
  std::string                   m_name;
  bool                          m_symmetric;      // layer is symmteric for reflection on the XY plane (Z <--> -Z, η <--> -η)
  MaterialAccountingStep        m_accounting;
  MaterialAccountingStep        m_errors;
  unsigned int                  m_tracks;
  bool                          m_counted;
  MaterialAccountingStep        m_buffer;
  
  // plots of material effects distribution
  TH1F * m_dedx_spectrum;
  TH1F * m_radlen_spectrum;
  // plots of material effects vs. η / Z / R
  TProfile * m_dedx_vs_eta;
  TProfile * m_dedx_vs_z;
  TProfile * m_dedx_vs_r;
  TProfile * m_radlen_vs_eta;
  TProfile * m_radlen_vs_z;
  TProfile * m_radlen_vs_r;

  // file to store plots into
  mutable TFile * m_file;
};

#endif // MaterialAccountingLayer_h
