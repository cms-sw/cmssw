#ifndef MaterialAccountingGroup_h
#define MaterialAccountingGroup_h

#include <string>
#include <stdexcept>
#include <utility>

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingDetector.h"

class TH1;
class TH1F;
class TProfile;
class TFile;

class MaterialAccountingGroup {
private:

  // quick implementation of a bounding cilinder
  class BoundingBox {
  public:

    BoundingBox(double min_r, double max_r, double min_z, double max_z) :
      r_min(min_r),
      r_max(max_r),
      z_min(min_z),
      z_max(max_z)
    { }

    BoundingBox() :
      r_min(0.),
      r_max(0.),
      z_min(0.),
      z_max(0.)
    { }

    void grow(double r, double z) {
      if (r < r_min) r_min = r;
      if (r > r_max) r_max = r;
      if (z < z_min) z_min = z;
      if (z > z_max) z_max = z;
    }

    void grow(double skin) {
      r_min -= skin;    // yes, we allow r_min to go negative
      r_max += skin;
      z_min -= skin;
      z_max += skin;
    }

    bool inside(double r, double z) const {
      return (r >= r_min and r <= r_max and z >= z_min and z <= z_max);
    }

    std::pair<double, double> range_r() const {
      return std::make_pair(r_min, r_max);
    }

    std::pair<double, double> range_z() const {
      return std::make_pair(z_min, z_max);
    }

  private:
    double r_min;
    double r_max;
    double z_min;
    double z_max;
  };
  
public:
  /// explicit constructors
  MaterialAccountingGroup( const std::string & name, const DDCompactView & geometry );
    
  /// destructor
  ~MaterialAccountingGroup( void );

private:
  /// stop default copy ctor
  MaterialAccountingGroup(const MaterialAccountingGroup & layer);

  /// stop default assignment operator
  MaterialAccountingGroup& operator=( const MaterialAccountingGroup & layer);

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

  /// get some infos
  std::string info(void) const;

  /// save the plots
  void savePlots(const char * directory);
 
private:
  void savePlot(TH1F * plot, const std::string & name);
  void savePlot(TProfile * plot, float average, const std::string & name);
 
  std::string                   m_name;
  std::vector<GlobalPoint>      m_elements;
  BoundingBox                   m_boundingbox;
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

  // 100um should be small enough that no elements from different layers/groups are so close
  static double s_tolerance;
};

#endif // MaterialAccountingGroup_h
