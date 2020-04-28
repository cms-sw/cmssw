#ifndef DD4hep_MaterialAccountingGroup_h
#define DD4hep_MaterialAccountingGroup_h

#include <string>
#include <stdexcept>
#include <utility>
#include <memory>

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingDetector.h"

#include "BoundingBox.h"

class TH1;
class TH1F;
class TProfile;
class TFile;
class DDCompactView;

class MaterialAccountingGroup {

 private:

  MaterialAccountingGroup(const MaterialAccountingGroup& layer) = delete;
  MaterialAccountingGroup& operator=(const MaterialAccountingGroup& layer) = delete;
  void savePlot_(std::shared_ptr<TH1F> plot, const std::string& name);
  void savePlot_(std::shared_ptr<TProfile> plot, float average, const std::string& name);

  std::string m_name;
  std::vector<GlobalPoint> m_elements;
  BoundingBox m_boundingbox;
  MaterialAccountingStep m_accounting;
  MaterialAccountingStep m_errors;
  unsigned int m_tracks;
  bool m_counted;
  MaterialAccountingStep m_buffer;

  std::shared_ptr<TH1F> m_dedx_spectrum;
  std::shared_ptr<TH1F> m_radlen_spectrum;
  std::shared_ptr<TProfile> m_dedx_vs_eta;
  std::shared_ptr<TProfile> m_dedx_vs_z;
  std::shared_ptr<TProfile> m_dedx_vs_r;
  std::shared_ptr<TProfile> m_radlen_vs_eta;
  std::shared_ptr<TProfile> m_radlen_vs_z;
  std::shared_ptr<TProfile> m_radlen_vs_r;

  mutable TFile* m_file;

  static constexpr double s_tolerance = 0.01;

 public:

  MaterialAccountingGroup(const std::string &name, const cms::DDCompactView& geometry);
  ~MaterialAccountingGroup(void);

  bool addDetector_(const MaterialAccountingDetector& detector);
  void endOfTrack_(void);
  bool isInside(const MaterialAccountingDetector& detector) const;
  std::pair<double, double> getBoundingR() const { return m_boundingbox.range_r(); };
  std::pair<double, double> getBoundingZ() const { return m_boundingbox.range_z(); };
  MaterialAccountingStep average(void) const;
  double averageLength(void) const;
  double averageRadiationLengths(void) const;
  double averageEnergyLoss(void) const;
  double sigmaLength(void) const;
  double sigmaRadiationLengths(void) const;
  double sigmaEnergyLoss(void) const;
  unsigned int tracks(void) const { return m_tracks; }
  const std::string& name(void) const { return m_name; }
  std::string info_(void) const; 
  void savePlots_(void);

  const std::vector<GlobalPoint>& elements(void) const { return m_elements; }

};

#endif
