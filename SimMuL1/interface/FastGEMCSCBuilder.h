#ifndef SimMuL1_FastGEMCSCBuilder_h
#define SimMuL1_FastGEMCSCBuilder_h

/**\class FastGEMCSCBuilder

 Description:

 Builds fast stubs out of CSC SimHits with extrapolation to given GEM z planes.

 Original Author:  "Vadim Khotilovich"
 $Id: $
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"

#include "GEMCode/GEMValidation/src/SimHitMatcher.h"

#include "TLinearFitter.h"

#include <memory>
#include <map>
#include <vector>
#include <iosfwd>



class CSCGeometry;


/// just an organizational data structure to encapsulate modeled stub information
struct SimStub
{
  SimStub(double z_gem);

  // ----- getters -----

  GlobalPoint globalPointAtZ(double z) const {return GlobalPoint( x0_ + x1_ * z, y0_ + y1_ * z, z);}

  // Global position of stub at CSC chamber key layer
  GlobalPoint globalPointCSC() const { return gp_csc_; }

  // Global position of linearly extrapolated stub at GEM z plane
  GlobalPoint globalPointGEMLinear() const { return gp_gem_lin_; }

  // Global position of propagated stub at GEM z plane
  GlobalPoint globalPointGEMPropagator() const { return gp_gem_prop_; }

  double dPhiGEMCSCLinear() const { return deltaPhi(gp_csc_.phi(), gp_gem_lin_.phi()); }

  double dPhiGEMCSCPropagator() const { return deltaPhi(gp_csc_.phi(), gp_gem_prop_.phi()); }

  // unit direction vector: available after setFitParameters
  GlobalVector direction() const { return direction_; }

  bool isValid() const { return min_hs_ <= max_hs_ && min_wg_ <= max_wg_; }
  bool hasHalfStrip(int hs) const { return hs >= min_hs_ && hs <= max_hs_; }
  bool hasWireGroup(int wg) const { return wg >= min_wg_ && wg <= max_wg_; }

  // ----- modifiers -----

  void setFitParameters(double x0, double x1, double y0, double y1);

  void addHalfStrips(std::set<int> hs);
  void addWireGroups(std::set<int> wg);

  void setCSC(GlobalPoint &gp) { gp_csc_ = gp; }
  void setGEMLinear(GlobalPoint &gp) { gp_gem_lin_ = gp; }
  void setGEMPropagator(GlobalPoint &gp) { gp_gem_prop_ = gp; }

  // ----- printing -----
  friend std::ostream& operator<<(std::ostream& os, const SimStub& s);

private:

  double x0_, x1_, y0_, y1_;
  double z_gem_;

  GlobalVector direction_;

  int min_hs_, max_hs_;
  int min_wg_, max_wg_;

  GlobalPoint gp_csc_;
  GlobalPoint gp_gem_lin_;
  GlobalPoint gp_gem_prop_;
};

std::ostream & operator<<(std::ostream & o, const SimStub& s);


class FastGEMCSCBuilder
{
public:

  explicit FastGEMCSCBuilder(const edm::ParameterSet&, CLHEP::HepRandomEngine& eng);

  ~FastGEMCSCBuilder() {}

  void setCSCGeometry(const CSCGeometry* g) { csc_geo_ = g; }

  void build(const SimHitMatcher& match_sh);

  std::vector<unsigned int> getChamberIds();
  std::vector<SimStub>& getStubs(unsigned int det_id) {return stubs_map_[det_id];}

private:

  // these configuration vectors have to have 10 elements corresponding to 10 chamber types
  // negative parameters values would mean chambers are not supposed to be used
  std::vector<double> zOddGEM_;
  std::vector<double> zEvenGEM_;
  std::vector<double> phiSmearCSC_;
  std::vector<double> phiSmearGEM_;

  std::unique_ptr<TLinearFitter> fitterXZ_;
  std::unique_ptr<TLinearFitter> fitterYZ_;

  std::unique_ptr<CLHEP::RandFlat> flat_;

  const CSCGeometry* csc_geo_;

  std::map<unsigned int, std::vector<SimStub> > stubs_map_;
};

#endif

