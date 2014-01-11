#ifndef SimMuL1_MuGeometryArea_h
#define SimMuL1_MuGeometryArea_h

/**\file MuGeometryHelpers

 Description:

 Utils to summarize and access some of the muon detectors' information, sensitive volumes areas in particular.
*/

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include <vector>
#include <cmath>
#include <map>
#include <set>

class CSCGeometry;
class GEMGeometry;
class RPCGeometry;
class DTGeometry;


namespace mugeo {

// check valid region
inline bool isME1bEtaRegion(float eta, float eta_min = 1.64, float eta_max = 2.14){return fabs(eta) >= eta_min && fabs(eta) <= eta_max;}
inline bool isME1abEtaRegion(float eta, float eta_min = 1.64){return fabs(eta) >= eta_min;}
inline bool isME1aEtaRegion(float eta, float eta_min = 2.14){return fabs(eta) >= eta_min;}
inline bool isME42EtaRegion(float eta){return fabs(eta)>=1.2499 && fabs(eta)<=1.8;}
inline bool isME42RPCEtaRegion(float eta){return fabs(eta)>=1.2499 && fabs(eta)<=1.6;}

// constants
enum ETrigCSC {MAX_CSC_STATIONS = 4, CSC_TYPES = 10};
enum ETrigGEM {MAX_GEM_STATIONS = 1, GEM_TYPES = 1};
enum ETrigDT  {MAX_DT_STATIONS = 4, DT_TYPES = 12};
enum ETrigRPCF {MAX_RPCF_STATIONS = 4, RPCF_TYPES = 12};
enum ETrigRPCB {MAX_RPCB_STATIONS = 4, RPCB_TYPES = 12};

// chamber types
inline int type(CSCDetId &d)  {return  d.iChamberType();}
inline int type(GEMDetId &d)  {return  3*d.station() + d.ring() - 3;}
inline int type(RPCDetId &d)
{
  if (d.region()==0) return  3*d.station() + abs(d.ring()) - 2;
  else return  3*d.station() + d.ring() - 3;
}
inline int type(DTWireId &d)  {return  3*d.station() + abs(d.wheel()) - 2;}

// labels for chamber types
const std::string csc_type[CSC_TYPES+1] =
  { "all", "ME1/a", "ME1/b", "ME1/2", "ME1/3", "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1", "ME4/2"};
const std::string csc_type_[CSC_TYPES+1] =
  { "all", "ME1a", "ME1b", "ME12", "ME13", "ME21", "ME22", "ME31", "ME32", "ME41", "ME42"};

const std::string gem_type[GEM_TYPES+1] =
  { "all", "GE1/1"};
const std::string gem_type_[GEM_TYPES+1] =
  { "all", "GE11"};

const std::string dt_type[DT_TYPES+1] =
  { "all", "MB1/0", "MB1/1", "MB1/2", "MB2/0", "MB2/1", "MB2/2", "MB3/0", "MB3/1", "MB3/2", "MB4/0", "MB4/1", "MB4/2",};
const std::string dt_type_[DT_TYPES+1] =
  { "all", "MB10", "MB11", "MB12", "MB20", "MB21", "MB22", "MB30", "MB31", "MB32", "MB40", "MB41", "MB42",};

const std::string rpcf_type[RPCF_TYPES+1] =
  { "all", "RE1/1", "RE1/2", "RE1/3", "RE2/1", "RE2/2", "RE2/3", "RE3/1", "RE3/2", "RE3/3", "RE4/1", "RE4/2", "RE4/3"};
const std::string rpcf_type_[RPCF_TYPES+1] =
  { "all", "RE11", "RE12", "RE13", "RE21", "RE22", "RE23", "RE31", "RE32", "RE33", "RE41", "RE42", "RE43"};

const std::string rpcb_type[RPCB_TYPES+1] =
  { "all", "RB1/0", "RB1/1", "RB1/2", "RB2/0", "RB2/1", "RB2/2", "RB3/0", "RB3/1", "RB3/2", "RB4/0", "RB4/1", "RB4/2",};
const std::string rpcb_type_[RPCB_TYPES+1] =
  { "all", "RB10", "RB11", "RB12", "RB20", "RB21", "RB22", "RB30", "RB31", "RB32", "RB40", "RB41", "RB42",};

const std::string csc_type_a[CSC_TYPES+2] =
   { "N/A", "ME1/a", "ME1/b", "ME1/2", "ME1/3", "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1", "ME4/2", "ME1/T"};
const std::string csc_type_a_[CSC_TYPES+2] =
   { "NA", "ME1A", "ME1B", "ME12", "ME13", "ME21", "ME22", "ME31", "ME32", "ME41", "ME42", "ME1T"};

// chamber radial segmentation numbers (including factor of 2 for non-zero wheels in barrel):
const double csc_radial_segm[CSC_TYPES+1] = {1, 36, 36, 36, 36, 18, 36, 18, 36, 18, 36};
const double gem_radial_segm[GEM_TYPES+1] = {1, 36};
const double dt_radial_segm[DT_TYPES+1]   = {1, 12, 12*2, 12*2, 12, 12*2, 12*2, 12, 12*2, 12*2, 14, 14*2, 14*2};
const double rpcb_radial_segm[RPCF_TYPES+1] = {1, 12, 12*2, 12*2, 12, 12*2, 12*2, 24, 24*2, 24*2, 12, 24*2, 24*2};
const double rpcf_radial_segm[RPCF_TYPES+1] = {1, 36, 36, 36, 18, 36, 36, 18, 36, 36, 18, 36, 36};

// DT # of superlayers in chamber
const double dt_n_superlayers[DT_TYPES+1]   = {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2};


///
class MuGeometryAreas
{
public:
  //MuGeometryAreas();

  void calculateCSCDetectorAreas(const CSCGeometry* g);
  void calculateGEMDetectorAreas(const GEMGeometry* g);
  void calculateDTDetectorAreas(const DTGeometry* g);
  void calculateRPCDetectorAreas(const RPCGeometry* g);

  float csc_total_areas_cm2[CSC_TYPES+1];
  float gem_total_areas_cm2[GEM_TYPES+1];
  float dt_total_areas_cm2[DT_TYPES+1];
  float rpcb_total_areas_cm2[RPCB_TYPES+1];
  float rpcf_total_areas_cm2[RPCF_TYPES+1];

  // vector index is over partitions
  std::vector<float> gem_total_part_areas_cm2[GEM_TYPES+1];  // partition total areas
  std::vector<float> gem_part_radius[GEM_TYPES+1];  // partition center radii
  std::vector<float> gem_part_halfheight[GEM_TYPES+1];  // partition half height

  // centers of CSC chamber positions in r
  static const float csc_ch_radius[CSC_TYPES+1];
  // half-spans of CSC stations in r
  static const float csc_ch_halfheight[CSC_TYPES+1];


  // centers of MB chamber types in |z|
  // Note: normally, wheel 0 chamber is centered at 0, but as we are looking at "half-detector"
  //       we take half-of-half of a wheel 0 position
  static const float dt_ch_z[DT_TYPES+1];
  // half-spans of MB chambers in |z|
  static const float dt_ch_halfspanz[DT_TYPES+1];

private:

};
 
typedef std::map<int, std::vector<float> > fidLUT;
 
class MuFiducial
{
 public:
  MuFiducial() {}
  ~MuFiducial() {}

  void buildGEMLUT();
  void buildCSCLUT();

  fidLUT getGEMLUT() const;
  fidLUT getCSCLUT() const;

  const GEMGeometry* getGEMGeometry() const {return gemGeometry;}
  const CSCGeometry* getCSCGeometry() const {return cscGeometry;}
  void setGEMGeometry(const GEMGeometry* geom) {gemGeometry = geom;}
  void setCSCGeometry(const CSCGeometry* geom) {cscGeometry = geom;}

/*   std::set<int> gemDetIds(math::XYZVectorD); */
/*   std::set<int> cscDetIds(math::XYZVectorD); */

 private:
  // Rmin, Rmax, PhiMin, PhiMax, Zmin, Zmax
  fidLUT gemLUT_;
  // Rmin, Rmax, PhiMin, PhiMax, Zmin, Zmax
  fidLUT cscLUT_;

  const CSCGeometry* cscGeometry;
  const GEMGeometry* gemGeometry;
};

} // namespace

#endif
