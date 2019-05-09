#ifndef SimDataFormatsValidationFormatsPHGCalValidInfo_h
#define SimDataFormatsValidationFormatsPHGCalValidInfo_h

#include <string>
#include <vector>
#include <memory>

///////////////////////////////////////////////////////////////////////////////
////// PHGCalValidInfo
///////////////////////////////////////////////////////////////////////////////

class PHGCalValidInfo {
public:
  PHGCalValidInfo() : edepEETot(0.0), edepHEFTot(0.0), edepHEBTot(0.0) {}
  virtual ~PHGCalValidInfo() {}

  float eeTotEdep() const { return edepEETot; }
  float hefTotEdep() const { return edepHEFTot; }
  float hebTotEdep() const { return edepHEBTot; }

  std::vector<float> eehgcEdep() const { return hgcEEedep; }
  std::vector<float> hefhgcEdep() const { return hgcHEFedep; }
  std::vector<float> hebhgcEdep() const { return hgcHEBedep; }
  std::vector<unsigned int> hitDets() const { return hgcHitDets; }
  std::vector<unsigned int> hitIndex() const { return hgcHitIndex; }
  std::vector<float> hitvtxX() const { return hgcHitVtxX; }
  std::vector<float> hitvtxY() const { return hgcHitVtxY; }
  std::vector<float> hitvtxZ() const { return hgcHitVtxZ; }

  void fillhgcHits(const std::vector<unsigned int>& hitdets,
                   const std::vector<unsigned int>& hitindex,
                   const std::vector<double>& hitvtxX,
                   const std::vector<double>& hitvtxY,
                   const std::vector<double>& hitvtxZ);

  void fillhgcLayers(const double edepEE,
                     const double edepHEF,
                     const double edepHEB,
                     const std::vector<double>& eedep,
                     const std::vector<double>& hefdep,
                     const std::vector<double>& hebdep);

private:
  float edepEETot, edepHEFTot, edepHEBTot;
  std::vector<float> hgcEEedep, hgcHEFedep, hgcHEBedep;
  std::vector<float> hgcHitVtxX, hgcHitVtxY, hgcHitVtxZ;
  std::vector<unsigned int> hgcHitDets, hgcHitIndex;
};

#endif
