#ifndef SimDataFormats_GeneratorProducts_WeightsInfo_h
#define SimDataFormats_GeneratorProducts_WeightsInfo_h

/** \class PdfInfo
 *
 */
#include <string>

namespace gen {
  struct WeightsInfo {
    WeightsInfo() : id(""), wgt(0.) {}
    WeightsInfo(const std::string& s, const double w) : id(s), wgt(w) {}
    std::string id;
    double wgt;
  };
}  // namespace gen

#endif  // SimDataFormats_GeneratorProducts_WeightsInfo_h
