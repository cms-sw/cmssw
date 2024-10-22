#ifndef SimDataFormats_GeneratorProducts_LHECommonBlocks_h
#define SimDataFormats_GeneratorProducts_LHECommonBlocks_h

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

extern "C" {
extern struct HEPRUP_ {
  int idbmup[2];
  double ebmup[2];
  int pdfgup[2];
  int pdfsup[2];
  int idwtup;
  int nprup;
  double xsecup[100];
  double xerrup[100];
  double xmaxup[100];
  int lprup[100];
} heprup_;

extern struct HEPEUP_ {
  int nup;
  int idprup;
  double xwgtup;
  double scalup;
  double aqedup;
  double aqcdup;
  int idup[500];
  int istup[500];
  int mothup[500][2];
  int icolup[500][2];
  double pup[500][5];
  double vtimup[500];
  double spinup[500];
} hepeup_;
}  // extern "C"

namespace lhef {

  class CommonBlocks {
  public:
    static void fillHEPRUP(const HEPRUP *heprup) {
      heprup_.idbmup[0] = heprup->IDBMUP.first;
      heprup_.idbmup[1] = heprup->IDBMUP.second;
      heprup_.ebmup[0] = heprup->EBMUP.first;
      heprup_.ebmup[1] = heprup->EBMUP.second;
      heprup_.pdfgup[0] = heprup->PDFGUP.first;
      heprup_.pdfgup[1] = heprup->PDFGUP.second;
      heprup_.pdfsup[0] = heprup->PDFSUP.first;
      heprup_.pdfsup[1] = heprup->PDFSUP.second;
      heprup_.idwtup = heprup->IDWTUP;
      heprup_.nprup = heprup->NPRUP;
      for (int i = 0; i < heprup->NPRUP; i++) {
        heprup_.xsecup[i] = heprup->XSECUP[i];
        heprup_.xerrup[i] = heprup->XERRUP[i];
        heprup_.xmaxup[i] = heprup->XMAXUP[i];
        heprup_.lprup[i] = heprup->LPRUP[i];
      }
    }

    static void fillHEPEUP(const HEPEUP *hepeup) {
      hepeup_.nup = hepeup->NUP;
      hepeup_.idprup = hepeup->IDPRUP;
      hepeup_.xwgtup = hepeup->XWGTUP;
      hepeup_.scalup = hepeup->SCALUP;
      hepeup_.aqedup = hepeup->AQEDUP;
      hepeup_.aqcdup = hepeup->AQCDUP;
      for (int i = 0; i < hepeup->NUP; i++) {
        hepeup_.idup[i] = hepeup->IDUP[i];
        hepeup_.istup[i] = hepeup->ISTUP[i];
        hepeup_.mothup[i][0] = hepeup->MOTHUP[i].first;
        hepeup_.mothup[i][1] = hepeup->MOTHUP[i].second;
        hepeup_.icolup[i][0] = hepeup->ICOLUP[i].first;
        hepeup_.icolup[i][1] = hepeup->ICOLUP[i].second;
        for (unsigned int j = 0; j < 5; j++)
          hepeup_.pup[i][j] = hepeup->PUP[i][j];
        hepeup_.vtimup[i] = hepeup->VTIMUP[i];
        hepeup_.spinup[i] = hepeup->SPINUP[i];
      }
    }

    static void readHEPRUP(HEPRUP *heprup) {
      heprup->resize(heprup_.nprup);
      heprup->IDBMUP.first = heprup_.idbmup[0];
      heprup->IDBMUP.second = heprup_.idbmup[1];
      heprup->EBMUP.first = heprup_.ebmup[0];
      heprup->EBMUP.second = heprup_.ebmup[1];
      heprup->PDFGUP.first = heprup_.pdfgup[0];
      heprup->PDFGUP.second = heprup_.pdfgup[1];
      heprup->PDFSUP.first = heprup_.pdfsup[0];
      heprup->PDFSUP.second = heprup_.pdfsup[1];
      heprup->IDWTUP = heprup_.idwtup;
      for (int i = 0; i < heprup->NPRUP; i++) {
        heprup->XSECUP[i] = heprup_.xsecup[i];
        heprup->XERRUP[i] = heprup_.xerrup[i];
        heprup->XMAXUP[i] = heprup_.xmaxup[i];
        heprup->LPRUP[i] = heprup_.lprup[i];
      }
    }

    static void readHEPEUP(HEPEUP *hepeup) {
      hepeup->resize(hepeup_.nup);
      hepeup->IDPRUP = hepeup_.idprup;
      hepeup->XWGTUP = hepeup_.xwgtup;
      hepeup->SCALUP = hepeup_.scalup;
      hepeup->AQEDUP = hepeup_.aqedup;
      hepeup->AQCDUP = hepeup_.aqcdup;
      for (int i = 0; i < hepeup->NUP; i++) {
        hepeup->IDUP[i] = hepeup_.idup[i];
        hepeup->ISTUP[i] = hepeup_.istup[i];
        hepeup->MOTHUP[i].first = hepeup_.mothup[i][0];
        hepeup->MOTHUP[i].second = hepeup_.mothup[i][1];
        hepeup->ICOLUP[i].first = hepeup_.icolup[i][0];
        hepeup->ICOLUP[i].second = hepeup_.icolup[i][1];
        for (unsigned int j = 0; j < 5; j++)
          hepeup->PUP[i][j] = hepeup_.pup[i][j];
        hepeup->VTIMUP[i] = hepeup_.vtimup[i];
        hepeup->SPINUP[i] = hepeup_.spinup[i];
      }
    }

    CommonBlocks() = delete;
    ~CommonBlocks() = delete;
  };

}  // namespace lhef

#endif  // SimDataFormats_GeneratorProducts_LHECommonBlocks_h
