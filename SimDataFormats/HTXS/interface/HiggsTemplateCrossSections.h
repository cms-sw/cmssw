#ifndef SimDataFormats_GeneratorProducts_HiggsTemplateCrossSections_h
#define SimDataFormats_GeneratorProducts_HiggsTemplateCrossSections_h

#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>

/// Higgs Template Cross Section namespace
namespace HTXS {

  /// Error code: whether the classification was successful or failed
  enum ErrorCode {
    UNDEFINED = -99,
    SUCCESS = 0,                     ///< successful classification
    PRODMODE_DEFINED = 1,            ///< production mode not defined
    MOMENTUM_CONSERVATION = 2,       ///< failed momentum conservation
    HIGGS_IDENTIFICATION = 3,        ///< failed to identify Higgs boson
    HIGGS_DECAY_IDENTIFICATION = 4,  ///< failed to identify Higgs boson decay products
    HS_VTX_IDENTIFICATION = 5,       ///< failed to identify hard scatter vertex
    VH_IDENTIFICATION = 6,           ///< failed to identify associated vector boson
    VH_DECAY_IDENTIFICATION = 7,     ///< failed to identify associated vector boson decay products
    TOP_W_IDENTIFICATION = 8         ///< failed to identify top decay
  };

  /// Higgs production modes, corresponding to input sample
  enum HiggsProdMode { UNKNOWN = 0, GGF = 1, VBF = 2, WH = 3, QQ2ZH = 4, GG2ZH = 5, TTH = 6, BBH = 7, TH = 8 };

  /// Additional identifier flag for TH production modes
  enum tH_type { noTH = 0, THQB = 1, TWH = 2 };

  ///   Two digit number of format PF
  ///   P is digit for the physics process
  ///   and F is 0 for |yH|>2.5 and 11 for |yH|<2.5 ("in fiducial")

  /// Namespace for Stage0 categorization
  namespace Stage0 {
    /// @enum Stage-0 ategorization: Two-digit number of format PF, with P for process and F being 0 for |yH|>2.5 and 1 for |yH|<2.5
    enum Category {
      UNKNOWN = 0,
      GG2H_FWDH = 10,
      GG2H = 11,
      VBF_FWDH = 20,
      VBF = 21,
      VH2HQQ_FWDH = 22,
      VH2HQQ = 23,
      QQ2HLNU_FWDH = 30,
      QQ2HLNU = 31,
      QQ2HLL_FWDH = 40,
      QQ2HLL = 41,
      GG2HLL_FWDH = 50,
      GG2HLL = 51,
      TTH_FWDH = 60,
      TTH = 61,
      BBH_FWDH = 70,
      BBH = 71,
      TH_FWDH = 80,
      TH = 81
    };
  }  // namespace Stage0

  /// Categorization Stage 1:
  /// Three digit integer of format PF
  /// Where P is a digit representing the process
  /// F is a unique integer ( F < 99 ) corresponding to each Stage1 phase-space region (bin)
  namespace Stage1 {
    enum Category {
      UNKNOWN = 0,
      // Gluon fusion
      GG2H_FWDH = 100,
      GG2H_VBFTOPO_JET3VETO = 101,
      GG2H_VBFTOPO_JET3 = 102,
      GG2H_0J = 103,
      GG2H_1J_PTH_0_60 = 104,
      GG2H_1J_PTH_60_120 = 105,
      GG2H_1J_PTH_120_200 = 106,
      GG2H_1J_PTH_GT200 = 107,
      GG2H_GE2J_PTH_0_60 = 108,
      GG2H_GE2J_PTH_60_120 = 109,
      GG2H_GE2J_PTH_120_200 = 110,
      GG2H_GE2J_PTH_GT200 = 111,
      // "VBF"
      QQ2HQQ_FWDH = 200,
      QQ2HQQ_VBFTOPO_JET3VETO = 201,
      QQ2HQQ_VBFTOPO_JET3 = 202,
      QQ2HQQ_VH2JET = 203,
      QQ2HQQ_REST = 204,
      QQ2HQQ_PTJET1_GT200 = 205,
      // qq -> WH
      QQ2HLNU_FWDH = 300,
      QQ2HLNU_PTV_0_150 = 301,
      QQ2HLNU_PTV_150_250_0J = 302,
      QQ2HLNU_PTV_150_250_GE1J = 303,
      QQ2HLNU_PTV_GT250 = 304,
      // qq -> ZH
      QQ2HLL_FWDH = 400,
      QQ2HLL_PTV_0_150 = 401,
      QQ2HLL_PTV_150_250_0J = 402,
      QQ2HLL_PTV_150_250_GE1J = 403,
      QQ2HLL_PTV_GT250 = 404,
      // gg -> ZH
      GG2HLL_FWDH = 500,
      GG2HLL_PTV_0_150 = 501,
      GG2HLL_PTV_GT150_0J = 502,
      GG2HLL_PTV_GT150_GE1J = 503,
      // ttH
      TTH_FWDH = 600,
      TTH = 601,
      // bbH
      BBH_FWDH = 700,
      BBH = 701,
      // tH
      TH_FWDH = 800,
      TH = 801
    };
  }  // namespace Stage1

  namespace Stage1_1 {
    enum Category {
      UNKNOWN = 0,
      // Gluon fusion
      GG2H_FWDH = 100,
      GG2H_PTH_GT200 = 101,
      GG2H_0J_PTH_0_10 = 102,
      GG2H_0J_PTH_GT10 = 103,
      GG2H_1J_PTH_0_60 = 104,
      GG2H_1J_PTH_60_120 = 105,
      GG2H_1J_PTH_120_200 = 106,
      GG2H_GE2J_MJJ_0_350_PTH_0_60 = 107,
      GG2H_GE2J_MJJ_0_350_PTH_60_120 = 108,
      GG2H_GE2J_MJJ_0_350_PTH_120_200 = 109,
      GG2H_MJJ_350_700_PTHJJ_0_25 = 110,
      GG2H_MJJ_350_700_PTHJJ_GT25 = 111,
      GG2H_MJJ_GT700_PTHJJ_0_25 = 112,
      GG2H_MJJ_GT700_PTHJJ_GT25 = 113,
      // "VBF"
      QQ2HQQ_FWDH = 200,
      QQ2HQQ_0J = 201,
      QQ2HQQ_1J = 202,
      QQ2HQQ_MJJ_0_60 = 203,
      QQ2HQQ_MJJ_60_120 = 204,
      QQ2HQQ_MJJ_120_350 = 205,
      QQ2HQQ_MJJ_GT350_PTH_GT200 = 206,
      QQ2HQQ_MJJ_350_700_PTHJJ_0_25 = 207,
      QQ2HQQ_MJJ_350_700_PTHJJ_GT25 = 208,
      QQ2HQQ_MJJ_GT700_PTHJJ_0_25 = 209,
      QQ2HQQ_MJJ_GT700_PTHJJ_GT25 = 210,
      // qq -> WH
      QQ2HLNU_FWDH = 300,
      QQ2HLNU_PTV_0_75 = 301,
      QQ2HLNU_PTV_75_150 = 302,
      QQ2HLNU_PTV_150_250_0J = 303,
      QQ2HLNU_PTV_150_250_GE1J = 304,
      QQ2HLNU_PTV_GT250 = 305,
      // qq -> ZH
      QQ2HLL_FWDH = 400,
      QQ2HLL_PTV_0_75 = 401,
      QQ2HLL_PTV_75_150 = 402,
      QQ2HLL_PTV_150_250_0J = 403,
      QQ2HLL_PTV_150_250_GE1J = 404,
      QQ2HLL_PTV_GT250 = 405,
      // gg -> ZH
      GG2HLL_FWDH = 500,
      GG2HLL_PTV_0_75 = 501,
      GG2HLL_PTV_75_150 = 502,
      GG2HLL_PTV_150_250_0J = 503,
      GG2HLL_PTV_150_250_GE1J = 504,
      GG2HLL_PTV_GT250 = 505,
      // ttH
      TTH_FWDH = 600,
      TTH = 601,
      // bbH
      BBH_FWDH = 700,
      BBH = 701,
      // tH
      TH_FWDH = 800,
      TH = 801
    };
  }  // namespace Stage1_1

  namespace Stage1_1_Fine {
    enum Category {
      UNKNOWN = 0,
      // Gluon fusion
      GG2H_FWDH = 100,
      GG2H_PTH_GT200 = 101,
      GG2H_0J_PTH_0_10 = 102,
      GG2H_0J_PTH_GT10 = 103,
      GG2H_1J_PTH_0_60 = 104,
      GG2H_1J_PTH_60_120 = 105,
      GG2H_1J_PTH_120_200 = 106,
      GG2H_GE2J_MJJ_0_350_PTH_0_60_PTHJJ_0_25 = 107,
      GG2H_GE2J_MJJ_0_350_PTH_60_120_PTHJJ_0_25 = 108,
      GG2H_GE2J_MJJ_0_350_PTH_120_200_PTHJJ_0_25 = 109,
      GG2H_GE2J_MJJ_0_350_PTH_0_60_PTHJJ_GT25 = 110,
      GG2H_GE2J_MJJ_0_350_PTH_60_120_PTHJJ_GT25 = 111,
      GG2H_GE2J_MJJ_0_350_PTH_120_200_PTHJJ_GT25 = 112,
      GG2H_MJJ_350_700_PTHJJ_0_25 = 113,
      GG2H_MJJ_350_700_PTHJJ_GT25 = 114,
      GG2H_MJJ_700_1000_PTHJJ_0_25 = 115,
      GG2H_MJJ_700_1000_PTHJJ_GT25 = 116,
      GG2H_MJJ_1000_1500_PTHJJ_0_25 = 117,
      GG2H_MJJ_1000_1500_PTHJJ_GT25 = 118,
      GG2H_MJJ_GT1500_PTHJJ_0_25 = 119,
      GG2H_MJJ_GT1500_PTHJJ_GT25 = 120,
      // "VBF"
      QQ2HQQ_FWDH = 200,
      QQ2HQQ_0J = 201,
      QQ2HQQ_1J = 202,
      QQ2HQQ_MJJ_0_60_PTHJJ_0_25 = 203,
      QQ2HQQ_MJJ_60_120_PTHJJ_0_25 = 204,
      QQ2HQQ_MJJ_120_350_PTHJJ_0_25 = 205,
      QQ2HQQ_MJJ_0_60_PTHJJ_GT25 = 206,
      QQ2HQQ_MJJ_60_120_PTHJJ_GT25 = 207,
      QQ2HQQ_MJJ_120_350_PTHJJ_GT25 = 208,
      QQ2HQQ_MJJ_350_700_PTHJJ_0_25 = 209,
      QQ2HQQ_MJJ_350_700_PTHJJ_GT25 = 210,
      QQ2HQQ_MJJ_700_1000_PTHJJ_0_25 = 211,
      QQ2HQQ_MJJ_700_1000_PTHJJ_GT25 = 212,
      QQ2HQQ_MJJ_1000_1500_PTHJJ_0_25 = 213,
      QQ2HQQ_MJJ_1000_1500_PTHJJ_GT25 = 214,
      QQ2HQQ_MJJ_GT1500_PTHJJ_0_25 = 215,
      QQ2HQQ_MJJ_GT1500_PTHJJ_GT25 = 216,
      QQ2HQQ_PTH_GT200_MJJ_350_700_PTHJJ_0_25 = 217,
      QQ2HQQ_PTH_GT200_MJJ_350_700_PTHJJ_GT25 = 218,
      QQ2HQQ_PTH_GT200_MJJ_700_1000_PTHJJ_0_25 = 219,
      QQ2HQQ_PTH_GT200_MJJ_700_1000_PTHJJ_GT25 = 220,
      QQ2HQQ_PTH_GT200_MJJ_1000_1500_PTHJJ_0_25 = 221,
      QQ2HQQ_PTH_GT200_MJJ_1000_1500_PTHJJ_GT25 = 222,
      QQ2HQQ_PTH_GT200_MJJ_GT1500_PTHJJ_0_25 = 223,
      QQ2HQQ_PTH_GT200_MJJ_GT1500_PTHJJ_GT25 = 224,
      // qq -> WH
      QQ2HLNU_FWDH = 300,
      QQ2HLNU_PTV_0_75_0J = 301,
      QQ2HLNU_PTV_75_150_0J = 302,
      QQ2HLNU_PTV_150_250_0J = 303,
      QQ2HLNU_PTV_250_400_0J = 304,
      QQ2HLNU_PTV_GT400_0J = 305,
      QQ2HLNU_PTV_0_75_1J = 306,
      QQ2HLNU_PTV_75_150_1J = 307,
      QQ2HLNU_PTV_150_250_1J = 308,
      QQ2HLNU_PTV_250_400_1J = 309,
      QQ2HLNU_PTV_GT400_1J = 310,
      QQ2HLNU_PTV_0_75_GE2J = 311,
      QQ2HLNU_PTV_75_150_GE2J = 312,
      QQ2HLNU_PTV_150_250_GE2J = 313,
      QQ2HLNU_PTV_250_400_GE2J = 314,
      QQ2HLNU_PTV_GT400_GE2J = 315,
      // qq -> ZH
      QQ2HLL_FWDH = 400,
      QQ2HLL_PTV_0_75_0J = 401,
      QQ2HLL_PTV_75_150_0J = 402,
      QQ2HLL_PTV_150_250_0J = 403,
      QQ2HLL_PTV_250_400_0J = 404,
      QQ2HLL_PTV_GT400_0J = 405,
      QQ2HLL_PTV_0_75_1J = 406,
      QQ2HLL_PTV_75_150_1J = 407,
      QQ2HLL_PTV_150_250_1J = 408,
      QQ2HLL_PTV_250_400_1J = 409,
      QQ2HLL_PTV_GT400_1J = 410,
      QQ2HLL_PTV_0_75_GE2J = 411,
      QQ2HLL_PTV_75_150_GE2J = 412,
      QQ2HLL_PTV_150_250_GE2J = 413,
      QQ2HLL_PTV_250_400_GE2J = 414,
      QQ2HLL_PTV_GT400_GE2J = 415,
      // gg -> ZH
      GG2HLL_FWDH = 500,
      GG2HLL_PTV_0_75_0J = 501,
      GG2HLL_PTV_75_150_0J = 502,
      GG2HLL_PTV_150_250_0J = 503,
      GG2HLL_PTV_250_400_0J = 504,
      GG2HLL_PTV_GT400_0J = 505,
      GG2HLL_PTV_0_75_1J = 506,
      GG2HLL_PTV_75_150_1J = 507,
      GG2HLL_PTV_150_250_1J = 508,
      GG2HLL_PTV_250_400_1J = 509,
      GG2HLL_PTV_GT400_1J = 510,
      GG2HLL_PTV_0_75_GE2J = 511,
      GG2HLL_PTV_75_150_GE2J = 512,
      GG2HLL_PTV_150_250_GE2J = 513,
      GG2HLL_PTV_250_400_GE2J = 514,
      GG2HLL_PTV_GT400_GE2J = 515,
      // ttH
      TTH_FWDH = 600,
      TTH = 601,
      // bbH
      BBH_FWDH = 700,
      BBH = 701,
      // tH
      TH_FWDH = 800,
      TH = 801
    };
  }  // namespace Stage1_1_Fine

  /// Categorization Stage 1.2:
  /// Three digit integer of format PF
  /// Where P is a digit representing the process
  /// F is a unique integer ( F < 99 ) corresponding to each Stage1_2 phase-space region (bin)
  namespace Stage1_2 {
    enum Category {
      UNKNOWN = 0,
      // Gluon fusion
      GG2H_FWDH = 100,
      GG2H_PTH_200_300 = 101,
      GG2H_PTH_300_450 = 102,
      GG2H_PTH_450_650 = 103,
      GG2H_PTH_GT650 = 104,
      GG2H_0J_PTH_0_10 = 105,
      GG2H_0J_PTH_GT10 = 106,
      GG2H_1J_PTH_0_60 = 107,
      GG2H_1J_PTH_60_120 = 108,
      GG2H_1J_PTH_120_200 = 109,
      GG2H_GE2J_MJJ_0_350_PTH_0_60 = 110,
      GG2H_GE2J_MJJ_0_350_PTH_60_120 = 111,
      GG2H_GE2J_MJJ_0_350_PTH_120_200 = 112,
      GG2H_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25 = 113,
      GG2H_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25 = 114,
      GG2H_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25 = 115,
      GG2H_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25 = 116,
      // "VBF"
      QQ2HQQ_FWDH = 200,
      QQ2HQQ_0J = 201,
      QQ2HQQ_1J = 202,
      QQ2HQQ_GE2J_MJJ_0_60 = 203,
      QQ2HQQ_GE2J_MJJ_60_120 = 204,
      QQ2HQQ_GE2J_MJJ_120_350 = 205,
      QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200 = 206,
      QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25 = 207,
      QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25 = 208,
      QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25 = 209,
      QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25 = 210,
      // qq -> WH
      QQ2HLNU_FWDH = 300,
      QQ2HLNU_PTV_0_75 = 301,
      QQ2HLNU_PTV_75_150 = 302,
      QQ2HLNU_PTV_150_250_0J = 303,
      QQ2HLNU_PTV_150_250_GE1J = 304,
      QQ2HLNU_PTV_GT250 = 305,
      // qq -> ZH
      QQ2HLL_FWDH = 400,
      QQ2HLL_PTV_0_75 = 401,
      QQ2HLL_PTV_75_150 = 402,
      QQ2HLL_PTV_150_250_0J = 403,
      QQ2HLL_PTV_150_250_GE1J = 404,
      QQ2HLL_PTV_GT250 = 405,
      // gg -> ZH
      GG2HLL_FWDH = 500,
      GG2HLL_PTV_0_75 = 501,
      GG2HLL_PTV_75_150 = 502,
      GG2HLL_PTV_150_250_0J = 503,
      GG2HLL_PTV_150_250_GE1J = 504,
      GG2HLL_PTV_GT250 = 505,
      // ttH
      TTH_FWDH = 600,
      TTH_PTH_0_60 = 601,
      TTH_PTH_60_120 = 602,
      TTH_PTH_120_200 = 603,
      TTH_PTH_200_300 = 604,
      TTH_PTH_GT300 = 605,
      // bbH
      BBH_FWDH = 700,
      BBH = 701,
      // tH
      TH_FWDH = 800,
      TH = 801
    };
  }  // namespace Stage1_2

  namespace Stage1_2_Fine {
    enum Category {
      UNKNOWN = 0,
      // Gluon fusion
      GG2H_FWDH = 100,
      GG2H_PTH_200_300_PTHJoverPTH_0_15 = 101,
      GG2H_PTH_300_450_PTHJoverPTH_0_15 = 102,
      GG2H_PTH_450_650_PTHJoverPTH_0_15 = 103,
      GG2H_PTH_GT650_PTHJoverPTH_0_15 = 104,
      GG2H_PTH_200_300_PTHJoverPTH_GT15 = 105,
      GG2H_PTH_300_450_PTHJoverPTH_GT15 = 106,
      GG2H_PTH_450_650_PTHJoverPTH_GT15 = 107,
      GG2H_PTH_GT650_PTHJoverPTH_GT15 = 108,
      GG2H_0J_PTH_0_10 = 109,
      GG2H_0J_PTH_GT10 = 110,
      GG2H_1J_PTH_0_60 = 111,
      GG2H_1J_PTH_60_120 = 112,
      GG2H_1J_PTH_120_200 = 113,
      GG2H_GE2J_MJJ_0_350_PTH_0_60_PTHJJ_0_25 = 114,
      GG2H_GE2J_MJJ_0_350_PTH_60_120_PTHJJ_0_25 = 115,
      GG2H_GE2J_MJJ_0_350_PTH_120_200_PTHJJ_0_25 = 116,
      GG2H_GE2J_MJJ_0_350_PTH_0_60_PTHJJ_GT25 = 117,
      GG2H_GE2J_MJJ_0_350_PTH_60_120_PTHJJ_GT25 = 118,
      GG2H_GE2J_MJJ_0_350_PTH_120_200_PTHJJ_GT25 = 119,
      GG2H_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25 = 120,
      GG2H_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25 = 121,
      GG2H_GE2J_MJJ_700_1000_PTH_0_200_PTHJJ_0_25 = 122,
      GG2H_GE2J_MJJ_700_1000_PTH_0_200_PTHJJ_GT25 = 123,
      GG2H_GE2J_MJJ_1000_1500_PTH_0_200_PTHJJ_0_25 = 124,
      GG2H_GE2J_MJJ_1000_1500_PTH_0_200_PTHJJ_GT25 = 125,
      GG2H_GE2J_MJJ_GT1500_PTH_0_200_PTHJJ_0_25 = 126,
      GG2H_GE2J_MJJ_GT1500_PTH_0_200_PTHJJ_GT25 = 127,
      // "VBF"
      QQ2HQQ_FWDH = 200,
      QQ2HQQ_0J = 201,
      QQ2HQQ_1J = 202,
      QQ2HQQ_GE2J_MJJ_0_60_PTHJJ_0_25 = 203,
      QQ2HQQ_GE2J_MJJ_60_120_PTHJJ_0_25 = 204,
      QQ2HQQ_GE2J_MJJ_120_350_PTHJJ_0_25 = 205,
      QQ2HQQ_GE2J_MJJ_0_60_PTHJJ_GT25 = 206,
      QQ2HQQ_GE2J_MJJ_60_120_PTHJJ_GT25 = 207,
      QQ2HQQ_GE2J_MJJ_120_350_PTHJJ_GT25 = 208,
      QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25 = 209,
      QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25 = 210,
      QQ2HQQ_GE2J_MJJ_700_1000_PTH_0_200_PTHJJ_0_25 = 211,
      QQ2HQQ_GE2J_MJJ_700_1000_PTH_0_200_PTHJJ_GT25 = 212,
      QQ2HQQ_GE2J_MJJ_1000_1500_PTH_0_200_PTHJJ_0_25 = 213,
      QQ2HQQ_GE2J_MJJ_1000_1500_PTH_0_200_PTHJJ_GT25 = 214,
      QQ2HQQ_GE2J_MJJ_GT1500_PTH_0_200_PTHJJ_0_25 = 215,
      QQ2HQQ_GE2J_MJJ_GT1500_PTH_0_200_PTHJJ_GT25 = 216,
      QQ2HQQ_GE2J_MJJ_350_700_PTH_GT200_PTHJJ_0_25 = 217,
      QQ2HQQ_GE2J_MJJ_350_700_PTH_GT200_PTHJJ_GT25 = 218,
      QQ2HQQ_GE2J_MJJ_700_1000_PTH_GT200_PTHJJ_0_25 = 219,
      QQ2HQQ_GE2J_MJJ_700_1000_PTH_GT200_PTHJJ_GT25 = 220,
      QQ2HQQ_GE2J_MJJ_1000_1500_PTH_GT200_PTHJJ_0_25 = 221,
      QQ2HQQ_GE2J_MJJ_1000_1500_PTH_GT200_PTHJJ_GT25 = 222,
      QQ2HQQ_GE2J_MJJ_GT1500_PTH_GT200_PTHJJ_0_25 = 223,
      QQ2HQQ_GE2J_MJJ_GT1500_PTH_GT200_PTHJJ_GT25 = 224,
      // qq -> WH
      QQ2HLNU_FWDH = 300,
      QQ2HLNU_PTV_0_75_0J = 301,
      QQ2HLNU_PTV_75_150_0J = 302,
      QQ2HLNU_PTV_150_250_0J = 303,
      QQ2HLNU_PTV_250_400_0J = 304,
      QQ2HLNU_PTV_GT400_0J = 305,
      QQ2HLNU_PTV_0_75_1J = 306,
      QQ2HLNU_PTV_75_150_1J = 307,
      QQ2HLNU_PTV_150_250_1J = 308,
      QQ2HLNU_PTV_250_400_1J = 309,
      QQ2HLNU_PTV_GT400_1J = 310,
      QQ2HLNU_PTV_0_75_GE2J = 311,
      QQ2HLNU_PTV_75_150_GE2J = 312,
      QQ2HLNU_PTV_150_250_GE2J = 313,
      QQ2HLNU_PTV_250_400_GE2J = 314,
      QQ2HLNU_PTV_GT400_GE2J = 315,
      // qq -> ZH
      QQ2HLL_FWDH = 400,
      QQ2HLL_PTV_0_75_0J = 401,
      QQ2HLL_PTV_75_150_0J = 402,
      QQ2HLL_PTV_150_250_0J = 403,
      QQ2HLL_PTV_250_400_0J = 404,
      QQ2HLL_PTV_GT400_0J = 405,
      QQ2HLL_PTV_0_75_1J = 406,
      QQ2HLL_PTV_75_150_1J = 407,
      QQ2HLL_PTV_150_250_1J = 408,
      QQ2HLL_PTV_250_400_1J = 409,
      QQ2HLL_PTV_GT400_1J = 410,
      QQ2HLL_PTV_0_75_GE2J = 411,
      QQ2HLL_PTV_75_150_GE2J = 412,
      QQ2HLL_PTV_150_250_GE2J = 413,
      QQ2HLL_PTV_250_400_GE2J = 414,
      QQ2HLL_PTV_GT400_GE2J = 415,
      // gg -> ZH
      GG2HLL_FWDH = 500,
      GG2HLL_PTV_0_75_0J = 501,
      GG2HLL_PTV_75_150_0J = 502,
      GG2HLL_PTV_150_250_0J = 503,
      GG2HLL_PTV_250_400_0J = 504,
      GG2HLL_PTV_GT400_0J = 505,
      GG2HLL_PTV_0_75_1J = 506,
      GG2HLL_PTV_75_150_1J = 507,
      GG2HLL_PTV_150_250_1J = 508,
      GG2HLL_PTV_250_400_1J = 509,
      GG2HLL_PTV_GT400_1J = 510,
      GG2HLL_PTV_0_75_GE2J = 511,
      GG2HLL_PTV_75_150_GE2J = 512,
      GG2HLL_PTV_150_250_GE2J = 513,
      GG2HLL_PTV_250_400_GE2J = 514,
      GG2HLL_PTV_GT400_GE2J = 515,
      // ttH
      TTH_FWDH = 600,
      TTH_PTH_0_60 = 601,
      TTH_PTH_60_120 = 602,
      TTH_PTH_120_200 = 603,
      TTH_PTH_200_300 = 604,
      TTH_PTH_300_450 = 605,
      TTH_PTH_GT450 = 606,
      // bbH
      BBH_FWDH = 700,
      BBH = 701,
      // tH
      TH_FWDH = 800,
      TH = 801
    };
  }  // namespace Stage1_2_Fine

  //#ifdef ROOT_TLorentzVector
  //typedef TLorentzVector TLV;
  typedef math::XYZTLorentzVectorD TLV;
  typedef std::vector<TLV> TLVs;

  template <class vec4>
  TLV MakeTLV(vec4 const p) {
    return TLV(p.px(), p.py(), p.pz(), p.E());
  }

  template <class Vvec4>
  inline TLVs MakeTLVs(Vvec4 const &rivet_jets) {
    TLVs jets;
    for (const auto &jet : rivet_jets)
      jets.push_back(MakeTLV(jet));
    return jets;
  }

  // Structure holding information about the current event:
  // Four-momenta and event classification according to the
  // Higgs Template Cross Section
  struct HiggsClassification {
    // Higgs production mode
    HTXS::HiggsProdMode prodMode;
    // The Higgs boson
    TLV higgs;
    // The Higgs boson decay products
    TLV p4decay_higgs;
    // Associated vector bosons
    TLV V;
    // The V-boson decay products
    TLV p4decay_V;
    // Jets are built ignoring Higgs decay products and leptons from V decays
    // jets with pT > 25 GeV and 30 GeV
    TLVs jets25, jets30;
    // Event categorization according to YR4 wrtietup
    // https://cds.cern.ch/record/2138079
    HTXS::Stage0::Category stage0_cat;
    HTXS::Stage1::Category stage1_cat_pTjet25GeV;
    HTXS::Stage1::Category stage1_cat_pTjet30GeV;
    HTXS::Stage1_1::Category stage1_1_cat_pTjet25GeV;
    HTXS::Stage1_1::Category stage1_1_cat_pTjet30GeV;
    HTXS::Stage1_1_Fine::Category stage1_1_fine_cat_pTjet25GeV;
    HTXS::Stage1_1_Fine::Category stage1_1_fine_cat_pTjet30GeV;
    HTXS::Stage1_2::Category stage1_2_cat_pTjet25GeV;
    HTXS::Stage1_2::Category stage1_2_cat_pTjet30GeV;
    HTXS::Stage1_2_Fine::Category stage1_2_fine_cat_pTjet25GeV;
    HTXS::Stage1_2_Fine::Category stage1_2_fine_cat_pTjet30GeV;
    // Flag for Z->vv decay mode (needed to split QQ2ZH and GG2ZH)
    bool isZ2vvDecay;
    // Error code :: classification was succesful or some error occured
    HTXS::ErrorCode errorCode;
  };

  template <class category>
  inline HTXS::HiggsClassification Rivet2Root(category const &htxs_cat_rivet) {
    HTXS::HiggsClassification cat;
    cat.prodMode = htxs_cat_rivet.prodMode;
    cat.errorCode = htxs_cat_rivet.errorCode;
    cat.higgs = MakeTLV(htxs_cat_rivet.higgs);
    cat.V = MakeTLV(htxs_cat_rivet.V);
    cat.p4decay_higgs = MakeTLV(htxs_cat_rivet.p4decay_higgs);
    cat.p4decay_V = MakeTLV(htxs_cat_rivet.p4decay_V);
    cat.jets25 = MakeTLVs(htxs_cat_rivet.jets25);
    cat.jets30 = MakeTLVs(htxs_cat_rivet.jets30);
    cat.stage0_cat = htxs_cat_rivet.stage0_cat;
    cat.stage1_cat_pTjet25GeV = htxs_cat_rivet.stage1_cat_pTjet25GeV;
    cat.stage1_cat_pTjet30GeV = htxs_cat_rivet.stage1_cat_pTjet30GeV;
    cat.stage1_1_cat_pTjet25GeV = htxs_cat_rivet.stage1_1_cat_pTjet25GeV;
    cat.stage1_1_cat_pTjet30GeV = htxs_cat_rivet.stage1_1_cat_pTjet30GeV;
    cat.stage1_1_fine_cat_pTjet25GeV = htxs_cat_rivet.stage1_1_fine_cat_pTjet25GeV;
    cat.stage1_1_fine_cat_pTjet30GeV = htxs_cat_rivet.stage1_1_fine_cat_pTjet30GeV;
    cat.stage1_2_cat_pTjet25GeV = htxs_cat_rivet.stage1_2_cat_pTjet25GeV;
    cat.stage1_2_cat_pTjet30GeV = htxs_cat_rivet.stage1_2_cat_pTjet30GeV;
    cat.stage1_2_fine_cat_pTjet25GeV = htxs_cat_rivet.stage1_2_fine_cat_pTjet25GeV;
    cat.stage1_2_fine_cat_pTjet30GeV = htxs_cat_rivet.stage1_2_fine_cat_pTjet30GeV;
    cat.isZ2vvDecay = htxs_cat_rivet.isZ2vvDecay;
    return cat;
  }

  inline int HTXSstage1_to_HTXSstage1FineIndex(HTXS::Stage1::Category stage1, HiggsProdMode prodMode, tH_type tH) {
    if (stage1 == HTXS::Stage1::Category::UNKNOWN)
      return 0;
    int P = (int)(stage1 / 100);
    int F = (int)(stage1 % 100);
    // 1.a spit tH categories
    if (prodMode == HiggsProdMode::TH) {
      // check that tH splitting is valid for Stage-1 FineIndex
      // else return unknown category
      if (tH == tH_type::noTH)
        return 0;
      // check if forward tH
      int fwdH = F == 0 ? 0 : 1;
      return (49 + 2 * (tH - 1) + fwdH);
    }
    // 1.b QQ2HQQ --> split into VBF, WH, ZH -> HQQ
    // offset vector 1: input is the Higgs prodMode
    // first two indicies are dummies, given that this is only called for prodMode=2,3,4
    std::vector<int> pMode_offset = {0, 0, 13, 19, 25};
    if (P == 2)
      return (F + pMode_offset[prodMode]);
    // 1.c remaining categories
    // offset vector 2: input is the Stage-1 category P
    // third index is dummy, given that this is called for category P=0,1,3,4,5,6,7
    std::vector<int> catP_offset = {0, 1, 0, 31, 36, 41, 45, 47};
    return (F + catP_offset[P]);
  }

  inline int HTXSstage1_to_HTXSstage1FineIndex(const HiggsClassification &stxs,
                                               tH_type tH = noTH,
                                               bool jets_pT25 = false) {
    HTXS::Stage1::Category stage1 = jets_pT25 == false ? stxs.stage1_cat_pTjet30GeV : stxs.stage1_cat_pTjet25GeV;
    return HTXSstage1_to_HTXSstage1FineIndex(stage1, stxs.prodMode, tH);
  }

  inline int HTXSstage1_to_index(HTXS::Stage1::Category stage1) {
    // the Stage-1 categories
    int P = (int)(stage1 / 100);
    int F = (int)(stage1 % 100);
    std::vector<int> offset{0, 1, 13, 19, 24, 29, 33, 35, 37, 39};
    // convert to linear values
    return (F + offset[P]);
  }

  inline int HTXSstage1_2_to_HTXSstage1_2_FineIndex(HTXS::Stage1_2::Category stage1_2,
                                                    HiggsProdMode prodMode,
                                                    tH_type tH) {
    if (stage1_2 == HTXS::Stage1_2::Category::UNKNOWN)
      return 0;
    int P = (int)(stage1_2 / 100);
    int F = (int)(stage1_2 % 100);
    // 1.a spit tH categories
    if (prodMode == HiggsProdMode::TH) {
      // check that tH splitting is valid for Stage-1 FineIndex
      // else return unknown category
      if (tH == tH_type::noTH)
        return 0;
      // check if forward tH
      int fwdH = F == 0 ? 0 : 1;
      return (94 + 2 * (tH - 1) + fwdH);
    }
    // 1.b QQ2HQQ --> split into VBF, WH, ZH -> HQQ
    // offset vector 1: input is the Higgs prodMode
    // first two indicies are dummies, given that this is only called for prodMode=2,3,4
    std::vector<int> pMode_offset = {0, 0, 35, 46, 57};
    if (P == 2)
      return (F + pMode_offset[prodMode]);
    // 1.c GG2ZH split into gg->ZH-had and gg->ZH-lep
    if (prodMode == HiggsProdMode::GG2ZH && P == 1)
      return F + 18;
    // 1.d remaining categories
    // offset vector 2: input is the Stage-1 category P
    // third index is dummy, given that this is called for category P=0,1,3,4,5,6,7
    std::vector<int> catP_offset = {0, 1, 0, 68, 74, 80, 86, 92};
    return (F + catP_offset[P]);
  }

  inline int HTXSstage1_2_to_HTXSstage1_2_FineIndex(const HiggsClassification &stxs,
                                                    tH_type tH = noTH,
                                                    bool jets_pT25 = false) {
    HTXS::Stage1_2::Category stage1_2 =
        jets_pT25 == false ? stxs.stage1_2_cat_pTjet30GeV : stxs.stage1_2_cat_pTjet25GeV;
    return HTXSstage1_2_to_HTXSstage1_2_FineIndex(stage1_2, stxs.prodMode, tH);
  }

  inline int HTXSstage1_2_to_index(HTXS::Stage1_2::Category stage1_2) {
    // the Stage-1 categories
    int P = (int)(stage1_2 / 100);
    int F = (int)(stage1_2 % 100);
    //std::vector<int> offset{0,1,13,19,24,29,33,35,37,39};
    std::vector<int> offset{0, 1, 18, 29, 35, 41, 47, 53, 55, 57};
    // convert to linear values
    return (F + offset[P]);
  }

  //Same for Stage1_2_Fine categories
  inline int HTXSstage1_2_Fine_to_HTXSstage1_2_Fine_FineIndex(HTXS::Stage1_2_Fine::Category Stage1_2_Fine,
                                                              HiggsProdMode prodMode,
                                                              tH_type tH) {
    if (Stage1_2_Fine == HTXS::Stage1_2_Fine::Category::UNKNOWN)
      return 0;
    int P = (int)(Stage1_2_Fine / 100);
    int F = (int)(Stage1_2_Fine % 100);
    // 1.a spit tH categories
    if (prodMode == HiggsProdMode::TH) {
      // check that tH splitting is valid for Stage-1 FineIndex
      // else return unknown category
      if (tH == tH_type::noTH)
        return 0;
      // check if forward tH
      int fwdH = F == 0 ? 0 : 1;
      return (189 + 2 * (tH - 1) + fwdH);
    }
    // 1.b QQ2HQQ --> split into VBF, WH, ZH -> HQQ
    // offset vector 1: input is the Higgs prodMode
    // first two indicies are dummies, given that this is only called for prodMode=2,3,4
    std::vector<int> pMode_offset = {0, 0, 57, 82, 107};
    if (P == 2)
      return (F + pMode_offset[prodMode]);
    // 1.c GG2ZH split into gg->ZH-had and gg->ZH-lep
    if (prodMode == HiggsProdMode::GG2ZH && P == 1)
      return F + 29;
    // 1.d remaining categories
    // offset vector 2: input is the Stage-1 category P
    // third index is dummy, given that this is called for category P=0,1,3,4,5,6,7
    std::vector<int> catP_offset = {0, 1, 0, 132, 148, 164, 180, 187};
    return (F + catP_offset[P]);
  }

  inline int HTXSstage1_2_Fine_to_HTXSstage1_2_Fine_FineIndex(const HiggsClassification &stxs,
                                                              tH_type tH = noTH,
                                                              bool jets_pT25 = false) {
    HTXS::Stage1_2_Fine::Category Stage1_2_Fine =
        jets_pT25 == false ? stxs.stage1_2_fine_cat_pTjet30GeV : stxs.stage1_2_fine_cat_pTjet25GeV;
    return HTXSstage1_2_Fine_to_HTXSstage1_2_Fine_FineIndex(Stage1_2_Fine, stxs.prodMode, tH);
  }

  inline int HTXSstage1_2_Fine_to_index(HTXS::Stage1_2_Fine::Category Stage1_2_Fine) {
    // the Stage-1_2_Fine categories
    int P = (int)(Stage1_2_Fine / 100);
    int F = (int)(Stage1_2_Fine % 100);
    std::vector<int> offset{0, 1, 29, 54, 70, 86, 102, 109, 111, 113};
    // convert to linear values
    return (F + offset[P]);
  }

  // #endif

}  // namespace HTXS

#ifdef RIVET_Particle_HH
//#ifdef HIGGSTRUTHCLASSIFIER_HIGGSTRUTHCLASSIFIER_CC
//#include "Rivet/Particle.hh"
namespace Rivet {

  /// @struct HiggsClassification
  /// @brief Structure holding information about the current event:
  ///        Four-momenta and event classification according to the
  ///        Higgs Template Cross Section
  struct HiggsClassification {
    /// Higgs production mode
    HTXS::HiggsProdMode prodMode;
    /// The Higgs boson
    Rivet::Particle higgs;
    /// Vector boson produced in asscoiation with the Higgs
    Rivet::Particle V;
    /// The four momentum sum of all stable decay products orignating from the Higgs boson
    Rivet::FourMomentum p4decay_higgs;
    /// The four momentum sum of all stable decay products orignating from the vector boson in associated production
    Rivet::FourMomentum p4decay_V;
    /// Jets built ignoring Higgs decay products and leptons from V decays, pT thresholds at 25 GeV and 30 GeV
    Rivet::Jets jets25, jets30;
    /// Stage-0 HTXS event classifcation, see: https://cds.cern.ch/record/2138079
    HTXS::Stage0::Category stage0_cat;
    /// Stage-1 HTXS event classifcation, see: https://cds.cern.ch/record/2138079
    HTXS::Stage1::Category stage1_cat_pTjet25GeV;
    /// Stage-1 HTXS event classifcation, see: https://cds.cern.ch/record/2138079
    HTXS::Stage1::Category stage1_cat_pTjet30GeV;
    /// Error code: Whether classification was succesful or some error occured
    HTXS::Stage1_1::Category stage1_1_cat_pTjet25GeV;
    /// Stage-1 STXS event classifcation, see: https://cds.cern.ch/record/2138079
    HTXS::Stage1_1::Category stage1_1_cat_pTjet30GeV;
    /// Stage-1_1 STXS event classifcation, see: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHXSWGFiducialAndSTXS#Stage_1_1
    HTXS::Stage1_1_Fine::Category stage1_1_fine_cat_pTjet25GeV;
    /// Stage-1_1 STXS event classifcation, see: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHXSWGFiducialAndSTXS#Stage_1_1
    HTXS::Stage1_1_Fine::Category stage1_1_fine_cat_pTjet30GeV;
    /// Stage-1 STXS event classifcation, see: https://cds.cern.ch/record/2138079
    HTXS::Stage1_2::Category stage1_2_cat_pTjet25GeV;
    /// Stage-1 STXS event classifcation, see: https://cds.cern.ch/record/2138079
    HTXS::Stage1_2::Category stage1_2_cat_pTjet30GeV;
    /// Stage-1_2 STXS event classifcation, see: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHXSWGFiducialAndSTXS#Stage_1_1
    HTXS::Stage1_2_Fine::Category stage1_2_fine_cat_pTjet25GeV;
    /// Stage-1_2 STXS event classifcation, see: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHXSWGFiducialAndSTXS#Stage_1_1
    HTXS::Stage1_2_Fine::Category stage1_2_fine_cat_pTjet30GeV;
    /// Flag to distiguish the Z->vv and Z->l+l- decay modes
    bool isZ2vvDecay;
    /// Error code: Whether classification was succesful or some error occured
    HTXS::ErrorCode errorCode;
  };
}  // namespace Rivet
#endif

#endif
