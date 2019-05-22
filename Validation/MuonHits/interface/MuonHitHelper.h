#ifndef Validation_MuonHits_MuonHitHelper_h
#define Validation_MuonHits_MuonHitHelper_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

class MuonHitHelper {
public:
  /// CSC chamber types, according to CSCDetId::iChamberType()
  enum CSCType {
    CSC_ALL = 0,
    CSC_ME11,
    CSC_ME1a,
    CSC_ME1b,
    CSC_ME12,
    CSC_ME13,
    CSC_ME21,
    CSC_ME22,
    CSC_ME31,
    CSC_ME32,
    CSC_ME41,
    CSC_ME42
  };

  /// GEM chamber types
  enum GEMType { GEM_ALL = 0, GEM_ME11, GEM_ME21 };

  /// RPC endcap chamber types
  enum RPCType {
    RPC_ALL = 0,
    RPC_ME12,
    RPC_ME13,
    RPC_ME22,
    RPC_ME23,
    RPC_ME31,
    RPC_ME32,
    RPC_ME33,
    RPC_ME41,
    RPC_ME42,
    RPC_ME43,
    RPC_MB01,
    RPC_MB02,
    RPC_MB03,
    RPC_MB04,
    RPC_MB11p,
    RPC_MB12p,
    RPC_MB13p,
    RPC_MB14p,
    RPC_MB21p,
    RPC_MB22p,
    RPC_MB23p,
    RPC_MB24p,
    RPC_MB11n,
    RPC_MB12n,
    RPC_MB13n,
    RPC_MB14n,
    RPC_MB21n,
    RPC_MB22n,
    RPC_MB23n,
    RPC_MB24n
  };

  /// DT chamber types
  enum DTType {
    DT_ALL = 0,
    DT_MB01,
    DT_MB02,
    DT_MB03,
    DT_MB04,
    DT_MB11p,
    DT_MB12p,
    DT_MB13p,
    DT_MB14p,
    DT_MB21p,
    DT_MB22p,
    DT_MB23p,
    DT_MB24p,
    DT_MB11n,
    DT_MB12n,
    DT_MB13n,
    DT_MB14n,
    DT_MB21n,
    DT_MB22n,
    DT_MB23n,
    DT_MB24n
  };

  /// check detid type
  static bool isDT(unsigned int detId);
  static bool isGEM(unsigned int detId);
  static bool isCSC(unsigned int detId);
  static bool isRPC(unsigned int detId);
  static bool isME0(unsigned int detId);

  // return MuonType for a particular DetId
  static int toGEMType(int st, int ri);
  static int toRPCType(int re, int st, int ri);
  static int toDTType(int wh, int st);
  static int toCSCType(int st, int ri);

  // get chamber number
  static int chamber(const DetId& id);
};

#endif
