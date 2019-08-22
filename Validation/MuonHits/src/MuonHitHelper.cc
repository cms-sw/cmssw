#include "Validation/MuonHits/interface/MuonHitHelper.h"

bool MuonHitHelper::isDT(unsigned int detId) {
  return (DetId(detId)).det() == DetId::Muon && (DetId(detId)).subdetId() == MuonSubdetId::DT;
}

bool MuonHitHelper::isGEM(unsigned int detId) {
  return (DetId(detId)).det() == DetId::Muon && (DetId(detId)).subdetId() == MuonSubdetId::GEM;
}

bool MuonHitHelper::isCSC(unsigned int detId) {
  return (DetId(detId)).det() == DetId::Muon && (DetId(detId)).subdetId() == MuonSubdetId::CSC;
}

bool MuonHitHelper::isRPC(unsigned int detId) {
  return (DetId(detId)).det() == DetId::Muon && (DetId(detId)).subdetId() == MuonSubdetId::RPC;
}

bool MuonHitHelper::isME0(unsigned int detId) {
  return (DetId(detId)).det() == DetId::Muon && (DetId(detId)).subdetId() == MuonSubdetId::ME0;
}

int MuonHitHelper::chamber(const DetId& id) {
  if (id.det() != DetId::Detector::Muon)
    return -99;
  int chamberN = 0;
  switch (id.subdetId()) {
    case MuonSubdetId::GEM:
      chamberN = GEMDetId(id).chamber();
      break;
    case MuonSubdetId::RPC:
      // works only for endcap!!
      chamberN = RPCDetId(id).sector();
      break;
    case MuonSubdetId::CSC:
      chamberN = CSCDetId(id).chamber();
      break;
    case MuonSubdetId::ME0:
      chamberN = ME0DetId(id).chamber();
      break;
    case MuonSubdetId::DT:
      chamberN = DTChamberId(id).sector();
      break;
  };
  return chamberN;
}

// return MuonType for a particular DetId
int MuonHitHelper::toGEMType(int st, int ri) {
  if (st == 1) {
    if (ri == 1)
      return GEM_ME11;
  } else if (st == 2) {
    if (ri == 1)
      return GEM_ME21;
  }
  return GEM_ALL;
}

int MuonHitHelper::toRPCType(int re, int st, int ri) {
  // endcap
  if (std::abs(re) == 1) {
    if (st == 1) {
      if (ri == 2)
        return RPC_ME12;
      if (ri == 3)
        return RPC_ME13;
    } else if (st == 2) {
      if (ri == 2)
        return RPC_ME22;
      if (ri == 3)
        return RPC_ME23;
    } else if (st == 3) {
      if (ri == 1)
        return RPC_ME31;
      if (ri == 2)
        return RPC_ME32;
      if (ri == 3)
        return RPC_ME33;
    } else if (st == 4) {
      if (ri == 1)
        return RPC_ME41;
      if (ri == 2)
        return RPC_ME42;
      if (ri == 3)
        return RPC_ME43;
    }
  }
  // Barrel
  else {
    if (ri == -2) {
      if (st == 1)
        return RPC_MB21n;
      if (st == 2)
        return RPC_MB22n;
      if (st == 3)
        return RPC_MB23n;
      if (st == 4)
        return RPC_MB24n;
    } else if (ri == -1) {
      if (st == 1)
        return RPC_MB11n;
      if (st == 2)
        return RPC_MB12n;
      if (st == 3)
        return RPC_MB13n;
      if (st == 4)
        return RPC_MB14n;
    } else if (ri == 0) {
      if (st == 1)
        return RPC_MB01;
      if (st == 2)
        return RPC_MB02;
      if (st == 3)
        return RPC_MB03;
      if (st == 4)
        return RPC_MB04;
    } else if (ri == 1) {
      if (st == 1)
        return RPC_MB11p;
      if (st == 2)
        return RPC_MB12p;
      if (st == 3)
        return RPC_MB13p;
      if (st == 4)
        return RPC_MB14p;
    } else if (ri == 2) {
      if (st == 1)
        return RPC_MB21p;
      if (st == 2)
        return RPC_MB22p;
      if (st == 3)
        return RPC_MB23p;
      if (st == 4)
        return RPC_MB24p;
    }
  }
  return RPC_ALL;
}

int MuonHitHelper::toDTType(int wh, int st) {
  if (wh == -2) {
    if (st == 1)
      return DT_MB21n;
    if (st == 2)
      return DT_MB22n;
    if (st == 3)
      return DT_MB23n;
    if (st == 4)
      return DT_MB24n;
  }
  if (wh == -1) {
    if (st == 1)
      return DT_MB11n;
    if (st == 2)
      return DT_MB12n;
    if (st == 3)
      return DT_MB13n;
    if (st == 4)
      return DT_MB14n;
  }
  if (wh == 0) {
    if (st == 1)
      return DT_MB01;
    if (st == 2)
      return DT_MB02;
    if (st == 3)
      return DT_MB03;
    if (st == 4)
      return DT_MB04;
  }
  if (wh == 1) {
    if (st == 1)
      return DT_MB11p;
    if (st == 2)
      return DT_MB12p;
    if (st == 3)
      return DT_MB13p;
    if (st == 4)
      return DT_MB14p;
  }
  if (wh == 2) {
    if (st == 1)
      return DT_MB21p;
    if (st == 2)
      return DT_MB22p;
    if (st == 3)
      return DT_MB23p;
    if (st == 4)
      return DT_MB24p;
  }
  return DT_ALL;
}

int MuonHitHelper::toCSCType(int st, int ri) {
  if (st == 1) {
    if (ri == 0)
      return CSC_ME11;
    if (ri == 1)
      return CSC_ME1b;
    if (ri == 2)
      return CSC_ME12;
    if (ri == 3)
      return CSC_ME13;
    if (ri == 4)
      return CSC_ME1a;
  } else if (st == 2) {
    if (ri == 1)
      return CSC_ME21;
    if (ri == 2)
      return CSC_ME22;
  } else if (st == 3) {
    if (ri == 1)
      return CSC_ME31;
    if (ri == 2)
      return CSC_ME32;
  } else if (st == 4) {
    if (ri == 1)
      return CSC_ME41;
    if (ri == 2)
      return CSC_ME42;
  }
  return CSC_ALL;
}
