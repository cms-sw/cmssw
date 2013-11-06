#ifndef SimMuL1_MuNtupleClasses_h
#define SimMuL1_MuNtupleClasses_h

/**\file MuNtupleClasses

Various data classes to use for analysis ntuples

*/

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "TTree.h"

class CSCGeometry;
class GEMGeometry;
class RPCGeometry;
class DTGeometry;


// ================================================================================================
// CSC classes

struct MyCSCDetId
{
  void init(CSCDetId &id);
  void book(TTree* t)
  {
    t->Branch("id", &e,"e/S:s:r:c:l:t");
  }
  Short_t e, s, r, c, l;
  Short_t t; // type 1-10: ME1/a,1/b,1/2,1/3,2/1...4/2
};

struct MyCSCSimHit
{
  void init(PSimHit &sh, const CSCGeometry* csc_g, const ParticleDataTable * pdt);
  void book(TTree* t)
  {
    t->Branch("sh", &x,"x/F:y:z:r:eta:phi:gx:gy:gz:e:p:m:t:trid/I:pdg:w:s");
  }
  float eKin() {return sqrt(p*p + m*m) - m;}
  bool operator < (const MyCSCSimHit &rhs) const;
  Float_t x, y, z;    // local
  Float_t r, eta, phi, gx, gy, gz; // global
  Float_t e;          // energy deposit
  Float_t p;          // particle momentum
  Float_t m;          // particle mass
  Float_t t;          // TOF
  Int_t trid;         // trackId
  Int_t pdg;          // PDG
  Int_t w, s;         // WG & Strip
};

struct MyCSCCluster
{
  void init(std::vector<MyCSCSimHit> &shits);
  void book(TTree* t)
  {
    t->Branch("cl", &nh,"nh/I:r/F:eta:phi:gx:gy:gz:e:p:m:mint:maxt:meant:sigmat:mintrid/I:maxtrid:pdg/I:minw:maxw:mins:maxs");
  }
  float eKin() {return sqrt(p*p + m*m) - m;}
  std::vector<MyCSCSimHit> hits;
  Int_t nh;             // # of hits
  Float_t r, eta, phi, gx, gy, gz; // globals fot 1st hit
  Float_t e;            // total energy deposit
  Float_t p, m;         // particle mass and initial momentum
  Float_t mint, maxt;   // min/max TOF
  Float_t meant, sigmat;// mean&stdev of TOF
  Int_t mintrid, maxtrid;// trackId
  Int_t pdg;            // PDG
  Int_t minw, maxw, mins, maxs; // min/max WG & Strip
};

struct MyCSCLayer
{
  void init(int l, std::vector<MyCSCCluster> &sclusters);
  void book(TTree* t)
  {
    t->Branch("la", &nh,"nh/I:nclu:mint/F:maxt:mintrid/I:maxtrid:minw:maxw:mins:maxs");
  }
  std::vector<MyCSCCluster> clusters;
  Int_t ln;             // layer #, not stored
  Int_t nh;             // # of hits
  Int_t nclu;           // # of clusters
  Float_t mint, maxt;   // min/max TOF
  Int_t mintrid, maxtrid;// trackId
  Int_t minw, maxw, mins, maxs; // min/max WG & Strip
};

struct MyCSCChamber
{
  void init(std::vector<MyCSCLayer> &slayers);
  void book(TTree* t)
  {
    t->Branch("ch", &nh,"nh/I:nclu:nl:l1:ln:mint/F:maxt:minw/I:maxw:mins:maxs");
  }
  Int_t nh;             // # of hits
  Int_t nclu;           // # of clusters
  Int_t nl, l1, ln;     // # of layers, 1st and last layer #
  Float_t mint, maxt;   // min/max TOF
  Int_t minw, maxw, mins, maxs; // min/max WG & Strip
};

struct MyCSCEvent
{
  void init(std::vector<MyCSCChamber> &schambers);
  void book(TTree* t)
  {
    t->Branch("ev", &nh,"nh/I:nclu:nch:nch2:nch3:nch4:nch5:nch6");
  }
  Int_t nh;        // #hits
  Int_t nclu;      // #clusters
  Int_t nch;       // #chambers w/ hits
  Int_t nch2, nch3, nch4, nch5, nch6; // #chambers w/ at least 2,3... hits
};


// ================================================================================================
// GEM classes

struct MyGEMDetId
{
  void init(GEMDetId &id);
  void book(TTree* t)
  {
    t->Branch("id", &reg,"reg/S:ring:st:layer:ch:part:t");
  }
  Short_t reg, ring, st, layer, ch, part;
  Short_t t; // type 1: GE1/1
};

struct MyGEMSimHit
{
  void init(PSimHit &sh, const GEMGeometry* gem_g, const ParticleDataTable * pdt);
  void book(TTree* t)
  {
    t->Branch("sh", &x,"x/F:y:z:r:eta:phi:gx:gy:gz:e:p:m:t:trid/I:pdg:s");
  }
  float eKin() {return sqrt(p*p + m*m) - m;}
  bool operator < (const MyGEMSimHit &rhs) const;
  Float_t x, y, z;    // local
  Float_t r, eta, phi, gx, gy, gz; // global
  Float_t e;          // energy deposit
  Float_t p;          // particle momentum
  Float_t m;          // particle mass
  Float_t t;          // TOF
  Int_t trid;         // trackId
  Int_t pdg;          // PDG
  Int_t s;            // Strip
};

struct MyGEMCluster
{
  void init(std::vector<MyGEMSimHit> &shits);
  void book(TTree* t)
  {
    t->Branch("cl", &nh,"nh/I:r/F:eta:phi:gx:gy:gz:e:p:m:mint:maxt:meant:sigmat:mintrid/I:maxtrid:pdg/I:mins:maxs");
  }
  float eKin() {return sqrt(p*p + m*m) - m;}
  std::vector<MyGEMSimHit> hits;
  Int_t nh;             // # of hits
  Float_t r, eta, phi, gx, gy, gz; // globals fot 1st hit
  Float_t e;            // total energy deposit
  Float_t p, m;         // 1st particle mass and initial momentum
  Float_t mint, maxt;   // min/max TOF
  Float_t meant, sigmat;// mean&stdev of TOF
  Int_t mintrid, maxtrid;// trackId
  Int_t pdg;            // PDG
  Int_t mins, maxs; // min/max strip
};

struct MyGEMPart
{
  void init(int r, int l, std::vector<MyGEMCluster> &sclusters);
  void book(TTree* t)
  {
    t->Branch("part", &nh,"nh/I:nclu:mint/F:maxt:mintrid/I:maxtrid:mins:maxs");
  }
  std::vector<MyGEMCluster> clusters;
  Int_t pn;             // partition #, not stored
  Int_t ln;             // layer #, not stored
  Int_t nh;             // # of hits
  Int_t nclu;           // # of clusters
  Float_t mint, maxt;   // min/max TOF
  Int_t mintrid, maxtrid;// trackId
  Int_t mins, maxs;      // min/max strip
};

struct MyGEMChamber
{
  void init(std::vector<MyGEMPart> &sparts);
  void book(TTree* t)
  {
    t->Branch("ch", &nh,"nh/I:nclu:np:nl:mint/F:maxt");
  }
  Int_t nh;             // # of hits
  Int_t nclu;           // # of clusters
  Int_t np;             // # of partitions
  Int_t nl;             // # of layers
  Float_t mint, maxt;   // min/max TOF
};

struct MyGEMEvent
{
  void init(std::vector<MyGEMChamber> &schambers);
  void book(TTree* t)
  {
    t->Branch("ev", &nh,"nh/I:nclu:np:nch");
  }
  Int_t nh;        // #hits
  Int_t nclu;      // #clusters
  Int_t np;        // #partitions
  Int_t nch;       // #chambers w/ hits
  //Short_t nch2, nch3, nch4, nch5, nch6; // #chambers w/ at least 2,3... hits
};


// ================================================================================================
// PRC classes

struct MyRPCDetId
{
  void init(RPCDetId &id);
  void book(TTree* t)
  {
    t->Branch("id", &reg,"reg/S:ring:st:sec:layer:subsec:roll:t");
  }
  Short_t reg, ring, st, sec, layer, subsec, roll;
  Short_t t; // type 1-8: RE1/2,1/3,2/2,2/3,3/2,3/3,4/2,4/3
};

struct MyRPCSimHit
{
  void init(PSimHit &sh, const RPCGeometry* rpc_g, const ParticleDataTable * pdt);
  void book(TTree* t)
  {
    t->Branch("sh", &x,"x/F:y:z:r:eta:phi:gx:gy:gz:e:p:m:t:trid/I:pdg:s");
  }
  float eKin() {return sqrt(p*p + m*m) - m;}
  bool operator < (const MyRPCSimHit &rhs) const;
  Float_t x, y, z;    // local
  Float_t r, eta, phi, gx, gy, gz; // global
  Float_t e;          // energy deposit
  Float_t p;          // particle momentum
  Float_t m;          // particle mass
  Float_t t;          // TOF
  Int_t trid;         // trackId
  Int_t pdg;          // PDG
  Int_t s;            // Strip
};

struct MyRPCCluster
{
  void init(std::vector<MyRPCSimHit> &shits);
  void book(TTree* t)
  {
    t->Branch("cl", &nh,"nh/I:r/F:eta:phi:gx:gy:gz:e:p:m:mint:maxt:meant:sigmat:mintrid/I:maxtrid:pdg/I:mins:maxs");
  }
  float eKin() {return sqrt(p*p + m*m) - m;}
  std::vector<MyRPCSimHit> hits;
  Int_t nh;             // # of hits
  Float_t r, eta, phi, gx, gy, gz; // globals fot 1st hit
  Float_t e;            // total energy deposit
  Float_t p, m;         // 1st particle mass and initial momentum
  Float_t mint, maxt;   // min/max TOF
  Float_t meant, sigmat;// mean&stdev of TOF
  Int_t mintrid, maxtrid;// trackId
  Int_t pdg;            // PDG
  Int_t mins, maxs; // min/max strip
};

struct MyRPCRoll
{
  void init(int r, int l, std::vector<MyRPCCluster> &sclusters);
  void book(TTree* t)
  {
    t->Branch("rl", &nh,"nh/I:nclu:mint/F:maxt:mintrid/I:maxtrid:mins:maxs");
  }
  std::vector<MyRPCCluster> clusters;
  Int_t rn;             // roll #, not stored
  Int_t ln;             // layer #, not stored
  Int_t nh;             // # of hits
  Int_t nclu;           // # of clusters
  Float_t mint, maxt;   // min/max TOF
  Int_t mintrid, maxtrid;// trackId
  Int_t mins, maxs;      // min/max strip
};

struct MyRPCChamber
{
  void init(std::vector<MyRPCRoll> &srolls);
  void book(TTree* t)
  {
    t->Branch("ch", &nh,"nh/I:nclu:nr:nl:mint/F:maxt");
  }
  Int_t nh;             // # of hits
  Int_t nclu;           // # of clusters
  Int_t nr;             // # of rolls
  Int_t nl;             // # of layers
  Float_t mint, maxt;   // min/max TOF
};

struct MyRPCEvent
{
  void init(std::vector<MyRPCChamber> &schambers);
  void book(TTree* t)
  {
    t->Branch("ev", &nh,"nh/I:nclu:nr:nch");
  }
  Int_t nh;        // #hits
  Int_t nclu;      // #clusters
  Int_t nr;        // #rolls
  Int_t nch;       // #chambers w/ hits
  //Short_t nch2, nch3, nch4, nch5, nch6; // #chambers w/ at least 2,3... hits
};


// ================================================================================================
// DT classes

struct MyDTDetId
{
  void init(DTWireId &id);
  void book(TTree* t)
  {
    t->Branch("id", &st,"st/I:wh:sec:sl:l:wire:t");
  }
  Short_t st, wh, sec, sl, l, wire;
  Short_t t; //
};

struct MyDTSimHit
{
  void init(PSimHit &sh, const DTGeometry* dt_g, const ParticleDataTable * pdt);
  void book(TTree* t)
  {
    t->Branch("sh", &x,"x/F:y:z:r:eta:phi:gx:gy:gz:e:p:m:t:trid/I:pdg");
  }
  float eKin() {return sqrt(p*p + m*m) - m;}
  Float_t x, y, z;    // local
  Float_t r, eta, phi, gx, gy, gz; // global
  Float_t e;          // energy deposit
  Float_t p;          // particle momentum
  Float_t m;          // particle mass
  Float_t t;          // TOF
  Int_t trid;         // trackId
  Int_t pdg;          // PDG
  //Int_t w, s;         // WG & Strip
};


#endif
