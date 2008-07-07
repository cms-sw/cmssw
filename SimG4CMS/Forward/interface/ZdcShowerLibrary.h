#ifndef SimG4CMS_ZdcShowerLibrary_h
#define SimG4CMS_ZdcShowerLibrary_h 1
///////////////////////////////////////////////////////////////////////////////
// File: ZdcShowerLibrary.h
// Description: Gets information from a shower library
// E. Garcia June 2008
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/ForwardGeometry/src/ZdcHardcodeGeometryData.h"

#include "G4ParticleTable.hh"
#include "G4ThreeVector.hh"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
 
//ROOT
#include "TFile.h"
#include "TH2I.h"
#include "TRandom.h"

#include <string>
#include <memory>


class G4Step;
class DDCompactView;    
class ZdcShowerLibrary {

public:
  
  //Constructor and Destructor
  ZdcShowerLibrary(std::string & name, const DDCompactView & cpv, edm::ParameterSet const & p);
  ~ZdcShowerLibrary();

 public:

  TH1I* binInfo;
  TH1I* maxBitsInfo;  
  TH1I* lutPartIDLut; 
  TH2I* lutMatrixEAverage;
  TH2I* lutMatrixESigma;
  TH2I* lutMatrixEDist;
  

  TRandom* randomGen;

  struct Hit {
    Hit() {}
    G4ThreeVector             entryLocal;
    G4ThreeVector             position;
    int                       depth;
    double                    time;
    int                       detID;
    double                    DeHad;
    double                    DeEM;
  };


  void                        initRun(G4ParticleTable * theParticleTable);
  std::vector<Hit>            getHits(G4Step * aStep, bool & ok);
  int                         getEnergyFromLibrary(G4ThreeVector posHit, G4ThreeVector momDir, double energy,
						   int parCode,HcalZDCDetId::Section section, bool side, int channel);
  unsigned long               encode1(int iphi, int itheta, int ix, int iy, int iz);
  unsigned long               encode2(int ien,  int isec,  int isid, int icha, int iparID);  
  int                         encodeParID(int parID);
  void              decode1(const unsigned long & lutidx, int& iphi,int& itheta, int& ix, int& iy, int& iz); 
  void              decode2(const unsigned long & lutidx, int& ien, int& isec, int& isid, int& icha, int& iparID);
  int               photonFluctuation(double eav, double esig,double edis);

protected:

private:

  TFile *                    zdc;
  int                        ienergyBin,ithetaBin,iphiBin, isideBin, isectionBin, ichannelBin,ixBin ,iyBin, izBin, iPIDBin;
  int                        maxBitsEnergy, maxBitsTheta, maxBitsPhi, maxBitsSide, maxBitsSection, maxBitsChannel, maxBitsX, maxBitsY, maxBitsZ, maxBitsPID;
  bool                       verbose;
  int                        emPDG, epPDG, gammaPDG;
  int                        pi0PDG, etaPDG, nuePDG, numuPDG, nutauPDG;
  int                        anuePDG, anumuPDG, anutauPDG, geantinoPDG;
  unsigned long              iLutIndex1;
  unsigned long              iLutIndex2;
};
#endif
