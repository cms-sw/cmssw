#ifndef PValidationFormats_h
#define PValidationFormats_h

///////////////////////////////////////////////////////////////////////////////
// PGlobalSimHit
///////////////////////////////////////////////////////////////////////////////
#ifndef PGlobalSimHit_h
#define PGlobalSimHit_h

/** \class PGlobalSimHit
 *  
 *  DataFormat class to hold the information for the Global Hit Validation
 *
 *  $Date: 2013/04/22 22:33:07 $
 *  $Revision: 1.4 $
 *  \author M. Strang SUNY-Buffalo
 */

#include <vector>
#include <memory>

class PGlobalSimHit
{

 public:

  PGlobalSimHit(): nRawGenPart(0), nG4Vtx(0), nG4Trk(0),  
    nECalHits(0), nPreShHits(0), nHCalHits(0), nPxlFwdHits(0),
    nPxlBrlHits(0), nSiFwdHits(0), nSiBrlHits(0),
    nMuonDtHits(0), nMuonCscHits(0), nMuonRpcFwdHits(0),
    nMuonRpcBrlHits(0) {}
  virtual ~PGlobalSimHit(){}

  struct Vtx
  {
    Vtx(): x(0), y(0), z(0) {}
    float x;
    float y;
    float z;
  };

  struct Trk
  {
    Trk() : pt(0), e(0) {}
    float pt;
    float e;
  };

  struct CalHit
  {
    CalHit() : e(0), tof(0), phi(0), eta(0) {}
    float e;
    float tof;
    float phi;
    float eta;
  };

  struct FwdHit
  {
    FwdHit() : tof(0), z(0), phi(0), eta(0) {}
    float tof;
    float z;
    float phi;
    float eta;
  };

  struct BrlHit
  {
    BrlHit() : tof(0), r(0), phi(0), eta(0) {}
    float tof;
    float r;
    float phi;
    float eta;
  };

  typedef std::vector<Vtx> VtxVector;
  typedef std::vector<Trk> TrkVector;
  typedef std::vector<CalHit> CalVector;
  typedef std::vector<FwdHit> FwdVector;
  typedef std::vector<BrlHit> BrlVector;

  // put functions
  void putRawGenPart(int n);
  void putG4Vtx(const std::vector<float>& x, const std::vector<float>& y, 
		 const std::vector<float>& z);
  void putG4Trk(const std::vector<float>& pt, const std::vector<float>& e);
  void putECalHits(const std::vector<float>& e, const std::vector<float>& tof,
		    const std::vector<float>& phi, const std::vector<float>& eta);
  void putPreShHits(const std::vector<float>& e, const std::vector<float>& tof,
		     const std::vector<float>& phi, const std::vector<float>& eta);
  void putHCalHits(const std::vector<float>& e, const std::vector<float>& tof,
		    const std::vector<float>& phi, const std::vector<float>& eta);
  void putPxlFwdHits(const std::vector<float>& tof, const std::vector<float>& z,
		       const std::vector<float>& phi, const std::vector<float>& eta);
  void putPxlBrlHits(const std::vector<float>& tof, const std::vector<float>& r,
		      const std::vector<float>& phi, const std::vector<float>& eta);
  void putSiFwdHits(const std::vector<float>& tof, const std::vector<float>& z,
		      const std::vector<float>& phi, const std::vector<float>& eta);
  void putSiBrlHits(const std::vector<float>& tof, const std::vector<float>& r,
		     const std::vector<float>& phi, const std::vector<float>& eta);
  void putMuonCscHits(const std::vector<float>& tof, const std::vector<float>& z,
		       const std::vector<float>& phi, const std::vector<float>& eta);
  void putMuonDtHits(const std::vector<float>& tof, const std::vector<float>& r,
		      const std::vector<float>& phi, const std::vector<float>& eta);
  void putMuonRpcFwdHits(const std::vector<float>& tof, const std::vector<float>& z,
			   const std::vector<float>& phi, const std::vector<float>& eta);
  void putMuonRpcBrlHits(const std::vector<float>& tof, const std::vector<float>& r,
			  const std::vector<float>& phi, const std::vector<float>& eta);  

  int getnRawGenPart() const {return nRawGenPart;}
  int getnG4Vtx() const {return nG4Vtx;}
  VtxVector getG4Vtx() const {return G4Vtx;}
  int getnG4Trk() const {return nG4Trk;}
  TrkVector getG4Trk() const {return G4Trk;}
  int getnECalHits() const {return nECalHits;}
  CalVector getECalHits() const {return ECalHits;}
  int getnPreShHits() const {return nPreShHits;}
  CalVector getPreShHits() const {return PreShHits;}
  int getnHCalHits() const {return nHCalHits;}
  CalVector getHCalHits() const {return HCalHits;}
  int getnPxlFwdHits() const {return nPxlFwdHits;}
  FwdVector getPxlFwdHits() const {return PxlFwdHits;}
  int getnPxlBrlHits() const {return nPxlBrlHits;}
  BrlVector getPxlBrlHits() const {return PxlBrlHits;}
  int getnSiFwdHits() const {return nSiFwdHits;}
  FwdVector getSiFwdHits() const {return SiFwdHits;}
  int getnSiBrlHits() const {return nSiBrlHits;}
  BrlVector getSiBrlHits() const {return SiBrlHits;}  
  int getnMuonDtHits() const {return nMuonDtHits;}
  BrlVector getMuonDtHits() const {return MuonDtHits;}
  int getnMuonCscHits() const {return nMuonCscHits;}
  FwdVector getMuonCscHits() const {return MuonCscHits;}
  int getnMuonRpcFwdHits() const {return nMuonRpcFwdHits;}
  FwdVector getMuonRpcFwdHits() const {return MuonRpcFwdHits;}
  int getnMuonRpcBrlHits() const {return nMuonRpcBrlHits;}
  BrlVector getMuonRpcBrlHits() const {return MuonRpcBrlHits;}  

 private:

  // G4MC info
  int nRawGenPart;
  int nG4Vtx;
  VtxVector G4Vtx; 
  int nG4Trk; 
  TrkVector G4Trk; 

  // ECal info
  int nECalHits;
  CalVector ECalHits; 
  int nPreShHits;
  CalVector PreShHits; 

  // HCal info
  int nHCalHits;
  CalVector HCalHits;

  // Tracker info
  int nPxlFwdHits;
  FwdVector PxlFwdHits; 
  int nPxlBrlHits;
  BrlVector PxlBrlHits;
  int nSiFwdHits;
  FwdVector SiFwdHits; 
  int nSiBrlHits;
  BrlVector SiBrlHits;  

  // Muon info
  int nMuonDtHits;
  BrlVector MuonDtHits;
  int nMuonCscHits;
  FwdVector MuonCscHits;
  int nMuonRpcFwdHits;
  FwdVector MuonRpcFwdHits;  
  int nMuonRpcBrlHits;
  BrlVector MuonRpcBrlHits;

}; // end class declaration

#endif // endif PGlobalHit_h

///////////////////////////////////////////////////////////////////////////////
// PGlobalDigi
///////////////////////////////////////////////////////////////////////////////

#ifndef PGlobalDigi_h
#define PGlobalDigi_h

class PGlobalDigi
{
 public:

  PGlobalDigi(): nEBCalDigis(0), nEECalDigis(0), nESCalDigis(0),
    nHBCalDigis(0), nHECalDigis(0), nHOCalDigis(0), nHFCalDigis(0),
    nTIBL1Digis(0), nTIBL2Digis(0), nTIBL3Digis(0), nTIBL4Digis(0),
    nTOBL1Digis(0), nTOBL2Digis(0), nTOBL3Digis(0), nTOBL4Digis(0),
    nTIDW1Digis(0), nTIDW2Digis(0), nTIDW3Digis(0),
    nTECW1Digis(0), nTECW2Digis(0), nTECW3Digis(0), nTECW4Digis(0), 
    nTECW5Digis(0), nTECW6Digis(0), nTECW7Digis(0), nTECW8Digis(0),
    nBRL1Digis(0), nBRL2Digis(0), nBRL3Digis(0), 
    nFWD1pDigis(0), nFWD1nDigis(0), nFWD2pDigis(0), nFWD2nDigis(0),
    nMB1Digis(0), nMB2Digis(0), nMB3Digis(0), nMB4Digis(0),
    nCSCstripDigis(0), nCSCwireDigis(0) {}
  virtual ~PGlobalDigi(){}

  ////////////
  // ECal Info
  ////////////
  struct ECalDigi
  {
    ECalDigi(): maxPos(0), AEE(0), SHE(0) {}
    int maxPos;
    double AEE; //maximum analog equivalent energy
    float SHE; //simhit energy sum
  };
  typedef std::vector<ECalDigi> ECalDigiVector;
  struct ESCalDigi
  {
    ESCalDigi(): ADC0(0), ADC1(0), ADC2(0), SHE(0) {}
    float ADC0, ADC1, ADC2; //ADC counts
    float SHE; //sum simhit energy    
  };
  typedef std::vector<ESCalDigi> ESCalDigiVector;
  //put functions
  void putEBCalDigis(const std::vector<int>& maxpos,
		     const std::vector<double>& aee, const std::vector<float>& she);
  void putEECalDigis(const std::vector<int>& maxpos,
		     const std::vector<double>& aee, const std::vector<float>& she);
  void putESCalDigis(const std::vector<float>& adc0, const std::vector<float>& adc1,
		     const std::vector<float>& adc2, const std::vector<float>& she);
  //get functions
  int getnEBCalDigis() const {return nEBCalDigis;}  
  int getnEECalDigis() const {return nEECalDigis;}
  int getnESCalDigis() const {return nESCalDigis;}  
  ECalDigiVector getEBCalDigis() const {return EBCalDigis;}  
  ECalDigiVector getEECalDigis() const {return EECalDigis;}
  ESCalDigiVector getESCalDigis() const {return ESCalDigis;}  

  ////////////
  // HCal Info
  ////////////
  struct HCalDigi
  {
    HCalDigi(): AEE(0), SHE(0) {}
    float AEE; //sum analog equivalent energy in fC
    float SHE; //simhit energy sum
  };
  typedef std::vector<HCalDigi> HCalDigiVector;
  //put functions
  void putHBCalDigis(const std::vector<float>& aee, const std::vector<float>& she);
  void putHECalDigis(const std::vector<float>& aee, const std::vector<float>& she);
  void putHOCalDigis(const std::vector<float>& aee, const std::vector<float>& she);
  void putHFCalDigis(const std::vector<float>& aee, const std::vector<float>& she);
  //get functions
  int getnHBCalDigis() const {return nHBCalDigis;}  
  int getnHECalDigis() const {return nHECalDigis;}  
  int getnHOCalDigis() const {return nHOCalDigis;}  
  int getnHFCalDigis() const {return nHFCalDigis;}  
  HCalDigiVector getHBCalDigis() const {return HBCalDigis;}  
  HCalDigiVector getHECalDigis() const {return HECalDigis;}  
  HCalDigiVector getHOCalDigis() const {return HOCalDigis;}  
  HCalDigiVector getHFCalDigis() const {return HFCalDigis;}  

  ////////////////////////
  // Silicon Tracker info
  ///////////////////////

  ///////////////
  // SiStrip info
  ///////////////
  struct SiStripDigi
  {
    SiStripDigi(): ADC(0), STRIP(0) {}
    float ADC; //adc value
    int STRIP; //strip number
  };
  typedef std::vector<SiStripDigi> SiStripDigiVector;
  //put functions
  void putTIBL1Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTIBL2Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTIBL3Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTIBL4Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTOBL1Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTOBL2Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTOBL3Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTOBL4Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTIDW1Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTIDW2Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTIDW3Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTECW1Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTECW2Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTECW3Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTECW4Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTECW5Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTECW6Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTECW7Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  void putTECW8Digis(const std::vector<float>& adc, const std::vector<int>& strip);
  //get functions
  int getnTIBL1Digis() const {return nTIBL1Digis;}  
  int getnTIBL2Digis() const {return nTIBL2Digis;}  
  int getnTIBL3Digis() const {return nTIBL3Digis;}  
  int getnTIBL4Digis() const {return nTIBL4Digis;}  
  int getnTOBL1Digis() const {return nTOBL1Digis;}  
  int getnTOBL2Digis() const {return nTOBL2Digis;}  
  int getnTOBL3Digis() const {return nTOBL3Digis;}  
  int getnTOBL4Digis() const {return nTOBL4Digis;}
  int getnTIDW1Digis() const {return nTIDW1Digis;}
  int getnTIDW2Digis() const {return nTIDW2Digis;}
  int getnTIDW3Digis() const {return nTIDW3Digis;} 
  int getnTECW1Digis() const {return nTECW1Digis;}
  int getnTECW2Digis() const {return nTECW2Digis;}
  int getnTECW3Digis() const {return nTECW3Digis;}
  int getnTECW4Digis() const {return nTECW4Digis;}
  int getnTECW5Digis() const {return nTECW5Digis;}
  int getnTECW6Digis() const {return nTECW6Digis;}
  int getnTECW7Digis() const {return nTECW7Digis;}
  int getnTECW8Digis() const {return nTECW8Digis;} 
  SiStripDigiVector getTIBL1Digis() const {return TIBL1Digis;}  
  SiStripDigiVector getTIBL2Digis() const {return TIBL2Digis;}  
  SiStripDigiVector getTIBL3Digis() const {return TIBL3Digis;}  
  SiStripDigiVector getTIBL4Digis() const {return TIBL4Digis;}
  SiStripDigiVector getTOBL1Digis() const {return TOBL1Digis;}  
  SiStripDigiVector getTOBL2Digis() const {return TOBL2Digis;}  
  SiStripDigiVector getTOBL3Digis() const {return TOBL3Digis;}  
  SiStripDigiVector getTOBL4Digis() const {return TOBL4Digis;}   
  SiStripDigiVector getTIDW1Digis() const {return TIDW1Digis;}
  SiStripDigiVector getTIDW2Digis() const {return TIDW2Digis;}
  SiStripDigiVector getTIDW3Digis() const {return TIDW3Digis;} 
  SiStripDigiVector getTECW1Digis() const {return TECW1Digis;}
  SiStripDigiVector getTECW2Digis() const {return TECW2Digis;}
  SiStripDigiVector getTECW3Digis() const {return TECW3Digis;}
  SiStripDigiVector getTECW4Digis() const {return TECW4Digis;}
  SiStripDigiVector getTECW5Digis() const {return TECW5Digis;}
  SiStripDigiVector getTECW6Digis() const {return TECW6Digis;}
  SiStripDigiVector getTECW7Digis() const {return TECW7Digis;}
  SiStripDigiVector getTECW8Digis() const {return TECW8Digis;}

  ///////////////
  // SiPixel info
  ///////////////
  struct SiPixelDigi
  {
    SiPixelDigi(): ADC(0), ROW(0), COLUMN(0) {}
    float ADC; //adc value
    int ROW; //row number
    int COLUMN; //column number
  };
  typedef std::vector<SiPixelDigi> SiPixelDigiVector;
  //put functions
  void putBRL1Digis(const std::vector<float>& adc, const std::vector<int>& row,
		    const std::vector<int>& column);
  void putBRL2Digis(const std::vector<float>& adc, const std::vector<int>& row,
		    const std::vector<int>& column);
  void putBRL3Digis(const std::vector<float>& adc, const std::vector<int>& row,
		    const std::vector<int>& column);
  void putFWD1pDigis(const std::vector<float>& adc, const std::vector<int>& row,
		    const std::vector<int>& column);
  void putFWD1nDigis(const std::vector<float>& adc, const std::vector<int>& row,
		    const std::vector<int>& column);
  void putFWD2pDigis(const std::vector<float>& adc, const std::vector<int>& row,
		    const std::vector<int>& column);
  void putFWD2nDigis(const std::vector<float>& adc, const std::vector<int>& row,
		    const std::vector<int>& column);
  //get functions
  int getnBRL1Digis() const {return nBRL1Digis;}  
  int getnBRL2Digis() const {return nBRL2Digis;}  
  int getnBRL3Digis() const {return nBRL3Digis;}
  int getnFWD1pDigis() const {return nFWD1pDigis;}  
  int getnFWD1nDigis() const {return nFWD1nDigis;}    
  int getnFWD2pDigis() const {return nFWD2pDigis;}  
  int getnFWD2nDigis() const {return nFWD2nDigis;}  
  SiPixelDigiVector getBRL1Digis() const {return BRL1Digis;}  
  SiPixelDigiVector getBRL2Digis() const {return BRL2Digis;}  
  SiPixelDigiVector getBRL3Digis() const {return BRL3Digis;}  
  SiPixelDigiVector getFWD1pDigis() const {return FWD1pDigis;}
  SiPixelDigiVector getFWD1nDigis() const {return FWD1nDigis;} 
  SiPixelDigiVector getFWD2pDigis() const {return FWD2pDigis;}
  SiPixelDigiVector getFWD2nDigis() const {return FWD2nDigis;} 

  ////////////
  // Muon info
  ////////////

  //////////
  // DT Info
  ////////// 
  struct DTDigi
  {
    DTDigi(): SLAYER(0), TIME(0), LAYER(0) {}
    int SLAYER; //superlayer number
    float TIME; //time of hit
    int LAYER; //layer number
  };
  typedef std::vector<DTDigi> DTDigiVector;
  //put functions
  void putMB1Digis(const std::vector<int>& slayer, const std::vector<float>& time, 
		   const std::vector<int>& layer);
  void putMB2Digis(const std::vector<int>& slayer, const std::vector<float>& time, 
		   const std::vector<int>& layer);
  void putMB3Digis(const std::vector<int>& slayer, const std::vector<float>& time, 
		   const std::vector<int>& layer);
  void putMB4Digis(const std::vector<int>& slayer, const std::vector<float>& time, 
		   const std::vector<int>& layer);
  //get functions
  int getnMB1Digis() const {return nMB1Digis;}  
  int getnMB2Digis() const {return nMB2Digis;}  
  int getnMB3Digis() const {return nMB3Digis;}  
  int getnMB4Digis() const {return nMB4Digis;}  
  DTDigiVector getMB1Digis() const {return MB1Digis;}  
  DTDigiVector getMB2Digis() const {return MB2Digis;}  
  DTDigiVector getMB3Digis() const {return MB3Digis;}  
  DTDigiVector getMB4Digis() const {return MB4Digis;}  

  /////////////////
  // CSC Strip info
  /////////////////
  struct CSCstripDigi
  {
    CSCstripDigi(): ADC(0) {}
    float ADC; //ped subtracted amplitude
  };
  typedef std::vector<CSCstripDigi> CSCstripDigiVector;
  //put functions
  void putCSCstripDigis(const std::vector<float>& adc);
  //get functions
  int getnCSCstripDigis() const {return nCSCstripDigis;}  
  CSCstripDigiVector getCSCstripDigis() const {return CSCstripDigis;}  

  /////////////////
  // CSC Wire info
  /////////////////
  struct CSCwireDigi
  {
    CSCwireDigi(): TIME(0) {}
    float TIME; //time
  };
  typedef std::vector<CSCwireDigi> CSCwireDigiVector;
  //put functions
  void putCSCwireDigis(const std::vector<float>& time);
  //get functions
  int getnCSCwireDigis() const {return nCSCwireDigis;}  
  CSCwireDigiVector getCSCwireDigis() const {return CSCwireDigis;} 

 private:

  ////////////
  // ECal info
  ////////////
  int nEBCalDigis;
  ECalDigiVector EBCalDigis;
  int nEECalDigis;
  ECalDigiVector EECalDigis;
  int nESCalDigis;
  ESCalDigiVector ESCalDigis;

  ////////////
  // HCal info
  ////////////
  int nHBCalDigis;
  HCalDigiVector HBCalDigis;
  int nHECalDigis;
  HCalDigiVector HECalDigis;
  int nHOCalDigis;
  HCalDigiVector HOCalDigis;
  int nHFCalDigis;
  HCalDigiVector HFCalDigis;

  ////////////////////////
  // Silicon Tracker info
  ///////////////////////

  //////////////
  //SiStrip info
  //////////////
  int nTIBL1Digis;  
  SiStripDigiVector TIBL1Digis;
  int nTIBL2Digis;  
  SiStripDigiVector TIBL2Digis;
  int nTIBL3Digis; 
  SiStripDigiVector TIBL3Digis;
  int nTIBL4Digis;  
  SiStripDigiVector TIBL4Digis;
  int nTOBL1Digis;
  SiStripDigiVector TOBL1Digis;
  int nTOBL2Digis;  
  SiStripDigiVector TOBL2Digis;
  int nTOBL3Digis;  
  SiStripDigiVector TOBL3Digis;
  int nTOBL4Digis; 
  SiStripDigiVector TOBL4Digis;
  int nTIDW1Digis;   
  SiStripDigiVector TIDW1Digis;
  int nTIDW2Digis;
  SiStripDigiVector TIDW2Digis;
  int nTIDW3Digis;
  SiStripDigiVector TIDW3Digis; 
  int nTECW1Digis;
  SiStripDigiVector TECW1Digis;
  int nTECW2Digis;
  SiStripDigiVector TECW2Digis;
  int nTECW3Digis;
  SiStripDigiVector TECW3Digis;
  int nTECW4Digis;
  SiStripDigiVector TECW4Digis;
  int nTECW5Digis;
  SiStripDigiVector TECW5Digis;
  int nTECW6Digis;
  SiStripDigiVector TECW6Digis;
  int nTECW7Digis;
  SiStripDigiVector TECW7Digis;
  int nTECW8Digis;
  SiStripDigiVector TECW8Digis;

  //////////////
  //SiPixel info
  //////////////
  int nBRL1Digis;
  SiPixelDigiVector BRL1Digis;
  int nBRL2Digis;  
  SiPixelDigiVector BRL2Digis; 
  int nBRL3Digis; 
  SiPixelDigiVector BRL3Digis; 
  int nFWD1pDigis; 
  SiPixelDigiVector FWD1pDigis;
  int nFWD1nDigis;
  SiPixelDigiVector FWD1nDigis; 
  int nFWD2pDigis;
  SiPixelDigiVector FWD2pDigis;
  int nFWD2nDigis;
  SiPixelDigiVector FWD2nDigis; 

  ////////////
  // Muon info
  ////////////

  //////////
  // DT Info
  ////////// 
  int nMB1Digis;
  DTDigiVector MB1Digis; 
  int nMB2Digis;
  DTDigiVector MB2Digis; 
  int nMB3Digis;
  DTDigiVector MB3Digis; 
  int nMB4Digis; 
  DTDigiVector MB4Digis; 

  /////////////////
  // CSC Strip info
  ////////////////
  int nCSCstripDigis;
  CSCstripDigiVector CSCstripDigis;

  /////////////////
  // CSC Wire info
  ////////////////
  int nCSCwireDigis;
  CSCwireDigiVector CSCwireDigis;
 
}; // end class declaration

#endif //PGlobalDigiHit_h

///////////////////////////////////////////////////////////////////////////////
// PGlobalRecHit
///////////////////////////////////////////////////////////////////////////////

#ifndef PGlobalRecHit_h
#define PGlobalRecHit_h

class PGlobalRecHit
{
 public:

  PGlobalRecHit(): nEBCalRecHits(0), nEECalRecHits(0), nESCalRecHits(0),
    nHBCalRecHits(0), nHECalRecHits(0), nHOCalRecHits(0), nHFCalRecHits(0),
    nTIBL1RecHits(0), nTIBL2RecHits(0), nTIBL3RecHits(0), nTIBL4RecHits(0),
    nTOBL1RecHits(0), nTOBL2RecHits(0), nTOBL3RecHits(0), nTOBL4RecHits(0),
    nTIDW1RecHits(0), nTIDW2RecHits(0), nTIDW3RecHits(0),
    nTECW1RecHits(0), nTECW2RecHits(0), nTECW3RecHits(0), nTECW4RecHits(0), 
    nTECW5RecHits(0), nTECW6RecHits(0), nTECW7RecHits(0), nTECW8RecHits(0),
    nBRL1RecHits(0), nBRL2RecHits(0), nBRL3RecHits(0), 
    nFWD1pRecHits(0), nFWD1nRecHits(0), nFWD2pRecHits(0), nFWD2nRecHits(0),
    nDTRecHits(0), nCSCRecHits(0), nRPCRecHits(0) {}
  virtual ~PGlobalRecHit(){}

  ////////////
  // ECal Info
  ////////////
  struct ECalRecHit
  {
    ECalRecHit(): RE(0), SHE(0) {}
    float RE; //reconstructed energy
    float SHE; //simhit energy
  };
  typedef std::vector<ECalRecHit> ECalRecHitVector;
  //put functions
  void putEBCalRecHits(const std::vector<float>& re, const std::vector<float>& she);
  void putEECalRecHits(const std::vector<float>& re, const std::vector<float>& she);
  void putESCalRecHits(const std::vector<float>& re, const std::vector<float>& she);
  //get functions
  int getnEBCalRecHits() const {return nEBCalRecHits;}  
  int getnEECalRecHits() const {return nEECalRecHits;}
  int getnESCalRecHits() const {return nESCalRecHits;}  
  ECalRecHitVector getEBCalRecHits() const {return EBCalRecHits;}  
  ECalRecHitVector getEECalRecHits() const {return EECalRecHits;}
  ECalRecHitVector getESCalRecHits() const {return ESCalRecHits;}  

  ////////////
  // HCal Info
  ////////////
  struct HCalRecHit
  {
    HCalRecHit(): REC(0), R(0), SHE(0) {}
    float REC; // reconstructed energy
    float R;   // distance in cone 
    float SHE; // simhit energy
  };
  typedef std::vector<HCalRecHit> HCalRecHitVector;
  //put functions
  void putHBCalRecHits(const std::vector<float>& rec, const std::vector<float>& r, 
		       const std::vector<float>& she);
  void putHECalRecHits(const std::vector<float>& rec, const std::vector<float>& r, 
		       const std::vector<float>& she);
  void putHOCalRecHits(const std::vector<float>& rec, const std::vector<float>& r, 
		       const std::vector<float>& she);
  void putHFCalRecHits(const std::vector<float>& rec, const std::vector<float>& r, 
		       const std::vector<float>& she);
  //get functions
  int getnHBCalRecHits() const {return nHBCalRecHits;}  
  int getnHECalRecHits() const {return nHECalRecHits;}  
  int getnHOCalRecHits() const {return nHOCalRecHits;}  
  int getnHFCalRecHits() const {return nHFCalRecHits;}  
  HCalRecHitVector getHBCalRecHits() const {return HBCalRecHits;}  
  HCalRecHitVector getHECalRecHits() const {return HECalRecHits;}  
  HCalRecHitVector getHOCalRecHits() const {return HOCalRecHits;}  
  HCalRecHitVector getHFCalRecHits() const {return HFCalRecHits;}  

  ////////////////////////
  // Silicon Tracker info
  ///////////////////////

  ///////////////
  // SiStrip info
  ///////////////
  struct SiStripRecHit
  {
    SiStripRecHit(): RX(0), RY(0), SX(0), SY(0) {}
    float RX; //reconstructed x
    float RY; //reconstructed y
    float SX; //simulated x
    float SY; //simulated y
  };
  typedef std::vector<SiStripRecHit> SiStripRecHitVector;
  //put functions
  void putTIBL1RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTIBL2RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTIBL3RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTIBL4RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTOBL1RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTOBL2RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTOBL3RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTOBL4RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTIDW1RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTIDW2RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTIDW3RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTECW1RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTECW2RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTECW3RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTECW4RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTECW5RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTECW6RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTECW7RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putTECW8RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  //get functions
  int getnTIBL1RecHits() const {return nTIBL1RecHits;}  
  int getnTIBL2RecHits() const {return nTIBL2RecHits;}  
  int getnTIBL3RecHits() const {return nTIBL3RecHits;}  
  int getnTIBL4RecHits() const {return nTIBL4RecHits;}  
  int getnTOBL1RecHits() const {return nTOBL1RecHits;}  
  int getnTOBL2RecHits() const {return nTOBL2RecHits;}  
  int getnTOBL3RecHits() const {return nTOBL3RecHits;}  
  int getnTOBL4RecHits() const {return nTOBL4RecHits;}
  int getnTIDW1RecHits() const {return nTIDW1RecHits;}
  int getnTIDW2RecHits() const {return nTIDW2RecHits;}
  int getnTIDW3RecHits() const {return nTIDW3RecHits;} 
  int getnTECW1RecHits() const {return nTECW1RecHits;}
  int getnTECW2RecHits() const {return nTECW2RecHits;}
  int getnTECW3RecHits() const {return nTECW3RecHits;}
  int getnTECW4RecHits() const {return nTECW4RecHits;}
  int getnTECW5RecHits() const {return nTECW5RecHits;}
  int getnTECW6RecHits() const {return nTECW6RecHits;}
  int getnTECW7RecHits() const {return nTECW7RecHits;}
  int getnTECW8RecHits() const {return nTECW8RecHits;} 
  SiStripRecHitVector getTIBL1RecHits() const {return TIBL1RecHits;}  
  SiStripRecHitVector getTIBL2RecHits() const {return TIBL2RecHits;}  
  SiStripRecHitVector getTIBL3RecHits() const {return TIBL3RecHits;}  
  SiStripRecHitVector getTIBL4RecHits() const {return TIBL4RecHits;}
  SiStripRecHitVector getTOBL1RecHits() const {return TOBL1RecHits;}  
  SiStripRecHitVector getTOBL2RecHits() const {return TOBL2RecHits;}  
  SiStripRecHitVector getTOBL3RecHits() const {return TOBL3RecHits;}  
  SiStripRecHitVector getTOBL4RecHits() const {return TOBL4RecHits;}   
  SiStripRecHitVector getTIDW1RecHits() const {return TIDW1RecHits;}
  SiStripRecHitVector getTIDW2RecHits() const {return TIDW2RecHits;}
  SiStripRecHitVector getTIDW3RecHits() const {return TIDW3RecHits;} 
  SiStripRecHitVector getTECW1RecHits() const {return TECW1RecHits;}
  SiStripRecHitVector getTECW2RecHits() const {return TECW2RecHits;}
  SiStripRecHitVector getTECW3RecHits() const {return TECW3RecHits;}
  SiStripRecHitVector getTECW4RecHits() const {return TECW4RecHits;}
  SiStripRecHitVector getTECW5RecHits() const {return TECW5RecHits;}
  SiStripRecHitVector getTECW6RecHits() const {return TECW6RecHits;}
  SiStripRecHitVector getTECW7RecHits() const {return TECW7RecHits;}
  SiStripRecHitVector getTECW8RecHits() const {return TECW8RecHits;}

  ///////////////
  // SiPixel info
  ///////////////
  struct SiPixelRecHit
  {
    SiPixelRecHit(): RX(0), RY(0), SX(0), SY(0) {}
    float RX; //reconstructed x
    float RY; //reconstructed y
    float SX; //simulated x
    float SY; //simulated y
  };
  typedef std::vector<SiPixelRecHit> SiPixelRecHitVector;
  //put functions
  void putBRL1RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putBRL2RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putBRL3RecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putFWD1pRecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putFWD1nRecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putFWD2pRecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  void putFWD2nRecHits(const std::vector<float>& rx, const std::vector<float>& ry,
		       const std::vector<float>& sx, const std::vector<float>& sy);
  //get functions
  int getnBRL1RecHits() const {return nBRL1RecHits;}  
  int getnBRL2RecHits() const {return nBRL2RecHits;}  
  int getnBRL3RecHits() const {return nBRL3RecHits;}
  int getnFWD1pRecHits() const {return nFWD1pRecHits;}  
  int getnFWD1nRecHits() const {return nFWD1nRecHits;}    
  int getnFWD2pRecHits() const {return nFWD2pRecHits;}  
  int getnFWD2nRecHits() const {return nFWD2nRecHits;}  
  SiPixelRecHitVector getBRL1RecHits() const {return BRL1RecHits;}  
  SiPixelRecHitVector getBRL2RecHits() const {return BRL2RecHits;}  
  SiPixelRecHitVector getBRL3RecHits() const {return BRL3RecHits;}  
  SiPixelRecHitVector getFWD1pRecHits() const {return FWD1pRecHits;}
  SiPixelRecHitVector getFWD1nRecHits() const {return FWD1nRecHits;} 
  SiPixelRecHitVector getFWD2pRecHits() const {return FWD2pRecHits;}
  SiPixelRecHitVector getFWD2nRecHits() const {return FWD2nRecHits;} 

  ////////////
  // Muon info
  ////////////

  //////////
  // DT Info
  ////////// 
  struct DTRecHit
  {
    DTRecHit(): RHD(0), SHD(0) {}
    float RHD; //distance of rechit from wire
    float SHD; //distance of simhit from wire
  };
  typedef std::vector<DTRecHit> DTRecHitVector;
  //put functions
  void putDTRecHits(const std::vector<float>& rhd, const std::vector<float>& shd);
  //get functions
  int getnDTRecHits() const {return nDTRecHits;}  
  DTRecHitVector getDTRecHits() const {return DTRecHits;}  

  /////////////////
  // CSC info
  /////////////////
  struct CSCRecHit
  {
    CSCRecHit(): RHPHI(0), RHPERP(0), SHPHI(0) {}
    float RHPHI; //reconstructed hit phi
    float RHPERP; //reconstructed hit perp
    float SHPHI; //simulated hit phi
  };
  typedef std::vector<CSCRecHit> CSCRecHitVector;
  //put functions
  void putCSCRecHits(const std::vector<float>& rhphi, const std::vector<float>& rhperp, 
		     const std::vector<float>& shphi);
  //get functions
  int getnCSCRecHits() const {return nCSCRecHits;}  
  CSCRecHitVector getCSCRecHits() const {return CSCRecHits;}  

  /////////////////
  // RPC info
  /////////////////
  struct RPCRecHit
  {
    RPCRecHit(): RHX(0), SHX(0) {}
    float RHX; //reconstructed hit x
    float SHX; //simulated hit x
  };
  typedef std::vector<RPCRecHit> RPCRecHitVector;
  //put functions
  void putRPCRecHits(const std::vector<float>& rhx, const std::vector<float>& shx);
  //get functions
  int getnRPCRecHits() const {return nRPCRecHits;}  
  RPCRecHitVector getRPCRecHits() const {return RPCRecHits;} 

 private:

  ////////////
  // ECal info
  ////////////
  int nEBCalRecHits;
  ECalRecHitVector EBCalRecHits;
  int nEECalRecHits;
  ECalRecHitVector EECalRecHits;
  int nESCalRecHits;
  ECalRecHitVector ESCalRecHits;

  ////////////
  // HCal info
  ////////////
  int nHBCalRecHits;
  HCalRecHitVector HBCalRecHits;
  int nHECalRecHits;
  HCalRecHitVector HECalRecHits;
  int nHOCalRecHits;
  HCalRecHitVector HOCalRecHits;
  int nHFCalRecHits;
  HCalRecHitVector HFCalRecHits;

  ////////////////////////
  // Silicon Tracker info
  ///////////////////////

  //////////////
  //SiStrip info
  //////////////
  int nTIBL1RecHits;  
  SiStripRecHitVector TIBL1RecHits;
  int nTIBL2RecHits;  
  SiStripRecHitVector TIBL2RecHits;
  int nTIBL3RecHits; 
  SiStripRecHitVector TIBL3RecHits;
  int nTIBL4RecHits;  
  SiStripRecHitVector TIBL4RecHits;
  int nTOBL1RecHits;
  SiStripRecHitVector TOBL1RecHits;
  int nTOBL2RecHits;  
  SiStripRecHitVector TOBL2RecHits;
  int nTOBL3RecHits;  
  SiStripRecHitVector TOBL3RecHits;
  int nTOBL4RecHits; 
  SiStripRecHitVector TOBL4RecHits;
  int nTIDW1RecHits;   
  SiStripRecHitVector TIDW1RecHits;
  int nTIDW2RecHits;
  SiStripRecHitVector TIDW2RecHits;
  int nTIDW3RecHits;
  SiStripRecHitVector TIDW3RecHits; 
  int nTECW1RecHits;
  SiStripRecHitVector TECW1RecHits;
  int nTECW2RecHits;
  SiStripRecHitVector TECW2RecHits;
  int nTECW3RecHits;
  SiStripRecHitVector TECW3RecHits;
  int nTECW4RecHits;
  SiStripRecHitVector TECW4RecHits;
  int nTECW5RecHits;
  SiStripRecHitVector TECW5RecHits;
  int nTECW6RecHits;
  SiStripRecHitVector TECW6RecHits;
  int nTECW7RecHits;
  SiStripRecHitVector TECW7RecHits;
  int nTECW8RecHits;
  SiStripRecHitVector TECW8RecHits;

  //////////////
  //SiPixel info
  //////////////
  int nBRL1RecHits;
  SiPixelRecHitVector BRL1RecHits;
  int nBRL2RecHits;  
  SiPixelRecHitVector BRL2RecHits; 
  int nBRL3RecHits; 
  SiPixelRecHitVector BRL3RecHits; 
  int nFWD1pRecHits; 
  SiPixelRecHitVector FWD1pRecHits;
  int nFWD1nRecHits;
  SiPixelRecHitVector FWD1nRecHits; 
  int nFWD2pRecHits;
  SiPixelRecHitVector FWD2pRecHits;
  int nFWD2nRecHits;
  SiPixelRecHitVector FWD2nRecHits; 

  ////////////
  // Muon info
  ////////////

  //////////
  // DT Info
  ////////// 
  int nDTRecHits;
  DTRecHitVector DTRecHits; 

  /////////////////
  // CSC info
  ////////////////
  int nCSCRecHits;
  CSCRecHitVector CSCRecHits;

  /////////////////
  // RPC info
  ////////////////
  int nRPCRecHits;
  RPCRecHitVector RPCRecHits;
 
}; // end class declaration

#endif //PGlobalRecHitHit_h

///////////////////////////////////////////////////////////////////////////////
// PEcalValidInfo
///////////////////////////////////////////////////////////////////////////////

#ifndef  PEcalValidInfo_H
#define  PEcalValidInfo_H

/*----------------------------------------------------------
Class Description:
      The Class, PEcalValidInfo, includes all the quantities 
    needed to validate for the Simulation of Eletromagnetic 
    Calorimetor. 
       The Objects of this class will be save into regular 
    Root file vis EDProducer.

Author: X.HUANG ( huangxt@fnal.gov )
Date:  Dec, 2005

---------------------------------------------------------*/

#include <string>
#include <vector>
#include "DataFormats/Math/interface/LorentzVector.h"

class EcalTestAnalysis; 

class PEcalValidInfo 
{
   friend  class   EcalTestAnalysis;
   friend  class   PreshowerTestAnalysis;
   friend  class   SimHitSingleTest;
   friend  class   EcalSimHitsValidProducer;
   typedef  std::vector<float>   FloatVector;

public:
   PEcalValidInfo()
  :ee1(0.0),ee4(0.0),ee9(0.0),ee16(0.0),ee25(0.0),
   eb1(0.0),eb4(0.0),eb9(0.0),eb16(0.0),eb25(0.0),
   totalEInEE(0.0), totalEInEB(0.0), totalEInES(0.0),
   totalHits(0), nHitsInEE(0),nHitsInEB(0),nHitsInES(0),nHitsIn1ES(0),nHitsIn2ES(0) 
{

 }


   ~PEcalValidInfo() {} 

   // Get functions.
   float  ee1x1() const { return ee1; }
   float  ee2x2() const { return ee4; }
   float  ee3x3() const { return ee9; }
   float  ee4x4() const { return ee16;}
   float  ee5x5() const { return ee25;}

   float  eb1x1() const { return eb1; }
   float  eb2x2() const { return eb4; }
   float  eb3x3() const { return eb9; }
   float  eb4x4() const { return eb16;}
   float  eb5x5() const { return eb25;}

   float  eInEE()  const { return totalEInEE; }
   float  eInEB()  const { return totalEInEB; }
   float  eInES()  const { return totalEInES; }

   float  eInEEzp()  const { return totalEInEEzp; }
   float  eInEEzm()  const { return totalEInEEzm; }

   float  eInESzp()  const { return totalEInESzp; }
   float  eInESzm()  const { return totalEInESzm; }

   int    hitsInEcal() const { return totalHits; }
   int    hitsInEE()   const { return nHitsInEE; }
   int    hitsInEB()   const { return nHitsInEB; }
   int    hitsInES()   const { return nHitsInES; }
   int    hitsIn1ES()  const { return nHitsIn1ES;}
   int    hitsIn2ES()  const { return nHitsIn2ES;}
  
   int    hitsIn1ESzp()  const { return nHitsIn1ESzp;}
   int    hitsIn1ESzm()  const { return nHitsIn1ESzm;}
   int    hitsIn2ESzp()  const { return nHitsIn2ESzp;}
   int    hitsIn2ESzm()  const { return nHitsIn2ESzm;}       

   int    crystalInEB()   const { return nCrystalInEB;}
   int    crystalInEEzp() const { return nCrystalInEEzp; }
   int    crystalInEEzm() const { return nCrystalInEEzm; }

   FloatVector  bX0() const { return eBX0; }
   FloatVector  eX0() const { return eEX0; }


   FloatVector  eIn1ES() const { return eOf1ES; }
   FloatVector  eIn2ES() const { return eOf2ES; }
   FloatVector  zOfInES()  const { return zOfES;  }

   FloatVector  eIn1ESzp() const { return eOf1ESzp; }
   FloatVector  eIn1ESzm() const { return eOf1ESzm; }
 
   FloatVector  eIn2ESzp() const { return eOf2ESzp; }
   FloatVector  eIn2ESzm() const { return eOf2ESzm; }

   FloatVector  phiOfEEHits() const { return phiOfEECaloG4Hit; }
   FloatVector  etaOfEEHits() const { return etaOfEECaloG4Hit; }
   FloatVector  tOfEEHits()   const { return tOfEECaloG4Hit;   }
   FloatVector  eOfEEHits()   const { return eOfEECaloG4Hit;   }
   FloatVector  eOfEEPlusHits()    const { return eOfEEPlusCaloG4Hit;   }
   FloatVector  eOfEEMinusHits()   const { return eOfEEMinusCaloG4Hit;   }


   FloatVector  phiOfEBHits() const { return phiOfEBCaloG4Hit; }
   FloatVector  etaOfEBHits() const { return etaOfEBCaloG4Hit; }
   FloatVector  tOfEBHits()   const { return tOfEBCaloG4Hit;   }
   FloatVector  eOfEBHits()   const { return eOfEBCaloG4Hit;   }

   FloatVector  phiOfiESHits() const { return phiOfESCaloG4Hit; }
   FloatVector  etaOfESHits() const { return etaOfESCaloG4Hit; }
   FloatVector  tOfESHits()   const { return tOfESCaloG4Hit;   }
   FloatVector  eOfESHits()   const { return eOfESCaloG4Hit;   }

   math::XYZTLorentzVector momentum() const { return theMomentum; }
   math::XYZTLorentzVector vertex() const  { return theVertex; }
   
   int pId()  const { return thePID; }   

private:
 
   float  ee1;       //Energy deposition in cluser1x1
   float  ee4;       //Energy deposition in cluser2x2
   float  ee9;       //Energy deposition in cluser3x3
   float  ee16;      //Energy deposition in cluser4x4
   float  ee25;      //Energy deposition in cluser5x5

   float  eb1;       //Energy deposition in cluser1x1
   float  eb4;       //Energy deposition in cluser2x2
   float  eb9;       //Energy deposition in cluser3x3
   float  eb16;      //Energy deposition in cluser4x4
   float  eb25;      //Energy deposition in cluser5x5


 
   float  totalEInEE;       //The Total Energy deposited in EE;
   float  totalEInEB;       //The Total Energy deposited in EB;
   float  totalEInES;       //The Total Energy deposited in ES;
 
   float  totalEInEEzp;
   float  totalEInEEzm;
   float  totalEInESzp;
   float  totalEInESzm;



   int totalHits;          //Total number of Hits.
   int nHitsInEE;          //Total number of Hits in EE.
   int nHitsInEB;          //Total number of Hits in EB.
   int nHitsInES;          //Total number of Hits in ES.
   int nHitsIn1ES;         //Total number of Hits in 1st Layer of ES;
   int nHitsIn2ES;         //Total number of Hits in 2nd Layer of ES;

   int nHitsIn1ESzp;
   int nHitsIn1ESzm;
   int nHitsIn2ESzp;
   int nHitsIn2ESzm;       

   int nCrystalInEB;
   int nCrystalInEEzp;
   int nCrystalInEEzm;


   FloatVector eBX0;       // longitudinal Energy deposition In EB.
   FloatVector eEX0;       // longitudinal Energy deposition In EE.

   FloatVector  eOf1ES;    // Energy deposition of Hits in 1st layer of ES;
   FloatVector  eOf2ES;    // Energy deposition of Hits in 2nd layer of ES;              
   FloatVector  zOfES;


   FloatVector  eOf1ESzp;
   FloatVector  eOf1ESzm;
   FloatVector  eOf2ESzp;
   FloatVector  eOf2ESzm;

   FloatVector  phiOfEECaloG4Hit;    // Phi of Hits.
   FloatVector  etaOfEECaloG4Hit;    // Eta of Hits.
   FloatVector  tOfEECaloG4Hit;      // Tof of Hits.
   FloatVector  eOfEECaloG4Hit;      // Energy depostion of Hits.
   FloatVector  eOfEEPlusCaloG4Hit;       // Energy depostion of Hits.
   FloatVector  eOfEEMinusCaloG4Hit;      // Energy depostion of Hits.

   FloatVector  phiOfESCaloG4Hit;    // Phi of Hits.
   FloatVector  etaOfESCaloG4Hit;    // Eta of Hits.
   FloatVector  tOfESCaloG4Hit;      // Tof of Hits.
   FloatVector  eOfESCaloG4Hit;      // Energy depostion of Hits.

   FloatVector  phiOfEBCaloG4Hit;    // Phi of Hits.
   FloatVector  etaOfEBCaloG4Hit;    // Eta of Hits.
   FloatVector  tOfEBCaloG4Hit;      // Tof of Hits.
   FloatVector  eOfEBCaloG4Hit;      // Energy depostion of Hits.



   int thePID;                      // add more ??
   math::XYZTLorentzVector theMomentum;  
   math::XYZTLorentzVector theVertex;
};


#endif // endif PECal

///////////////////////////////////////////////////////////////////////////////
// PHcalValidInfoJets
///////////////////////////////////////////////////////////////////////////////

#ifndef  PHcalValidInfoJets_H
#define  PHcalValidInfoJets_H

#include <string>
#include <vector>
#include <memory>

class SimG4HcalValidation;

class PHcalValidInfoJets {

  friend class SimG4HcalValidation;

public:
       
  PHcalValidInfoJets(): nJetHit(0), nJet(0), ecalJet(0.), hcalJet(0.),
			hoJet(0.), etotJet(0.), detaJet(0.), dphiJet(0.),
			drJet(0.), dijetM(0.) {}
  virtual ~PHcalValidInfoJets() {}

  // acceess

  std::vector<float> jethite() const {return jetHite;}
  std::vector<float> jethitr() const {return jetHitr;}
  std::vector<float> jethitt() const {return jetHitt;}
  int                njethit() const {return nJetHit;}

  std::vector<float> jete()    const {return jetE;}
  std::vector<float> jeteta()  const {return jetEta;}
  std::vector<float> jetphi()  const {return jetPhi;}
  int                njet()    const {return nJet;} 

  float              ecaljet() const {return ecalJet;}
  float              hcaljet() const {return hcalJet;}
  float                hojet() const {return   hoJet;}
  float              etotjet() const {return etotJet;}

  float              detajet() const {return detaJet;}
  float              dphijet() const {return dphiJet;}
  float                drjet() const {return   drJet;}
  float               dijetm() const {return  dijetM;}

  // fill
  void fillTProfileJet      (double e, double r, double t);
  void fillEcollectJet      (double ee, double he, double hoe, double etot);
  void fillEtaPhiProfileJet (double eta0, double phi0, double eta,
                             double phi, double dist);
  void fillJets             (const std::vector<double>& enj, const std::vector<double>& etaj,
			     const std::vector<double>& phij);
  void fillDiJets           (double mass);

private:

  int                 nJetHit, nJet;
  float               ecalJet, hcalJet, hoJet, etotJet;
  float               detaJet, dphiJet, drJet, dijetM;
  std::vector<float>  jetHite;
  std::vector<float>  jetHitr;
  std::vector<float>  jetHitt;
  std::vector<float>  jetE;
  std::vector<float>  jetEta;
  std::vector<float>  jetPhi;

};

#endif

///////////////////////////////////////////////////////////////////////////////
// PHcalValidInfoLayer
///////////////////////////////////////////////////////////////////////////////

#ifndef  PHcalValidInfoLayer_H
#define  PHcalValidInfoLayer_H

#include <string>
#include <vector>
#include <memory>

class SimG4HcalValidation;

class PHcalValidInfoLayer {

  friend class SimG4HcalValidation;

public:
       
  PHcalValidInfoLayer(): hitN(0), eHO(0.0),eHBHE(0.0),eEBEE(0.0),elongHF(0.0),
			 eshortHF(0.0), eEcalHF(0.0), eHcalHF(0.0) {}
  virtual ~PHcalValidInfoLayer() {}

  // access
  int                    nHit()   const {return hitN;}

  float                   eho()   const {return eHO;}    
  float                 ehbhe()   const {return eHBHE;}    
  float                 eebee()   const {return eEBEE;}    
  float               elonghf()   const {return elongHF;}    
  float              eshorthf()   const {return eshortHF;}    
  float               eecalhf()   const {return eEcalHF;}    
  float               ehcalhf()   const {return eHcalHF;}    

  std::vector<float>   elayer()   const {return eLayer;}
  std::vector<float>   edepth()   const {return eDepth;}

  std::vector<float>   etaHit()   const {return hitEta;} 
  std::vector<float>   phiHit()   const {return hitPhi;} 
  std::vector<float>     eHit()   const {return hitE;} 
  std::vector<float>     tHit()   const {return hitTime;} 
  std::vector<float> layerHit()   const {return hitLayer;} 
  std::vector<float>    idHit()   const {return hitId;} 

  // filling
  void fillLayers (double el[], double ed[], double ho, double hbhe,
		   double ebee);
  void fillHF     (double fibl, double fibs, double enec, double enhc);
  void fillHits   (int Nhits, int lay, int unitID, double eta, double phi, 
		   double ehit, double t); 
  //  void clear();


private:

  int                hitN;
  float              eHO, eHBHE, eEBEE;
  float              elongHF, eshortHF, eEcalHF, eHcalHF;
  std::vector<float> eLayer;
  std::vector<float> eDepth;
  // SimHits parameters
  std::vector<float> hitLayer; // float for int
  std::vector<float> hitId;    // float for int
  std::vector<float> hitEta;
  std::vector<float> hitPhi;
  std::vector<float> hitE;
  std::vector<float> hitTime;

};

#endif

///////////////////////////////////////////////////////////////////////////////
// PHcalValidInfoNxN
///////////////////////////////////////////////////////////////////////////////

#ifndef  PHcalValidInfoNxN_H
#define  PHcalValidInfoNxN_H

#include <string>
#include <vector>
#include <memory>

class SimG4HcalValidation;


class PHcalValidInfoNxN {

  friend class SimG4HcalValidation;

public:
       
  PHcalValidInfoNxN(): nNxN(0), ecalNxNr(0), hcalNxNr(0.), hoNxNr(0.), 
    etotNxNr(0.), ecalNxN(0.), hcalNxN(0.), hoNxN(0.), etotNxN(0.) {}
  virtual ~PHcalValidInfoNxN() {}

  // access
  std::vector<float> idnxn() const {return idNxN;}
  std::vector<float>  enxn() const {return  eNxN;}
  std::vector<float>  tnxn() const {return  tNxN;}
  int                 nnxn() const {return  nNxN;}
  
  float           ecalnxnr() const {return ecalNxNr;}
  float           hcalnxnr() const {return hcalNxNr;}
  float             honxnr() const {return   hoNxNr;}
  float           etotnxnr() const {return etotNxNr;}

  float           ecalnxn () const {return ecalNxN ;}
  float           hcalnxn () const {return hcalNxN ;}
  float             honxn () const {return   hoNxN ;}
  float           etotnxn () const {return etotNxN ;}
  

  // fill
  void fillHvsE        (double ee, double he, double hoe, double etot);
  void fillEcollectNxN (double een, double hen, double hoen, double etotn);
  void fillTProfileNxN (double e, int i, double t);

private:

  int                nNxN;
  float              ecalNxNr, hcalNxNr, hoNxNr, etotNxNr;
  float              ecalNxN,  hcalNxN,  hoNxN,  etotNxN;
  std::vector<float> idNxN; // float for int
  std::vector<float> eNxN;
  std::vector<float> tNxN;

};

#endif

///////////////////////////////////////////////////////////////////////////////
// PMuonSimHit
///////////////////////////////////////////////////////////////////////////////

#ifndef PMuonSimHit_h
#define PMuonSimHit_h

#include <vector>
#include <memory>

/// Class PMuonSimHit defines structure of simulated hits data in CSC,DT,RPC
/// for validation. It also includes vertex and track info.

class PMuonSimHit
{
 public:

  PMuonSimHit(): nRawGenPart(0), nG4Vtx(0), nG4Trk(0), 
                 nCSCHits(0), nDTHits(0), nRPCHits(0) {}
  virtual ~PMuonSimHit(){}

  struct Vtx
  {
    Vtx(): x(0), y(0), z(0) {}
    float x;
    float y;
    float z;
  };

  struct Trk
  {
    Trk() : pt(0), e(0), eta(0), phi(0) {}
    float pt;
    float e;
    float eta;
    float phi;
  };


  struct CSC
  {
    CSC() :
         _cscId(0), 
         _detUnitId(0),   _trackId(0),     _processType(0), 
         _particleType(0),_pabs(0),
         _globposz(0),    _globposphi(0),  _globposeta(0), 
	 _locposx(0),     _locposy(0),     _locposz(0), 
	 _locdirx(0),     _locdiry(0),     _locdirz(0), 
         _locdirtheta(0), _locdirphi(0),
	 _exitpointx(0),  _exitpointy(0),  _exitpointz(0),
	 _entrypointx(0), _entrypointy(0), _entrypointz(0), 
         _enloss(0),      _tof(0) {}
 
    int   _cscId;
    unsigned int _detUnitId;
    float _trackId;
    float _processType;
    float _particleType;
    float _pabs;
    float _globposz;
    float _globposphi;
    float _globposeta;
    float _locposx;
    float _locposy;
    float _locposz;
    float _locdirx;
    float _locdiry;
    float _locdirz;
    float _locdirtheta;
    float _locdirphi;
    float _exitpointx;
    float _exitpointy;
    float _exitpointz;
    float _entrypointx;
    float _entrypointy;
    float _entrypointz;
    float _enloss;
    float _tof;
  };

  struct DT
  {
    DT() : 
         _detUnitId(0),   _trackId(0),     _processType(0), 
         _particleType(0),_pabs(0), 
         _globposz(0),    _globposphi(0),  _globposeta(0),
	 _locposx(0),     _locposy(0),     _locposz(0), 
	 _locdirx(0),     _locdiry(0),     _locdirz(0), 
         _locdirtheta(0), _locdirphi(0),
	 _exitpointx(0),  _exitpointy(0),  _exitpointz(0),
	 _entrypointx(0), _entrypointy(0), _entrypointz(0), 
         _enloss(0),      _tof(0) {}

    unsigned int _detUnitId;
    float _trackId;
    float _processType;
    float _particleType;
    float _pabs;
    float _globposz;
    float _globposphi;
    float _globposeta;
    float _locposx;
    float _locposy;
    float _locposz;
    float _locdirx;
    float _locdiry;
    float _locdirz;
    float _locdirtheta;
    float _locdirphi;
    float _exitpointx;
    float _exitpointy;
    float _exitpointz;
    float _entrypointx;
    float _entrypointy;
    float _entrypointz;
    float _enloss;
    float _tof;
  };

  struct RPC
  {
    RPC() : 
         _detUnitId(0),   _trackId(0),     _processType(0), 
         _particleType(0),_pabs(0), 
         _globposz(0),    _globposphi(0),  _globposeta(0),
	 _locposx(0),     _locposy(0),     _locposz(0), 
	 _locdirx(0),     _locdiry(0),     _locdirz(0), 
         _locdirtheta(0), _locdirphi(0),
	 _exitpointx(0),  _exitpointy(0),  _exitpointz(0),
	 _entrypointx(0), _entrypointy(0), _entrypointz(0), 
         _enloss(0),      _tof(0) {}

    unsigned int _detUnitId;
    float _trackId;
    float _processType;
    float _particleType;
    float _pabs;
    float _globposz;
    float _globposphi;
    float _globposeta;
    float _locposx;
    float _locposy;
    float _locposz;
    float _locdirx;
    float _locdiry;
    float _locdirz;
    float _locdirtheta;
    float _locdirphi;
    float _exitpointx;
    float _exitpointy;
    float _exitpointz;
    float _entrypointx;
    float _entrypointy;
    float _entrypointz;
    float _enloss;
    float _tof;
  };

  typedef std::vector<Vtx> VtxVector;
  typedef std::vector<Trk> TrkVector;

  typedef std::vector<CSC> CSCVector;
  typedef std::vector<DT>  DTVector;
  typedef std::vector<RPC>  RPCVector;

  /// put functions

  void putRawGenPart(int n);

  void putG4Vtx(const std::vector<float>& x,   const std::vector<float>& y,
                const std::vector<float>& z);
  void putG4Trk(const std::vector<float>& pt,  const std::vector<float>& e,
                const std::vector<float>& eta, const std::vector<float>& phi);  

  void putCSCHits(
               const std::vector<int>&  _cscId,
               const std::vector<unsigned int>& _detUnitId,
	       const std::vector<float>& _trackId , 
               const std::vector<float>& _processType,
	       const std::vector<float>& _particleType, 
               const std::vector<float>& _pabs,
	       const std::vector<float>& _globposz, 
               const std::vector<float>& _globposphi, 
               const std::vector<float>& _globposeta,
	       const std::vector<float>& _locposx, 
               const std::vector<float>& _locposy, 
               const std::vector<float>& _locposz,
	       const std::vector<float>& _locdirx, 
               const std::vector<float>& _locdiry, 
               const std::vector<float>& _locdirz,
	       const std::vector<float>& _locdirtheta, 
               const std::vector<float>& _locdirphi, 
	       const std::vector<float>& _exitpointx, 
               const std::vector<float>& _exitpointy, 
               const std::vector<float>& _exitpointz,
	       const std::vector<float>& _entrypointx, 
               const std::vector<float>& _entrypointy, 
               const std::vector<float>& _entrypointz,
	       const std::vector<float>& _enloss, 
               const std::vector<float>& _tof);   

  void putDTHits(
               const std::vector<unsigned int>& _detUnitId,
	       const std::vector<float>& _trackId , 
               const std::vector<float>& _processType,
	       const std::vector<float>& _particleType, 
               const std::vector<float>& _pabs,
	       const std::vector<float>& _globposz, 
               const std::vector<float>& _globposphi, 
               const std::vector<float>& _globposeta,
	       const std::vector<float>& _locposx, 
               const std::vector<float>& _locposy, 
               const std::vector<float>& _locposz,
	       const std::vector<float>& _locdirx, 
               const std::vector<float>& _locdiry, 
               const std::vector<float>& _locdirz,
	       const std::vector<float>& _locdirtheta, 
               const std::vector<float>& _locdirphi, 
	       const std::vector<float>& _exitpointx, 
               const std::vector<float>& _exitpointy, 
               const std::vector<float>& _exitpointz,
	       const std::vector<float>& _entrypointx, 
               const std::vector<float>& _entrypointy, 
               const std::vector<float>& _entrypointz,
	       const std::vector<float>& _enloss, 
               const std::vector<float>& _tof); 

  void putRPCHits(
               const std::vector<unsigned int>& _detUnitId,
	       const std::vector<float>& _trackId , 
               const std::vector<float>& _processType,
	       const std::vector<float>& _particleType, 
               const std::vector<float>& _pabs,
	       const std::vector<float>& _globposz, 
               const std::vector<float>& _globposphi, 
               const std::vector<float>& _globposeta,
	       const std::vector<float>& _locposx, 
               const std::vector<float>& _locposy, 
               const std::vector<float>& _locposz,
	       const std::vector<float>& _locdirx, 
               const std::vector<float>& _locdiry, 
               const std::vector<float>& _locdirz,
	       const std::vector<float>& _locdirtheta, 
               const std::vector<float>& _locdirphi, 
	       const std::vector<float>& _exitpointx, 
               const std::vector<float>& _exitpointy, 
               const std::vector<float>& _exitpointz,
	       const std::vector<float>& _entrypointx, 
               const std::vector<float>& _entrypointy, 
               const std::vector<float>& _entrypointz,
	       const std::vector<float>& _enloss, 
               const std::vector<float>& _tof); 

  /// get functions

  int getnRawGenPart() {return nRawGenPart;}
  int getnG4Vtx() {return nG4Vtx;}
  int getnG4Trk() {return nG4Trk;}

  VtxVector getG4Vtx() {return G4Vtx;}
  TrkVector getG4Trk() {return G4Trk;}

  int getnCSCHits() {return nCSCHits;}
  CSCVector getCSCHits() {return CSCHits;}

  int getnDTHits() {return nDTHits;}
  DTVector getDTHits() {return DTHits;}

  int getnRPCHits() {return nRPCHits;}
  RPCVector getRPCHits() {return RPCHits;}

 
private:

  /// G4MC info

  int nRawGenPart;
  int nG4Vtx;
  VtxVector G4Vtx; 
  int nG4Trk; 
  TrkVector G4Trk;
 
  /// Hit info

  int nCSCHits;
  CSCVector CSCHits; 

  int nDTHits;
  DTVector DTHits; 

  int nRPCHits;
  RPCVector RPCHits; 

};

#endif

///////////////////////////////////////////////////////////////////////////////
// PTrackerSimHit
///////////////////////////////////////////////////////////////////////////////

#ifndef PTrackerSimHit_h
#define PTrackerSimHit_h

#include <vector>
#include <memory>

class PTrackerSimHit
{

 public:

  PTrackerSimHit(): nRawGenPart(0), nG4Vtx(0), nG4Trk(0), nHits(0) {}
  virtual ~PTrackerSimHit(){}

  struct Vtx
  {
    Vtx(): x(0), y(0), z(0) {}
    float x;
    float y;
    float z;
  };

  struct Trk
  {
    Trk() : pt(0), e(0), eta(0), phi(0) {}
    float pt;
    float e;
    float eta;
    float phi;
  };


  struct Hit
  {
    Hit() : _sysID(0), _detUnitId(0), _trackId(0), _processType(0), 
            _particleType(0), _pabs(0), 
	    _lpx(0), _lpy(0), _lpz(0), 
	    _ldx(0), _ldy(0), _ldz(0), _ldtheta(0), _ldphi(0),
	    _exx(0), _exy(0), _exz(0),
	    _enx(0), _eny(0), _enz(0), _eloss(0), _tof(0) {}
    int   _sysID; 
    float _detUnitId;
    float _trackId;
    float _processType;
    float _particleType;
    float _pabs;
    float _lpx;
    float _lpy;
    float _lpz;
    float _ldx;
    float _ldy;
    float _ldz;
    float _ldtheta;
    float _ldphi;
    float _exx;
    float _exy;
    float _exz;
    float _enx;
    float _eny;
    float _enz;
    float _eloss;
    float _tof;
  };


  typedef std::vector<Vtx> VtxVector;
  typedef std::vector<Trk> TrkVector;
  typedef std::vector<Hit> HitVector;

  // put functions
  void putRawGenPart(int n);
  void putG4Vtx(const std::vector<float> &x, const std::vector<float> &y, const std::vector<float> &z);
  void putG4Trk(const std::vector<float> &pt, const std::vector<float> &e, const std::vector<float> &eta, const std::vector<float> &phi);  
  void putHits(const std::vector<int> &_sysID, const std::vector<float> &_detUnitId,
	       const std::vector<float>&_trackId , const std::vector<float>&_processType,
	       const std::vector<float>&_particleType, const std::vector<float> &_pabs,
	       const std::vector<float>&_lpx, const std::vector<float>&_lpy, const std::vector<float>&_lpz,
	       const std::vector<float>&_ldx, const std::vector<float>&_ldy, const std::vector<float>&_ldz,
	       const std::vector<float>&_ldtheta, const std::vector<float>&_ldphi, 
	       const std::vector<float>&_exx, const std::vector<float>&_exy, const std::vector<float>&_exz,
	       const std::vector<float>&_enx, const std::vector<float>&_eny, const std::vector<float>&_enz,
	       const std::vector<float>&_eloss, const std::vector<float>&_tof);   

  // get functions
  int getnRawGenPart() {return nRawGenPart;}
  int getnG4Vtx() {return nG4Vtx;}
  VtxVector getG4Vtx() {return G4Vtx;}
  int getnG4Trk() {return nG4Trk;}
  TrkVector getG4Trk() {return G4Trk;}
  int getnHits() {return nHits;}
  HitVector getHits() {return Hits;}

 private:

  // G4MC info
  int nRawGenPart;
  int nG4Vtx;
  VtxVector G4Vtx; 
  int nG4Trk; 
  TrkVector G4Trk; 
  // Tracker info
  int nHits;
  HitVector Hits; 


}; // end class declaration

#endif

#endif // endif PValidationFormats_h
