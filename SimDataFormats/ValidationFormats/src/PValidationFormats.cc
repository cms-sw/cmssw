/** \file PValidationFormats.cc
 *  
 *  See header file for description of classes
 *  conglomoration of all Validation SimDataFormats
 *
 *  $Date: 2007/10/04 19:26:49 $
 *  $Revision: 1.1 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"

///////////////////////////////////////////////////////////////////////////////
// PGlobalSimHit
///////////////////////////////////////////////////////////////////////////////

void PGlobalSimHit::putRawGenPart(int n)
{
  nRawGenPart = n;
  return;
}

void PGlobalSimHit::putG4Vtx(std::vector<float> x, std::vector<float> y, 
	       std::vector<float> z)
{
  nG4Vtx = x.size();
  G4Vtx.resize(nG4Vtx);
  for (int i = 0; i < nG4Vtx; ++i) {
    G4Vtx[i].x = x[i];
    G4Vtx[i].y = y[i];
    G4Vtx[i].z = z[i];
  }

  return;
}

void PGlobalSimHit::putG4Trk(std::vector<float> pt, std::vector<float> e)
{
  nG4Trk = pt.size();
  G4Trk.resize(nG4Trk);
  for (int i = 0; i < nG4Trk; ++i) {
    G4Trk[i].pt = pt[i];
    G4Trk[i].e = e[i];
  }

  return;
}

void PGlobalSimHit::putECalHits(std::vector<float> e, std::vector<float> tof,
				 std::vector<float> phi, 
				 std::vector<float> eta)
{
  nECalHits = e.size();
  ECalHits.resize(nECalHits);
  for (int i = 0; i < nECalHits; ++i) {
    ECalHits[i].e = e[i];
    ECalHits[i].tof = tof[i];
    ECalHits[i].phi = phi[i];
    ECalHits[i].eta = eta[i];
  }

  return;
}

void PGlobalSimHit::putPreShHits(std::vector<float> e, std::vector<float> tof,
				  std::vector<float> phi, 
				  std::vector<float> eta)
{
  nPreShHits = e.size();
  PreShHits.resize(nPreShHits);
  for (int i = 0; i < nPreShHits; ++i) {
    PreShHits[i].e = e[i];
    PreShHits[i].tof = tof[i];
    PreShHits[i].phi = phi[i];
    PreShHits[i].eta = eta[i];
  }

  return;
}

void PGlobalSimHit::putHCalHits(std::vector<float> e, std::vector<float> tof,
				 std::vector<float> phi, 
				 std::vector<float> eta)
{
  nHCalHits = e.size();
  HCalHits.resize(nHCalHits);
  for (int i = 0; i < nHCalHits; ++i) {
    HCalHits[i].e = e[i];
    HCalHits[i].tof = tof[i];
    HCalHits[i].phi = phi[i];
    HCalHits[i].eta = eta[i];
  }

  return;
}

void PGlobalSimHit::putPxlFwdHits(std::vector<float> tof, 
				   std::vector<float> z,
				   std::vector<float> phi, 
				   std::vector<float> eta)
{
  nPxlFwdHits = tof.size();
  PxlFwdHits.resize(nPxlFwdHits);
  for (int i = 0; i < nPxlFwdHits; ++i) {
    PxlFwdHits[i].tof = tof[i];
    PxlFwdHits[i].z = z[i];
    PxlFwdHits[i].phi = phi[i];
    PxlFwdHits[i].eta = eta[i];
  }

  return;
}

void PGlobalSimHit::putPxlBrlHits(std::vector<float> tof, 
				   std::vector<float> r,
				   std::vector<float> phi, 
				   std::vector<float> eta)
{
  nPxlBrlHits = tof.size(); 
  PxlBrlHits.resize(nPxlBrlHits);
  for (int i = 0; i < nPxlBrlHits; ++i) {
    PxlBrlHits[i].tof = tof[i];
    PxlBrlHits[i].r = r[i];
    PxlBrlHits[i].phi = phi[i];
    PxlBrlHits[i].eta = eta[i];
  }

  return;
}

void PGlobalSimHit::putSiFwdHits(std::vector<float> tof, 
				  std::vector<float> z,
				  std::vector<float> phi, 
				  std::vector<float> eta)
{
  nSiFwdHits = tof.size();
  SiFwdHits.resize(nSiFwdHits);
  for (int i = 0; i < nSiFwdHits; ++i) {
    SiFwdHits[i].tof = tof[i];
    SiFwdHits[i].z = z[i];
    SiFwdHits[i].phi = phi[i];
    SiFwdHits[i].eta = eta[i];
  }

  return;
}

void PGlobalSimHit::putSiBrlHits(std::vector<float> tof, std::vector<float> r,
				  std::vector<float> phi, 
				  std::vector<float> eta)
{
  nSiBrlHits = tof.size();
  SiBrlHits.resize(nSiBrlHits);
  for (int i = 0; i < nSiBrlHits; ++i) {
    SiBrlHits[i].tof = tof[i];
    SiBrlHits[i].r = r[i];
    SiBrlHits[i].phi = phi[i];
    SiBrlHits[i].eta = eta[i];
  }

  return;
}

void PGlobalSimHit::putMuonCscHits(std::vector<float> tof, 
				    std::vector<float> z,
				    std::vector<float> phi, 
				    std::vector<float> eta)
{
  nMuonCscHits = tof.size();
  MuonCscHits.resize(nMuonCscHits);
  for (int i = 0; i < nMuonCscHits; ++i) {
    MuonCscHits[i].tof = tof[i];
    MuonCscHits[i].z = z[i];
    MuonCscHits[i].phi = phi[i];
    MuonCscHits[i].eta = eta[i];
  }

  return;
}

void PGlobalSimHit::putMuonDtHits(std::vector<float> tof, 
				   std::vector<float> r,
				   std::vector<float> phi, 
				   std::vector<float> eta)
{
  nMuonDtHits = tof.size();
  MuonDtHits.resize(nMuonDtHits);
  for (int i = 0; i < nMuonDtHits; ++i) {
    MuonDtHits[i].tof = tof[i];
    MuonDtHits[i].r = r[i];
    MuonDtHits[i].phi = phi[i];
    MuonDtHits[i].eta = eta[i];
  }

  return;
}

void PGlobalSimHit::putMuonRpcFwdHits(std::vector<float> tof, 
				       std::vector<float> z,
				       std::vector<float> phi, 
				       std::vector<float> eta)
{
  nMuonRpcFwdHits = tof.size();
  MuonRpcFwdHits.resize(nMuonRpcFwdHits);
  for (int i = 0; i < nMuonRpcFwdHits; ++i) {
    MuonRpcFwdHits[i].tof = tof[i];
    MuonRpcFwdHits[i].z = z[i];
    MuonRpcFwdHits[i].phi = phi[i];
    MuonRpcFwdHits[i].eta = eta[i];
  }

  return;
}

void PGlobalSimHit::putMuonRpcBrlHits(std::vector<float> tof, 
				       std::vector<float> r,
				       std::vector<float> phi, 
				       std::vector<float> eta)
{
  nMuonRpcBrlHits = tof.size();
  MuonRpcBrlHits.resize(nMuonRpcBrlHits);
  for (int i = 0; i < nMuonRpcBrlHits; ++i) {
    MuonRpcBrlHits[i].tof = tof[i];
    MuonRpcBrlHits[i].r = r[i];
    MuonRpcBrlHits[i].phi = phi[i];
    MuonRpcBrlHits[i].eta = eta[i];
  }

  return;
}

///////////////////////////////////////////////////////////////////////////////
// PGlobalDigi
///////////////////////////////////////////////////////////////////////////////

void PGlobalDigi::putEBCalDigis(std::vector<int> maxpos,
				std::vector<double> aee,
				std::vector<float> she)
{
  nEBCalDigis = maxpos.size();
  EBCalDigis.resize(nEBCalDigis);
  for (int i = 0; i < nEBCalDigis; ++i) {
    EBCalDigis[i].maxPos = maxpos[i];
    EBCalDigis[i].AEE = aee[i];
    EBCalDigis[i].SHE = she[i];
  }

  return;
}

void PGlobalDigi::putEECalDigis(std::vector<int> maxpos,
				std::vector<double> aee,
				std::vector<float> she)
{
  nEECalDigis = maxpos.size();
  EECalDigis.resize(nEECalDigis);
  for (int i = 0; i < nEECalDigis; ++i) {
    EECalDigis[i].maxPos = maxpos[i];
    EECalDigis[i].AEE = aee[i];
    EECalDigis[i].SHE = she[i];
  }

  return;
}

void PGlobalDigi::putESCalDigis(std::vector<float> adc0,
				std::vector<float> adc1,
				std::vector<float> adc2,
				std::vector<float> she)
{
  nESCalDigis = adc0.size();
  ESCalDigis.resize(nESCalDigis);
  for (int i = 0; i < nESCalDigis; ++i) {
    ESCalDigis[i].ADC0 = adc0[i];
    ESCalDigis[i].ADC1 = adc1[i];
    ESCalDigis[i].ADC2 = adc2[i];
    ESCalDigis[i].SHE = she[i];
  }

  return;
}

void PGlobalDigi::putHBCalDigis(std::vector<float> aee,
				std::vector<float> she)
{
  nHBCalDigis = aee.size();
  HBCalDigis.resize(nHBCalDigis);
  for (int i = 0; i < nHBCalDigis; ++i) {
    HBCalDigis[i].AEE = aee[i];
    HBCalDigis[i].SHE = she[i];
  }

  return;
}

void PGlobalDigi::putHECalDigis(std::vector<float> aee,
				std::vector<float> she)
{
  nHECalDigis = aee.size();
  HECalDigis.resize(nHECalDigis);
  for (int i = 0; i < nHECalDigis; ++i) {
    HECalDigis[i].AEE = aee[i];
    HECalDigis[i].SHE = she[i];
  }

  return;
}

void PGlobalDigi::putHOCalDigis(std::vector<float> aee,
				std::vector<float> she)
{
  nHOCalDigis = aee.size();
  HOCalDigis.resize(nHOCalDigis);
  for (int i = 0; i < nHOCalDigis; ++i) {
    HOCalDigis[i].AEE = aee[i];
    HOCalDigis[i].SHE = she[i];
  }

  return;
}

void PGlobalDigi::putHFCalDigis(std::vector<float> aee,
				std::vector<float> she)
{
  nHFCalDigis = aee.size();
  HFCalDigis.resize(nHFCalDigis);
  for (int i = 0; i < nHFCalDigis; ++i) {
    HFCalDigis[i].AEE = aee[i];
    HFCalDigis[i].SHE = she[i];
  }

  return;
}

void PGlobalDigi::putTIBL1Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTIBL1Digis = adc.size();
  TIBL1Digis.resize(nTIBL1Digis);
  for (int i = 0; i < nTIBL1Digis; ++i) {
    TIBL1Digis[i].ADC = adc[i];
    TIBL1Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTIBL2Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTIBL2Digis = adc.size();
  TIBL2Digis.resize(nTIBL2Digis);
  for (int i = 0; i < nTIBL2Digis; ++i) {
    TIBL2Digis[i].ADC = adc[i];
    TIBL2Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTIBL3Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTIBL3Digis = adc.size();
  TIBL3Digis.resize(nTIBL3Digis);
  for (int i = 0; i < nTIBL3Digis; ++i) {
    TIBL3Digis[i].ADC = adc[i];
    TIBL3Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTIBL4Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTIBL4Digis = adc.size();
  TIBL4Digis.resize(nTIBL4Digis);
  for (int i = 0; i < nTIBL4Digis; ++i) {
    TIBL4Digis[i].ADC = adc[i];
    TIBL4Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTOBL1Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTOBL1Digis = adc.size();
  TOBL1Digis.resize(nTOBL1Digis);
  for (int i = 0; i < nTOBL1Digis; ++i) {
    TOBL1Digis[i].ADC = adc[i];
    TOBL1Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTOBL2Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTOBL2Digis = adc.size();
  TOBL2Digis.resize(nTOBL2Digis);
  for (int i = 0; i < nTOBL2Digis; ++i) {
    TOBL2Digis[i].ADC = adc[i];
    TOBL2Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTOBL3Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTOBL3Digis = adc.size();
  TOBL3Digis.resize(nTOBL3Digis);
  for (int i = 0; i < nTOBL3Digis; ++i) {
    TOBL3Digis[i].ADC = adc[i];
    TOBL3Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTOBL4Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTOBL4Digis = adc.size();
  TOBL4Digis.resize(nTOBL4Digis);
  for (int i = 0; i < nTOBL4Digis; ++i) {
    TOBL4Digis[i].ADC = adc[i];
    TOBL4Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTIDW1Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTIDW1Digis = adc.size();
  TIDW1Digis.resize(nTIDW1Digis);
  for (int i = 0; i < nTIDW1Digis; ++i) {
    TIDW1Digis[i].ADC = adc[i];
    TIDW1Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTIDW2Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTIDW2Digis = adc.size();
  TIDW2Digis.resize(nTIDW2Digis);
  for (int i = 0; i < nTIDW2Digis; ++i) {
    TIDW2Digis[i].ADC = adc[i];
    TIDW2Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTIDW3Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTIDW3Digis = adc.size();
  TIDW3Digis.resize(nTIDW3Digis);
  for (int i = 0; i < nTIDW3Digis; ++i) {
    TIDW3Digis[i].ADC = adc[i];
    TIDW3Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTECW1Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTECW1Digis = adc.size();
  TECW1Digis.resize(nTECW1Digis);
  for (int i = 0; i < nTECW1Digis; ++i) {
    TECW1Digis[i].ADC = adc[i];
    TECW1Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTECW2Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTECW2Digis = adc.size();
  TECW2Digis.resize(nTECW2Digis);
  for (int i = 0; i < nTECW2Digis; ++i) {
    TECW2Digis[i].ADC = adc[i];
    TECW2Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTECW3Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTECW3Digis = adc.size();
  TECW3Digis.resize(nTECW3Digis);
  for (int i = 0; i < nTECW3Digis; ++i) {
    TECW3Digis[i].ADC = adc[i];
    TECW3Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTECW4Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTECW4Digis = adc.size();
  TECW4Digis.resize(nTECW4Digis);
  for (int i = 0; i < nTECW4Digis; ++i) {
    TECW4Digis[i].ADC = adc[i];
    TECW4Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTECW5Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTECW5Digis = adc.size();
  TECW5Digis.resize(nTECW5Digis);
  for (int i = 0; i < nTECW5Digis; ++i) {
    TECW5Digis[i].ADC = adc[i];
    TECW5Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTECW6Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTECW6Digis = adc.size();
  TECW6Digis.resize(nTECW6Digis);
  for (int i = 0; i < nTECW6Digis; ++i) {
    TECW6Digis[i].ADC = adc[i];
    TECW6Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTECW7Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTECW7Digis = adc.size();
  TECW7Digis.resize(nTECW7Digis);
  for (int i = 0; i < nTECW7Digis; ++i) {
    TECW7Digis[i].ADC = adc[i];
    TECW7Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putTECW8Digis(std::vector<float> adc,
			        std::vector<int> strip)
{
  nTECW8Digis = adc.size();
  TECW8Digis.resize(nTECW8Digis);
  for (int i = 0; i < nTECW8Digis; ++i) {
    TECW8Digis[i].ADC = adc[i];
    TECW8Digis[i].STRIP = strip[i];
  }

  return;
}

void PGlobalDigi::putBRL1Digis(std::vector<float> adc,
			       std::vector<int> row,
			       std::vector<int> column)
{
  nBRL1Digis = adc.size();
  BRL1Digis.resize(nBRL1Digis);
  for (int i = 0; i < nBRL1Digis; ++i) {
    BRL1Digis[i].ADC = adc[i];
    BRL1Digis[i].ROW = row[i];
    BRL1Digis[i].COLUMN = column[i];
  }

  return;
}

void PGlobalDigi::putBRL2Digis(std::vector<float> adc,
			       std::vector<int> row,
			       std::vector<int> column)
{
  nBRL2Digis = adc.size();
  BRL2Digis.resize(nBRL2Digis);
  for (int i = 0; i < nBRL2Digis; ++i) {
    BRL2Digis[i].ADC = adc[i];
    BRL2Digis[i].ROW = row[i];
    BRL2Digis[i].COLUMN = column[i];
  }

  return;
}

void PGlobalDigi::putBRL3Digis(std::vector<float> adc,
			       std::vector<int> row,
			       std::vector<int> column)
{
  nBRL3Digis = adc.size();
  BRL3Digis.resize(nBRL3Digis);
  for (int i = 0; i < nBRL3Digis; ++i) {
    BRL3Digis[i].ADC = adc[i];
    BRL3Digis[i].ROW = row[i];
    BRL3Digis[i].COLUMN = column[i];
  }

  return;
}

void PGlobalDigi::putFWD1pDigis(std::vector<float> adc,
				std::vector<int> row,
				std::vector<int> column)
{
  nFWD1pDigis = adc.size();
  FWD1pDigis.resize(nFWD1pDigis);
  for (int i = 0; i < nFWD1pDigis; ++i) {
    FWD1pDigis[i].ADC = adc[i];
    FWD1pDigis[i].ROW = row[i];
    FWD1pDigis[i].COLUMN = column[i];
  }

  return;
}

void PGlobalDigi::putFWD1nDigis(std::vector<float> adc,
				std::vector<int> row,
				std::vector<int> column)
{
  nFWD1nDigis = adc.size();
  FWD1nDigis.resize(nFWD1nDigis);
  for (int i = 0; i < nFWD1nDigis; ++i) {
    FWD1nDigis[i].ADC = adc[i];
    FWD1nDigis[i].ROW = row[i];
    FWD1nDigis[i].COLUMN = column[i];
  }

  return;
}

void PGlobalDigi::putFWD2pDigis(std::vector<float> adc,
				std::vector<int> row,
				std::vector<int> column)
{
  nFWD2pDigis = adc.size();
  FWD2pDigis.resize(nFWD2pDigis);
  for (int i = 0; i < nFWD2pDigis; ++i) {
    FWD2pDigis[i].ADC = adc[i];
    FWD2pDigis[i].ROW = row[i];
    FWD2pDigis[i].COLUMN = column[i];
  }

  return;
}

void PGlobalDigi::putFWD2nDigis(std::vector<float> adc,
				std::vector<int> row,
				std::vector<int> column)
{
  nFWD2nDigis = adc.size();
  FWD2nDigis.resize(nFWD2nDigis);
  for (int i = 0; i < nFWD2nDigis; ++i) {
    FWD2nDigis[i].ADC = adc[i];
    FWD2nDigis[i].ROW = row[i];
    FWD2nDigis[i].COLUMN = column[i];
  }

  return;
}

void PGlobalDigi::putMB1Digis(std::vector<int> slayer,
			      std::vector<float> time,
			      std::vector<int> layer)
{
  nMB1Digis = slayer.size();
  MB1Digis.resize(nMB1Digis);
  for (int i = 0; i < nMB1Digis; ++i) {
    MB1Digis[i].SLAYER = slayer[i];
    MB1Digis[i].TIME = time[i];
    MB1Digis[i].LAYER = layer[i];
  }

  return;
}

void PGlobalDigi::putMB2Digis(std::vector<int> slayer,
			      std::vector<float> time,
			      std::vector<int> layer)
{
  nMB2Digis = slayer.size();
  MB2Digis.resize(nMB2Digis);
  for (int i = 0; i < nMB2Digis; ++i) {
    MB2Digis[i].SLAYER = slayer[i];
    MB2Digis[i].TIME = time[i];
    MB2Digis[i].LAYER = layer[i];
  }

  return;
}

void PGlobalDigi::putMB3Digis(std::vector<int> slayer,
			      std::vector<float> time,
			      std::vector<int> layer)
{
  nMB3Digis = slayer.size();
  MB3Digis.resize(nMB3Digis);
  for (int i = 0; i < nMB3Digis; ++i) {
    MB3Digis[i].SLAYER = slayer[i];
    MB3Digis[i].TIME = time[i];
    MB3Digis[i].LAYER = layer[i];
  }

  return;
}

void PGlobalDigi::putMB4Digis(std::vector<int> slayer,
			      std::vector<float> time,
			      std::vector<int> layer)
{
  nMB4Digis = slayer.size();
  MB4Digis.resize(nMB4Digis);
  for (int i = 0; i < nMB4Digis; ++i) {
    MB4Digis[i].SLAYER = slayer[i];
    MB4Digis[i].TIME = time[i];
    MB4Digis[i].LAYER = layer[i];
  }

  return;
}

void PGlobalDigi::putCSCstripDigis(std::vector<float> adc)
{
  nCSCstripDigis = adc.size();
  CSCstripDigis.resize(nCSCstripDigis);
  for (int i = 0; i < nCSCstripDigis; ++i) {
    CSCstripDigis[i].ADC = adc[i];
  }

  return;
}

void PGlobalDigi::putCSCwireDigis(std::vector<float> time)
{
  nCSCwireDigis = time.size();
  CSCwireDigis.resize(nCSCwireDigis);
  for (int i = 0; i < nCSCwireDigis; ++i) {
    CSCwireDigis[i].TIME = time[i];
  }

  return;
}

///////////////////////////////////////////////////////////////////////////////
// PGlobalRecHit
///////////////////////////////////////////////////////////////////////////////

void PGlobalRecHit::putEBCalRecHits(std::vector<float> re,
				    std::vector<float> she)
{
  nEBCalRecHits = re.size();
  EBCalRecHits.resize(nEBCalRecHits);
  for (int i = 0; i < nEBCalRecHits; ++i) {
    EBCalRecHits[i].RE = re[i];
    EBCalRecHits[i].SHE = she[i];
  }

  return;
}

void PGlobalRecHit::putEECalRecHits(std::vector<float> re,
				    std::vector<float> she)
{
  nEECalRecHits = re.size();
  EECalRecHits.resize(nEECalRecHits);
  for (int i = 0; i < nEECalRecHits; ++i) {
    EECalRecHits[i].RE = re[i];
    EECalRecHits[i].SHE = she[i];
  }

  return;
}

void PGlobalRecHit::putESCalRecHits(std::vector<float> re,
				    std::vector<float> she)
{
  nESCalRecHits = re.size();
  ESCalRecHits.resize(nESCalRecHits);
  for (int i = 0; i < nESCalRecHits; ++i) {
    ESCalRecHits[i].RE = re[i];
    ESCalRecHits[i].SHE = she[i];
  }

  return;
}

void PGlobalRecHit::putHBCalRecHits(std::vector<float> rec,
				    std::vector<float> r,
				    std::vector<float> she)
{
  nHBCalRecHits = rec.size();
  HBCalRecHits.resize(nHBCalRecHits);
  for (int i = 0; i < nHBCalRecHits; ++i) {
    HBCalRecHits[i].REC = rec[i];
    HBCalRecHits[i].R = r[i];
    HBCalRecHits[i].SHE = she[i];
  }

  return;
}

void PGlobalRecHit::putHECalRecHits(std::vector<float> rec,
				    std::vector<float> r,
				    std::vector<float> she)
{
  nHECalRecHits = rec.size();
  HECalRecHits.resize(nHECalRecHits);
  for (int i = 0; i < nHECalRecHits; ++i) {
    HECalRecHits[i].REC = rec[i];
    HECalRecHits[i].R = r[i];
    HECalRecHits[i].SHE = she[i];
  }

  return;
}

void PGlobalRecHit::putHOCalRecHits(std::vector<float> rec,
				    std::vector<float> r,
				    std::vector<float> she)
{
  nHOCalRecHits = rec.size();
  HOCalRecHits.resize(nHOCalRecHits);
  for (int i = 0; i < nHOCalRecHits; ++i) {
    HOCalRecHits[i].REC = rec[i];
    HOCalRecHits[i].R = r[i];
    HOCalRecHits[i].SHE = she[i];
  }

  return;
}

void PGlobalRecHit::putHFCalRecHits(std::vector<float> rec,
				    std::vector<float> r,
				    std::vector<float> she)
{
  nHFCalRecHits = rec.size();
  HFCalRecHits.resize(nHFCalRecHits);
  for (int i = 0; i < nHFCalRecHits; ++i) {
    HFCalRecHits[i].REC = rec[i];
    HFCalRecHits[i].R = r[i];
    HFCalRecHits[i].SHE = she[i];
  }

  return;
}

void PGlobalRecHit::putTIBL1RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTIBL1RecHits = rx.size();
  TIBL1RecHits.resize(nTIBL1RecHits);
  for (int i = 0; i < nTIBL1RecHits; ++i) {
    TIBL1RecHits[i].RX = rx[i];
    TIBL1RecHits[i].RY = ry[i];
    TIBL1RecHits[i].SX = sx[i];
    TIBL1RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTIBL2RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTIBL2RecHits = rx.size();
  TIBL2RecHits.resize(nTIBL2RecHits);
  for (int i = 0; i < nTIBL2RecHits; ++i) {
    TIBL2RecHits[i].RX = rx[i];
    TIBL2RecHits[i].RY = ry[i];
    TIBL2RecHits[i].SX = sx[i];
    TIBL2RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTIBL3RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTIBL3RecHits = rx.size();
  TIBL3RecHits.resize(nTIBL3RecHits);
  for (int i = 0; i < nTIBL3RecHits; ++i) {
    TIBL3RecHits[i].RX = rx[i];
    TIBL3RecHits[i].RY = ry[i];
    TIBL3RecHits[i].SX = sx[i];
    TIBL3RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTIBL4RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTIBL4RecHits = rx.size();
  TIBL4RecHits.resize(nTIBL4RecHits);
  for (int i = 0; i < nTIBL4RecHits; ++i) {
    TIBL4RecHits[i].RX = rx[i];
    TIBL4RecHits[i].RY = ry[i];
    TIBL4RecHits[i].SX = sx[i];
    TIBL4RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTOBL1RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTOBL1RecHits = rx.size();
  TOBL1RecHits.resize(nTOBL1RecHits);
  for (int i = 0; i < nTOBL1RecHits; ++i) {
    TOBL1RecHits[i].RX = rx[i];
    TOBL1RecHits[i].RY = ry[i];
    TOBL1RecHits[i].SX = sx[i];
    TOBL1RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTOBL2RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTOBL2RecHits = rx.size();
  TOBL2RecHits.resize(nTOBL2RecHits);
  for (int i = 0; i < nTOBL2RecHits; ++i) {
    TOBL2RecHits[i].RX = rx[i];
    TOBL2RecHits[i].RY = ry[i];
    TOBL2RecHits[i].SX = sx[i];
    TOBL2RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTOBL3RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTOBL3RecHits = rx.size();
  TOBL3RecHits.resize(nTOBL3RecHits);
  for (int i = 0; i < nTOBL3RecHits; ++i) {
    TOBL3RecHits[i].RX = rx[i];
    TOBL3RecHits[i].RY = ry[i];
    TOBL3RecHits[i].SX = sx[i];
    TOBL3RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTOBL4RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTOBL4RecHits = rx.size();
  TOBL4RecHits.resize(nTOBL4RecHits);
  for (int i = 0; i < nTOBL4RecHits; ++i) {
    TOBL4RecHits[i].RX = rx[i];
    TOBL4RecHits[i].RY = ry[i];
    TOBL4RecHits[i].SX = sx[i];
    TOBL4RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTIDW1RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTIDW1RecHits = rx.size();
  TIDW1RecHits.resize(nTIDW1RecHits);
  for (int i = 0; i < nTIDW1RecHits; ++i) {
    TIDW1RecHits[i].RX = rx[i];
    TIDW1RecHits[i].RY = ry[i];
    TIDW1RecHits[i].SX = sx[i];
    TIDW1RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTIDW2RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTIDW2RecHits = rx.size();
  TIDW2RecHits.resize(nTIDW2RecHits);
  for (int i = 0; i < nTIDW2RecHits; ++i) {
    TIDW2RecHits[i].RX = rx[i];
    TIDW2RecHits[i].RY = ry[i];
    TIDW2RecHits[i].SX = sx[i];
    TIDW2RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTIDW3RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTIDW3RecHits = rx.size();
  TIDW3RecHits.resize(nTIDW3RecHits);
  for (int i = 0; i < nTIDW3RecHits; ++i) {
    TIDW3RecHits[i].RX = rx[i];
    TIDW3RecHits[i].RY = ry[i];
    TIDW3RecHits[i].SX = sx[i];
    TIDW3RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTECW1RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTECW1RecHits = rx.size();
  TECW1RecHits.resize(nTECW1RecHits);
  for (int i = 0; i < nTECW1RecHits; ++i) {
    TECW1RecHits[i].RX = rx[i];
    TECW1RecHits[i].RY = ry[i];
    TECW1RecHits[i].SX = sx[i];
    TECW1RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTECW2RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTECW2RecHits = rx.size();
  TECW2RecHits.resize(nTECW2RecHits);
  for (int i = 0; i < nTECW2RecHits; ++i) {
    TECW2RecHits[i].RX = rx[i];
    TECW2RecHits[i].RY = ry[i];
    TECW2RecHits[i].SX = sx[i];
    TECW2RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTECW3RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTECW3RecHits = rx.size();
  TECW3RecHits.resize(nTECW3RecHits);
  for (int i = 0; i < nTECW3RecHits; ++i) {
    TECW3RecHits[i].RX = rx[i];
    TECW3RecHits[i].RY = ry[i];
    TECW3RecHits[i].SX = sx[i];
    TECW3RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTECW4RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTECW4RecHits = rx.size();
  TECW4RecHits.resize(nTECW4RecHits);
  for (int i = 0; i < nTECW4RecHits; ++i) {
    TECW4RecHits[i].RX = rx[i];
    TECW4RecHits[i].RY = ry[i];
    TECW4RecHits[i].SX = sx[i];
    TECW4RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTECW5RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTECW5RecHits = rx.size();
  TECW5RecHits.resize(nTECW5RecHits);
  for (int i = 0; i < nTECW5RecHits; ++i) {
    TECW5RecHits[i].RX = rx[i];
    TECW5RecHits[i].RY = ry[i];
    TECW5RecHits[i].SX = sx[i];
    TECW5RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTECW6RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTECW6RecHits = rx.size();
  TECW6RecHits.resize(nTECW6RecHits);
  for (int i = 0; i < nTECW6RecHits; ++i) {
    TECW6RecHits[i].RX = rx[i];
    TECW6RecHits[i].RY = ry[i];
    TECW6RecHits[i].SX = sx[i];
    TECW6RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTECW7RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTECW7RecHits = rx.size();
  TECW7RecHits.resize(nTECW7RecHits);
  for (int i = 0; i < nTECW7RecHits; ++i) {
    TECW7RecHits[i].RX = rx[i];
    TECW7RecHits[i].RY = ry[i];
    TECW7RecHits[i].SX = sx[i];
    TECW7RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putTECW8RecHits(std::vector<float> rx, 
				    std::vector<float> ry,
				    std::vector<float> sx, 
				    std::vector<float> sy)
{
  nTECW8RecHits = rx.size();
  TECW8RecHits.resize(nTECW8RecHits);
  for (int i = 0; i < nTECW8RecHits; ++i) {
    TECW8RecHits[i].RX = rx[i];
    TECW8RecHits[i].RY = ry[i];
    TECW8RecHits[i].SX = sx[i];
    TECW8RecHits[i].SY = sy[i];    
  }

  return;
}

void PGlobalRecHit::putBRL1RecHits(std::vector<float> rx, 
				   std::vector<float> ry,
				   std::vector<float> sx, 
				   std::vector<float> sy)
{
  nBRL1RecHits = rx.size();
  BRL1RecHits.resize(nBRL1RecHits);
  for (int i = 0; i < nBRL1RecHits; ++i) {
    BRL1RecHits[i].RX = rx[i];
    BRL1RecHits[i].RY = ry[i];
    BRL1RecHits[i].SX = sx[i];
    BRL1RecHits[i].SY = sy[i];
  }

  return;
}

void PGlobalRecHit::putBRL2RecHits(std::vector<float> rx, 
				   std::vector<float> ry,
				   std::vector<float> sx, 
				   std::vector<float> sy)
{
  nBRL2RecHits = rx.size();
  BRL2RecHits.resize(nBRL2RecHits);
  for (int i = 0; i < nBRL2RecHits; ++i) {
    BRL2RecHits[i].RX = rx[i];
    BRL2RecHits[i].RY = ry[i];
    BRL2RecHits[i].SX = sx[i];
    BRL2RecHits[i].SY = sy[i];
  }

  return;
}

void PGlobalRecHit::putBRL3RecHits(std::vector<float> rx, 
				   std::vector<float> ry,
				   std::vector<float> sx, 
				   std::vector<float> sy)
{
  nBRL3RecHits = rx.size();
  BRL3RecHits.resize(nBRL3RecHits);
  for (int i = 0; i < nBRL3RecHits; ++i) {
    BRL3RecHits[i].RX = rx[i];
    BRL3RecHits[i].RY = ry[i];
    BRL3RecHits[i].SX = sx[i];
    BRL3RecHits[i].SY = sy[i];
  }

  return;
}

void PGlobalRecHit::putFWD1pRecHits(std::vector<float> rx, 
				   std::vector<float> ry,
				   std::vector<float> sx, 
				   std::vector<float> sy)
{
  nFWD1pRecHits = rx.size();
  FWD1pRecHits.resize(nFWD1pRecHits);
  for (int i = 0; i < nFWD1pRecHits; ++i) {
    FWD1pRecHits[i].RX = rx[i];
    FWD1pRecHits[i].RY = ry[i];
    FWD1pRecHits[i].SX = sx[i];
    FWD1pRecHits[i].SY = sy[i];
  }

  return;
}

void PGlobalRecHit::putFWD1nRecHits(std::vector<float> rx, 
				   std::vector<float> ry,
				   std::vector<float> sx, 
				   std::vector<float> sy)
{
  nFWD1nRecHits = rx.size();
  FWD1nRecHits.resize(nFWD1nRecHits);
  for (int i = 0; i < nFWD1nRecHits; ++i) {
    FWD1nRecHits[i].RX = rx[i];
    FWD1nRecHits[i].RY = ry[i];
    FWD1nRecHits[i].SX = sx[i];
    FWD1nRecHits[i].SY = sy[i];
  }

  return;
}

void PGlobalRecHit::putFWD2pRecHits(std::vector<float> rx, 
				   std::vector<float> ry,
				   std::vector<float> sx, 
				   std::vector<float> sy)
{
  nFWD2pRecHits = rx.size();
  FWD2pRecHits.resize(nFWD2pRecHits);
  for (int i = 0; i < nFWD2pRecHits; ++i) {
    FWD2pRecHits[i].RX = rx[i];
    FWD2pRecHits[i].RY = ry[i];
    FWD2pRecHits[i].SX = sx[i];
    FWD2pRecHits[i].SY = sy[i];
  }

  return;
}

void PGlobalRecHit::putFWD2nRecHits(std::vector<float> rx, 
				   std::vector<float> ry,
				   std::vector<float> sx, 
				   std::vector<float> sy)
{
  nFWD2nRecHits = rx.size();
  FWD2nRecHits.resize(nFWD2nRecHits);
  for (int i = 0; i < nFWD2nRecHits; ++i) {
    FWD2nRecHits[i].RX = rx[i];
    FWD2nRecHits[i].RY = ry[i];
    FWD2nRecHits[i].SX = sx[i];
    FWD2nRecHits[i].SY = sy[i];
  }

  return;
}

void PGlobalRecHit::putDTRecHits(std::vector<float> rhd, 
				 std::vector<float> shd)
{
  nDTRecHits = rhd.size();
  DTRecHits.resize(nDTRecHits);
  for (int i = 0; i < nDTRecHits; ++i) {
    DTRecHits[i].RHD = rhd[i];
    DTRecHits[i].SHD = shd[i];
  }

  return;
}

void PGlobalRecHit::putCSCRecHits(std::vector<float> rhphi, 
				  std::vector<float> rhperp, 
				  std::vector<float> shphi)
{
  nCSCRecHits = rhphi.size();
  CSCRecHits.resize(nCSCRecHits);
  for (int i = 0; i < nCSCRecHits; ++i) {
    CSCRecHits[i].RHPHI = rhphi[i];
    CSCRecHits[i].RHPERP = rhperp[i];
    CSCRecHits[i].SHPHI = shphi[i];
  }

  return;
}

void PGlobalRecHit::putRPCRecHits(std::vector<float> rhx, 
				  std::vector<float> shx)
{
  nRPCRecHits = rhx.size();
  RPCRecHits.resize(nRPCRecHits);
  for (int i = 0; i < nRPCRecHits; ++i) {
    RPCRecHits[i].RHX = rhx[i];
    RPCRecHits[i].SHX = shx[i];
  }

  return;
}

///////////////////////////////////////////////////////////////////////////////
// PHcalValidInfoJets
///////////////////////////////////////////////////////////////////////////////

void PHcalValidInfoJets::fillTProfileJet(double e, double r,  double t) {
  jetHite.push_back((float)e);
  jetHitr.push_back((float)r);
  jetHitt.push_back((float)t);
  nJetHit++;

  //  std::cout << " fillTProfileJet - nJetHit = " << nJetHit << std::endl;
  
}

void PHcalValidInfoJets::fillEcollectJet(double ee, double he, 
					 double hoe, double etot) {
  // hardest jet properties

  ecalJet = (float)ee;
  hcalJet = (float)he;
    hoJet = (float)hoe;
  etotJet = (float)etot;
}

void PHcalValidInfoJets::fillEtaPhiProfileJet(double eta0, double phi0, 
					      double eta,  double phi,
					      double dist) {
  detaJet = (float)(eta-eta0);
  dphiJet = (float)(phi-phi0);
    drJet = (float)dist;
}

void PHcalValidInfoJets::fillJets(std::vector<double> en,
				  std::vector<double> eta,
				  std::vector<double> phi) {
  nJet = en.size();
  for (int i = 0; i < nJet; i++) {
    jetE.push_back((float)en[i]);
    jetEta.push_back((float)eta[i]);
    jetPhi.push_back((float)phi[i]);
  }

  //  std::cout << " fillJets - nJet = " << nJet << std::endl;

}

void PHcalValidInfoJets::fillDiJets(double mass) {
  dijetM = (float)mass;
}

///////////////////////////////////////////////////////////////////////////////
// PHcalValidInfoLayer
///////////////////////////////////////////////////////////////////////////////

void PHcalValidInfoLayer::fillLayers(double el[], double ed[], double ho,
				     double hbhe, double ebee) {

  for (int i = 0; i < 20; i++) {
    double en  = 0.001*el[i]; // GeV
    eLayer.push_back((float)en);
  }
  for (int i = 0; i < 4; i++) {
    double en  = 0.001*ed[i]; // GeV
    eDepth.push_back((float)en);
  }
  eHO   = (float)ho;
  eHBHE = (float)hbhe; // MeV
  eEBEE = (float)ebee;
}

void PHcalValidInfoLayer::fillHF(double fibl, double fibs, double enec,
				 double enhc) {
  elongHF  = (float)fibl;
  eshortHF = (float)fibs;
  eEcalHF  = (float)enec;
  eHcalHF  = (float)enhc;
}

void PHcalValidInfoLayer::fillHits(int nHits, int lay, int unitID, double eta,
				   double phi, double ehit, double t){

  hitLayer.push_back((float)lay);
  hitId.push_back((float)unitID);
  hitEta.push_back((float)eta);
  hitPhi.push_back((float)phi);
  hitE.push_back((float)ehit);
  hitTime.push_back((float)t);
  hitN++;

  //  std::cout << " fillHits: nHits,hitN = " << nHits << "," << hitN << std::endl;

}

///////////////////////////////////////////////////////////////////////////////
// PHcalValidInfoNxN
///////////////////////////////////////////////////////////////////////////////

void PHcalValidInfoNxN::fillHvsE(double ee, double he, double hoe, 
				 double etot) {
  ecalNxNr = (float)ee;
  hcalNxNr = (float)he;
  hoNxNr   = (float)hoe;
  etotNxNr = (float)etot;
}

void PHcalValidInfoNxN::fillEcollectNxN(double een, double hen, double hoen,
					double etotn) {
  ecalNxN = (float)een;
  hcalNxN = (float)hen;
  hoNxN   = (float)hoen;
  etotNxN = (float)etotn;
}

void PHcalValidInfoNxN::fillTProfileNxN (double e, int i, double t) {  
  idNxN.push_back((float)i);
  eNxN.push_back((float)e);
  tNxN.push_back((float)t);
  nNxN++;

  //  std::cout << " fillTProfileNxN - nNxN = " << nNxN << std::endl;

}

///////////////////////////////////////////////////////////////////////////////
// PMuonSimHit
///////////////////////////////////////////////////////////////////////////////

void PMuonSimHit::putRawGenPart(int n)
{
  nRawGenPart = n;
  return;
}

void PMuonSimHit::putG4Vtx(std::vector<float> x, std::vector<float> y, 
	                   std::vector<float> z)
{
  nG4Vtx = x.size();
  G4Vtx.resize(nG4Vtx);
  for (int i = 0; i < nG4Vtx; ++i) {
    G4Vtx[i].x = x[i];
    G4Vtx[i].y = y[i];
    G4Vtx[i].z = z[i];
  }
  return;
}

void PMuonSimHit::putG4Trk(std::vector<float> pt,  std::vector<float> e,
		           std::vector<float> eta, std::vector<float> phi)
{
  nG4Trk = pt.size();
  G4Trk.resize(nG4Trk);
  for (int i = 0; i < nG4Trk; ++i) {
    G4Trk[i].pt  = pt[i];
    G4Trk[i].e   = e[i];
    G4Trk[i].eta = eta[i];
    G4Trk[i].phi = phi[i];
  }
  return;
}

void PMuonSimHit::putCSCHits (
                              std::vector<int>   _cscId,
                              std::vector<unsigned int> _detUnitId,
                              std::vector<float> _trackId , 
                              std::vector<float> _processType,
                              std::vector<float> _particleType, 
                              std::vector<float> _pabs,
                              std::vector<float> _globposz,
                              std::vector<float> _globposphi,
                              std::vector<float> _globposeta,
                              std::vector<float> _locposx, 
                              std::vector<float> _locposy, 
                              std::vector<float> _locposz,
                              std::vector<float> _locdirx, 
                              std::vector<float> _locdiry, 
                              std::vector<float> _locdirz,
                              std::vector<float> _locdirtheta, 
                              std::vector<float> _locdirphi, 
                              std::vector<float> _exitpointx, 
                              std::vector<float> _exitpointy, 
                              std::vector<float> _exitpointz,
                              std::vector<float> _entrypointx, 
                              std::vector<float> _entrypointy, 
                              std::vector<float> _entrypointz,
                              std::vector<float> _enloss, 
                              std::vector<float> _tof)   

{
  nCSCHits = _tof.size();
  CSCHits.resize(nCSCHits);
  for (int i = 0; i < nCSCHits; ++i) {
    CSCHits[i]._cscId           = _cscId[i];
    CSCHits[i]._detUnitId       = _detUnitId[i];
    CSCHits[i]._trackId         = _trackId[i];
    CSCHits[i]._processType     = _processType[i];
    CSCHits[i]._particleType    = _particleType[i];
    CSCHits[i]._pabs            = _pabs[i];
    CSCHits[i]._globposz        = _globposz[i];
    CSCHits[i]._globposphi      = _globposphi[i];
    CSCHits[i]._globposeta      = _globposeta[i];
    CSCHits[i]._locposx         = _locposx[i];
    CSCHits[i]._locposy         = _locposy[i];
    CSCHits[i]._locposz         = _locposz[i];
    CSCHits[i]._locdirx         = _locdirx[i];
    CSCHits[i]._locdiry         = _locdiry[i];
    CSCHits[i]._locdirz         = _locdirz[i];
    CSCHits[i]._locdirtheta     = _locdirtheta[i];
    CSCHits[i]._locdirphi       = _locdirphi[i];
    CSCHits[i]._exitpointx      = _exitpointx[i];
    CSCHits[i]._exitpointy      = _exitpointy[i];
    CSCHits[i]._exitpointz      = _exitpointz[i];
    CSCHits[i]._entrypointx     = _entrypointx[i];
    CSCHits[i]._entrypointy     = _entrypointy[i];
    CSCHits[i]._entrypointz     = _entrypointz[i];
    CSCHits[i]._enloss          = _enloss[i];
    CSCHits[i]._tof             = _tof[i];
  }
  return;
}

void PMuonSimHit::putDTHits  (std::vector<unsigned int> _detUnitId,
                              std::vector<float> _trackId , 
                              std::vector<float> _processType,
                              std::vector<float> _particleType, 
                              std::vector<float> _pabs,
                              std::vector<float> _globposz,
                              std::vector<float> _globposphi,
                              std::vector<float> _globposeta,
                              std::vector<float> _locposx, 
                              std::vector<float> _locposy, 
                              std::vector<float> _locposz,
                              std::vector<float> _locdirx, 
                              std::vector<float> _locdiry, 
                              std::vector<float> _locdirz,
                              std::vector<float> _locdirtheta, 
                              std::vector<float> _locdirphi, 
                              std::vector<float> _exitpointx, 
                              std::vector<float> _exitpointy, 
                              std::vector<float> _exitpointz,
                              std::vector<float> _entrypointx, 
                              std::vector<float> _entrypointy, 
                              std::vector<float> _entrypointz,
                              std::vector<float> _enloss, 
                              std::vector<float> _tof)   

{
  nDTHits = _tof.size();
  DTHits.resize(nDTHits);
  for (int i = 0; i < nDTHits; ++i) {
    DTHits[i]._detUnitId       = _detUnitId[i];
    DTHits[i]._trackId         = _trackId[i];
    DTHits[i]._processType     = _processType[i];
    DTHits[i]._particleType    = _particleType[i];
    DTHits[i]._pabs            = _pabs[i];
    DTHits[i]._globposz        = _globposz[i];
    DTHits[i]._globposphi      = _globposphi[i];
    DTHits[i]._globposeta      = _globposeta[i];
    DTHits[i]._locposx         = _locposx[i];
    DTHits[i]._locposy         = _locposy[i];
    DTHits[i]._locposz         = _locposz[i];
    DTHits[i]._locdirx         = _locdirx[i];
    DTHits[i]._locdiry         = _locdiry[i];
    DTHits[i]._locdirz         = _locdirz[i];
    DTHits[i]._locdirtheta     = _locdirtheta[i];
    DTHits[i]._locdirphi       = _locdirphi[i];
    DTHits[i]._exitpointx      = _exitpointx[i];
    DTHits[i]._exitpointy      = _exitpointy[i];
    DTHits[i]._exitpointz      = _exitpointz[i];
    DTHits[i]._entrypointx     = _entrypointx[i];
    DTHits[i]._entrypointy     = _entrypointy[i];
    DTHits[i]._entrypointz     = _entrypointz[i];
    DTHits[i]._enloss          = _enloss[i];
    DTHits[i]._tof             = _tof[i];
  }
  return;
}

void PMuonSimHit::putRPCHits (std::vector<unsigned int> _detUnitId,
                              std::vector<float> _trackId , 
                              std::vector<float> _processType,
                              std::vector<float> _particleType, 
                              std::vector<float> _pabs,
                              std::vector<float> _globposz,
                              std::vector<float> _globposphi,
                              std::vector<float> _globposeta,
                              std::vector<float> _locposx, 
                              std::vector<float> _locposy, 
                              std::vector<float> _locposz,
                              std::vector<float> _locdirx, 
                              std::vector<float> _locdiry, 
                              std::vector<float> _locdirz,
                              std::vector<float> _locdirtheta, 
                              std::vector<float> _locdirphi, 
                              std::vector<float> _exitpointx, 
                              std::vector<float> _exitpointy, 
                              std::vector<float> _exitpointz,
                              std::vector<float> _entrypointx, 
                              std::vector<float> _entrypointy, 
                              std::vector<float> _entrypointz,
                              std::vector<float> _enloss, 
                              std::vector<float> _tof)   

{
  nRPCHits = _tof.size();
  RPCHits.resize(nRPCHits);
  for (int i = 0; i < nRPCHits; ++i) {
    RPCHits[i]._detUnitId       = _detUnitId[i];
    RPCHits[i]._trackId         = _trackId[i];
    RPCHits[i]._processType     = _processType[i];
    RPCHits[i]._particleType    = _particleType[i];
    RPCHits[i]._pabs            = _pabs[i];
    RPCHits[i]._globposz        = _globposz[i];
    RPCHits[i]._globposphi      = _globposphi[i];
    RPCHits[i]._globposeta      = _globposeta[i];
    RPCHits[i]._locposx         = _locposx[i];
    RPCHits[i]._locposy         = _locposy[i];
    RPCHits[i]._locposz         = _locposz[i];
    RPCHits[i]._locdirx         = _locdirx[i];
    RPCHits[i]._locdiry         = _locdiry[i];
    RPCHits[i]._locdirz         = _locdirz[i];
    RPCHits[i]._locdirtheta     = _locdirtheta[i];
    RPCHits[i]._locdirphi       = _locdirphi[i];
    RPCHits[i]._exitpointx      = _exitpointx[i];
    RPCHits[i]._exitpointy      = _exitpointy[i];
    RPCHits[i]._exitpointz      = _exitpointz[i];
    RPCHits[i]._entrypointx     = _entrypointx[i];
    RPCHits[i]._entrypointy     = _entrypointy[i];
    RPCHits[i]._entrypointz     = _entrypointz[i];
    RPCHits[i]._enloss          = _enloss[i];
    RPCHits[i]._tof             = _tof[i];
  }
  return;

}

///////////////////////////////////////////////////////////////////////////////
// PTrackerSimHit
///////////////////////////////////////////////////////////////////////////////

void PTrackerSimHit::putRawGenPart(int n)
{
  nRawGenPart = n;
  return;
}

void PTrackerSimHit::putG4Vtx(std::vector<float> x, std::vector<float> y, 
	       std::vector<float> z)
{
  nG4Vtx = x.size();
  G4Vtx.resize(nG4Vtx);
  for (int i = 0; i < nG4Vtx; ++i) {
    G4Vtx[i].x = x[i];
    G4Vtx[i].y = y[i];
    G4Vtx[i].z = z[i];
  }

  return;
}

void PTrackerSimHit::putG4Trk(std::vector<float> pt, std::vector<float> e,
		              std::vector<float> eta, std::vector<float> phi)
{
  nG4Trk = pt.size();
  G4Trk.resize(nG4Trk);
  for (int i = 0; i < nG4Trk; ++i) {
    G4Trk[i].pt = pt[i];
    G4Trk[i].e = e[i];
    G4Trk[i].eta = eta[i];
    G4Trk[i].phi = phi[i];
  }

  return;
}


void PTrackerSimHit::putHits (std::vector<int> _sysID, std::vector<float> _detUnitId,
	       std::vector<float>_trackId , std::vector<float>_processType,
	       std::vector<float>_particleType, std::vector<float> _pabs,
	       std::vector<float>_lpx, std::vector<float>_lpy, std::vector<float>_lpz,
	       std::vector<float>_ldx, std::vector<float>_ldy, std::vector<float>_ldz,
	       std::vector<float>_ldtheta, std::vector<float>_ldphi, 
	       std::vector<float>_exx, std::vector<float>_exy, std::vector<float>_exz,
	       std::vector<float>_enx, std::vector<float>_eny, std::vector<float>_enz,
	       std::vector<float>_eloss, std::vector<float>_tof)   

{
  nHits = _tof.size();
  Hits.resize(nHits);
  for (int i = 0; i < nHits; ++i) {
    Hits[i]._sysID = _sysID[i];
    Hits[i]._detUnitId = _detUnitId[i];
    Hits[i]._trackId = _trackId[i];
    Hits[i]._processType = _processType[i];
    Hits[i]._particleType = _particleType[i];
    Hits[i]._pabs = _pabs[i];
    Hits[i]._lpx = _lpx[i];
    Hits[i]._lpy = _lpy[i];
    Hits[i]._lpz = _lpz[i];
    Hits[i]._ldx = _ldx[i];
    Hits[i]._ldy = _ldy[i];
    Hits[i]._ldz = _ldz[i];
    Hits[i]._ldtheta = _ldtheta[i];
    Hits[i]._ldphi = _ldphi[i];
    Hits[i]._exx = _exx[i];
    Hits[i]._exy = _exy[i];
    Hits[i]._exz = _exz[i];
    Hits[i]._enx = _enx[i];
    Hits[i]._eny = _eny[i];
    Hits[i]._enz = _enz[i];
    Hits[i]._eloss = _eloss[i];
    Hits[i]._tof = _tof[i];
  }

  return;
}

