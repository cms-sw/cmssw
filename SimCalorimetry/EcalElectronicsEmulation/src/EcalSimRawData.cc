#include "SimCalorimetry/EcalElectronicsEmulation/interface/EcalSimRawData.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

#include <fstream> //used for debugging
#include <iostream>
#include <iomanip>
#include <cmath>
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

const int EcalSimRawData::ttType[nTtsAlongEbEta] = {
  0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1, //EE-
  0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1//EE+
};

const int EcalSimRawData::stripCh2Phi[nTtTypes][ttEdge][ttEdge] = {
  //TT type 0:
  /*ch-->*/
  {{4,3,2,1,0}, /*strip*/		
   {0,1,2,3,4}, /*|*/
   {4,3,2,1,0}, /*|*/
   {0,1,2,3,4}, /*|*/
   {4,3,2,1,0}},/*V*/
  //TT type 1:
  {{0,1,2,3,4},
   {4,3,2,1,0},
   {0,1,2,3,4},
   {4,3,2,1,0},
   {0,1,2,3,4}}
};

const int EcalSimRawData::strip2Eta[nTtTypes][ttEdge] = {
  {4,3,2,1,0}, //TT type 0
  {0,1,2,3,4}  //TT type 1
};

EcalSimRawData::EcalSimRawData(const edm::ParameterSet& params)
  //  : params_(params)
{
  //sets up parameters:
  digiProducer_ = params.getParameter<string>("digiProducer");
  ebdigiCollection_ = params.getParameter<std::string>("EBdigiCollection");
  eedigiCollection_ = params.getParameter<std::string>("EEdigiCollection");
  //ebSRPdigiCollection_ = params.getParameter<std::string>("EBSRPdigiCollection");
  //eeSRPdigiCollection_ = params.getParameter<std::string>("EESRPdigiCollection");
  tpDigiCollection_ = params.getParameter<std::string>("tpdigiCollection");
  trigPrimProducer_ = params.getParameter<string>("trigPrimProducer");
  xtalVerbose_ = params.getUntrackedParameter<bool>("xtalVerbose", false);
  tpVerbose_ = params.getUntrackedParameter<bool>("tpVerbose", false);
  tcc2dcc_ = params.getUntrackedParameter<bool>("tcc2dccData", true);
  srp2dcc_ = params.getUntrackedParameter<bool>("srp2dccData", true);
  thrs_.push_back(params.getParameter<double>("srpLowTowerThreshold"));
  thrs_.push_back(params.getParameter<double>("srpHighTowerThreshold"));
  dEta_ = params.getParameter<int>("deltaEta");
  dPhi_ = params.getParameter<int>("deltaPhi");
  assert(thrs_.size()==2);
  dccNum_ = -1;//params.getUntrackedParameter<int>("dccNum", -1);
  tccNum_ = -1;//params.getUntrackedParameter<int>("tccNum", -1);
		  
  string writeMode = params.getParameter<string>("writeMode");

  if(writeMode==string("littleEndian")){
    writeMode_ = littleEndian;
  } else if(writeMode==string("bigEndian")){
    writeMode_ = bigEndian;
  } else{
    writeMode_ = ascii;
  }
  //trigPrimBypass_ = params.getParameter<bool>("trigPrimBypass");
  //dumpFlags_ = params.getUntrackedParameter<int>("dumpFlags", 0);


  ttfFile.open("TTF.txt", ios::ate);
  if(!ttfFile) throw cms::Exception("Failed to create file TTF.txt");

  srfFile.open("SRF.txt", ios::ate);
  if(!srfFile) throw cms::Exception("Failed to create file SRF.txt");
}

void
EcalSimRawData::analyze(const edm::Event& event,
			const edm::EventSetup& es) 
{
  
  static int iEvent = 1;
    
  edm::Handle<EBDigiCollection> hEbDigis;
  event.getByLabel(digiProducer_, ebdigiCollection_, hEbDigis);
  assert(hEbDigis.isValid());
  const EBDigiCollection& ebDigis = *hEbDigis.product();

  if(xtalVerbose_){
    cout << "======================================================================\n"
      " Event " << iEvent << "\n"
	 << "---------------------------------------------------------------------\n";
  }
  
  if(ebDigis.size()==0) return;
  
  int nSamples = ebDigis.begin()->size();
  
  vector<uint16_t> adc[nEbEta][nEbPhi];

  const uint16_t suppressed = 0xFFFF;

  adc[0][0] = vector<uint16_t>(nSamples, suppressed);
  
  for(int iEbEta=0; iEbEta<nEbEta; ++iEbEta){
    for(int iEbPhi=0; iEbPhi<nEbPhi; ++iEbPhi){
      adc[iEbEta][iEbPhi] = adc[0][0];
    }
  }

  if(xtalVerbose_){
    cout << setfill('0');
  }
  for(EBDigiCollection::const_iterator it = ebDigis.begin();
      it != ebDigis.end(); ++it){
    const EBDataFrame& frame = *it;

    int iEta0 = iEta2cIndex((frame.id()).ieta());
    int iPhi0 = iPhi2cIndex((frame.id()).iphi());

//     cout << "xtl indices conv: (" << frame.id().ieta() << ","
// 	 << frame.id().iphi() << ") -> ("
// 	 << iEta0 << "," << iPhi0 << ")\n";
    
    if(iEta0<0 || iEta0>=nEbEta){
      cout << "iEta0 (= " << iEta0 << ") is out of range ("
           << "[0," << nEbEta -1 << "]\n";
    }
    if(iPhi0<0 || iPhi0>=nEbPhi){
      cout << "iPhi0 (= " << iPhi0 << ") is out of range ("
           << "[0," << nEbPhi -1 << "]\n";
    }
    
    if(xtalVerbose_){
      cout << iEta0 << "\t" << iPhi0 << ":\t";
      cout << hex;
    }
    assert(nSamples==frame.size());
    for(int iSample=0; iSample<nSamples; ++iSample){
      const EcalMGPASample& sample = frame.sample(iSample);
      uint16_t encodedAdc = sample.raw();
      adc[iEta0][iPhi0][iSample] = encodedAdc;  
      if(xtalVerbose_){
	cout << (iSample>0?" ":"") << "0x" << setw(4) 
	     << encodedAdc;
      }
    }
    if(xtalVerbose_) cout << "\n" << dec;
  }
  if(xtalVerbose_) cout << setfill(' ');
  genFeData("ecal", iEvent, adc);

  //Trigger primitives:
  edm::Handle<EcalTrigPrimDigiCollection> hTpDigis;
  event.getByLabel(trigPrimProducer_, tpDigiCollection_, hTpDigis);
  if(hTpDigis.isValid()&&hTpDigis->size()>0){
    const EcalTrigPrimDigiCollection& tpDigis = *hTpDigis.product();

    uint16_t tps[nTtsAlongEta][nTtsAlongPhi];
    EcalSelectiveReadout::ttFlag_t ttf[nTtsAlongEta][nTtsAlongPhi];
    for(int iTtEta0=0; iTtEta0 < nTtsAlongEta; ++iTtEta0){
      for(int iTtPhi0=0; iTtPhi0 < nTtsAlongPhi; ++iTtPhi0){
	tps[iTtEta0][iTtPhi0] = 0xFFFF;
	ttf[iTtEta0][iTtPhi0] = EcalSelectiveReadout::TTF_UNKNOWN;
      }
    }
    if(tpVerbose_){
      cout << setfill('0');
    }
    for(EcalTrigPrimDigiCollection::const_iterator it = tpDigis.begin();
	it != tpDigis.end(); ++it){
      const EcalTriggerPrimitiveDigi& tp = *it;
      int iTtEta0 = iTtEta2cIndex(tp.id().ieta());
      int iTtPhi0 = iTtPhi2cIndex(tp.id().iphi());
      if(iTtEta0<0 || iTtEta0>=nTtsAlongEta){
	cout << "iTtEta0 (= " << iTtEta0 << ") is out of range ("
	     << "[0," << nTtsAlongEbEta -1 << "]\n";
      }
      if(iTtPhi0<0 || iTtPhi0>=nTtsAlongPhi){
	cout << "iTtPhi0 (= " << iTtPhi0 << ") is out of range ("
	     << "[0," << nTtsAlongPhi -1 << "]\n";
      }
      //FIXME: here it is assumed that no compression is applied
      int adc = tp.compressedEt();
      if(tpVerbose_){
	for(int i=0; i<tp.size(); ++i){
	  cout << (i==0?"TP:":"") << "\t"
	       << tp.sample(i).compressedEt();
	}
	for(int i=0; i<tp.size(); ++i){
	  cout << (i==0?"\tTTF:":"") << "\t" << tp.sample(i).ttFlag();
	}
	cout << "\n";
      }
      int fgvb = tp.fineGrain();
      tps[iTtEta0][iTtPhi0] =
	(tp.ttFlag() & 0x7) <<9 
	| (fgvb&0x1) <<8
	| (adc&0xFF);
      if(tpVerbose_){
	cout << "tps[" << iTtEta0 << "][" << iTtPhi0 << "] = #"
	     << oct << tps[iTtEta0][iTtPhi0] << "o"
	     << " - " << "#" << tp.ttFlag() << "o" << dec << "\n";
      }
      ttf[iTtEta0][iTtPhi0] = (EcalSelectiveReadout::ttFlag_t)tp.ttFlag();
      if(tpVerbose_){
	cout << "TP(" << iTtEta0 << "," << iTtPhi0 << ") = "
	     << "0x" << setw(4) 
	     << tps[iTtEta0][iTtPhi0]
	     << "\tcmssw indices: "
	     << tp.id().ieta() << " " << tp.id().iphi() << "\n";
      }
    }//next TP
    if(tpVerbose_) cout << setfill(' ');
    
    //TTF file:
    printTTFlags(ttf, iEvent, ttfFile);
    
    genTcpData("ecal", iEvent, tps);


    //SR flags:
    EcalSelectiveReadout::towerInterest_t ebSrf[nTtsAlongEta][nTtsAlongPhi];
    EcalSelectiveReadout::towerInterest_t eeSrf[nEndcaps][nScX][nScY];
    getSrfs(ttf, ebSrf, eeSrf, es);
    //SRF file:
    printSRFlags(ebSrf, eeSrf, iEvent, srfFile);
    if(srp2dcc_){
      genSrData("ecal", iEvent, ebSrf);
    }
  } else{//TP digis not found
    static bool tpErr = false;
    if(!tpErr){
    cout << "Warning TP digis not found! No TP will be produed" << endl;
    tpErr = true;
    }
  }
  ++iEvent; //event counter
}

void EcalSimRawData::elec2GeomNum(int ittEta0, int ittPhi0, int strip1,
				  int ch1, int& iEta0, int& iPhi0) const{
  assert(0<=ittEta0 && ittEta0<nTtsAlongEbEta);
  assert(0<=ittPhi0 && ittPhi0<nTtsAlongPhi);
  assert(1<=strip1&& strip1<=ttEdge);
  assert(1<=ch1 && ch1<=ttEdge);
  const int type = ttType[ittEta0];
  iEta0 = ittEta0*ttEdge + strip2Eta[type][strip1-1];
  iPhi0 = ittPhi0*ttEdge + stripCh2Phi[type][strip1-1][ch1-1];
  assert(0<=iEta0 && iEta0<nEbEta);
  assert(0<=iPhi0 && iPhi0<nEbPhi);
}

void EcalSimRawData::fwrite(ofstream& f, uint16_t data,
			    int& iWord, bool hpar) const{

  if(hpar){
    //set horizontal odd parity bit:
    setHParity(data);
  }
  
  switch(writeMode_){
  case littleEndian:
    {
      char c = data&0x00FF;
      f.write(&c, sizeof(c));
      c = data&0xFF00;
      f.write(&c, sizeof(c));
    }
    break;
  case bigEndian:
    {
      char c = data&0xFF00; 
      f.write(&c, sizeof(c));
      c = data&0x00FF;
      f.write(&c, sizeof(c));
    }
    break;
  case ascii:
    f << ((iWord%8==0&&iWord!=0)?"\n":"")
      << "0x" << setfill('0') << setw(4) << hex << data << "\t"
      << dec << setfill(' ');
    break;
  }
  ++iWord;
}

string EcalSimRawData::getExt() const{
  switch(writeMode_){
  case littleEndian:
    return ".le";
  case bigEndian:
    return ".be";
  case ascii:
    return ".txt";
  default:
    return".?";
  }  
}

void EcalSimRawData::genFeData(string basename, int iEvent,
			       const vector<uint16_t> adcCount[nEbEta][nEbPhi]
			       ) const{
  int smf = 0;
  int gmf = 0;
  int nPendingEvt = 0;
  int monitorFlag = 0;
  int chFrameLen = adcCount[0][0].size() + 1;
  
  int iWord = 0;
  
  for(int iZ0 = 0; iZ0<2; ++iZ0){
    for(int iDccPhi0 = 0; iDccPhi0<nDccInPhi; ++iDccPhi0){
      int iDcc1 = iDccPhi0 + iZ0*nDccInPhi + nDccEndcap + 1;

      if(dccNum_!=-1  && dccNum_!=iDcc1) continue;
      
      stringstream s;
      s.str("");
      const string& ext = getExt();
      s << basename << "_fe2dcc" << setfill('0') << setw(2) << iDcc1
	<< setfill(' ') << ext;
      ofstream f(s.str().c_str(), (iEvent==1?ios::ate:ios::app));

      if(!f) return;


      if(writeMode_==ascii){
	f << (iEvent==1?"":"\n") << "[Event:" << iEvent << "]\n";
      }
      
      for(int iTtEtaInSm0 = 0; iTtEtaInSm0 < nTtsAlongSmEta; ++iTtEtaInSm0){
	int iTtEta0 = iZ0*nTtsAlongSmEta + iTtEtaInSm0;
	for(int iTtPhiInSm0 = 0; iTtPhiInSm0 < nTtsAlongSmPhi; ++iTtPhiInSm0){
	  //phi=0deg at middle of 1st barrel DCC:
	  int iTtPhi0 = -nTtPhisPerEbDcc/2 + iDccPhi0*nTtPhisPerEbDcc
	    + iTtPhiInSm0;
	  if(iTtPhi0<0) iTtPhi0 += nTtsAlongPhi;
	  for(int stripId1 = 1; stripId1 <= ttEdge; ++stripId1){
	    uint16_t stripHeader =
	      0xF << 11
	      | (nPendingEvt & 0x3F) << 5
	      | (gmf & 0x1) << 4
	      | (smf & 0x1) << 3
	      | (stripId1 & 0x7);
	    ///	    stripHeader |= parity(stripHeader) << 15;
	    fwrite(f,stripHeader, iWord);

	    for(int xtalId1 = 1; xtalId1 <= ttEdge; ++xtalId1){
	      
	      uint16_t crystalHeader =
		1 <<14
		| (chFrameLen & 0xFF) <<4
		| (monitorFlag & 0x1) <<3
		| (xtalId1 & 0x7);
	      //	      crystalHeader |=parity(crystalHeader) << 15;
	      fwrite(f, crystalHeader, iWord);
	      
	      int iEta0;
	      int iPhi0;
	      elec2GeomNum(iTtEta0, iTtPhi0, stripId1, xtalId1,
			   iEta0, iPhi0);
	      if(xtalVerbose_){
		cout << dec
		     << "iDcc1 = " << iDcc1 << "\t"
		     << "iEbTtEta0 = " << iTtEta0 << "\t"
		     << "iEbTtPhi0 = " << iTtPhi0 << "\t"
		     << "stripId1 = " << stripId1 << "\t"
		     << "xtalId1 = " << xtalId1 << "\t"
		     << "iEta0 = " << iEta0 << "\t"
		     << "iPhi0 = " << iPhi0 << "\t"
		     << "adc[5] = 0x" << hex << adcCount[iEta0][iPhi0][5]
		     << dec << "\n";
	      }
	      

	      const vector<uint16_t>& adc = adcCount[iEta0][iPhi0];
	      for(unsigned iSample=0; iSample  < adc.size(); ++iSample){
		uint16_t data = adc[iSample] & 0x3FFF;
		//		data |= parity(data);
		fwrite(f, data, iWord);
	      } //next time sample
	    } //next crystal in strip
	  } //next strip in TT
	} //next TT along phi
      } //next TT along eta
    } //next DCC
  } //next half-barrel
}

void EcalSimRawData::genSrData(string basename, int iEvent,
			       EcalSelectiveReadout::towerInterest_t srf[nTtsAlongEbEta][nTtsAlongPhi]
			       ) const{
  for(int iZ0 = 0; iZ0<2; ++iZ0){
    for(int iDccPhi0 = 0; iDccPhi0<nDccInPhi; ++iDccPhi0){
      int iDcc1 = iDccPhi0 + iZ0*nDccInPhi + nDccEndcap + 1;
      if(dccNum_!=-1  && dccNum_!=iDcc1) continue;
      stringstream s;
      s.str("");
      s << basename << "_ab2dcc" << setfill('0') << setw(2) << iDcc1
	<< setfill(' ') << getExt();
      ofstream f(s.str().c_str(), (iEvent==1?ios::ate:ios::app));
      
      if(!f) throw cms::Exception(string("Cannot create/open file ")
				  + s.str() + ".");
      
      int iWord = 0;
  
      if(writeMode_==ascii){
	f << (iEvent==1?"":"\n") << "[Event:" << iEvent << "]\n";
      }

      const uint16_t le1 = 0;
      const uint16_t le0 = 0;
      const uint16_t h1 = 1;
      const uint16_t nFlags = 68;
      uint16_t data =  (h1 & 0x1)<< 14
	| (le1 & 0x1) << 12
	| (le0 & 0x1) << 11
	| (nFlags & 0x7F);
      
      fwrite(f, data, iWord, true);
      
      int iFlag = 0;
      data = 0;

      cout << "iDcc1: " << iDcc1 << "\n";
      
      for(int iTtEtaInSm0 = 0; iTtEtaInSm0 < nTtsAlongSmEta; ++iTtEtaInSm0){
	//	int iTtEbEta0 = iZ0*nTtsAlongSmEta + iTtEtaInSm0;
	int iTtEta0 = nTtsAlongEeEta + iZ0*nTtsAlongSmEta + iTtEtaInSm0;
	for(int iTtPhiInSm0 = 0; iTtPhiInSm0 < nTtsAlongSmPhi; ++iTtPhiInSm0){
	  //phi=0deg at middle of 1st barrel DCC:
	  int iTtPhi0 = -nTtPhisPerEbDcc/2 + iDccPhi0*nTtPhisPerEbDcc
	    + iTtPhiInSm0;
	  if(iTtPhi0<0) iTtPhi0 += nTtsAlongPhi;
	  //flags are packed by four:
	  //|15 |14 |13-12 |11      9|8      6|5      3|2      0| 
	  //| P | 0 | X  X |  srf i+3| srf i+2| srf i+1| srf i  |
	  //|   |   |      | field 3 |field 2 | field 1| field 0|
	  const int field = iFlag%4;
	  cout << "TtEta0: " << iTtEta0 << "\tTtPhi0: " << iTtPhi0 << "\n";
	  cout << "#" << oct << (int)srf[iTtEta0][iTtPhi0] << "o ****> #" << oct << (srf[iTtEta0][iTtPhi0] << (field*3)) << "o\n" << dec;
	  
	  data |= srf[iTtEta0][iTtPhi0] << (field*3);

	  if(field==3){

	    cout <<  srf[iTtEta0][iTtPhi0] << "----> 0x" << hex << data << "\n";
	    
	    fwrite(f, data, iWord, true);
	    data = 0;
	  }
	  ++iFlag;
	} //next TT along phi
      } //next TT along eta
    } //next DCC
  } //next half-barrel
}


void EcalSimRawData::genTcpData(string basename, int iEvent,
				const uint16_t tps[nTtsAlongEta][nTtsAlongPhi]
				) const{
  int iDccWord = 0;

  for(int iZ0 = 0; iZ0<2; ++iZ0){
    for(int iTccPhi0 = 0; iTccPhi0<nTccInPhi; ++iTccPhi0){
      int iTcc1 = iTccPhi0 + iZ0*nTccInPhi + nTccEndcap + 1;

      if(tccNum_!=-1  && tccNum_!=iTcc1) continue;
      
      stringstream s;
      s.str("");
      char* ext = ".txt"; //only ascii mode supported for TCP

      s << basename << "_tcc" << setfill('0') << setw(2) << iTcc1
	<< setfill(' ') << ext;
      ofstream fe2tcc(s.str().c_str(), (iEvent==1?ios::ate:ios::app));

      if(!fe2tcc) throw cms::Exception(string("Failed to create file ")
				       + s.str() + ".");

      auto_ptr<ofstream> dccF;
      if(tcc2dcc_){
	s.str("");
	s << basename << "_tcc2dcc" << setfill('0') << setw(2) << iTcc1
	  << setfill(' ') << getExt();
	dccF = auto_ptr<ofstream>(new ofstream(s.str().c_str(),
					       (iEvent==1?ios::ate:ios::app)));
	if(!*dccF){
	  cout << "Warning: failed to create or open file " << s << ".\n";
	  dccF.reset();
	}
      }

      if(dccF.get()!=0){
	const uint16_t h1 = 1;
	const uint16_t le1 = 0;
	const uint16_t le0 = 0;
	const uint16_t nSamples = 1;
	const uint16_t nTts = 68;
	const uint16_t data = (h1 & 0x1) << 14
	  | (le1 & 0x1) << 12
	  | (le0 & 0x1) << 11
	  | (nSamples & 0xF) << 7
	  | (nTts & 0x7F);	
	*dccF << (iEvent==1?"":"\n") << "[Event:" << iEvent << "]\n";
	fwrite(*dccF, data, iDccWord, false);
      }
      
      //      if(writeMode_==ascii){
      //	fe2tcc << "[Event:" << iEvent << "]\n";
      //}
      int memPos = iEvent-1;
      int iCh1 = 1;
      for(int iTtEtaInSm0 = 0; iTtEtaInSm0 < nTtsAlongSmEta; ++iTtEtaInSm0){
	int iTtEta0 = nTtsAlongEeEta + iZ0*nTtsAlongSmEta + iTtEtaInSm0;
	for(int iTtPhiInSm0 = 0; iTtPhiInSm0 < nTtsAlongSmPhi; ++iTtPhiInSm0){
	  //phi=0deg at middle of 1st barrel DCC:
	  int iTtPhi0 = -nTtPhisPerEbTcc/2 + iTccPhi0*nTtPhisPerEbTcc
	    + iTtPhiInSm0;
	  if(iTtPhi0<0) iTtPhi0 += nTtsAlongPhi;

	  uint16_t tp_fe2tcc = tps[iTtEta0][iTtPhi0]&0xFF | (tps[iTtEta0][iTtPhi0]&0x600)>>1 ;
	  
	  if(tpVerbose_){
	    cout << dec
		 << "iTcc1 = " << iTcc1 << "\t"
		 << "iTtEta0 = " << iTtEta0 << "\t"
		 << "iTtPhi0 = " << iTtPhi0 << "\t"
		 << "iCh1 = " << iCh1 << "\t"
		 << "memPos = " << memPos << "\t" 
		 << "tp = 0x" << hex << tps[iTtEta0][iTtPhi0]
		 << dec << "\n";
	  }
	  fe2tcc << iCh1 << "\t"
		 << memPos << "\t"
		 << setfill('0') << hex
		 << "0x" << setw(4) << tp_fe2tcc << "\t"
		 << "0"
		 << dec << setfill(' ') << "\n";
	  if(dccF.get()!=0){
	    fwrite(*dccF, tps[iTtEta0][iTtPhi0], iDccWord, false);
	  }
	  ++iCh1;
	  } //next TT along phi
	} //next TT along eta
      } //next TCC
    } //next half-barrel
}


void EcalSimRawData::setHParity(uint16_t& a) const{
  const int odd = 1 <<15;
  const int even = 0;
  //parity bit of numbers from 0x0 to 0xF:
  //                    0   1   2    3   4    5    6   7   8    9    A   B    C   D   E    F           
  const int p[16] = {even,odd,odd,even,odd,even,even,odd,odd,even,even,odd,even,odd,odd,even};
  //inverts parity bit (LSB) of 'a' in case of even parity:
  a ^= p[a&0xF] ^ p[(a>>4)&0xF] ^ p[(a>>8)&0xF] ^ p[a>>12&0xF] ^ odd;   
}

void EcalSimRawData::getSrfs(const EcalSelectiveReadout::ttFlag_t ttf[nTtsAlongEta][nTtsAlongPhi],
			     EcalSelectiveReadout::towerInterest_t ebSrf[nTtsAlongEta][nTtsAlongPhi],
			     EcalSelectiveReadout::towerInterest_t eeSrf[nEndcaps][nScX][nScY],
			     const edm::EventSetup& es){
  static bool firstCall = true;
  if(firstCall){
    esr_ = auto_ptr<EcalSelectiveReadout>
      (new EcalSelectiveReadout(thrs_, dEta_, dPhi_));
    firstCall = false;
  }
  checkGeometry(es);
  checkTriggerMap(es);
  esr_->runSelectiveReadout0(ttf);
  cout << "======================================================================\n";
  cout << "SRP Result: \n";
  cout << *esr_;
  cout << "======================================================================\n";
  for(int iTtEta0 = 0; iTtEta0 < nTtsAlongEta; ++iTtEta0){
    for(int iTtPhi0 = 0; iTtPhi0 < nTtsAlongPhi; ++iTtPhi0){
      int iTtEta = cIndex2iTtEta(iTtEta0);
      int zside = (iTtEta0<0)?-1:1;
      int iTtPhi = cIndex2TtPhi(iTtPhi0);
      ebSrf[iTtEta0][iTtPhi0] =
	esr_->getTowerInterest(EcalTrigTowerDetId(zside, EcalBarrel,
						  abs(iTtEta),
						  iTtPhi));
    }
  }
  const float center = nScX/2-.5;
  for(int iEE = 0; iEE < nEndcaps; ++iEE){
    for(int iX0 = 0; iX0 < nScX; ++iX0){
      for(int iY0 = 0; iY0 < nScY; ++iY0){
	//distance to center of endcap to the square:
	float dCenter2 = pow(iX0-center,2)+pow(iY0-center,2);
	//some trick to get an existing crystal of SC even for partial SC:
	cout << "PGPGPGP>>> " << (dCenter2<25*25) << "\t"
	     << (iX0*5+5) << "," << (iY0*5+5) << "\t"
	     << (iX0*5+1) << "," << (iY0*5+1) << "\n";
	try{
	  EEDetId aCrystalOfSc = (dCenter2<25*25)?EEDetId(iX0*5+5, iY0*5+5, (iEE==0?-1:1)):EEDetId(iX0*5+1, iY0*5+1, (iEE==0?-1:1));
	  eeSrf[iEE][iX0][iY0] =
	    esr_->getCrystalInterest(aCrystalOfSc);
	} catch(cms::Exception e){
	  eeSrf[iEE][iX0][iY0] = EcalSelectiveReadout::UNKNOWN;
	}
      }
    }
  }
}

void EcalSimRawData::checkGeometry(const edm::EventSetup& es){
  edm::ESHandle<CaloGeometry> hGeometry;
  es.get<IdealGeometryRecord>().get(hGeometry);
  
  const CaloGeometry * pGeometry = &*hGeometry;
  
  // see if we need to update
  if(pGeometry != theGeometry) {
    theGeometry = pGeometry;
    esr_->setGeometry(theGeometry);
  }
}


void EcalSimRawData::checkTriggerMap(const edm::EventSetup&
					      es){
  edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap;
  es.get<IdealGeometryRecord>().get(eTTmap);
  
  const EcalTrigTowerConstituentsMap * pMap = &*eTTmap;
  
  // see if we need to update
  if(pMap!= theTriggerTowerMap) {
    theTriggerTowerMap = pMap;
    esr_->setTriggerMap(theTriggerTowerMap);
  }
}

void EcalSimRawData::printTTFlags(const EcalSelectiveReadout::ttFlag_t
				  ttf[nTtsAlongEta][nTtsAlongPhi],
				  int iEvent, ostream& os) const{
  const char tccFlagMarker[] = { '?', '.', 'S', '?', 'C', 'E', 'E', 'E', 'E'};
  const int nEta = EcalSelectiveReadout::nTriggerTowersInEta;
  const int nPhi = EcalSelectiveReadout::nTriggerTowersInPhi;
  
  if(iEvent==1){
    os << "# TCC flag map\n#\n"
      "# +-->Phi            " << tccFlagMarker[1] << ": 000 (low interest)\n"
      "# |                  " << tccFlagMarker[2] << ": 001 (mid interest)\n"
      "# |                  " << tccFlagMarker[3] << ": 010 (not valid)\n"
      "# V Eta              " << tccFlagMarker[5] << ": 011 (high interest)\n"
      "#                    " << tccFlagMarker[6] << ": 1xx forced readout (Hw error)\n";
  }

  os << "#\n#Event " << iEvent << "\n";
  
  for(int iEta=0; iEta<nEta; ++iEta){
    for(int iPhi=0; iPhi<nPhi; ++iPhi){
      os << tccFlagMarker[ttf[iEta][iPhi]+1];
    }
    os << "\n";
  }
}
  

void EcalSimRawData::printSRFlags(EcalSelectiveReadout::towerInterest_t
				  ebSrf[nTtsAlongEbEta][nTtsAlongPhi],
				  EcalSelectiveReadout::towerInterest_t
				  eeSrf[nEndcaps][nScX][nScY],
				  int iEvent, ostream& os) const{
  const char srpFlagMarker[] = {'.', 'S', 'N', 'C', '4','5','6','7'};
  if(iEvent==1){
    time_t t;
    time(&t);
    const char* date = ctime(&t);
    os << "#SRP flag map\n#\n"
      "# Generatied on: " << date << "\n#\n"
      "# Low TT Et Threshold:  " <<  thrs_[0] << " GeV\n"
      "# High TT Et Threshold: " << thrs_[1] << " GeV\n"
      "# Algorithm type: " << 2*dEta_+1 << "x" << 2*dPhi_+1 << "\n"
      "# +-->Phi/Y " << srpFlagMarker[0] << ": low interest\n"
      "# |         " << srpFlagMarker[1] << ": single\n"
      "# |         " << srpFlagMarker[2] << ": neighbour\n"
      "# V Eta/X   " << srpFlagMarker[3] << ": center\n"
      "#\n";    
  }

  //EE-,EB,EE+ map wil be written onto file in following format:
  //
  //      72
  // <-------------->
  //  20
  // <--->
  //  EEE                A             +-----> Y
  // EEEEE               |             |
  // EE EE               | 20   EE-    |
  // EEEEE               |             |
  //  EEE                V             V X
  // BBBBBBBBBBBBBBBBB   A
  // BBBBBBBBBBBBBBBBB   |             +-----> Phi
  // BBBBBBBBBBBBBBBBB   |             |
  // BBBBBBBBBBBBBBBBB   | 34  EB      |
  // BBBBBBBBBBBBBBBBB   |             |
  // BBBBBBBBBBBBBBBBB   |             V Eta
  // BBBBBBBBBBBBBBBBB   |
  // BBBBBBBBBBBBBBBBB   |
  // BBBBBBBBBBBBBBBBB   V
  //  EEE                A             +-----> Y
  // EEEEE               |             |
  // EE EE               | 20 EE+      |
  // EEEEE               |             |
  //  EEE                V             V X
  //
  //
  //
  //
  //event header:
  os << "# Event " << iEvent << "\n";

  esr_->print(os);
  
#if 0
  for(int iX0=0; iX0<nScX; ++iX0){
    for(int iY0=0; iY0<nScY; ++iY0){
      EcalSelectiveReadout::towerInterest_t srFlag = eeSrf[0][iX0][iY0];
      if(!((unsigned)srFlag<sizeof(srpFlagMarker)/sizeof(srpFlagMarker[0])
	   || srFlag==EcalSelectiveReadout::UNKNOWN)){

	cout << "==========> "
	     << srFlag << " " << EcalSelectiveReadout::UNKNOWN
	     << " " << iEvent << " " << iX0 << " " << iY0 << endl;
	 
      }
      
      assert((unsigned)srFlag<sizeof(srpFlagMarker)/sizeof(srpFlagMarker[0])
	     || srFlag==EcalSelectiveReadout::UNKNOWN);
      os << (srFlag==EcalSelectiveReadout::UNKNOWN?
	     ' ':srpFlagMarker[srFlag]);
    }
    os << "\n"; //one Y supercystal column per line
  } //next supercrystal X-index
   
  //EB
  for(int iEta0 = 0;
      iEta0 < nTtsAlongEbEta;
      ++iEta0){
    for(int iPhi0 = 0; iPhi0 < nTtsAlongPhi; ++iPhi0){
      EcalSelectiveReadout::towerInterest_t srFlag = ebSrf[iEta0][iPhi0];
      assert((unsigned)srFlag
	     < sizeof(srpFlagMarker)/sizeof(srpFlagMarker[0]));
      os << srpFlagMarker[srFlag];
    }
    os << "\n"; //one phi per line
  }
   
  //EE+
  for(int iX0=0; iX0<nScX; ++iX0){
    for(int iY0=0; iY0<nScY; ++iY0){
      EcalSelectiveReadout::towerInterest_t srFlag
	= eeSrf[1][iX0][iY0];
      assert((unsigned)srFlag<sizeof(srpFlagMarker)/sizeof(srpFlagMarker[0])
	     ||srFlag==EcalSelectiveReadout::UNKNOWN);
      os << (srFlag==EcalSelectiveReadout::UNKNOWN?
	     ' ':srpFlagMarker[srFlag]);
    }
    os << "\n"; //one Y supercystal column per line
  } //next supercrystal X-index
#endif
  //event trailer:
  os << "\n";
}
