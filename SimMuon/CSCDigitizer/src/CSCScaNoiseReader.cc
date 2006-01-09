#include "SimMuon/CSCDigitizer/src/CSCScaNoiseReader.h"
#include "Utilities/General/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandFlat.h"
#include<fstream>
#include<string>

using namespace std;

const int N_SCA_BINS = 16;

CSCScaNoiseReader::CSCScaNoiseReader() : 
  CSCScaNoiseGenerator(N_SCA_BINS)
{

  string path( getenv( "CMSSW_SEARCH_PATH" ) );
  string scaNoiseFile = "SimMuon/CSCDigitizer/data/scaNoise.dat";
  FileInPath f1( path, scaNoiseFile );
  if (f1() == 0 ) {
    edm::LogError("CSCScaNoiseReader") << "Input file " <<  scaNoiseFile <<  " not found"
    << " in path " << path
    << "\n Set Muon:Endcap:ScaNoiseFile in .orcarc to the "
         " location of the file relative to CMSSW_DATA_PATH.";
    throw cms::Exception( " Endcap Muon SCA noise file not found.");
  }
  else
  {
    edm::LogInfo("CSCScaNoiseReader") << "Reading " << f1.name();
  }

    //  ifstream fin;
  ifstream & fin = *f1();

  LogDebug("CSCScaNoiseReader") << "CSCScaNoiseReader: opening file " << scaNoiseFile.c_str();

  //  fin.open(scaNoiseFile.c_str(), ios::in);

  if (fin == 0) {
    string errorMessage =  "Cannot open input file " +  path + scaNoiseFile;
    edm::LogError("CSCScaNoiseReader") << errorMessage;
    //edm::LogError(errorMessage);
    throw cms::Exception(errorMessage);
  } else {
    // first just count the lines so we can allocate array
    string str;
    nStripEvents = 0;
    while(!fin.eof()) {
      getline(fin, str, '\n');
      ++nStripEvents;
    }
    --nStripEvents;

    // allocate the array
    theData.resize(N_SCA_BINS*nStripEvents);

    // rewind and read it in
    //    fin.close();
    //    fin.open(scaNoiseFile.c_str(), ios::in);
    fin.clear( );                  // Clear eof read status
    fin.seekg( 0, ios::beg );      // Position at start of file
    for(int iline = 0; iline< nStripEvents; ++iline) {
      for(int i = 0; i < N_SCA_BINS; ++i) {
        fin >> theData[iline*N_SCA_BINS + i];
      }
    }
  }
  fin.close();
}  

CSCScaNoiseReader::~CSCScaNoiseReader() {
}


vector<int>
CSCScaNoiseReader::getNoise() const {
  std::vector<int> result(N_SCA_BINS);
  int iEvent = (int) (RandFlat::shoot() * nStripEvents);
  // just to be safe, in case random  # is 1.
  if(iEvent == nStripEvents) iEvent = 0;
  // typically, test beam will have 16 SCA bins, real
  int startingBin = 0;
  for(int i = 0; i < N_SCA_BINS; ++i) {
    result[i] = theData[iEvent*N_SCA_BINS + startingBin + i];
  }
  return result;
}

