#include "SimMuon/CSCDigitizer/src/CSCConfigurableStripConditions.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCConfigurableStripConditions::CSCConfigurableStripConditions(const edm::ParameterSet & p)
: theAnalogNoise(  p.getParameter<double>("analogNoise") )
{
  theNoisifiers.resize(9);
  makeNoisifier(1, p.getParameter<std::vector<double> >("me11") );
  makeNoisifier(2, p.getParameter<std::vector<double> >("me12") );
  makeNoisifier(3, p.getParameter<std::vector<double> >("me13") );
  makeNoisifier(4, p.getParameter<std::vector<double> >("me11a") ); // not sure about this one
  makeNoisifier(5, p.getParameter<std::vector<double> >("me21") );
  makeNoisifier(6, p.getParameter<std::vector<double> >("me22") );
  makeNoisifier(7, p.getParameter<std::vector<double> >("me31") );
  makeNoisifier(8, p.getParameter<std::vector<double> >("me32") );
  makeNoisifier(9, p.getParameter<std::vector<double> >("me31") ); // for lack of a better idea
}


CSCConfigurableStripConditions::~CSCConfigurableStripConditions()
{
  for(int i = 0; i < 9; ++i)
  {
    delete theNoisifiers[i];
  }
}


void CSCConfigurableStripConditions::fetchNoisifier(const CSCDetId & detId, int istrip)
{
  //TODO get this moved toCSCDetId
  int chamberType = CSCChamberSpecs::whatChamberType(detId.station(), detId.ring());
  theNoisifier = theNoisifiers[chamberType-1];
}


void CSCConfigurableStripConditions::makeNoisifier(int chamberType, const std::vector<double> & correlations)
{
std::cout << "Make noisifiera " << chamberType << std::endl;

  // format is 33, 34, 44, 35, 45, 55
  //           46, 56, 66, 57, 67, 77
  if(correlations.size() != 12)
  {
    throw cms::Exception("CSCConfigurableStripConditions")
      << "Expect 12 noise correlation coefficients, but got "
      << correlations.size();
  }

  HepSymMatrix matrix(8);
  matrix[3][3] = correlations[0];
  matrix[3][4] = correlations[1];
  matrix[4][4] = correlations[2];
  matrix[3][5] = correlations[3];
  matrix[4][5] = correlations[4];
  matrix[5][5] = correlations[5];
  matrix[4][6] = correlations[6];
  matrix[5][6] = correlations[7];
  matrix[6][6] = correlations[8];
  matrix[5][7] = correlations[9];
  matrix[6][7] = correlations[10];
  matrix[7][7] = correlations[11];

  // since I don't know how to correlate the pedestal samples,
  // take as constant
  matrix[0][0] = theAnalogNoise * theAnalogNoise;
  matrix[1][1] = theAnalogNoise * theAnalogNoise;
  matrix[2][2] = theAnalogNoise * theAnalogNoise;
  theNoisifiers[chamberType-1] = new CorrelatedNoisifier(matrix);

}
