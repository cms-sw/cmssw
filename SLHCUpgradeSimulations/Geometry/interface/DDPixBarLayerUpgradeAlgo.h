#ifndef DD_PixBarLayerUpgradeAlgo_h
#define DD_PixBarLayerUpgradeAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDPixBarLayerUpgradeAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDPixBarLayerUpgradeAlgo(); 
  virtual ~DDPixBarLayerUpgradeAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::string              idNameSpace; //Namespace of this and ALL sub-parts
  std::string              genMat;      //Name of general material
  int                      number;      //Number of ladders in phi
  double                   layerDz;     //Length of the layer
  double                   sensorEdge;  //Distance from edge for a half sensor
  double                   coolDz;      //Length of the cooling piece
  double                   coolWidth;   //Width                       
  double                   coolSide;    //Side length
  double                   coolThick;   //Thickness of the shell     
  double                   coolDist;    //Radial distance between centres of 2 
  std::string              coolMat;     //Cooling fluid material name
  std::string              tubeMat;     //Cooling piece material name
  std::string              ladder;      //Name  of ladder
  double                   ladderWidth; //Width of ladder 
  double                   ladderThick; //Thicknes of ladder 
  int                      outerFirst;  //Controller of the placement of ladder
};

#endif
