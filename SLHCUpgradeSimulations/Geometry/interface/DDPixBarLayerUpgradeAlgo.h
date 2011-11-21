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
  double                   coolDz;      //Length of the cooling piece
  double                   coolThick;   //Thickness of the shell     
  double                   coolRadius;   //Cool tube external radius    
  double                   coolDist;    //Radial distance between centres of 2 
  double                   cool1Offset;    //cooling pipe 1 offset for ladder at interface
  double                   cool2Offset;    //cooling pipe 2 offset for ladder at interface
  std::string              coolMat;     //Cooling fluid material name
  std::string              tubeMat;     //Cooling piece material name
  std::string              ladder;      //Name  of ladder
  double                   ladderWidth; //Width of ladder 
  double                   ladderThick; //Thicknes of ladder 
  double                   ladderOffset; //ladder dispacement at interface 
  int                      outerFirst;  //Controller of the placement of ladder
};

#endif
