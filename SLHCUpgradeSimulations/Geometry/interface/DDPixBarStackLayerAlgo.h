#ifndef DD_PixBarStackLayerAlgo_h
#define DD_PixBarStackLayerAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDPixBarStackLayerAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDPixBarStackLayerAlgo(); 
  virtual ~DDPixBarStackLayerAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute();

private:

  std::string              idNameSpace; //Namespace of this and ALL sub-parts
  std::string              VolumeMaterial;      //Name of general material
  int                      number;      //Number of ladders in phi
  double                   layerDz;     //Length of the layer
  double                   sensorEdge;  //Distance from edge for a half sensor
  double                   coolDz;      //Length of the cooling piece
  double                   coolWidth;   //Width                       
  double                   coolSide;    //Side length
  double                   coolThick;   //Thickness of the shell     
  double                   moduleRadius;    //Radial distance between centres of 2 
  std::string              coolMat;     //Cooling fluid material name
  std::string              tubeMat;     //Cooling piece material name
  std::string              ladderNameUp;      //Names     of ladders
  std::string              ladderNameDown;      //Names     of ladders
  double                   ladderWidth; //Widths         ...
  double                   ladderThick; //Thickness      ...
  double                   module_offset; //offset of module from radial/tangential vector
};

#endif
