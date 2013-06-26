#ifndef DD_PixBarTPGStackLayerAlgo_h
#define DD_PixBarTPGStackLayerAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDPixBarTPGStackLayerAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDPixBarTPGStackLayerAlgo(); 
  virtual ~DDPixBarTPGStackLayerAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::string              idNameSpace; 	//Namespace of this and ALL sub-parts
  std::string              VolumeMaterial;      //Name of general material
  int                      number;      	//Number of ladders in phi
  double                   layerDz;     	//Length of the layer
  double                   sensorEdge;  	//Distance from edge for a half sensor
  double                   coolDz;      	//Length of the cooling piece
  double                   coolWidth;   	//Width                       
  double                   coolSide;    	//Side length
  double                   coolThick;   	//Thickness of the shell     
  double                   coolZ;               //Centre of cooling tube(s)
  int                      coolNumber;          // = 1 normally, =2 for LB ring layers
  double                   hybridThick; 	//Thickness of hybrid that determines cooling pipe 1 offset
  double                   moduleRadius;    	//Radial distance of mid point
  std::string              coolMat;     	//Cooling fluid material name
  std::string              tubeMat;     	//Cooling piece material name
  std::string              ladderNameUp;      	//Names of upper ladder
  std::string              ladderNameDown;      //Names of lower ladder
  double                   ladderWidth; 	//Up/Down Ladder Width
  double                   ladderThick; 	//Up/Down Ladder Thickness
  double                   module_offset; 	//Offset of module from radial/tangential vector
  double                   layout; 		//Layout type (0=TIB-like,1=TOB-like)
  double		   activeWidth;		//Up/Down Ladder active Width

};

#endif
