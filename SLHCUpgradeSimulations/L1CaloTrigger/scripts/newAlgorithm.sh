#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: `basename ${0}` <AlgorithmName>"
else
	filename="plugins/${1}.cc"
	
	if [ -e "$filename" ]; then	
		echo "$filename already exists"
	else
		echo "
#include \"SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h\"

class ${1}:public L1CaloAlgoBase < ...InputCollection... , ...OutputCollection... > 
{
  public:
	${1}( const edm::ParameterSet & );
	 ~${1}(  );

//	void initialize(  );

	void algorithm( const int &, const int & );

  private:

};

${1}::${1}( const edm::ParameterSet & aConfig ):
L1CaloAlgoBase < ...InputCollection... , ...OutputCollection... > ( aConfig )
{
}

${1}::~${1}(  )
{
}

/*
void ${1}::initialize(  )
{
}
*/

void ${1}::algorithm( const int &aEta, const int &aPhi )
{
}

DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<${1}>,\"${1}\");
DEFINE_FWK_PSET_DESC_FILLER(${1});
" >> $filename
	fi
fi




