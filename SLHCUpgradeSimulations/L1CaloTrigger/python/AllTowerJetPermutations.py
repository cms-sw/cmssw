import FWCore.ParameterSet.Config as cms

def NewNamedModule( process , Name , Object ):
	setattr( process , Name , Object )
	return getattr( process , Name )


def CreateAllTowerJetPermutations( process ):

	RingSubtractionMethods = [ "Mean" , "Median" , "Constant" ]

	TowerJetSizes = [ 8 , 9 , 10 , 12 ]
	TowerJetShapes = [ "Circle" , "Square" ]


	lReturnSequence = cms.Sequence()

	for RingSubtractionMethod in RingSubtractionMethods:
		lReturnSequence += NewNamedModule(
								process , 
								RingSubtractionMethod+"RingSubtractedTower" , 
								cms.EDProducer( "L1RingSubtractionProducer" , src = cms.InputTag("L1CaloTowerProducer") , RingSubtractionType = cms.string(RingSubtractionMethod) ) 
							)
 


	for TowerJetShape in TowerJetShapes:
		for TowerJetSize in TowerJetSizes:
	
			lReturnSequence += NewNamedModule(
									process , 
									"TowerJet"+TowerJetShape+str(TowerJetSize)+"FromL1CaloTower" , 
									cms.EDProducer( "L1TowerJetProducer" , src = cms.InputTag("L1CaloTowerProducer") , JetDiameter = cms.uint32( TowerJetSize ) , JetShape = cms.string( TowerJetShape ) ) 
								)

			for RingSubtractionMethod in RingSubtractionMethods:
				lReturnSequence += NewNamedModule(
										process , 
										"TowerJet"+TowerJetShape+str(TowerJetSize)+"From"+RingSubtractionMethod+"RingSubtractedTower" , 
										cms.EDProducer( "L1TowerJetProducer" , src = cms.InputTag( RingSubtractionMethod+"RingSubtractedTower" ) , JetDiameter = cms.uint32( TowerJetSize ) , JetShape = cms.string( TowerJetShape ) ) 
									)

	return lReturnSequence

