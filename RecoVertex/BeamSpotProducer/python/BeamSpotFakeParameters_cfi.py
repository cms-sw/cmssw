
import FWCore.ParameterSet.Config as cms

BeamSpotFakeConditions = cms.ESSource("BeamSpotFakeConditions",

 getDataFromFile = cms.bool(False), # set to true if you want to read text file
 InputFilename = cms.FileInPath("RecoVertex/BeamSpotProducer/test/EarlyCollision.txt"), # beam spot results
 # units are in cm
 X0 = cms.double(0.246815),
 Y0 = cms.double(0.398387),
 Z0 = cms.double(-0.617015),
 dxdz = cms.double(5.37945e-05),
 dydz = cms.double(-6.85109e-05),
 sigmaZ = cms.double(3.84749),
 widthX = cms.double(0.0293),
 widthY = cms.double(0.0293),
 emittanceX = cms.double(0.),
 emittanceY = cms.double(0.),
 betaStar = cms.double(0.),
 errorX0 = cms.double(0.0012087),
 errorY0 = cms.double(0.00163803),
 errorZ0 = cms.double(0.309234),
 errordxdz = cms.double(0.000307662),
 errordydz = cms.double(0.000408403),
 errorSigmaZ = cms.double(0.251521 ),
 errorWidth = cms.double(0.01)

)
