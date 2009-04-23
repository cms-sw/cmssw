import FWCore.ParameterSet.Config as cms

#
# pat tuple extra sequence for rec hit egamma
#

## modules to make the default ValueMaps for the electron
from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDepsModules_cff import eleIsoFromDepsEcalFromHits
## modules to make the default ValueMaps for the photon
from RecoEgamma.EgammaIsolationAlgos.gamIsoFromDepsModules_cff import gamIsoFromDepsEcalFromHits

## change the default vetos
eleIsoFromDepsEcalFromHits.deposits[0].vetos = cms.vstring('EcalBarrel:0.045' ,
                                               'EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)',
                                               'EcalBarrel:ThresholdFromTransverse(0.08)',
                                               'EcalEndcaps:ThresholdFromTransverse(0.3)',
                                               'EcalEndcaps:0.070',
                                               'EcalEndcaps:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)'
                                               )
gamIsoFromDepsEcalFromHits.deposits[0].vetos = cms.vstring('EcalBarrel:0.045',
                                               'EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)',
                                               'EcalBarrel:ThresholdFromTransverse(0.08)',
                                               'EcalEndcaps:ThresholdFromTransverse(0.3)',
                                               'EcalEndcaps:0.070',
                                               'EcalEndcaps:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)'
                                               )

## std sequence for isoDeposits from rec hits
isoDepositsFromRecHits = cms.Sequence(eleIsoFromDepsEcalFromHits +
                                      gamIsoFromDepsEcalFromHits
                                      )
