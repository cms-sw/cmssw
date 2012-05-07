import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.stringResolutionProvider_cfi import *

print "*** Including object resolutions derived from Fall11 MC for:"
print "*** - electrons   - muons   - udscJetsPF     - bJetsPF     - pfMET"
print "*** Please make sure that you are really using resolutions that are suited for the objects in your analysis!"

udscResolutionPF = stringResolution.clone(parametrization = 'EtEtaPhi',
                                          functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.087'),
    et  = cms.string('et * (sqrt(0.0591^2 + (1/sqrt(et))^2 + (0.891/et)^2))'),
    eta  = cms.string('sqrt(0.00915^2 + (1.51/et)^2)'),
    phi  = cms.string('sqrt(0.01^2 + (1.6/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.087<=abs(eta) && abs(eta)<0.174'),
    et  = cms.string('et * (sqrt(0.0619^2 + (0.975/sqrt(et))^2 + (1.54/et)^2))'),
    eta  = cms.string('sqrt(0.00887^2 + (1.53/et)^2)'),
    phi  = cms.string('sqrt(0.00982^2 + (1.61/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.174<=abs(eta) && abs(eta)<0.261'),
    et  = cms.string('et * (sqrt(0.0574^2 + (1/sqrt(et))^2 + (1.49e-05/et)^2))'),
    eta  = cms.string('sqrt(0.00865^2 + (1.54/et)^2)'),
    phi  = cms.string('sqrt(0.0101^2 + (1.59/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.261<=abs(eta) && abs(eta)<0.348'),
    et  = cms.string('et * (sqrt(0.0569^2 + (1.01/sqrt(et))^2 + (1.22e-07/et)^2))'),
    eta  = cms.string('sqrt(0.00867^2 + (1.55/et)^2)'),
    phi  = cms.string('sqrt(0.00988^2 + (1.6/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.348<=abs(eta) && abs(eta)<0.435'),
    et  = cms.string('et * (sqrt(0.057^2 + (1/sqrt(et))^2 + (2.17e-08/et)^2))'),
    eta  = cms.string('sqrt(0.00907^2 + (1.55/et)^2)'),
    phi  = cms.string('sqrt(0.0102^2 + (1.59/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.435<=abs(eta) && abs(eta)<0.522'),
    et  = cms.string('et * (sqrt(0.0522^2 + (1.02/sqrt(et))^2 + (2.64e-05/et)^2))'),
    eta  = cms.string('sqrt(0.00844^2 + (1.59/et)^2)'),
    phi  = cms.string('sqrt(0.00982^2 + (1.6/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.522<=abs(eta) && abs(eta)<0.609'),
    et  = cms.string('et * (sqrt(0.0502^2 + (1.02/sqrt(et))^2 + (2.6e-06/et)^2))'),
    eta  = cms.string('sqrt(0.00915^2 + (1.57/et)^2)'),
    phi  = cms.string('sqrt(0.00979^2 + (1.6/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.609<=abs(eta) && abs(eta)<0.696'),
    et  = cms.string('et * (sqrt(0.053^2 + (1.03/sqrt(et))^2 + (4.87e-07/et)^2))'),
    eta  = cms.string('sqrt(0.00856^2 + (1.58/et)^2)'),
    phi  = cms.string('sqrt(0.00925^2 + (1.62/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.696<=abs(eta) && abs(eta)<0.783'),
    et  = cms.string('et * (sqrt(0.051^2 + (1.03/sqrt(et))^2 + (7.53e-06/et)^2))'),
    eta  = cms.string('sqrt(0.00897^2 + (1.58/et)^2)'),
    phi  = cms.string('sqrt(0.00973^2 + (1.61/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.783<=abs(eta) && abs(eta)<0.870'),
    et  = cms.string('et * (sqrt(0.0549^2 + (1.04/sqrt(et))^2 + (5.62e-08/et)^2))'),
    eta  = cms.string('sqrt(0.0095^2 + (1.6/et)^2)'),
    phi  = cms.string('sqrt(0.00971^2 + (1.62/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.870<=abs(eta) && abs(eta)<0.957'),
    et  = cms.string('et * (sqrt(0.0544^2 + (1.06/sqrt(et))^2 + (1.07e-05/et)^2))'),
    eta  = cms.string('sqrt(0.00836^2 + (1.65/et)^2)'),
    phi  = cms.string('sqrt(0.00916^2 + (1.64/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.957<=abs(eta) && abs(eta)<1.044'),
    et  = cms.string('et * (sqrt(0.0519^2 + (1.09/sqrt(et))^2 + (8.43e-06/et)^2))'),
    eta  = cms.string('sqrt(0.00782^2 + (1.68/et)^2)'),
    phi  = cms.string('sqrt(0.00959^2 + (1.66/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.044<=abs(eta) && abs(eta)<1.131'),
    et  = cms.string('et * (sqrt(0.0539^2 + (1.12/sqrt(et))^2 + (1.97e-07/et)^2))'),
    eta  = cms.string('sqrt(0.0093^2 + (1.65/et)^2)'),
    phi  = cms.string('sqrt(0.00964^2 + (1.67/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.131<=abs(eta) && abs(eta)<1.218'),
    et  = cms.string('et * (sqrt(0.0492^2 + (1.16/sqrt(et))^2 + (1.37e-08/et)^2))'),
    eta  = cms.string('sqrt(0.00986^2 + (1.69/et)^2)'),
    phi  = cms.string('sqrt(0.00969^2 + (1.71/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.218<=abs(eta) && abs(eta)<1.305'),
    et  = cms.string('et * (sqrt(0.0489^2 + (1.18/sqrt(et))^2 + (3.44e-07/et)^2))'),
    eta  = cms.string('sqrt(0.0124^2 + (1.72/et)^2)'),
    phi  = cms.string('sqrt(0.00992^2 + (1.76/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.305<=abs(eta) && abs(eta)<1.392'),
    et  = cms.string('et * (sqrt(0.0414^2 + (1.25/sqrt(et))^2 + (1.98e-07/et)^2))'),
    eta  = cms.string('sqrt(0.0181^2 + (1.63/et)^2)'),
    phi  = cms.string('sqrt(0.0124^2 + (1.79/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.392<=abs(eta) && abs(eta)<1.479'),
    et  = cms.string('et * (sqrt(0.0373^2 + (1.26/sqrt(et))^2 + (5.4e-07/et)^2))'),
    eta  = cms.string('sqrt(0.0121^2 + (1.69/et)^2)'),
    phi  = cms.string('sqrt(0.0135^2 + (1.8/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.479<=abs(eta) && abs(eta)<1.566'),
    et  = cms.string('et * (sqrt(0.0125^2 + (1.24/sqrt(et))^2 + (1e-06/et)^2))'),
    eta  = cms.string('sqrt(0.0122^2 + (1.69/et)^2)'),
    phi  = cms.string('sqrt(0.0107^2 + (1.85/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.566<=abs(eta) && abs(eta)<1.653'),
    et  = cms.string('et * (sqrt(1.37e-07^2 + (1.08/sqrt(et))^2 + (3.06/et)^2))'),
    eta  = cms.string('sqrt(0.00975^2 + (1.69/et)^2)'),
    phi  = cms.string('sqrt(0.00895^2 + (1.84/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.653<=abs(eta) && abs(eta)<1.740'),
    et  = cms.string('et * (sqrt(2.37e-07^2 + (1.04/sqrt(et))^2 + (3.01/et)^2))'),
    eta  = cms.string('sqrt(0.00881^2 + (1.71/et)^2)'),
    phi  = cms.string('sqrt(0.00902^2 + (1.81/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.740<=abs(eta) && abs(eta)<1.830'),
    et  = cms.string('et * (sqrt(2.3e-07^2 + (1/sqrt(et))^2 + (3.1/et)^2))'),
    eta  = cms.string('sqrt(0.00938^2 + (1.75/et)^2)'),
    phi  = cms.string('sqrt(0.00861^2 + (1.79/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.830<=abs(eta) && abs(eta)<1.930'),
    et  = cms.string('et * (sqrt(1.25e-07^2 + (0.965/sqrt(et))^2 + (3.14/et)^2))'),
    eta  = cms.string('sqrt(0.00894^2 + (1.8/et)^2)'),
    phi  = cms.string('sqrt(0.00877^2 + (1.75/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.930<=abs(eta) && abs(eta)<2.043'),
    et  = cms.string('et * (sqrt(5.78e-08^2 + (0.924/sqrt(et))^2 + (3.14/et)^2))'),
    eta  = cms.string('sqrt(0.00893^2 + (1.83/et)^2)'),
    phi  = cms.string('sqrt(0.00791^2 + (1.73/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.043<=abs(eta) && abs(eta)<2.172'),
    et  = cms.string('et * (sqrt(4.25e-08^2 + (0.923/sqrt(et))^2 + (2.85/et)^2))'),
    eta  = cms.string('sqrt(0.0099^2 + (1.82/et)^2)'),
    phi  = cms.string('sqrt(0.00775^2 + (1.73/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.172<=abs(eta) && abs(eta)<2.322'),
    et  = cms.string('et * (sqrt(0.00601^2 + (0.881/sqrt(et))^2 + (3.23/et)^2))'),
    eta  = cms.string('sqrt(0.00944^2 + (1.8/et)^2)'),
    phi  = cms.string('sqrt(0.00807^2 + (1.71/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.322<=abs(eta) && abs(eta)<2.500'),
    et  = cms.string('et * (sqrt(4.94e-08^2 + (0.86/sqrt(et))^2 + (3.56/et)^2))'),
    eta  = cms.string('sqrt(0.0103^2 + (2.15/et)^2)'),
    phi  = cms.string('sqrt(0.0103^2 + (1.81/et)^2)'),
    ),
    ),
                                          constraints = cms.vdouble(0)
                                          )

bjetResolutionPF = stringResolution.clone(parametrization = 'EtEtaPhi',
                                          functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.087'),
    et  = cms.string('et * (sqrt(0.0686^2 + (1.03/sqrt(et))^2 + (1.68/et)^2))'),
    eta  = cms.string('sqrt(0.00605^2 + (1.63/et)^2)'),
    phi  = cms.string('sqrt(0.00787^2 + (1.74/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.087<=abs(eta) && abs(eta)<0.174'),
    et  = cms.string('et * (sqrt(0.0737^2 + (1.01/sqrt(et))^2 + (1.74/et)^2))'),
    eta  = cms.string('sqrt(0.00592^2 + (1.64/et)^2)'),
    phi  = cms.string('sqrt(0.00766^2 + (1.74/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.174<=abs(eta) && abs(eta)<0.261'),
    et  = cms.string('et * (sqrt(0.0657^2 + (1.07/sqrt(et))^2 + (5.16e-06/et)^2))'),
    eta  = cms.string('sqrt(0.00584^2 + (1.65/et)^2)'),
    phi  = cms.string('sqrt(0.00755^2 + (1.74/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.261<=abs(eta) && abs(eta)<0.348'),
    et  = cms.string('et * (sqrt(0.062^2 + (1.07/sqrt(et))^2 + (0.000134/et)^2))'),
    eta  = cms.string('sqrt(0.00593^2 + (1.65/et)^2)'),
    phi  = cms.string('sqrt(0.00734^2 + (1.74/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.348<=abs(eta) && abs(eta)<0.435'),
    et  = cms.string('et * (sqrt(0.0605^2 + (1.07/sqrt(et))^2 + (1.84e-07/et)^2))'),
    eta  = cms.string('sqrt(0.00584^2 + (1.68/et)^2)'),
    phi  = cms.string('sqrt(0.00734^2 + (1.75/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.435<=abs(eta) && abs(eta)<0.522'),
    et  = cms.string('et * (sqrt(0.059^2 + (1.08/sqrt(et))^2 + (9.06e-09/et)^2))'),
    eta  = cms.string('sqrt(0.00646^2 + (1.67/et)^2)'),
    phi  = cms.string('sqrt(0.00767^2 + (1.74/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.522<=abs(eta) && abs(eta)<0.609'),
    et  = cms.string('et * (sqrt(0.0577^2 + (1.08/sqrt(et))^2 + (5.46e-06/et)^2))'),
    eta  = cms.string('sqrt(0.00661^2 + (1.67/et)^2)'),
    phi  = cms.string('sqrt(0.00742^2 + (1.75/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.609<=abs(eta) && abs(eta)<0.696'),
    et  = cms.string('et * (sqrt(0.0525^2 + (1.09/sqrt(et))^2 + (4.05e-05/et)^2))'),
    eta  = cms.string('sqrt(0.00724^2 + (1.65/et)^2)'),
    phi  = cms.string('sqrt(0.00771^2 + (1.73/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.696<=abs(eta) && abs(eta)<0.783'),
    et  = cms.string('et * (sqrt(0.0582^2 + (1.09/sqrt(et))^2 + (1.17e-05/et)^2))'),
    eta  = cms.string('sqrt(0.00763^2 + (1.67/et)^2)'),
    phi  = cms.string('sqrt(0.00758^2 + (1.76/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.783<=abs(eta) && abs(eta)<0.870'),
    et  = cms.string('et * (sqrt(0.0649^2 + (1.08/sqrt(et))^2 + (7.85e-06/et)^2))'),
    eta  = cms.string('sqrt(0.00746^2 + (1.7/et)^2)'),
    phi  = cms.string('sqrt(0.00789^2 + (1.75/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.870<=abs(eta) && abs(eta)<0.957'),
    et  = cms.string('et * (sqrt(0.0654^2 + (1.1/sqrt(et))^2 + (1.09e-07/et)^2))'),
    eta  = cms.string('sqrt(0.00807^2 + (1.7/et)^2)'),
    phi  = cms.string('sqrt(0.00802^2 + (1.76/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.957<=abs(eta) && abs(eta)<1.044'),
    et  = cms.string('et * (sqrt(0.0669^2 + (1.11/sqrt(et))^2 + (1.87e-06/et)^2))'),
    eta  = cms.string('sqrt(0.00843^2 + (1.72/et)^2)'),
    phi  = cms.string('sqrt(0.0078^2 + (1.79/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.044<=abs(eta) && abs(eta)<1.131'),
    et  = cms.string('et * (sqrt(0.0643^2 + (1.15/sqrt(et))^2 + (2.76e-05/et)^2))'),
    eta  = cms.string('sqrt(0.00886^2 + (1.74/et)^2)'),
    phi  = cms.string('sqrt(0.00806^2 + (1.82/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.131<=abs(eta) && abs(eta)<1.218'),
    et  = cms.string('et * (sqrt(0.0645^2 + (1.16/sqrt(et))^2 + (1.04e-06/et)^2))'),
    eta  = cms.string('sqrt(0.0101^2 + (1.76/et)^2)'),
    phi  = cms.string('sqrt(0.00784^2 + (1.86/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.218<=abs(eta) && abs(eta)<1.305'),
    et  = cms.string('et * (sqrt(0.0637^2 + (1.19/sqrt(et))^2 + (1.08e-07/et)^2))'),
    eta  = cms.string('sqrt(0.0127^2 + (1.78/et)^2)'),
    phi  = cms.string('sqrt(0.00885^2 + (1.9/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.305<=abs(eta) && abs(eta)<1.392'),
    et  = cms.string('et * (sqrt(0.0695^2 + (1.21/sqrt(et))^2 + (5.75e-06/et)^2))'),
    eta  = cms.string('sqrt(0.0161^2 + (1.73/et)^2)'),
    phi  = cms.string('sqrt(0.0108^2 + (1.93/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.392<=abs(eta) && abs(eta)<1.479'),
    et  = cms.string('et * (sqrt(0.0748^2 + (1.2/sqrt(et))^2 + (5.15e-08/et)^2))'),
    eta  = cms.string('sqrt(0.0122^2 + (1.77/et)^2)'),
    phi  = cms.string('sqrt(0.0112^2 + (2/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.479<=abs(eta) && abs(eta)<1.566'),
    et  = cms.string('et * (sqrt(0.0624^2 + (1.23/sqrt(et))^2 + (2.28e-05/et)^2))'),
    eta  = cms.string('sqrt(0.0123^2 + (1.79/et)^2)'),
    phi  = cms.string('sqrt(0.0102^2 + (2.02/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.566<=abs(eta) && abs(eta)<1.653'),
    et  = cms.string('et * (sqrt(0.0283^2 + (1.25/sqrt(et))^2 + (4.79e-07/et)^2))'),
    eta  = cms.string('sqrt(0.0111^2 + (1.79/et)^2)'),
    phi  = cms.string('sqrt(0.00857^2 + (2.01/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.653<=abs(eta) && abs(eta)<1.740'),
    et  = cms.string('et * (sqrt(0.0316^2 + (1.21/sqrt(et))^2 + (5e-05/et)^2))'),
    eta  = cms.string('sqrt(0.0106^2 + (1.8/et)^2)'),
    phi  = cms.string('sqrt(0.00856^2 + (1.97/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.740<=abs(eta) && abs(eta)<1.830'),
    et  = cms.string('et * (sqrt(2.29e-07^2 + (1.2/sqrt(et))^2 + (1.71e-05/et)^2))'),
    eta  = cms.string('sqrt(0.0115^2 + (1.83/et)^2)'),
    phi  = cms.string('sqrt(0.00761^2 + (1.95/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.830<=abs(eta) && abs(eta)<1.930'),
    et  = cms.string('et * (sqrt(5.18e-09^2 + (1.14/sqrt(et))^2 + (1.7/et)^2))'),
    eta  = cms.string('sqrt(0.012^2 + (1.88/et)^2)'),
    phi  = cms.string('sqrt(0.00721^2 + (1.92/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.930<=abs(eta) && abs(eta)<2.043'),
    et  = cms.string('et * (sqrt(2.17e-07^2 + (1.09/sqrt(et))^2 + (2.08/et)^2))'),
    eta  = cms.string('sqrt(0.0131^2 + (1.91/et)^2)'),
    phi  = cms.string('sqrt(0.00722^2 + (1.86/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.043<=abs(eta) && abs(eta)<2.172'),
    et  = cms.string('et * (sqrt(3.65e-07^2 + (1.09/sqrt(et))^2 + (1.63/et)^2))'),
    eta  = cms.string('sqrt(0.0134^2 + (1.92/et)^2)'),
    phi  = cms.string('sqrt(0.00703^2 + (1.86/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.172<=abs(eta) && abs(eta)<2.322'),
    et  = cms.string('et * (sqrt(2.02e-07^2 + (1.09/sqrt(et))^2 + (1.68/et)^2))'),
    eta  = cms.string('sqrt(0.0132^2 + (1.89/et)^2)'),
    phi  = cms.string('sqrt(0.00845^2 + (1.86/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.322<=abs(eta) && abs(eta)<2.500'),
    et  = cms.string('et * (sqrt(5.27e-07^2 + (1.12/sqrt(et))^2 + (1.78/et)^2))'),
    eta  = cms.string('sqrt(0.0121^2 + (2.28/et)^2)'),
    phi  = cms.string('sqrt(0.00975^2 + (2/et)^2)'),
    ),
    ),
                                          constraints = cms.vdouble(0)
                                          )

muonResolution   = stringResolution.clone(parametrization = 'EtEtaPhi',
                                          functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.100'),
    et  = cms.string('et * (0.00517 + 0.000143 * et)'),
    eta  = cms.string('sqrt(0.000433^2 + (0.000161/sqrt(et))^2 + (0.00334/et)^2)'),
    phi  = cms.string('sqrt(7.21e-05^2 + (7e-05/sqrt(et))^2 + (0.00296/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.100<=abs(eta) && abs(eta)<0.200'),
    et  = cms.string('et * (0.00524 + 0.000143 * et)'),
    eta  = cms.string('sqrt(0.000381^2 + (0.000473/sqrt(et))^2 + (0.00259/et)^2)'),
    phi  = cms.string('sqrt(6.79e-05^2 + (0.000245/sqrt(et))^2 + (0.00274/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.200<=abs(eta) && abs(eta)<0.300'),
    et  = cms.string('et * (0.00585 + 0.000138 * et)'),
    eta  = cms.string('sqrt(0.000337^2 + (0.000381/sqrt(et))^2 + (0.0023/et)^2)'),
    phi  = cms.string('sqrt(7.08e-05^2 + (6.75e-05/sqrt(et))^2 + (0.00307/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.300<=abs(eta) && abs(eta)<0.400'),
    et  = cms.string('et * (0.0065 + 0.000133 * et)'),
    eta  = cms.string('sqrt(0.000308^2 + (0.000166/sqrt(et))^2 + (0.00249/et)^2)'),
    phi  = cms.string('sqrt(6.59e-05^2 + (0.000301/sqrt(et))^2 + (0.00281/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.400<=abs(eta) && abs(eta)<0.500'),
    et  = cms.string('et * (0.0071 + 0.000129 * et)'),
    eta  = cms.string('sqrt(0.000289^2 + (5.37e-09/sqrt(et))^2 + (0.00243/et)^2)'),
    phi  = cms.string('sqrt(6.27e-05^2 + (0.000359/sqrt(et))^2 + (0.00278/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.500<=abs(eta) && abs(eta)<0.600'),
    et  = cms.string('et * (0.00721 + 0.00013 * et)'),
    eta  = cms.string('sqrt(0.000279^2 + (0.000272/sqrt(et))^2 + (0.0026/et)^2)'),
    phi  = cms.string('sqrt(6.46e-05^2 + (0.00036/sqrt(et))^2 + (0.00285/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.600<=abs(eta) && abs(eta)<0.700'),
    et  = cms.string('et * (0.00757 + 0.000129 * et)'),
    eta  = cms.string('sqrt(0.000282^2 + (3.63e-10/sqrt(et))^2 + (0.00288/et)^2)'),
    phi  = cms.string('sqrt(6.54e-05^2 + (0.000348/sqrt(et))^2 + (0.00301/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.700<=abs(eta) && abs(eta)<0.800'),
    et  = cms.string('et * (0.0081 + 0.000127 * et)'),
    eta  = cms.string('sqrt(0.000265^2 + (0.000609/sqrt(et))^2 + (0.00212/et)^2)'),
    phi  = cms.string('sqrt(6.2e-05^2 + (0.000402/sqrt(et))^2 + (0.00304/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.800<=abs(eta) && abs(eta)<0.900'),
    et  = cms.string('et * (0.00916 + 0.000131 * et)'),
    eta  = cms.string('sqrt(0.000241^2 + (0.000678/sqrt(et))^2 + (0.00221/et)^2)'),
    phi  = cms.string('sqrt(6.26e-05^2 + (0.000458/sqrt(et))^2 + (0.0031/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.900<=abs(eta) && abs(eta)<1.000'),
    et  = cms.string('et * (0.0108 + 0.000151 * et)'),
    eta  = cms.string('sqrt(0.000228^2 + (0.000612/sqrt(et))^2 + (0.00245/et)^2)'),
    phi  = cms.string('sqrt(7.18e-05^2 + (0.000469/sqrt(et))^2 + (0.00331/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.000<=abs(eta) && abs(eta)<1.100'),
    et  = cms.string('et * (0.0115 + 0.000153 * et)'),
    eta  = cms.string('sqrt(0.000217^2 + (0.000583/sqrt(et))^2 + (0.00307/et)^2)'),
    phi  = cms.string('sqrt(6.98e-05^2 + (0.000507/sqrt(et))^2 + (0.00338/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.100<=abs(eta) && abs(eta)<1.200'),
    et  = cms.string('et * (0.013 + 0.000136 * et)'),
    eta  = cms.string('sqrt(0.000195^2 + (0.000751/sqrt(et))^2 + (0.00282/et)^2)'),
    phi  = cms.string('sqrt(6.21e-05^2 + (0.000584/sqrt(et))^2 + (0.00345/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.200<=abs(eta) && abs(eta)<1.300'),
    et  = cms.string('et * (0.0144 + 0.000131 * et)'),
    eta  = cms.string('sqrt(0.000183^2 + (0.000838/sqrt(et))^2 + (0.00227/et)^2)'),
    phi  = cms.string('sqrt(5.37e-05^2 + (0.000667/sqrt(et))^2 + (0.00352/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.300<=abs(eta) && abs(eta)<1.400'),
    et  = cms.string('et * (0.0149 + 0.000141 * et)'),
    eta  = cms.string('sqrt(0.000196^2 + (0.000783/sqrt(et))^2 + (0.00274/et)^2)'),
    phi  = cms.string('sqrt(5.37e-05^2 + (0.000711/sqrt(et))^2 + (0.00358/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.400<=abs(eta) && abs(eta)<1.500'),
    et  = cms.string('et * (0.014 + 0.000155 * et)'),
    eta  = cms.string('sqrt(0.0002^2 + (0.000832/sqrt(et))^2 + (0.00254/et)^2)'),
    phi  = cms.string('sqrt(5.98e-05^2 + (0.000713/sqrt(et))^2 + (0.00362/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.500<=abs(eta) && abs(eta)<1.600'),
    et  = cms.string('et * (0.0132 + 0.000169 * et)'),
    eta  = cms.string('sqrt(0.000205^2 + (0.0007/sqrt(et))^2 + (0.00304/et)^2)'),
    phi  = cms.string('sqrt(6.21e-05^2 + (0.000781/sqrt(et))^2 + (0.00348/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.600<=abs(eta) && abs(eta)<1.700'),
    et  = cms.string('et * (0.0129 + 0.0002 * et)'),
    eta  = cms.string('sqrt(0.000214^2 + (0.000747/sqrt(et))^2 + (0.00319/et)^2)'),
    phi  = cms.string('sqrt(6.92e-05^2 + (0.000865/sqrt(et))^2 + (0.00337/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.700<=abs(eta) && abs(eta)<1.800'),
    et  = cms.string('et * (0.0135 + 0.000264 * et)'),
    eta  = cms.string('sqrt(0.000238^2 + (0.000582/sqrt(et))^2 + (0.00343/et)^2)'),
    phi  = cms.string('sqrt(9.13e-05^2 + (0.000896/sqrt(et))^2 + (0.00348/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.800<=abs(eta) && abs(eta)<1.900'),
    et  = cms.string('et * (0.0144 + 0.00034 * et)'),
    eta  = cms.string('sqrt(0.000263^2 + (0.000721/sqrt(et))^2 + (0.00322/et)^2)'),
    phi  = cms.string('sqrt(0.000102^2 + (0.000994/sqrt(et))^2 + (0.00337/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.900<=abs(eta) && abs(eta)<2.000'),
    et  = cms.string('et * (0.0147 + 0.000441 * et)'),
    eta  = cms.string('sqrt(0.000284^2 + (0.000779/sqrt(et))^2 + (0.0031/et)^2)'),
    phi  = cms.string('sqrt(0.000123^2 + (0.00108/sqrt(et))^2 + (0.00315/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.000<=abs(eta) && abs(eta)<2.100'),
    et  = cms.string('et * (0.0154 + 0.000604 * et)'),
    eta  = cms.string('sqrt(0.000316^2 + (0.000566/sqrt(et))^2 + (0.00384/et)^2)'),
    phi  = cms.string('sqrt(0.000169^2 + (0.000947/sqrt(et))^2 + (0.00422/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.100<=abs(eta) && abs(eta)<2.200'),
    et  = cms.string('et * (0.0163 + 0.000764 * et)'),
    eta  = cms.string('sqrt(0.000353^2 + (0.000749/sqrt(et))^2 + (0.0038/et)^2)'),
    phi  = cms.string('sqrt(0.000176^2 + (0.00116/sqrt(et))^2 + (0.00423/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.200<=abs(eta) && abs(eta)<2.300'),
    et  = cms.string('et * (0.0173 + 0.000951 * et)'),
    eta  = cms.string('sqrt(0.000412^2 + (0.00102/sqrt(et))^2 + (0.00351/et)^2)'),
    phi  = cms.string('sqrt(0.000207^2 + (0.00115/sqrt(et))^2 + (0.00469/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.300<=abs(eta) && abs(eta)<2.400'),
    et  = cms.string('et * (0.0175 + 0.00126 * et)'),
    eta  = cms.string('sqrt(0.000506^2 + (0.000791/sqrt(et))^2 + (0.0045/et)^2)'),
    phi  = cms.string('sqrt(0.00027^2 + (0.00113/sqrt(et))^2 + (0.00528/et)^2)'),
    ),
    ),
                                          constraints = cms.vdouble(0)
                                          )

elecResolution   = stringResolution.clone(parametrization = 'EtEtaPhi',
                                          functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.174'),
    et  = cms.string('et * (sqrt(0.00534^2 + (0.079/sqrt(et))^2 + (0.163/et)^2))'),
    eta  = cms.string('sqrt(0.000452^2 + (0.000285/sqrt(et))^2 + (0.00376/et)^2)'),
    phi  = cms.string('sqrt(0.000101^2 + (0.0011/sqrt(et))^2 + (0.00346/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.174<=abs(eta) && abs(eta)<0.261'),
    et  = cms.string('et * (sqrt(0.00518^2 + (0.0749/sqrt(et))^2 + (0.227/et)^2))'),
    eta  = cms.string('sqrt(0.00038^2 + (0.000571/sqrt(et))^2 + (0.00276/et)^2)'),
    phi  = cms.string('sqrt(9.3e-05^2 + (0.00115/sqrt(et))^2 + (0.0035/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.261<=abs(eta) && abs(eta)<0.348'),
    et  = cms.string('et * (sqrt(0.00332^2 + (0.0879/sqrt(et))^2 + (0.12/et)^2))'),
    eta  = cms.string('sqrt(0.000351^2 + (1.36e-09/sqrt(et))^2 + (0.00324/et)^2)'),
    phi  = cms.string('sqrt(0.000103^2 + (0.00117/sqrt(et))^2 + (0.00333/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.348<=abs(eta) && abs(eta)<0.435'),
    et  = cms.string('et * (sqrt(0.00445^2 + (0.0895/sqrt(et))^2 + (0.186/et)^2))'),
    eta  = cms.string('sqrt(0.000319^2 + (0.00061/sqrt(et))^2 + (0.00182/et)^2)'),
    phi  = cms.string('sqrt(0.00011^2 + (0.00115/sqrt(et))^2 + (0.00365/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.435<=abs(eta) && abs(eta)<0.522'),
    et  = cms.string('et * (sqrt(0.00453^2 + (0.0893/sqrt(et))^2 + (0.21/et)^2))'),
    eta  = cms.string('sqrt(0.000301^2 + (0.000612/sqrt(et))^2 + (0.00146/et)^2)'),
    phi  = cms.string('sqrt(0.000105^2 + (0.00122/sqrt(et))^2 + (0.00343/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.522<=abs(eta) && abs(eta)<0.609'),
    et  = cms.string('et * (sqrt(0.00308^2 + (0.0886/sqrt(et))^2 + (0.188/et)^2))'),
    eta  = cms.string('sqrt(0.000297^2 + (0.000791/sqrt(et))^2 + (2.09e-08/et)^2)'),
    phi  = cms.string('sqrt(0.000102^2 + (0.00129/sqrt(et))^2 + (0.00328/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.609<=abs(eta) && abs(eta)<0.696'),
    et  = cms.string('et * (sqrt(0.00308^2 + (0.0914/sqrt(et))^2 + (0.182/et)^2))'),
    eta  = cms.string('sqrt(0.00032^2 + (0.000329/sqrt(et))^2 + (0.00325/et)^2)'),
    phi  = cms.string('sqrt(0.000103^2 + (0.00139/sqrt(et))^2 + (0.00253/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.696<=abs(eta) && abs(eta)<0.783'),
    et  = cms.string('et * (sqrt(0.00442^2 + (0.0914/sqrt(et))^2 + (0.231/et)^2))'),
    eta  = cms.string('sqrt(0.000309^2 + (0.000821/sqrt(et))^2 + (0.00119/et)^2)'),
    phi  = cms.string('sqrt(0.000115^2 + (0.00139/sqrt(et))^2 + (0.00293/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.783<=abs(eta) && abs(eta)<0.870'),
    et  = cms.string('et * (sqrt(0.00455^2 + (0.0949/sqrt(et))^2 + (0.335/et)^2))'),
    eta  = cms.string('sqrt(0.000293^2 + (0.000767/sqrt(et))^2 + (0.00211/et)^2)'),
    phi  = cms.string('sqrt(0.000121^2 + (0.00158/sqrt(et))^2 + (0.00151/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.870<=abs(eta) && abs(eta)<0.957'),
    et  = cms.string('et * (sqrt(0.00181^2 + (0.102/sqrt(et))^2 + (0.333/et)^2))'),
    eta  = cms.string('sqrt(0.000275^2 + (0.000765/sqrt(et))^2 + (0.00227/et)^2)'),
    phi  = cms.string('sqrt(0.000128^2 + (0.00169/sqrt(et))^2 + (1.93e-08/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.957<=abs(eta) && abs(eta)<1.044'),
    et  = cms.string('et * (sqrt(0.000764^2 + (0.108/sqrt(et))^2 + (0.42/et)^2))'),
    eta  = cms.string('sqrt(0.000274^2 + (0.000622/sqrt(et))^2 + (0.00299/et)^2)'),
    phi  = cms.string('sqrt(0.000145^2 + (0.00179/sqrt(et))^2 + (1.69e-08/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.044<=abs(eta) && abs(eta)<1.131'),
    et  = cms.string('et * (sqrt(0.00114^2 + (0.128/sqrt(et))^2 + (0.55/et)^2))'),
    eta  = cms.string('sqrt(0.000269^2 + (0.000929/sqrt(et))^2 + (0.00183/et)^2)'),
    phi  = cms.string('sqrt(0.000185^2 + (0.00182/sqrt(et))^2 + (2.99e-09/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.131<=abs(eta) && abs(eta)<1.218'),
    et  = cms.string('et * (sqrt(4.14e-09^2 + (0.155/sqrt(et))^2 + (0.674/et)^2))'),
    eta  = cms.string('sqrt(0.000268^2 + (0.000876/sqrt(et))^2 + (0.00234/et)^2)'),
    phi  = cms.string('sqrt(0.000194^2 + (0.002/sqrt(et))^2 + (2.39e-08/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.218<=abs(eta) && abs(eta)<1.305'),
    et  = cms.string('et * (sqrt(8.03e-09^2 + (0.144/sqrt(et))^2 + (0.8/et)^2))'),
    eta  = cms.string('sqrt(0.000258^2 + (0.000782/sqrt(et))^2 + (0.00246/et)^2)'),
    phi  = cms.string('sqrt(0.000226^2 + (0.00206/sqrt(et))^2 + (5.88e-08/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.305<=abs(eta) && abs(eta)<1.392'),
    et  = cms.string('et * (sqrt(0.00842^2 + (0.118/sqrt(et))^2 + (0.951/et)^2))'),
    eta  = cms.string('sqrt(0.000269^2 + (0.000817/sqrt(et))^2 + (0.00278/et)^2)'),
    phi  = cms.string('sqrt(0.000247^2 + (0.00225/sqrt(et))^2 + (1.47e-09/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.392<=abs(eta) && abs(eta)<1.479'),
    et  = cms.string('et * (sqrt(0.00684^2 + (0.144/sqrt(et))^2 + (0.892/et)^2))'),
    eta  = cms.string('sqrt(0.000267^2 + (0.000734/sqrt(et))^2 + (0.00327/et)^2)'),
    phi  = cms.string('sqrt(0.000234^2 + (0.00233/sqrt(et))^2 + (4.92e-09/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.479<=abs(eta) && abs(eta)<1.653'),
    et  = cms.string('et * (sqrt(0.0245^2 + (0.196/sqrt(et))^2 + (0.555/et)^2))'),
    eta  = cms.string('sqrt(0.000268^2 + (0.000757/sqrt(et))^2 + (0.00295/et)^2)'),
    phi  = cms.string('sqrt(0.00025^2 + (0.00268/sqrt(et))^2 + (7.5e-09/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.653<=abs(eta) && abs(eta)<1.740'),
    et  = cms.string('et * (sqrt(0.0174^2 + (0.127/sqrt(et))^2 + (0.894/et)^2))'),
    eta  = cms.string('sqrt(0.000274^2 + (1.77e-09/sqrt(et))^2 + (0.00435/et)^2)'),
    phi  = cms.string('sqrt(0.000284^2 + (0.00275/sqrt(et))^2 + (6.56e-09/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.740<=abs(eta) && abs(eta)<1.830'),
    et  = cms.string('et * (sqrt(0.0144^2 + (0.133/sqrt(et))^2 + (0.708/et)^2))'),
    eta  = cms.string('sqrt(0.000274^2 + (0.00101/sqrt(et))^2 + (0.000982/et)^2)'),
    phi  = cms.string('sqrt(0.000356^2 + (0.00279/sqrt(et))^2 + (0.00261/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.830<=abs(eta) && abs(eta)<1.930'),
    et  = cms.string('et * (sqrt(0.0149^2 + (0.126/sqrt(et))^2 + (0.596/et)^2))'),
    eta  = cms.string('sqrt(0.000299^2 + (0.000686/sqrt(et))^2 + (0.00341/et)^2)'),
    phi  = cms.string('sqrt(0.000347^2 + (0.00298/sqrt(et))^2 + (1.02e-08/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.930<=abs(eta) && abs(eta)<2.043'),
    et  = cms.string('et * (sqrt(0.0143^2 + (0.12/sqrt(et))^2 + (0.504/et)^2))'),
    eta  = cms.string('sqrt(0.000329^2 + (3.05e-10/sqrt(et))^2 + (0.00439/et)^2)'),
    phi  = cms.string('sqrt(0.000302^2 + (0.00322/sqrt(et))^2 + (5.22e-08/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.043<=abs(eta) && abs(eta)<2.172'),
    et  = cms.string('et * (sqrt(0.0162^2 + (0.0965/sqrt(et))^2 + (0.483/et)^2))'),
    eta  = cms.string('sqrt(0.00037^2 + (1.32e-08/sqrt(et))^2 + (0.00447/et)^2)'),
    phi  = cms.string('sqrt(0.000287^2 + (0.00349/sqrt(et))^2 + (3e-11/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.172<=abs(eta) && abs(eta)<2.322'),
    et  = cms.string('et * (sqrt(0.0122^2 + (0.13/sqrt(et))^2 + (0.207/et)^2))'),
    eta  = cms.string('sqrt(0.000442^2 + (4.03e-10/sqrt(et))^2 + (0.00544/et)^2)'),
    phi  = cms.string('sqrt(0.000214^2 + (0.00436/sqrt(et))^2 + (2.98e-09/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.322<=abs(eta) && abs(eta)<2.500'),
    et  = cms.string('et * (sqrt(0.0145^2 + (0.127/sqrt(et))^2 + (0.0782/et)^2))'),
    eta  = cms.string('sqrt(0.000577^2 + (0.000768/sqrt(et))^2 + (0.00331/et)^2)'),
    phi  = cms.string('sqrt(8.02e-05^2 + (0.00525/sqrt(et))^2 + (0.00581/et)^2)'),
    ),
    ),
                                          constraints = cms.vdouble(0)
                                          )

metResolutionPF  = stringResolution.clone(parametrization = 'EtEtaPhi',
                                          functions = cms.VPSet(
    cms.PSet(
    et  = cms.string('et * (sqrt(0.0337^2 + (0.888/sqrt(et))^2 + (19.6/et)^2))'),
    eta  = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (0/et)^2)'),
    phi  = cms.string('sqrt(1.28e-08^2 + (1.45/sqrt(et))^2 + (1.03/et)^2)'),
    ),
    ),
                                          constraints = cms.vdouble(0)
                                          )
