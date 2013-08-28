import FWCore.ParameterSet.Config as cms

#function to switch to an external weight producer for an analyzer in the validation chain
def useExternalWeight(process, analyzerName, vWeightProducerTags):
  analyzer = getattr(process,analyzerName)
  if analyzer != None:
    print "Changing weight in "+analyzerName
    analyzer.UseWeightFromHepMC = cms.bool(False)
    analyzer.genEventInfos = vWeightProducerTags
    setattr(process, analyzerName, analyzer)

#function to switch to an external weight producer for all analyzers in the validation chain
def useExternalWeightForValidation(process, vWeightProducerTags):
  useExternalWeight(process, "basicGenParticleValidation", vWeightProducerTags)
  useExternalWeight(process, "mbueAndqcdValidation", vWeightProducerTags)
  useExternalWeight(process, "basicHepMCValidation", vWeightProducerTags)
  useExternalWeight(process, "drellYanEleValidation", vWeightProducerTags)
  useExternalWeight(process, "drellYanMuoValidation", vWeightProducerTags)
  useExternalWeight(process, "wMinusEleValidation", vWeightProducerTags)
  useExternalWeight(process, "wPlusEleValidation", vWeightProducerTags)
  useExternalWeight(process, "wMinusMuoValidation", vWeightProducerTags)
  useExternalWeight(process, "wPlusMuoValidation", vWeightProducerTags)
  useExternalWeight(process, "tauValidation", vWeightProducerTags)
  useExternalWeight(process, "duplicationChecker", vWeightProducerTags)
 
#function to switch to an alternative gen source (default is "generator") for an analyzer in the validation chain
def switchGenSource(process, analyzerName, source):
  analyzer = getattr(process,analyzerName)
  if analyzer != None:
    print "Changing inputSource in "+analyzerName
    analyzer.hepmcCollection = source
    setattr(process, analyzerName, analyzer)

#function to switch to an alternative gen source (default is "generator") for all analyzers in the validation chain
def switchGenSourceForValidation(process, source):
  process.genParticles.src = 'lhe2HepMCConverter'
  switchGenSource(process, "basicGenParticleValidation", source)
  switchGenSource(process, "mbueAndqcdValidation", source)
  switchGenSource(process, "basicHepMCValidation", source)
  switchGenSource(process, "drellYanEleValidation", source)
  switchGenSource(process, "drellYanMuoValidation", source)
  switchGenSource(process, "wMinusEleValidation", source)
  switchGenSource(process, "wPlusEleValidation", source)
  switchGenSource(process, "wMinusMuoValidation", source)
  switchGenSource(process, "wPlusMuoValidation", source)
  switchGenSource(process, "tauValidation", source)
  switchGenSource(process, "duplicationChecker", source)    
