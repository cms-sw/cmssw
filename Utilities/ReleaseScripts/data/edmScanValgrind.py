#!/bin/env python
from __future__ import print_function
import sys

fileName = sys.argv[1]

f = open(fileName,'r')

messageToIgnore = [
    'edm::FunctorTask'
    ,'edm::FunctorWaitingTask'
    ,'edm::ModuleCallingContext::setContext'
    ,'edm::SerialTaskQueue::'
    ,'edm::SerialTaskQueueChain::'
    ,'edm::WaitingTaskList'
    ,'edm::Worker::RunModuleTask<'
    ,'edm::Worker::beginStream'
    ,'edm::eventsetup::EventSetupRecord::getFromProxy'
    ,'edm::GlobalSchedule::processOneGlobalAsync'
    ,'edm::SerialTaskQueueChain::push'
    ,'edm::Worker::doWorkNoPrefetchingAsync'
    ,'edm::ServiceRegistry::setContext'
    ,'edm::ServiceRegistry::presentToken()'
    ,'edm::service::InitRootHandlers::ThreadTracker::on_scheduler_entry'
    ,'__TBB_machine_fetchstore4'
    ,'__TBB_machine_cmpswp4'
    ,'__TBB_machine_fetchstore1'
    ,'acquire (spin_rw_mutex.h:118)'
    ,'reset_extra_state ('
    ,'priority (scheduler_common.h:130)'
    ,'edm::service::MessageLogger::'
#    ,'edm::service::MessageLogger::establishModule('
#    ,'edm::Run::Run(edm::RunPrincipal const&,'
    ,'edm::service::MessageLogger::unEstablishModule'
    ,'tbb::'
    ,'edm::RunForOutput::RunForOutput('
    ,'edm::stream::EDAnalyzerAdaptor<'
    ,'edm::EventSetup::find('
    ,'edm::eventsetup::EventSetupRecord::find('
    ,'edm::eventsetup::DataKey::operator<('
    ,'edm::eventsetup::SimpleStringTag::operator<('
    ,'std::__shared_ptr<edm::serviceregistry::ServicesManager'
    ,'try_acquire (spin_mutex.h:109)'
    ,'edm::Run::Run('
    ,'FastTimerService::preStreamBeginRun('
    ,'decltype ({parm#1}()) edm::convertException::wrap<bool edm::Worker::runModule'
    ,'edm::Worker::reset()'
    ,'edm::stream::ProducingModuleAdaptorBase<edm::stream::EDProducerBase>::doStreamBeginLuminosityBlock('
    ,'edm::stream::ProducingModuleAdaptorBase<edm::stream::EDFilterBase>::doStreamBeginLuminosityBlock('
    ,'edm::LuminosityBlock::LuminosityBlock(edm::LuminosityBlockPrincipal'
    ,'edm::StreamSchedule::processOneStreamAsync<'
    ,'edm::Worker::doWorkAsync<'
    ,'edm::StreamSchedule::processOneEventAsync('
    ,'edm::ParentContext::moduleCallingContext()'
    ,'edm::ModuleCallingContext::getTopModuleCallingContext'
    ,'edm::Event::Event('
    ,'edm::Path::workerFinished('
    ,'edm::Path::updateCounters('
    ,'edm::Path::recordStatus('
    ,'FastTimerService::postPathEvent('
    ,'edm::hash_detail::isCompactForm_('
    ,'edm::InputProductResolver::resolveProduct_'
    ,'edm::NoProcessProductResolver::dataValidFromResolver('
    ,'edm::DataManagingProductResolver::productWasFetchedAndIsValid_(bool)'
    ,'FastTimerService::postModuleEvent('
    ,'edm::UnscheduledProductResolver::prefetchAsync_'
#    ,'edm::NoProcessProductResolver::prefetchAsync_'
#    ,'edm::NoProcessProductResolver::resolveProduct_('
    ,'edm::NoProcessProductResolver::'
    ,'reco::Jet::detectorP4'
    ,'edm::EarlyDeleteHelper::moduleRan'
    ,'edm::clearLoggedErrorsSummary('
    ,'edm::ProductProvenanceRetriever::branchIDToProvenance('
    ,'HistogramProbabilityEstimator::probability' #protected by an atomic
    ,'edm::EventPrincipal::setLuminosityBlockPrincipal'
    ,'edm::DataManagingProductResolver::'
]

stackToIgnore = [
    'edm::service::MessageLogger::'
    ,'edm::MessageSender::ErrorObjDeleter'
    ,'edm::Run::runPrincipal() const'
    ,'edm::WaitingTaskList::'
    ,'edm::EventProcessor::beginJob()'
    ,'edm::StreamSchedule::processOneEventAsync'
    ,'edm::WorkerManager::resetAll()'
    ,'edm::ParentageRegistry::insertMapped('
    ,'edm::one::EDFilterBase::doEvent('
    ,'edm::one::EDProducerBase'
    ,'edm::EventBase::triggerNames_'
    ,'edm::EDFilter::doEvent('
    ,'edm::EDAnalyzer::doEvent('
    ,'edm::one::OutputModuleBase::doEvent'
    ,'edm::EDProducer::doEvent'
    ,'edm::Principal::clearPrincipal'
    ,'edm::RootOutputFile::writeOne'
    ,'edm::PrincipalCache::deleteRun('
    ,'edm::eventsetup::EventSetupProvider::eventSetupForInstance'
    ,'edm::EventPrincipal::clearEventPrincipal()'
    ,'FastTimerService::Resources::operator+='
    ,'FastTimerService::preSourceEvent(edm::StreamID)'
    ,'edm::EventPrincipal::fillEventPrincipal('
    ,'edm::InputProductResolver::putProduct_('
]

addressesToIgnore = [
#    'edm::eventsetup::makeEventSetupProvider('
#    ,' edm::eventsetup::DataProxy::get('
#    ,'cond::createPayload<'
#    ,'edm::pset::Registry::getMapped('
    'is in a rw- anonymous segment' #not sure about this one
#    ,'edm::RootFile::fillRunAuxiliary'
    ,'tbb::internal::arena::arena('
#    ,'edm::EventPrincipal::fillEventPrincipal('
#    ,'edm::Principal::addUnscheduledProduct('
#    ,'edm::RootDelayedReader::getProduct_'
#    ,'TBranchElement::GetEntry('
#    ,'edm::Event::put<'
#    ,'edm::stream::EDProducerAdaptorBase::doEvent'
#    ,'edm::stream::EDFilterAdaptorBase::doEvent('
#    ,'edm::EventProcessor::init(' #this may ignore too much, but needed to ignore member data of streams
#    ,'edm::global::EDProducerBase::doEvent'
#    ,'FastTimerService::postBeginJob()'
#    ,'edm::EDProducer::doEvent('
#    ,'_ZN3pat15PackedCandidate27covarianceParameterization_E'
#    ,'edm::RootOutputFile::writeOne'
    ,'DQMStore::book'
    ,'L1TdeCSCTF::L1TdeCSCTF' #legacy
    #,'MeasurementTrackerEventProducer::produce(' #MeasurementTrackerEvent ultimately hits edmNew::DetSetVector's lazy caching of DetSet which is supposed to be thread safe (but may not be?)
    ,'std::vector<reco::TrackExtra' #this is the cache in Ref
    ,'std::vector<reco::Track'
    ,'std::vector<reco::PFConversion'
]

addressesToIgnoreIfRead = [
    'edm::eventsetup::makeEventSetupProvider('
    ,' edm::eventsetup::DataProxy::get('
    ,'cond::createPayload<'
    ,'edm::pset::Registry::getMapped('
#    ,'is in a rw- anonymous segment' #not sure about this one
    ,'edm::RootFile::fillRunAuxiliary'
#    ,'tbb::internal::arena::arena('
    ,'edm::EventPrincipal::fillEventPrincipal('
    ,'edm::Principal::addUnscheduledProduct('
    ,'edm::RootDelayedReader::getProduct_'
    ,'TBranchElement::GetEntry('
    ,'edm::Event::put<'
    ,'edm::stream::EDProducerAdaptorBase::doEvent'
    ,'edm::stream::EDFilterAdaptorBase::doEvent('
    ,'edm::EventProcessor::init(' #this may ignore too much, but needed to ignore member data of streams
    ,'edm::global::EDProducerBase::doEvent'
    ,'FastTimerService::postBeginJob()'
    ,'edm::EDProducer::doEvent('
    ,'_ZN3pat15PackedCandidate27covarianceParameterization_E'
    ,'edm::RootOutputFile::writeOne'
    ,'BSS segment'
    ,'bytes inside data symbol' #this shows the writes but will miss the reads
    ,'FSQ::HandlerTemplate' #some function statics
#    ,'DQMStore::book'
    ,'TBufferFile::'
    ,'edm::service::MessageLogger::'
    ,'TClass::GetClass('
]

#startOfMessage ='-------------------'
endOfMessage ='-------------------'
startOfMessage = 'Possible data race'
startOfMessageLength = len(startOfMessage)
messageStarted = False
lineCount = 100
buffer = []
maxCount = 20
lookForAddress = False
foundAddress = False
addressCount = 100
possibleDataRaceRead = False
foundStartOfMessage = False
for l in f.readlines():
    if l[:2] != '==':
        continue
    if l.find(endOfMessage) != -1:
        foundAddress = False
        addressCount = 100
    if l.find(startOfMessage) != -1:
        lookForAddress = False
        foundAddress = False
        possibleDataRaceRead = (l.find('data race during read') != -1)
        if buffer:
            #print buffer
            print('---------------------')
            for b in buffer:
                print(b[:-1])
        
        buffer=[l]
        lineCount = 0
        continue
#    if lineCount == 2:
#        if l.find('data race') == -1:
#            buffer = []
#            lineCount = 100
#        possibleDataRaceRead = (l.find('data race during read') != -1)
    if lineCount < maxCount:
        skipThis = False
        for i in stackToIgnore:
            if l.find(i) != -1:
                lineCount = 100
                skipThis = True
                buffer = []
                break
        if skipThis:
            continue
        buffer.append(l)
        lineCount +=1
        if ' at 0x' in l:
            for i in messageToIgnore:
                if l.find(i) != -1:
                    buffer = []
                    lineCount = 100
                    break
        if lineCount == 100:
            continue
        if l.find('Address 0x') != -1:
            lookForAddress = True
            foundAddress = False
            lineCount = 100
    if lineCount == maxCount:
        lookForAddress = True
        foundAddress = False
        lineCount = 100
    if lookForAddress:
        if l.find('Address 0x') != -1:
            foundAddress = True
            lookForAddress = False
            addressCount = 0
            lineCount = 100
    if foundAddress:
        addressCount +=1
        if addressCount < maxCount:
            buffer.append(l)
            for i in addressesToIgnore:
                if l.find(i) != -1:
                    buffer = []
                    foundAddress = False
                    addressCount = 100
                    break
            if possibleDataRaceRead:
                for i in addressesToIgnoreIfRead:
                    if l.find(i) != -1:
                        buffer = []
                        foundAddress = False
                        addressCount = 100
                        break
            if l[-3:]=="== ":
                foundAddress = False
                addressCount = 100
