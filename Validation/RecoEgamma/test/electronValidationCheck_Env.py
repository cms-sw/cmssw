#! /usr/bin/env python3
# -*-coding: utf-8 -*-

import os, sys


class env:
    def checkSample(self):
        if ('DD_SAMPLE' not in os.environ) or (os.environ['DD_SAMPLE'] == ''):
            print('no existing sample name')
            print('nb of element : ', len(sys.argv))
            print('0 : ', sys.argv[0])
            print('1 : ', sys.argv[1])
            if (len(sys.argv) > 1):  # no else part since if sample does not exist, we had quit previously
                sampleName = str(sys.argv[1])
                os.environ['DD_SAMPLE'] = 'RelVal' + sampleName
                print('Sample name:', sampleName, ' - ', os.environ['DD_SAMPLE'])
            else:
                print('====================')
                print('no sample name, quit')
                print('====================')
                quit()
        else:
            print('Existing Sample name:', sampleName, ' - ', os.environ['DD_SAMPLE'])

    def beginTag(self):
        beginTag = 'Phase2'
        #beginTag = 'Run3'
        return beginTag

    def dd_tier(self):
        dd_tier = 'GEN-SIM-RECO'
        dd_tier = 'MINIAODSIM'
        return dd_tier

    def tag_startup(self):
        #tag_startup = '125X_mcRun3_2022_realistic_v3'
        tag_startup = '140X_mcRun4_realistic_v4_STD_2026D110_noPU'
        # tag_startup = '113X_mcRun3_2021_realistic_v7'
        return tag_startup

    def data_version(self):
        data_version = 'v1'
        return data_version

    def test_global_tag(self):
        test_global_tag = self.tag_startup()
        return test_global_tag

    def dd_cond(self):
        # dd_cond = 'PU25ns_' + self.test_global_tag() + '-' + self.data_version() # PU
        dd_cond = self.test_global_tag() + '-' + self.data_version()  # noPU
        return dd_cond

    def checkValues(self):
        print('-----')
        print(self.dd_tier())
        print(self.tag_startup())
        print(self.data_version())
        print(self.test_global_tag())
        print(self.dd_cond())
        print('-----')

        os.environ['beginTag'] = self.beginTag()

        if ('DD_TIER' not in os.environ) or (os.environ['DD_TIER'] == ''):
            os.environ['DD_TIER'] = self.dd_tier()  # 'GEN-SIM-RECO'
        if 'TAG_STARTUP' not in os.environ:  # TAG_STARTUP from OvalFile
            os.environ['TAG_STARTUP'] = self.tag_startup()  # '93X_upgrade2023_realistic_v0_D17PU200'
        if 'DATA_VERSION' not in os.environ:  # DATA_VERSION from OvalFile
            os.environ['DATA_VERSION'] = self.data_version()  # 'v1'
        if 'TEST_GLOBAL_TAG' not in os.environ:  # TEST_GLOBAL_TAG from OvalFile
            os.environ['TEST_GLOBAL_TAG'] = self.test_global_tag()  # os.environ['TAG_STARTUP']
        if ('DD_COND' not in os.environ) or (os.environ['DD_COND'] == ''):
            os.environ[
                'DD_COND'] = self.dd_cond()  # 'PU25ns_' + os.environ['TEST_GLOBAL_TAG'] + '-' + os.environ['DATA_VERSION']

        os.environ['DD_RELEASE'] = os.environ['CMSSW_VERSION']
        # os.environ['DD_RELEASE'] = "CMSSW_11_3_0_pre3"

        print('=====')
        if ('DD_SAMPLE_OUT' not in os.environ) or (os.environ['DD_SAMPLE_OUT'] == ''):
            os.environ['DD_SAMPLE_OUT'] = os.environ['DD_SAMPLE'].replace("RelVal", "ValFull")
        print('=====')

        os.environ['DD_SOURCE'] = '/eos/cms/store/relval/' + os.environ['DD_RELEASE'] + '/' + os.environ[
            'DD_SAMPLE'] + '/' + os.environ['DD_TIER'] + '/' + os.environ['DD_COND']
        os.environ['data'] = '/' + os.environ['DD_SAMPLE'] + '/' + os.environ['DD_RELEASE'] + '-' + os.environ[
            'DD_COND'] + '/' + os.environ['DD_TIER']
        os.environ['outputFile'] = 'electronHistos.' + os.environ['DD_SAMPLE_OUT'] + '_gedGsfE.root'
        if (os.environ['DD_TIER'] == 'MINIAODSIM'):
            os.environ['outputFile'] = 'electronHistos.' + os.environ['DD_SAMPLE_OUT'] + '_miniAOD.root'
        if ('inputPostFile' not in os.environ) or (os.environ['inputPostFile'] == ''):
            print('inputPostFile : %s' % os.environ['outputFile'])
            os.environ['inputPostFile'] = os.environ['outputFile']

        print('DD_RELEASE', os.environ['DD_RELEASE'])
        print('DD_SAMPLE', os.environ['DD_SAMPLE'])
        print('DD_SAMPLE_OUT', os.environ['DD_SAMPLE_OUT'])
        print('DD_COND', os.environ['DD_COND'])
        print('DD_TIER', os.environ['DD_TIER'])
        print('DD_SOURCE', os.environ['DD_SOURCE'])
        print('data', os.environ['data'])
        print('outputFile    :', os.environ['outputFile'])
        print('inputPostFile :', os.environ['inputPostFile'])
        print('beginTag : ', self.beginTag())
