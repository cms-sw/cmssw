#! /usr/bin/env python

from __future__ import print_function
import os
import shutil
import sys
import fileinput
import string
import new_userparams

##############################################################
# Input parameters
#

print("Reading input parameters") 

#
#specify macros used to plot here
#
macro='macro/new_TrackValHistoPublisher.C'
macroSeed='macro/SeedValHistoPublisher.C'
macroReco='macro/RecoValHistoPublisher.C'
macroIsol='macro/IsoValHistoPublisher.C'
macroMuonReco='macro/RecoMuonValHistoPublisher.C'


##############################################################
# Helper functions
#
def GetGuiRepository(param):
    splitrelease=param['Release'].split('_')
    genericrelease='_'.join(splitrelease[0:3])+'_x/'
    dqmguirepository=param['DqmGuiBaseRepo']+genericrelease
    return dqmguirepository

def GetEosRepository(param):
    splitrelease=param['Release'].split('_')
    genericrelease='_'.join(splitrelease[0:3])+'_x/'
    eosRepository=param['EOSBaseRepository']+genericrelease
    return eosRepository

def GetLabel(params):
    label = params['Condition']
    if (params['Condition'] == 'STARTUP'):
        label = 'START'
    elif (params['Condition']=='MC'):
        label ='MC_52_V1'
    label+=params['Label']

    if (params['FastSim']):
        label += '_FastSim'

    label += '-' + params['Version']
   
    if (params['PileUp']!='no'):
   	label = 'PU' + params['PileUp']+ '_' + label

    return label


def GetTag(params):
    tag = params['Condition']
    if (params['PileUp']=='no'):
        tag += '_noPU'
    else:
        tag += '_PU'+params['PileUp']
    if (params['FastSim']):
        tag +='_FSIM'
    return tag

def GetFastSimSuffix(params):
    if (params['FastSim']):
        return 'FS'
    else:
        return ''

def replace(map, filein, fileout):
    replace_items = map.items()
    while True:
        line = filein.readline()
        if not line: break
        for old, new in replace_items:
            line = string.replace(line, old, new)
        fileout.write(line)
    fileout.close()
    filein.close()

def myrootsubmit(cfgfile):
    command='root -b -q -l '+ cfgfile + '.C' + '>  macro.' + cfgfile +'.log'
    print('>> Executing \'' + command + '\'')
    os.system(command)
    print('   ... executed \'' + command + '\'');

def downloadfile(url):
    print('   + Downloading "' + url + '"...')
    #os.system('wget --ca-directory $X509_CERT_DIR/ --certificate=$X509_USER_PROXY --private-key=$X509_USER_PROXY '+url)
    stream = os.popen('/usr/bin/curl -k -O -L --capath $X509_CERT_DIR --key $X509_USER_PROXY --cert $X509_USER_PROXY -w "%{http_code}" '+ url)
    output=stream.readlines()
    if output[0]=='200':
        print('   + OK!')
	return True
    else:
        print('   + ERROR! ' + str(output[0]))
	print("Skipping " + url)
	print("Please check the name of the file in the repository: "+GetGuiRepository(new_userparams.NewParams))
        sys.exit('Exiting...');
	# return False

def GetSamplePath(params, sample):
    return params['Release']+'/'+GetTag(params)+'/'+sample

def GetLocalSampleName(params, sample):
    return GetSamplePath(params, sample) + '/val.' + sample + '.root'

def createSampleDirectory(params, sample):
    path = GetSamplePath(params, sample)
    print('>> Creating directory ' + path + '...')
    if(os.path.exists(path)==False):
        os.makedirs(path)
    else:
        print('NOTE: Directory \'' + path + '\' already exists')

    return path
    
def getSampleFiles(params, sample):
    path = GetSamplePath(params,sample)
    print('>> Creating directory ' + path + '...')
    if(os.path.exists(path)==False):
        os.makedirs(path)
    else:
        print('NOTE: Directory \'' + path + '\' already exists')

    print('\n')

    # Check if pdfs have already been produced
    checkFile = path + '/standAloneMuons.pdf'
    print('>> Checking if ' + checkFile + ' already exists...')
    if (os.path.isfile(checkFile)==True):
        print('   + Files of type ' + checkFile + ' exist already.')
        print('     Delete them first, if you really want to overwrite them')
        quit()
      
    print('   + The file does not exist so we continue')
    localsample=GetLocalSampleName(params, sample)
    sampleOnWeb=new_userparams.WebRepository+'/'+ localsample

    if (os.path.isfile(localsample)==True):
        print('   + ' + params['Type'] + ' sample file found at: ' +localsample + '. Using that one')

    elif (new_userparams.NewParams['GetFilesFrom']=='GUI'):
        theGuiSample = sample
        
        guiFileName='DQM_V0001_R000000001__'+theGuiSample+'__'+params['Release']+'-'+GetLabel(params)+'__'+params['Format']+'.root'
        guiFullURL=GetGuiRepository(params)+guiFileName
        print(">> Downloading file from the GUI: " + guiFullURL)

        if (downloadfile(guiFullURL)==True):
		print('   + Moving ' + guiFileName + ' to ' + localsample)
        	shutil.move(guiFileName,localsample)

    elif ((params['GetFilesFrom']=='WEB') & (os.path.isfile(sampleOnWeb))) :
        print("NOTE: New file found at: "+newSample+' -> Copy that one')
        os.system('cp '+sampleOnWeb+' '+path)

    elif ((params['GetFilesFrom']=='EOS')) :
        print("creating a symbolic link to a file from eos: ")
        eosFileName='DQM_V0001_R000000001__'+sample+'__'+params['Release']+'-'+GetLabel(params)+'__'+params['Format']+'.root'
        eosRepository=GetEosRepository(params)
        sampleOnEOS=eosRepository+eosFileName

        if (os.path.isfile(sampleOnEOS)) :
            print(sampleOnEOS) 
            #cp_command = 'cp '+sampleOnEOS+' '+path+'/val.'+ sample+'.root'
            #os.system(cp_command)
            ln_command = 'ln -s '+sampleOnEOS+' '+path+'/val.'+ sample+'.root'
            os.system(ln_command)
        else :
            print("ERROR: File "+sampleOnEOS+" NOT found.")
            quit()

    else:
        print('*** WARNING: no signal file was found')

def getReplaceMap(newparams, refparams, sample, datatype, cfgkey, cfgfile):
    newLocalSample=GetLocalSampleName(newparams, sample)
    refLocalSample=GetLocalSampleName(refparams, sample)
    replace_map = { 'DATATYPE': datatype,
                    'NEW_FILE':newLocalSample,
                    'REF_FILE':refLocalSample,
                    'REF_LABEL':sample,
                    'NEW_LABEL': sample,
                    'REF_RELEASE':refparams['Release'],
                    'NEW_RELEASE':newparams['Release'],
                    'REFSELECTION':GetTag(refparams),
                    'NEWSELECTION':GetTag(newparams),
                    cfgkey: cfgfile
                    }
    return replace_map

##############################################################
# Main program
#
# Initial checks
# + Copying missing parameters from New to Ref
print('>> Checking input parameters...')
for i in new_userparams.NewParams:
    if i in new_userparams.RefParams:
        print('   + ' + i + ' set to:')
        print('     -' + str(new_userparams.NewParams[i]) + ' (new)')
        print('     -' + str(new_userparams.RefParams[i]) + ' (ref)')
    else:
        new_userparams.RefParams[i]=new_userparams.NewParams[i]
        print('   + ' + i + ' set to ' + str(new_userparams.NewParams[i]) + ' (both)')

# + Things needed by GUI
if ((new_userparams.NewParams['GetFilesFrom']=='GUI')|(new_userparams.RefParams['GetFilesFrom']=='GUI')):
    if os.getenv('X509_USER_PROXY','') == '':
        print("ERROR: It seems you did not configure your environment to be able")
        print("       to download files from the GUI. Your should follow these steps:")
        print(" > source /cvmfs/cms.cern.ch/crab3/crab.csh")
        print(" > voms-proxy-init --voms cms")
        print(" > setenv X509_CERT_DIR $HOME/.globus")
        print(" > setenv X509_USER_PROXY /tmp/x509up_uVWXYZ (where VWXYZ = your unix Id on lxplus)")
        print(" >  or similarly for bash shell")
        quit()

if (new_userparams.NewParams['FastSim']|new_userparams.RefParams['FastSim']):
    new_userparams.ValidateHLT=False
    new_userparams.ValidateDQM=False

if (new_userparams.NewParams['HeavyIons']|new_userparams.RefParams['HeavyIons']):
    new_userparams.ValidateHLT=False
    new_userparams.ValidateRECO=False
    new_userparams.ValidateISO=False


# Iterate over new_userparams.samples
print('>> Selected samples:')
for sample in new_userparams.samples:
    print('   + ' + sample)
    
print('>> Processing new_userparams.samples:')
for sample in new_userparams.samples :
    print('###################################################')
    print('# ' + sample)
    print('###################################################')
    print('\n')

    # Create the directories to store the new_userparams.samples and pdfs
    newpath=createSampleDirectory(new_userparams.NewParams, sample)
    refpath=createSampleDirectory(new_userparams.RefParams, sample)
    print('\n')

    getSampleFiles(new_userparams.NewParams, sample)
    getSampleFiles(new_userparams.RefParams, sample)
        
    print('   + Producing macro files...')
    # root macro name should not contain "-"
    sample_m = sample
    if (sample == 'RelValJpsiMuMu_Pt-8'):
        sample_m = 'RelValJpsiMuMu_Pt8'
        
    cfgFileName=sample_m+'_'+new_userparams.NewParams['Release']+'_'+new_userparams.RefParams['Release']
    hltcfgFileName='HLT'+sample_m+'_'+new_userparams.NewParams['Release']+'_'+new_userparams.RefParams['Release']
    seedcfgFileName='DQMSEED'+sample_m+'_'+new_userparams.NewParams['Release']+'_'+new_userparams.RefParams['Release']
    recocfgFileName='DQMRECO'+sample_m+'_'+new_userparams.NewParams['Release']+'_'+new_userparams.RefParams['Release']
    recomuoncfgFileName='RECO'+sample_m+'_'+new_userparams.NewParams['Release']+'_'+new_userparams.RefParams['Release']
    isolcfgFileName='ISOL'+sample_m+'_'+new_userparams.NewParams['Release']+'_'+new_userparams.RefParams['Release']    

    if os.path.isfile(GetLocalSampleName(new_userparams.RefParams,sample)):
        replace_map = getReplaceMap(new_userparams.NewParams, new_userparams.RefParams, sample, 'RECO', 'new_TrackValHistoPublisher', cfgFileName)
        if (new_userparams.ValidateHLT):
            replace_map_HLT = getReplaceMap(new_userparams.NewParams, new_userparams.RefParams, sample, 'HLT', 'new_TrackValHistoPublisher', hltcfgFileName)
        if (new_userparams.ValidateDQM):
            replace_map_DIST = getReplaceMap(new_userparams.NewParams, new_userparams.RefParams, sample, 'RECO', 'RecoValHistoPublisher', recocfgFileName)
            replace_map_SEED = getReplaceMap(new_userparams.NewParams, new_userparams.RefParams, sample, 'RECO', 'SeedValHistoPublisher', seedcfgFileName)
        if (new_userparams.ValidateISO):
            replace_map_ISOL = getReplaceMap(new_userparams.NewParams, new_userparams.RefParams, sample, 'RECO', 'IsoValHistoPublisher', isolcfgFileName)
        if (new_userparams.ValidateRECO):
            replace_map_RECO = getReplaceMap(new_userparams.NewParams, new_userparams.RefParams, sample, 'RECO', 'RecoMuonValHistoPublisher', recomuoncfgFileName)
            replace_map_RECO['IS_FSIM']=''
    else:
        print("No reference file found at: ", refpath)
        replace_map = getReplaceMap(new_userparams.NewParams, new_userparams.NewParams, sample, 'RECO', 'new_TrackValHistoPublisher', cfgFileName)
        if (new_userparams.ValidateHLT):
            replace_map_HLT = getReplaceMap(new_userparams.NewParams, new_userparams.NewParams, sample, 'HLT', 'new_TrackValHistoPublisher', hltcfgFileName)
        if (new_userparams.ValidateDQM):
            replace_map_DIST = getReplaceMap(new_userparams.NewParams, new_userparams.NewParams, sample, 'RECO', 'RecoValHistoPublisher', recocfgFileName)
            replace_map_SEED = getReplaceMap(new_userparams.NewParams, new_userparams.NewParams, sample, 'RECO', 'SeedValHistoPublisher', seedcfgFileName)
        if (new_userparams.ValidateISO):
            replace_map_ISOL = getReplaceMap(new_userparams.NewParams, new_userparams.NewParams, sample, 'RECO', 'IsoValHistoPublisher', isolcfgFileName)
        if (new_userparams.ValidateRECO):
            replace_map_RECO = getReplaceMap(new_userparams.NewParams, new_userparams.NewParams, sample, 'RECO', 'RecoMuonValHistoPublisher', recomuoncfgFileName)
            replace_map_RECO['IS_FSIM']=''

    templatemacroFile = open(macro, 'r')
    macroFile = open(cfgFileName+'.C' , 'w' )
    replace(replace_map, templatemacroFile, macroFile)

    if (new_userparams.ValidateHLT):
        templatemacroFile = open(macro, 'r')
        hltmacroFile = open(hltcfgFileName+'.C' , 'w' )
        replace(replace_map_HLT, templatemacroFile, hltmacroFile)

    if (new_userparams.ValidateDQM):
        templatemacroFile = open(macroReco, 'r')
        recomacroFile = open(recocfgFileName+'.C' , 'w' )
        replace(replace_map_DIST, templatemacroFile, recomacroFile)
        templatemacroFile = open(macroSeed, 'r')
        seedmacroFile = open(seedcfgFileName+'.C' , 'w' )
        replace(replace_map_SEED, templatemacroFile, seedmacroFile)

    if (new_userparams.ValidateISO):
        templatemacroFile = open(macroIsol, 'r')
        isolmacroFile = open(isolcfgFileName+'.C' , 'w' )
        replace(replace_map_ISOL, templatemacroFile, isolmacroFile)

    if (new_userparams.ValidateRECO):
        templatemacroFile = open(macroMuonReco, 'r')
        recomuonmacroFile = open(recomuoncfgFileName+'.C' , 'w' )
        replace(replace_map_RECO, templatemacroFile, recomuonmacroFile)

    if(new_userparams.Submit):
        myrootsubmit(cfgFileName)
        if (new_userparams.ValidateHLT):
            myrootsubmit(hltcfgFileName)
        if (new_userparams.ValidateDQM):
            myrootsubmit(recocfgFileName)
            myrootsubmit(seedcfgFileName)
        if (new_userparams.ValidateISO):
            myrootsubmit(isolcfgFileName)
            if (new_userparams.NewParams['FastSim']&new_userparams.RefParams['FastSim']):
                shutil.move(newpath+'/MuonIsolationV_inc.pdf',newpath+'/MuonIsolationV_inc_FS.pdf')
        if (new_userparams.ValidateRECO):
            myrootsubmit(recomuoncfgFileName)
            if (new_userparams.NewParams['FastSim']&new_userparams.RefParams['FastSim']):
                if (os.path.isfile(newpath+'/RecoMuonV.pdf') == True):
                    os.rename(newpath+'/RecoMuonV.pdf',newpath+'/RecoMuonV_FS.pdf')
                else:
                    print('ERROR: Could not find "' + newpath + '/RecoMuonV.pdf')

    os.system('mkdir '+newpath+'/PDF')
    os.system('mv '+newpath+'/*.pdf '+newpath+'/PDF/.')

    if(new_userparams.Publish):
        newpath = GetSamplePath(new_userparams.NewParams,sample)
        newlocalsample = GetLocalSampleName(new_userparams.NewParams, sample)
        newdir=new_userparams.WebRepository + '/' + newpath
        print('>> Publishing to ' + newdir + '...')
        if(os.path.exists(newdir)==False):
            os.system('ssh '+new_userparams.User+'@lxplus.cern.ch mkdir -p ' + newdir)
            # os.makedirs(newdir)
            # os.system('rm '+ newlocalsample)  
            # os.system('scp -r '+newpath+'/* ' + newdir)
        os.system('scp -r '+newpath+'/* '+new_userparams.User+'@lxplus.cern.ch:' + newdir)

        if(new_userparams.Publish_rootfile):            
            os.system('scp -r '+newpath+'/val.*.root '+new_userparams.User+'@lxplus.cern.ch:' + newdir)

        print('New path is ' + newlocalsample + ' and ' + newpath)
