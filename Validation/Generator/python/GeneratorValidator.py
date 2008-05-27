
#! /usr/bin/env python
#############################
#  Authors: Kenneth James Smith and Victor E. Bazterra
#
#  Validator runs cfg files and compares to old releases.  It can also run Vista, by Steve Mrenna.
#
#
##############################
import pickle
import socket
import os
import shutil
import Publisher
import commands
import threading 
import time

import Configuration
from Tools import Template
from optparse import OptionParser


class JobManager:

    
    def __init__(self):
        
        self.__pythia_release = ""
        self.__herwig_release = ""
        self.__release_List = {}
        self.__jobNumber = {}
        
    ### Runs config files
    def new_config_run(self, run, process, compare):
        generator_List = []
        self.__release_List = {}
        os.chdir(Configuration.variables["HomeDirectory"]+'bin/')
        if run != None or process != None:
            ######### Find release for each generator used ##########
            dir = os.listdir(Configuration.variables["HomeDirectory"]+'data/')
            temp_dir = os.listdir(Configuration.variables["HomeDirectory"]+'templates/')
            if run.upper() != "ALL":
                temp_status, temp_output =  commands.getstatusoutput('scramv1 tool info '+run )
                if "Version" not in temp_output:
                    #print "Error obtaining "+run+" release, please make sure scramv1 is working"
                    self.__release_List[run] = run+'__external'
                    generator_List.append(run)
                for line in temp_output.split('\n'):
                    if "Version" in line:
                        self.__release_List[run] = run+"__"+line.split(':')[-1].strip(' ')
                        generator_List.append(run)
            if compare.upper() != "ALL" and compare.upper() != run.upper():
                temp_status, temp_output =  commands.getstatusoutput('scramv1 tool info '+compare )
                if "Version" not in temp_output:
                    #print "Error obtaining "+compare+" release, please make sure scramv1 is working"
                    self.__release_List[compare] = compare+'__external'
                    generator_List.append(compare)
                for line in temp_output.split('\n'):
                    if "Version" in line:
                        self.__release_List[compare] = compare+"__"+line.split(':')[-1].strip(' ')
                        generator_List.append(compare)
            if run.upper() == "ALL" or compare.upper() == "ALL":
                for probe in dir:
                    if len(probe.split('__')) == 1:
                        continue 
                    if probe.split('.')[0].split('__')[1] not in generator_List:
                        temp_status, temp_output =  commands.getstatusoutput('scramv1 tool info '+probe.split('.')[0].split('__')[1] )
                        if "Version" not in temp_output:
                            self.__release_List[compare] = probe.split('.')[0].split('__')[1]+'__external'
                            generator_List.append(probe.split('.')[0].split('__')[1])
                        for line in temp_output.split('\n'):
                            if "Version" in line:
                                self.__release_List[probe.split('.')[0].split('__')[1]] = probe.split('.')[0].split('__')[1]+"__"+line.split(':')[-1].strip(' ')
                                generator_List.append(probe.split('.')[0].split('__')[1])
            
            #########  Make cfg files from templates and run #############
            for file in dir:
                if '~' in file or '.cfi' not in file:
                    continue
                if run.upper() != "ALL" and compare.upper() != "ALL" and run.upper() not in file.upper() and compare.upper() not in file.upper():
                    continue
                if process.upper() in file.upper():
                        print "About to run cmsRun", file
                        splitter = file.split('.')[0].split('__')
                        if len(splitter) > 2:
                            directory = file.split('.')[0].split('__')[1]+'/'+file.split('.')[0].split('__')[1]+'__'+file.split('.')[0].split('__')[2]+'EDM'
                        else:
                            directory = file.split('.')[0].split('__')[1]+'/'+self.__release_List[file.split('.')[0].split('__')[1]]
                        if os.path.exists(Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+process+'/'+directory+'/') == False:
                            os.makedirs(Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+process+'/'+directory+'/')
                        scratch = Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+process+'/'+directory+'/'
                        for template in temp_dir:
                            if template.split('.')[0] == file.split('__')[0]:
                                tfile = open (Configuration.variables['HomeDirectory'] + '/templates/'+template ,'r')
                                template = Template(tfile.read())
                                tfile.close()
                                file_reader = open(Configuration.variables['HomeDirectory'] + '/data/'+file, 'r')
                                source_string = file_reader.read()
                                file_reader.close()
                                cfg_file = file.split('.')[0]+'.cfg'
                                cfile = open(scratch+cfg_file, 'w')
                                cfile.write(template.safe_substitute(source = source_string))
                                cfile.close()
                                os.chdir(Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+process+'/'+directory+'/')
                                os.system('cmsRun '+cfg_file+' >& '+file.split('.')[0]+'.log')
                                root_files = 0
                                for data_file in os.listdir(Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+process+'/'+directory+'/'):
                                    if '.root' in data_file or '.log' in data_file:
                                        if len(splitter) > 2:
                                            directory1 = file.split('.')[0].split('__')[1]+'/'+file.split('.')[0].split('__')[1]+'__'+file.split('.')[0].split('__')[2]+'EDM'
                                        else:
                                            directory1 = file.split('.')[0].split('__')[1]+'/'+self.__release_List[file.split('.')[0].split('__')[1]]
                                        if os.path.exists(Configuration.variables['ReleaseDirectory']+directory1+'/'+file.split('.')[0].split('__')[0]+'/') != True:
                                            os.makedirs(Configuration.variables['ReleaseDirectory']+directory1+'/'+file.split('.')[0].split('__')[0]+'/')
                                        shutil.copyfile(Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+process+'/'+directory+'/'+data_file, Configuration.variables['ReleaseDirectory']+directory1+'/'+file.split('.')[0].split('__')[0]+'/'+data_file)
                                        if '.root' in data_file:
                                            root_files = 1
                                os.system('rm '+Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+process+'/'+directory+'/*')
                                if root_files == 0:
                                    print file+" didn't run properly, please check log file"

    ### Runs config files
    def new_batch_run(self, run, process, compare, batch):
        generator_List = []
        self.__release_List = {}
        self.__jobNumber = {}
        process_List = []
        os.chdir(Configuration.variables["HomeDirectory"]+'bin/')
        if run != None or process != None:
            ####### Find release for each generator used ##########
            dir = os.listdir(Configuration.variables["HomeDirectory"]+'data/')
            temp_dir = os.listdir(Configuration.variables["HomeDirectory"]+'templates/')
            if run.upper() != "ALL":
                temp_status, temp_output =  commands.getstatusoutput('scramv1 tool info '+run )
                if "Version" not in temp_output:
                    #print "Error obtaining release, please make sure scramv1 is working"
                    self.__release_List[run] = run+'__external'
                    generator_List.append(run)
                for line in temp_output.split('\n'):
                    if "Version" in line:
                        self.__release_List[run] = run+"__"+line.split(':')[-1].strip(' ')
                        generator_List.append(run)
            if compare.upper() != "ALL" and compare.upper() != run.upper():
                temp_status, temp_output =  commands.getstatusoutput('scramv1 tool info '+compare )
                if "Version" not in temp_output:
                    #print "Error obtaining release, please make sure scramv1 is working"
                    self.__release_List[compare] = compare+'__external'
                    generator_List.append(compare)
                for line in temp_output.split('\n'):
                    if "Version" in line:
                        self.__release_List[compare] = compare+"__"+line.split(':')[-1].strip(' ')
                        generator_List.append(compare)
            if run.upper() == "ALL" or compare.upper() == "ALL":
                for probe in dir:
                    if probe.split('.')[0].split('__')[1] not in generator_List:
                        #print 'scramv1 tool info '+probe.split('.')[0].split('__')[1] 
                        temp_status, temp_output =  commands.getstatusoutput('scramv1 tool info '+probe.split('.')[0].split('__')[1] )
                        if "Version" not in temp_output:
                            #print "Error obtaining release, please make sure scramv1 is working"
                            self.__release_List[compare] = probe.split('.')[0].split('__')[1]+'__external'
                            generator_List.append(probe.split('.')[0].split('__')[1])
                        for line in temp_output.split('\n'):
                            if "Version" in line:
                                self.__release_List[probe.split('.')[0].split('__')[1]] = probe.split('.')[0].split('__')[1]+"__"+line.split(':')[-1].strip(' ')
                                generator_List.append(probe.split('.')[0].split('__')[1])


            ########  Make cfgs from templates and run ##############
            for file in dir:
                if '~' in file or '.cfi' not in file:
                    continue
                if 'cmsrun'  in file:
                    continue
                if 'condor' in file:
                    continue
                #print file, 'file'
                if process.upper() != "ALL" and process.upper() not in file.upper():
                    continue
                if file.split('.')[0].split('__')[0] not in process_List:
                    self.__jobNumber[file.split('.')[0].split('__')[0]] = []
                    process_List.append(file.split('.')[0].split('__')[0])
                splitter = file.split('.')[0].split('__')
                if len(splitter) > 2:
                    scratch = Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+splitter[0]+"/"+splitter[1]+'/'+splitter[2]+'EDM/'
                else:
                    scratch = Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+file.split('.')[0].split('__')[0]+"/"+file.split('.')[0].split('__')[1]+'/'+self.__release_List[file.split('.')[0].split('__')[1]]+'/'
                if os.path.exists(scratch+'/') == False:
                    os.makedirs(scratch+'/')
               
                CMSstatus, CMSoutput = commands.getstatusoutput('$CMSSW_BASE')
                CMS_dir = CMSoutput.split(':')[1].strip(' ')
                for temp in temp_dir:
                    print temp, 'template'
                    if temp.split('.')[0] == file.split('__')[0]:
                        tfile = open (Configuration.variables['HomeDirectory'] + '/templates/'+temp ,'r')
                        template = Template(tfile.read())
                        tfile.close()
                        file_reader = open(Configuration.variables['HomeDirectory'] + '/data/'+file, 'r')
                        source_string = file_reader.read()
                        file_reader.close()
                        cfg_filename = file.split('.')[0]+'.cfg'
                        cfile = open(scratch+cfg_filename, 'w')
                        cfile.write(template.safe_substitute(source = source_string))
                        cfile.close()
                        if batch.upper() == "FNAL":
                            tfile = open (Configuration.variables['HomeDirectory'] + '/interface/condor.template' ,'r')
                            template = Template(tfile.read())
                            tfile.close()
                            cfile = open(scratch+"/condor.submit", 'w')
                            cfile.write(template.safe_substitute(file = scratch+'/cmsruncondor'+file.split('.')[0].split('__')[0], scratch = scratch))
                            cfile.close()
                            tfile = open (Configuration.variables['HomeDirectory'] + '/interface/cmsruncondor.template' ,'r')
                            template = Template(tfile.read())
                            tfile.close()
                            cfile = open(scratch+"/cmsruncondor"+file.split('.')[0].split('__')[0], 'w')
                            cfile.write(template.safe_substitute(directory=scratch, cfg = scratch+'/'+cfg_filename, CMSSW=CMS_dir ))
                            cfile.close()
                            os.chmod(scratch+"/cmsruncondor"+file.split('.')[0].split('__')[0], 0755)
                            status, output = commands.getstatusoutput("condor_submit " + scratch+"/condor.submit")
                            for line in output.split('\n'):
                                if "submitted" in line:
                                    self.__jobNumber[file.split('.')[0].split('__')[0]].append(line.split(' ')[5])
                            if status != 0:
                                print file + " wasn't submitted properly"
                        if batch.upper() == "CERN":
                            tfile = open (Configuration.variables['HomeDirectory'] + '/interface/cmsrunbsub.template' ,'r')
                            template = Template(tfile.read())
                            tfile.close()
                            cfile = open(scratch+"/cmsrunbsub"+file.split('.')[0].split('__')[0], 'w')
                            cfile.write(template.safe_substitute(directory=scratch, cfg = scratch+'/'+cfg_filename, CMSSW=CMS_dir ))
                            cfile.close()
                            os.chmod(scratch+"/cmsrunbsub"+file.split('.')[0].split('__')[0], 0755)
                            status, output = commands.getstatusoutput("condor_submit " + scratch+"/cmsrunbsub"+file.split('.')[0].split('__')[0])
                            for line in output.split('\n'):
                                if "submitted" in line:
                                    self.__jobNumber[file.split('.')[0].split('__')[0]].append(line.split(' ')[2].lstrip('<').rstrip('>'))
                            if status != 0:
                                print file + " wasn't submitted properly"
                                    

    def Vista_run(self, Site, job):
        #
        self.__jobNumber = [] 
        List = os.listdir(Configuration.variables['HomeDirectory']+'data/')
        Vista_List = []
        process_List = []
        CMSstatus, CMSoutput = commands.getstatusoutput('$CMSSW_BASE')
        CMS_dir = CMSoutput.split(':')[1].strip(' ')
        for file in List:
            if "VISTA" in file.upper() and file not in Vista_List:
                Vista_List.append(file)
        print Vista_List, "Vista"
        Remaining = Vista_List
        head = """source = PoolSource \n
        { \n
        untracked vstring fileNames = \n 
        { \n
        """
        foot ="""  } \n
        } \n """
        while len(Remaining) != 0:
            for run in Remaining:
                if run.split('.')[0].split('__')[0] not in process_List:
                    self.__jobNumber[run.split('.')[0].split('__')[0]] = []
                    process_List.append(run.split('.')[0].split('__')[0])
                if "DBS" in run.upper():
                    stat, out = commands.getstatusoutput('./rssparser --dataset='+run.split('__')[0].strip("-Vista")+' --release='+run.split('__')[2])
                    body_string = ''
                    if '.root' in out:
                        scratch = Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+run.split('__')[0]+'/'+"Vista"+'/'+run.split('__')[2]
                        if os.path.exists(scratch) == False:
                            os.makedirs(scratch)
                        for line in out.split('\n'):
                            if '.root' in line:
                                body_string = body_string + line + '\n'
                        tfile = open (Configuration.variables['HomeDirectory'] + '/templates/runVista.template' ,'r')
                        template = Template(tfile.read())
                        tfile.close()
                        source_string = head + body_string + foot 
                        cfile = open(scratch+"/runVista.cfg", 'w')
                        cfile.write(template.safe_substitute(source = source_string ))
                        cfile.close()
                        if Site.upper() == "FNAL":
                            tfile = open (Configuration.variables['HomeDirectory'] + '/interface/condor.template' ,'r')
                            template = Template(tfile.read())
                            tfile.close()
                            cfile = open(scratch+"/condor.submit", 'w')
                            cfile.write(template.safe_substitute(file = scratch+'/cmsruncondor'+run.split('.')[0].split('__')[0], scratch = scratch))
                            cfile.close()
                            tfile = open (Configuration.variables['HomeDirectory'] + '/interface/cmsruncondor.template' ,'r')
                            template = Template(tfile.read())
                            tfile.close()
                            cfile = open(scratch+"/cmsruncondor"+run.split('.')[0].split('__')[0], 'w')
                            cfile.write(template.safe_substitute(directory=scratch, cfg = scratch+'/runVista.cfg', CMSSW=CMS_dir ))
                            cfile.close()
                            os.chmod(scratch+"/cmsruncondor"+run.split('.')[0].split('__')[0], 0755)
                            status, output = commands.getstatusoutput("condor_submit " + scratch+"/condor.submit")
                            Remaining.remove(run)
                        elif Site.upper() == "CERN":
                            tfile = open (Configuration.variables['HomeDirectory'] + '/interface/cmsrunbsub.template' ,'r')
                            template = Template(tfile.read())
                            tfile.close()
                            cfile = open(scratch+"/cmsrunbsub"+run.split('.')[0].split('__')[0], 'w')
                            cfile.write(template.safe_substitute(directory=scratch, cfg = scratch+'/runVista.cfg', CMSSW=CMS_dir ))
                            cfile.close()
                            os.chmod(scratch+"/cmsrunbsub"+run.split('.')[0].split('__')[0], 0755)
                            status, output = commands.getstatusoutput("bsub " + scratch+"/cmsrunbsub"+run.split('.')[0].split('__')[0])
                            for line in output.split('\n'):
                                if "submitted" in line:
                                    self.__jobNumber[run.split('.')[0].split('__')[0]].append(line.split(' ')[2].lstrip('<').rstrip('>'))
                            if status != 0:
                                print file + " wasn't submitted properly"
                            Remaining.remove(run)
                        else:
                            print run, job, "DBS"
                            os.chdir(scratch)
                            status, output = commands.getstatusoutput("cmsRun runVista.cfg >& "+run.split('.')[0].split('__')[0]+'.log')
                            os.chdir(Configuration.variables['HomeDirectory'])
                    elif 'ERROR' in out.upper():
                        print "Error reading DBS for "+run
                        Remaining.remove(run)
                    else:
                        time.sleep(40)
                else:
                    file = open(Configuration.variables['HomeDirectory']+'data/'+run, 'r')
                    source_string = file.read()
                    file.close()
                    scratch = Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+run.split('__')[0]+'/'+"Vista"+'/'+run.split('__')[2]
                    tfile = open (Configuration.variables['HomeDirectory'] + '/tempaltes/runVista.template' ,'r')
                    template = Template(tfile.read())
                    tfile.close()
                    cfile = open(scratch+"/runVista.cfg", 'w')
                    cfile.write(template.safe_substitute(source = source_string ))
                    cfile.close()
                    if job.upper() != "BATCH":
                        print run, job
                        os.chdir(scratch)
                        status, output = commands.getstatusoutput("cmsRun runVista.cfg >& "+run.split('.')[0].split('__')[0]+'.log')
                        os.chdir(Configuration.variables['HomeDirectory'])
                    else:
                        if Site.upper() == "FNAL":
                            tfile = open (Configuration.variables['HomeDirectory'] + '/interface/condor.template' ,'r')
                            template = Template(tfile.read())
                            tfile.close()
                            cfile = open(scratch+"/condor.submit", 'w')
                            cfile.write(template.safe_substitute(file = scratch+'/cmsrun'+run.split('.')[0].split('__')[0], scratch = scratch))
                            cfile.close()
                            tfile = open (Configuration.variables['HomeDirectory'] + '/interface/cmsrun.template' ,'r')
                            template = Template(tfile.read())
                            tfile.close()
                            cfile = open(scratch+"/cmsrun"+run.split('.')[0].split('__')[0], 'w')
                            cfile.write(template.safe_substitute(directory=scratch, cfg = scratch+'/runVista.cfg', CMSSW=CMS_dir ))
                            cfile.close()
                            os.chmod(scratch+"/cmsrun"+run.split('.')[0].split('__')[0], 0755)
                            status, output = commands.getstatusoutput("condor_submit " + scratch+"/condor.submit")
                            print run, "BATCH"
                            for line in output.split('\n'):
                                if "submitted" in line:
                                    self.__jobNumber[run.split('.')[0].split('__')[0]].append(line.split(' ')[5])
                        if Site.upper() == "CERN":
                            tfile = open (Configuration.variables['HomeDirectory'] + '/interface/cmsrunbsub.template' ,'r')
                            template = Template(tfile.read())
                            tfile.close()
                            cfile = open(scratch+"/cmsrunbsub"+run.split('.')[0].split('__')[0], 'w')
                            cfile.write(template.safe_substitute(directory=scratch, cfg = scratch+'/runVista.cfg', CMSSW=CMS_dir ))
                            cfile.close()
                            os.chmod(scratch+"/cmsrunbsub"+run.split('.')[0].split('__')[0], 0755)
                            status, output = commands.getstatusoutput("bsub " + scratch+"/cmsrunbsub"+run.split('.')[0].split('__')[0])
                            for line in output.split('\n'):
                                if "submitted" in line:
                                    self.__jobNumber[run.split('.')[0].split('__')[0]].append(line.split(' ')[2].lstrip('<').rstrip('>'))
                            if status != 0:
                                print file + " wasn't submitted properly"

                    Remaining.remove(run)   
        
    def NewPublisher(self, run, process, compare, all, job):
        os.chdir(Configuration.variables["HomeDirectory"]+'/bin/')
        if job.upper() != "LOCAL":
            condor_status = 0
            while condor_status == 0:
                Unfinished_Job = 0
                for jobid in self.__jobNumber[process]:
                    status, output = commands.getstatusoutput('condor_q '+jobid)
                    if jobid in output:
                        Unfinished_Job = Unfinished_Job + 1 
                if Unfinished_Job == 0:
                    condor_status = 1
                    print "All jobs for " +process+ " are done" 
                if Unfinished_Job > 0:
                    time.sleep(15)
            for generator in os.listdir(Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+process):
                for sub_gen in os.listdir(Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+process+'/'+generator+'/'):
                    for subdir in os.listdir(Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+process+'/'+generator+'/'+sub_gen+'/'):
                        if '.root' in subdir or '.log' in subdir:
                            if os.path.isdir(Configuration.variables['ReleaseDirectory']+generator+'/'+sub_gen+'/'+process+'/') == False:
                                os.makedirs(Configuration.variables['ReleaseDirectory']+generator+'/'+sub_gen+'/'+process+'/')
                            shutil.copyfile(Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+process+'/'+generator+'/'+sub_gen+'/'+subdir, Configuration.variables['ReleaseDirectory']+generator+'/'+sub_gen+'/'+process+'/'+subdir)
                    #os.system('rm '+Configuration.variables["HomeDirectory"]+'DropBox/scratch/'+process+'/'+generator+'/'+sub_gen+'/*')
                
        #print "Begin Publishing" , process
        ### Publishes and compares, may be able to replace 'data' too ###
        gen_List = os.listdir(Configuration.variables['ReleaseDirectory'])
        if compare.upper() == "SAME" or compare.upper() == "ALL":
            for List in gen_List:
                if compare.upper() == "SAME" and List not in run and run.upper() != "ALL":
                    continue
                sub_dir = os.listdir(Configuration.variables['ReleaseDirectory']+List)
                for i in range(len(sub_dir) - 1):
                    for j in range(i+1, len(sub_dir)):
                        if sub_dir[i] == sub_dir[j]:
                            continue
                        if os.path.isdir(Configuration.variables['ReleaseDirectory']+List+'/'+sub_dir[i]+'/'+process) == False:
                            continue
                        if os.path.isdir(Configuration.variables['ReleaseDirectory']+List+'/'+sub_dir[j]+'/'+process) == False:
                            continue
                        web_dir = Configuration.variables["WebDirectory"]+sub_dir[i]+'/'+sub_dir[j]+'/'+process+'/'
                        if os.path.isdir(web_dir) == True:
                            continue
                        if os.path.isdir(Configuration.variables["WebDirectory"]+sub_dir[j]+'/'+sub_dir[i]+'/'+process+'/') and os.listdir(Configuration.variables["WebDirectory"]+sub_dir[j]+'/'+sub_dir[i]+'/'+process+'/') < 2:
                            continue 
                        for rel_file in os.listdir(Configuration.variables['ReleaseDirectory']+List+'/'+sub_dir[i]+'/'+process):
                            if '.root' in rel_file:
                                for ref_file in os.listdir(Configuration.variables['ReleaseDirectory']+List+'/'+sub_dir[j]+'/'+process):
                                    if '.root' in ref_file:
                                        if os.path.isdir(web_dir) == False:
                                            os.makedirs(web_dir)
                                        
                                        Publisher.StaticWeb().plot(Configuration.variables['ReleaseDirectory']+List+'/'+sub_dir[i]+'/'+process+'/'+rel_file, Configuration.variables['ReleaseDirectory']+List+'/'+sub_dir[j]+'/'+process+'/'+ref_file, web_dir, process+'--'+sub_dir[i]+'--'+sub_dir[j], '', sub_dir[i], sub_dir[j])
                                        Publisher.StaticWeb().index(sub_dir[i], sub_dir[j], process, web_dir)
                                              

        # Compares new release to old ones.... #
        if compare.upper() != "SAME":# or compare.upper() == "ALL":
            for i in range(len(gen_List) - 1):
                for j in range(i+1,len(gen_List)):
                    if compare.upper() != "ALL":
                        if compare.upper() not in gen_List[i] and compare.upper() != "OPPOSITE":
                            continue
                        if compare.upper() not in gen_List[j] and compare.upper() != "OPPOSITE":
                            continue
                        if run.upper() != "ALL":
                            if run.upper() not in gen_List[i] or run.upper() not in gen_List[j]:
                                continue
                    rel_List = os.listdir(Configuration.variables['ReleaseDirectory']+gen_List[i])
                    ref_List = os.listdir(Configuration.variables['ReleaseDirectory']+gen_List[j])
                    for rel_dir in rel_List:
                        if os.path.isdir(Configuration.variables['ReleaseDirectory']+gen_List[i]+'/'+rel_dir+'/'+process) == False:
                            continue
                        for ref_dir in ref_List:
                            if os.path.isdir(Configuration.variables['ReleaseDirectory']+gen_List[j]+'/'+ref_dir+'/'+process) == False:
                                continue
                            if ref_dir == rel_dir:
                                continue 
                            if all.upper() == "FALSE":
                                if ref_dir != self.__release_List[gen_List[j]] and ref_dir != self.__release_List[gen_List[i]] and rel_dir != self.__release_List[gen_List[j]] and rel_dir != self.__release_List[gen_List[i]]:
                                    continue
                            web_dir = Configuration.variables["WebDirectory"]+rel_dir+'/'+ref_dir+'/'+process+'/'
                            if os.path.exists(web_dir) and os.listdir(web_dir) < 2:
                                continue
                            if os.path.exists(Configuration.variables["WebDirectory"]+ref_dir+'/'+rel_dir+'/'+process+'/'):
                                continue
                            rel_loc = Configuration.variables['ReleaseDirectory']+gen_List[i]+'/'+rel_dir+ '/' + process+'/'
                            ref_loc = Configuration.variables['ReleaseDirectory']+gen_List[j]+'/'+ref_dir+ '/' + process+'/'
                            #print gen_List[i], gen_List[j]
                            for rel_file in os.listdir(rel_loc):
                                if '.root' in rel_file:
                                    for ref_file in os.listdir(ref_loc):
                                        if '.root' in ref_file:
                                            if os.path.isdir(web_dir) == False:
                                                os.makedirs(web_dir)
                                            
                                            Publisher.StaticWeb().plot(rel_loc+'/'+rel_file, ref_loc+'/'+ref_file, web_dir, process+'--'+rel_dir+'--'+ref_dir, '', rel_dir, ref_dir)
                                            Publisher.StaticWeb().index(rel_dir, ref_dir, process, web_dir)
                                
                                                    
    
   
                                        
    ## Makes matrix for web site ## 
    def MatrixCreator(self, job):

        ### Test to make sure that all batch jobs are done
        if job.upper() == "BATCH":
            Jobs = True
            while Jobs == True:
                me_stat, me_out = commands.getstatusoutput('whoami')
                tempstatus, tempoutput = commands.getstatusoutput('condor_q '+me_out)
                if len(tempoutput.split('\n')) == 6:
                    Jobs = False
                else:
                    time.sleep(15)

        ########### Make Header string here ###########
        Header_String = ''' \n\
        <HTML>\n\
        <head>\n\
        <LINK REL="stylesheet" TYPE="text/css" HREF="http://lcgapp.cern.ch/project/simu/framework/GDML/style.css">\n\
        <title> Generator comparison </title>\n\
        </head>\n\
        <body>\n\
        <center><h1> Generator validation test </h1>\n\
        <h3>Comparisons of standard events between MC generators </h3></center>\n\
        \n'''

        ### Footer String ###
        Footer_String = '''\n\
        </tr>
        </table>
        </body>
        </html>
        \n'''
        
        ###  Now the real part, the actual table(s) #####
        Column_Head = '''\n\
        <table cellspacing=1 cellpadding=1 border=1>\n\
        <tr>\n\
        <td colspan="1" rowspan="2"></td>\n\
        \n
        '''

        Column_Foot = ''' \n\
        </tr>\n\
        <tr>\n\
        \n
        '''
        
        List1 = []
        TempList1 = os.listdir(Configuration.variables["WebDirectory"])
        for probe in TempList1:
            if '.html' in probe:
                continue
            List1.append(probe)
        Col_String = ""
        Col_Name = ""
        Row_Body_String = ""
        Max = 0
        Col_Total = ""
        List2 = []
        Sub_List = []
        #for dir in List1:
        #    for dir1 in os.listdir(Configuration.variables["WebDirectory"]+'/'+dir+'/'):
        #        if '.html' in dir1:
        #            continue
                #if dir1 not in List2:
                #    List2.append(dir1)
        #        for dir2 in os.listdir(Configuration.variables["WebDirectory"]+'/'+dir+'/'+dir1+'/'):
        #            if dir2 not in Sub_List:
        #                Sub_List.append(dir2)

        
        eliminate = 1
        ## Max is size subcolumns 
        Max = len(Sub_List)
        ## Can make more than one table here....                                                                        
        Generators = []                                                                                                 
        Sub_Generators = []                                                                                            
        for probe in List1:                                                                                              
            if probe.split('__')[0] not in Generators:                                                                   
                Generators.append(probe.split('__')[0])                                                                  
            for probe1 in os.listdir(Configuration.variables["WebDirectory"]+probe+'/'):                                 
                if probe1.split('__')[0] not in Sub_Generators:                                                          
                    Sub_Generators.append(probe1.split('__')[0])
        for col in Generators:
            List2 = []
            Sub_List = []
            for dir in List1:
                if col not in dir:
                    print col, dir
                    continue
                print col, dir, "made it"
                for dir1 in os.listdir(Configuration.variables["WebDirectory"]+'/'+dir+'/'):
                    if '.html' in dir1:
                        continue
                    if dir1 not in List2:
                        List2.append(dir1)
                    for dir2 in os.listdir(Configuration.variables["WebDirectory"]+'/'+dir+'/'+dir1+'/'):
                        if dir2 not in Sub_List:
                            Sub_List.append(dir2)
                        
            print List1
            for row in Sub_Generators:
                
                instance = 0 
                for probe in List1:
                    if col not in probe:
                        continue
                    for probe1 in os.listdir(Configuration.variables["WebDirectory"]+probe+'/'):
                        if row not in probe1:
                            continue
                        else:
                            instance = instance + 1
                #print instance, row, col
                if instance == 0:
                    continue
                
                # Begin new table 
                Table_Max = []
                Col_String = ""
                Col_Name = ""
                Row_Body_String = ""
                for poll1 in List1:
                    if col in poll1:
                        for poll2 in os.listdir(Configuration.variables["WebDirectory"]+poll1+'/'):
                            if row in poll2:
                                for sub_proc in os.listdir(Configuration.variables["WebDirectory"]+poll1+'/'+poll2+'/'):
                                    if sub_proc not in Table_Max:
                                        Table_Max.append(sub_proc)
                for col1 in List1:
                    if col in col1:
                        #print col1
                        for proc in Table_Max:
                            Col_Name = Col_Name + ' <td><FONT SIZE= "-1"><center>'+proc+ '</center></FONT></td> \n'
                        

                for column in List1:
                    if col in column:
                        if len(column.split('__')) > 1:
                            Col_String = Col_String + '<td colspan="'+str(len(Table_Max))+'" align=center ><FONT SIZE="-1"><b>'+column.split('__')[0][0]+column.split('__')[1]+'</b> </FONT></td> \n'
                        else:
                            Col_String = Col_String + '<td colspan="'+str(len(Table_Max))+'" align=center><FONT SIZE="-1"><b>'+column+'</b></FONT> </td> \n'
                
                for row_List in List2:
                    if row not in row_List:
                        continue
                    Row_Body_String = Row_Body_String + '''\n\
                    <tr>\n\
                    <td rowspan="1" align="left" valign="center"><FONT SIZE="-1"><b> '''+row_List.split('__')[0][0]+row_List.split('__')[1]+ '</b></FONT></td>\n\
                    \n\
                    '
                    for col1 in List1:
                        if col not in col1:
                            continue
                        for proc in Table_Max:
                            if os.path.isdir(Configuration.variables["WebDirectory"]+col1+'/'+row_List+'/'+proc+'/') == True and len(os.listdir(Configuration.variables["WebDirectory"]+col1+'/'+row_List+'/'+proc+'/')) > 1:
                                Row_Body_String = Row_Body_String + '<td bgcolor="white" align="center"><a href='+col1+'/'+row_List+'/'+proc+'/'+'>&diams;</a></td> \n'
                            else:
                                Row_Body_String = Row_Body_String + '<td bgcolor="grey"></td> \n'
                            
                String = '<br><center><b><FONT COLOR=RED>' +col.upper() + ' vs. ' +row.upper()+ '</FONT></b> </center><br>'
                Col_Total  = Col_Total +  Column_Head + String + Col_String + Column_Foot + Col_Name + Row_Body_String                 
        index_file = open(Configuration.variables["WebDirectory"]+'index.html', 'w')
        index_file.write('<center>'+Header_String+Col_Total+Footer_String+'<br><br></center>')
        index_file.close()
        
        
       
                        
            
        
        
                       
            
                    
                           
                
            
            
            
        
                                    
                                
                
                    
                
        
