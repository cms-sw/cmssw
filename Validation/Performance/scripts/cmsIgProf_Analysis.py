#! /usr/bin/env python

# Import statements. An '''Analogue''' of the include c++ statement,
# but really really not the same!
# In python everything is an object, even if we don't know. 
# os and time will become 2 objects for us.

import os
import time

def execute(command):
    print '[IgAnalysis] %s ...' %command
    exitstate=os.system(command)
    return exitstate
    
def analyse_prof_sim(profile_name,outdir):
    
    # A way of formatting strings similar to the c++ one

    #Eliminating mem live value peak analysis:
    #outfile1='%s/mem_live_peak.res' %outdir
    #outfile2='%s/doEvent_output.txt' %outdir
    #outfile3='%s/doBeginJob_output.txt' %outdir

    #Assume the profile_name (i.e. the input profile file) is handled properly by the user or the scripts
    #Use the outdir to determine what kind of analysis to make:

    AnalysisType=''
    if "IgProfperf" in outdir:
        AnalysisType='PERF_TICKS'
    elif "IgProfMemTotal" in outdir:
        AnalysisType='MEM_TOTAL'
    elif "IgProfMemLive" in outdir:
        AnalysisType='MEM_LIVE'

    #Use profile input file to determine the name of the output file:
    outfile=outdir+"/"+profile_name[:-3]+".res"
        
    #A first manipulation of the profile
    #command='igprof-analyse -d -v -g --value peak -r MEM_LIVE %s > %s'\
    #                                %(profile_name,outfile1)
    #execute(command) #we use the method system of the object os to run a command
    
    # now let's execute the segment and analyse commands                                                                
    #commands_list=(\
    #'igprof-segment edm::EDProducer::doEvent < %s |tee  -a %s' %(outfile1,outfile2),
    #
    #'igprof-segment edm::EDProducer::doBeginJob < %s |tee -a %s' %(outfile1,outfile3),
    #
    #'igprof-analyse -d -v -g -r MEM_LIVE %s |tee -a \%s'\
    #                                %(profile_name,outfile4),
    # 
    #'igprof-analyse -d -v -g -r MEM_TOTAL %s |tee -a \%s'\
    #                                %(profile_name,outfile5)
    #              )
    #
    #for command in commands_list: #no iterator can be clearer than this one
    #    #print command
    #    execute(command)

    #Launch the 1 command:
    
    command='igprof-analyse -d -v -g -r %s %s|tee -a \%s'%(AnalysisType,profile_name,outfile)
    execute(command) 
    # Now for every plain ascii file we make an html:
    titlestring='<b>Report executed with release %s on %s.</b>\n<br>\n<hr>\n'\
                                   %(os.environ['CMSSW_VERSION'],time.asctime())
    
    #for command,filename in map(None,commands_list,[outfile2,outfile3,outfile4,outfile5]):
    command_info='Command executed: <em>%s</em><br><br>\n' %command 
    
    # we open and read the txt ascii file
    txt_file=open(outfile,'r')
    txt_file_content=txt_file.readlines()#again:everything is an object
    txt_file.close()
    
    html_file_name='%s.html' %outfile[:-4]# a way to say the string until its last but 4th char
    html_file=open(html_file_name,'w')
    html_file.write('<html>\n<body>\n'+\
                    titlestring+\
                    command_info)
    for line in txt_file_content:
        html_file.write(line+'<br>\n')
    html_file.write('\n</body>\n</html>')
    html_file.close()

    # compress all the plain txt files!
    execute('pushd %s;gzip *txt;popd' %outdir)
                
#-------------------------------------------------------------------------------

# A procedure used for importing the function above with the import statement
# or to run it if the script is called: power python..
if __name__ == '__main__':
    
    import optparse
    
    # Here we define an option parser to handle commandline options..
    usage='IgProf_Analysis.py <options>'
    parser = optparse.OptionParser(usage)
    parser.add_option('-i', '--in_  profile',
                      help='The profile to manipulate' ,
                      default='',
                      dest='profile')
                      
    parser.add_option('-o', '--outdir',
                      help='The directory of the output' ,
                      default='',
                      dest='outdir')
                      
    (options,args) = parser.parse_args()
    
    # Now some fault control..If an error is found we raise an exception
    if options.profile=='' or\
       options.outdir=='':
        raise('Please select a profile and an output dir!')
    
    if not os.path.exists(options.profile) or\
       not os.path.exists(options.outdir):
        raise ('Outdir or input profile not present!')
    
    #launch the function!
    analyse_prof_sim(options.profile,options.outdir)        
 
