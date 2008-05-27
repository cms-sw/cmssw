# ValidationTools : Publisher
#   
# Developers:
#   Victor E. Bazterra
#   Kenneth James Smith
#
# Descrition:
#   StaticWeb, and automated email for ValidationTools



import os
import commands
import tempfile
import shutil
import smtplib
import Configuration
import ErrorManager

class StaticWeb:

  def __init__ ( self ):
    self.dropBox = None
    self.webLocation = None
    self.releaseDifference = None
  
  ## Given a root file plots all the histograms.  
  def plot ( self, mainfile, reffile, location, file_prefix, package, release, reference ):
    # Creates a temporal file for executing root
    rootFile = tempfile.NamedTemporaryFile('w',suffix='.C')    
    #rootFile = open ('Holanda.C','w')
    # Create string to plot histograms
    #string = 'void '
    string = ''
    string = string + rootFile.name.split("/")[-1].split(".")[0] + '() {\n'
    
    string = string + 'gROOT->SetStyle(\"Plain\");\n'    
    string = string + 'gSystem->Load(\"' + Configuration.variables['VTRoot'] + 'libMyMakePlots.so\");\n'
    if os.path.isfile(reffile) == False:
      string = string + 'MyMakePlots plot; \n'
      string = string + "plot.SetFilename(\""+mainfile+"\");\n"
      string = string + "plot.SetWebPath(\""+location+"\");\n"
      string = string + "plot.SetExtension(\"png\");\n"
      string = string + "plot.SetFilePrefix(\""+file_prefix+"\");\n"
    else:
      string = string + 'MyMakePlots plot; \n'
      string = string + "plot.SetFilename(\""+mainfile+"\");\n"
      string = string + "plot.SetCompare(true);\n"
      string = string + "plot.SetCompareFilename(\""+reffile+"\");\n"
      string = string + "plot.SetWebPath(\""+location+"\");\n"
      string = string + "plot.SetExtension(\"png\");\n"
      string = string + "plot.SetFilePrefix(\""+file_prefix+"\");\n"
      string = string + "plot.SetDirectory(\""+package+"\");\n"
      string = string + "plot.SetRelease(\""+release+"\");\n"
      string = string + "plot.SetReference(\""+reference+"\");\n"
    string = string + "plot.Draw();\n }"
    # Write the string
    rootFile.write( string )
    # Fluch the cache
    rootFile.flush()
    # Run root script 
    status, output = commands.getstatusoutput("root -l -b -q " + rootFile.name ) 
    # Close tmp file
    rootFile.close()
 
  def index(self, release,reference, process,  dir):
    # Begin .html file...  Can probably beautify others...  We'll see!!
    b = 0 
    str = '<html>\n\
    <head>\n\
    <title>Plots for Release:  ' + release + '   Reference:  ' + reference + '   Process:    ' +process+'</title>\n\
    </head>\n\
    <body>\n\n\n\
    <b><center>Plots for Release:  ' + release +'    Reference:    ' + reference + '   Process:    ' +process+'</center></b>\n'
    for file in os.listdir(dir):
      newDir = file.split('.')[0]
      if ' ' in file:
        file_string = file.replace(' ','')
      else:
        file_string = file
      temp_file = file_string
      if '<' in file:
        file_string = file_string.replace('<', 'lt')
        shutil.move(dir+'/'+file,dir+'/'+file_string)
        temp_file = file_string
      elif '>' in file:
        file_string = file_string.replace('>', 'gt')
        shutil.move(dir+'/'+file,dir+'/'+file_string)
        temp_file = file_string
      if '<' in newDir:
        newDir = newDir.replace('<', 'lt')
      if '>' in newDir:
        newDir = newDir.replace('>', 'gt')
      if os.path.isdir(  dir+'/'+newDir+'/') == False:
        os.mkdir(dir+'/'+newDir+'/')
      shutil.copyfile(dir+'/'+file_string, dir+'/'+newDir+'/'+file_string)
      file1 = open(dir+'/'+newDir+'/index.html', 'w')
      string = '<html>\n\
      <head>\n\
      <title>Plot for Release:  ' + release + '   Reference:  ' + reference + '   Process:    '+process+ '  Plot:    ' +file+'</title>\n\
      </head>\n\
      <body>\n\n\n\
      <b><center>Plots for Release:  ' + release +'    Reference:    ' + reference + '   Process:    '+process+ '   Plot:    '+file+'</center></b>\n'
      string = string + '<img align =\"center\" src='+file_string+'> </html>'
      file1.write(string)
      file1.close()
      b = b + 1
      str = str + '<p>\n\
      <a href=\"' + newDir + '/\">'
      if b % 2 !=0:
        str = str + '<img WIDTH="40%" src=' + temp_file + ' align="left"></a>\n\
        </p>\n '
      else:
        str = str + '<img WIDTH="40%" src=' + temp_file + ' align="right"></a>\n\
        </p>\n '
    str = str + "</body>\n\n\n\
    </html> \n "
    file = open( dir + '/'  + 'index.html', 'w' )
    file.write(str)
    file.close()

  #Define an automated email function 
  def mail(self, serverURL=None, sender='', to='', subject='', text=''):
    """
    Usage:
    mail('somemailserver.com', 'me@example.com', 'someone@example.com', 'test', 'This is a test')
    """
    headers = "From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n" % (sender, to, subject)
    message = headers + text
    mailServer = smtplib.SMTP(serverURL)
    mailServer.sendmail(sender, to, message)
    mailServer.quit()

  ## Make Automated Email Machinery
  def email(self, lock, File_Prefix = ""):
    EmailInfo = []
    Body_String = ""
    To_Email = ['kjsmith@buffalo.edu', 'atomheart41@yahoo.com']# 'baites@fnal.gov', 'avto@fnal.gov']
    if os.path.exists(File_Prefix+'HistFail.txt'):
    
      Fail_File = open(File_Prefix+'HistFail.txt', 'r')
      EmailInfo = File_Prefix.split("__")
      Email_String = ' This is an automated email.  The Process:  '+EmailInfo[0]+'; has failed a test between release:   '+EmailInfo[1]+'  and reference:   '+EmailInfo[2]+'.  The following histograms are suspect: \n\
      \n'
      for Line in Fail_File:
        Body_String = Body_String + "Link:: " + Configuration.variables['HTTPLocation']+EmailInfo[1]+'/'+EmailInfo[2]+'/'+EmailInfo[0]+'/'+Line.split(":")[0]+'\n'
    
      Email_String = Email_String + Body_String + '\n\
      \n Please make any necessary changes that need to be made to your package.  If there is an error in this email please contact us at, username@fnal.gov\n\
      \n\
      \n Thank you,\n\
      \n SVSuite Team \n\
      \n'
      if Body_String != "":
        #lock.acquire()
        # Get names of developers 
        #if not os.path.exists('Validation/'+EmailInfo[0]+'/.admin/developers'):
        #  os.system('cvs co  Validation/'+EmailInfo[0]+'/.admin/developers')
        #lock.release()
        #Administrators = open('Validation/'+EmailInfo[0]+'/.admin/developers', 'r')
        Line_Count = 0
        Email_Address = []
        Line_Split = []
        for line in Administrators:
          if Line_Count == 2:
            Line_Split = line.split(':')
            Email_Address.append(Line_Split[2])
          if line[0] == '>':
            Line_Count = Line_Count + 1
        Administrators.close()
        #for Address in To_Email:
        #  self.mail('smtp.fnal.gov', 'kjsmith@fnal.gov', Address, 'Failed Software Validation for '+EmailInfo[0], Email_String)
      os.system('rm '+File_Prefix+'HistFail.txt')
    #os.system('rm -r Validation/')
      
  ## Run the static web publisher  
  def run( self, lock, package, release,dataset,  references = [ ] ):
    if len(references) == 0:
      dirname = release
      lock.acquire()
      if os.path.isdir( Configuration.variables['WebLocation'] + '/' + dirname  ) == False:
        os.mkdir ( Configuration.variables['WebLocation'] + '/' + dirname )
      lock.release()
      releaseDir = Configuration.variables['DropBox'] + '/Releases/' + release + '/'  + package
      file_prefix = package+'_'+release
      if os.path.isdir( releaseDir ) == True:
        weblocation = Configuration.variables['WebLocation'] + package + '/'+ dataset + '/' + dirname + '/' 
        os.makedirs(weblocation)
        for file in os.listdir(releaseDir):
          if file.split('.')[-1] == 'root':
            self.plot( releaseDir + '/' + file, "", weblocation, file_prefix )
        self.index(package, release, weblocation)
    for reference in references:
      dirname = release + '/' + reference
      # File prefix stores data to pass on to other parts of the class....  
      file_prefix = package+'__'+release+'__'+reference+'__'+dataset
      releaseDir = Configuration.variables['PackageDirectory']+release+'/'+dataset+'/'
      referenceDir = Configuration.variables['PackageDirectory']+reference+'/'+dataset+'/'
      # Check to see that both directories exist
      #print releaseDir, referenceDir
      if os.path.isdir( releaseDir ) == True and os.path.isdir( referenceDir ) == True:
        
        weblocation = Configuration.variables['WebLocation'] + '/' + package+ '/'+ dataset +'/' + dirname
        #weblocation = '/uscms_data/d1/kjsmith/NewWork/UserCode/ksmith/DropBox/Web/'+package+'/'+dataset+'/'+dirname+'/' # temp web directory
        # Create web directory
        #lock.acquire()
        if os.path.isdir(weblocation) == False:
          os.makedirs(weblocation)
        #lock.release()
        # look for root files to compare
        for file in os.listdir(releaseDir):
          if file.split('.')[-1] == 'root':
            # send for comparison
            self.plot( releaseDir + '/' + file, referenceDir + '/'  + file, weblocation, file_prefix , package+'V')
        # Build html file 
        self.index(package, release, weblocation)
        # Automate Email , this must run after plot becuase 'plot' creates a txt files needed for the automated email.
        self.email(lock, file_prefix )

