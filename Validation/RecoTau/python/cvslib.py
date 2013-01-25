import os
import subprocess
import hashlib

def showtags():
    'prints out the output of showtags'
    initialDir = os.getcwd()
    os.chdir(os.environ['CMSSW_BASE']+'/src') #move to the base src directory
    cmd = 'showtags'.split(' ')
    proc = subprocess.Popen(cmd,stdout=subprocess.PIPE)
    proc.wait()
    os.chdir(initialDir) #Get Back
    return proc.communicate()[0]


def showtags_c():
    'prints out the output of showtags -c | grep -v "?", therefore eliminating files that are not known by cvs'
    initialDir = os.getcwd()
    os.chdir(os.environ['CMSSW_BASE']+'/src') #move to the base src directory
    cmd  = 'showtags -c'.split(' ')
    proc = subprocess.Popen(cmd,stdout=subprocess.PIPE)
    proc.wait()
    os.chdir(initialDir) #Get Back
    stdout = proc.communicate()
    return '\n'.join( filter(lambda x: '?' not in x, stdout[0].split('\n') ) )

def get_cvs_version(path):
    'prints the cvs version of a file: path starts from src/'
    initialDir = os.getcwd()
    os.chdir(os.environ['CMSSW_BASE']+'/src') #move to the base src directory
    cmd  = ['cvs', 'status', path]
    proc = subprocess.Popen(cmd,stdout=subprocess.PIPE)
    proc.wait()
    stdout = proc.communicate()
    os.chdir(initialDir) #Get Back
    return stdout[0].split('\n')[3] #gets the '   Working revision:\t###'

def get_file_hash(path):
    'prints the md5 hash of a file'
    hasher = hashlib.md5()
    with open(os.environ['CMSSW_BASE']+'/src/'+path) as sfile:
        hasher.update(sfile.read())
    return hasher.hexdigest()    

def rich_showtags():
    'prints out the output of showtags -c | grep -v "?", therefore eliminating files that are not known by cvs. For updated and patched files the cvs version is also printed, for locally modified files the md5 hash value is printed'
    showtags = showtags_c().split('\n')
    rich_tags = []
    for line in showtags:
        if ' U ' in line or ' P ' in line:
            line += get_cvs_version(line.split(' ')[-1])
        elif ' M ' in line:
            line += '  md5 hash: '+ get_file_hash(line.split(' ')[-1])
        rich_tags.append(line)
    return '\n'.join( rich_tags )
