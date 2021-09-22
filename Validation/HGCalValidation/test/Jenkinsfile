pipeline {
    agent {
        label 'llrgrhgtrig.in2p3.fr'
    }

    stages {
        stage('Build') {
            steps {
                echo 'Building..'
                sh '''
                uname -a
                whoami
                pwd
                ls -l
                '''
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
                sh 'true'
            }
        }
    }
}
