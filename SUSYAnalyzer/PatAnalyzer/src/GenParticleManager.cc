#include "SUSYAnalyzer/PatAnalyzer/interface/GenParticleManager.h"

using namespace std;
const GenParticle* GenParticleManager::getMother(const GenParticle *p)
{
    
    if( p->numberOfMothers() == 0 )
    {
        //cerr<<"[ERROR] Trying to access mother of particle that has no mother" << endl;
        return NULL;
    }
    return &(*Collection.product())[p->motherRef(0).key()];
}

const GenParticle* GenParticleManager::getMother(const GenParticle *p, int i)
{
    
    if( p->numberOfMothers() == 0 )
    {
        //cerr<<"[ERROR] Trying to access mother of particle that has no mother" << endl;
        return NULL;
    }
    return &(*Collection.product())[p->motherRef(i).key()];
}

const GenParticle* GenParticleManager::getMotherParton(const GenParticle *p)
{
    
    if( p->numberOfMothers() == 0 )
    {
        //cerr<<"[ERROR] Trying to access mother of particle that has no mother" << endl;
        return NULL;
    }
    
    std::vector<const GenParticle*> moms;
    unsigned int counter = 0;
    const GenParticle* mom = getMother(p);
        
    while( mom && mom->pdgId() == p->pdgId() )
    {
        mom = getMother(mom);
    }
    
    moms.push_back(mom);
    counter++;
    
    bool found = false;
    while( mom && !found)
    {
        const GenParticle* mom0 = getMother(mom);
        for (unsigned int i = 0; i<mom->numberOfMothers(); ++i) {
            mom0 = getMother(mom,i);
            moms.push_back(mom0);
            //std::cout<<mom0->pdgId()<<" "<<mom0->status()<<std::endl;
            
            if (mom0) {
                if (fabs(mom0->pdgId()) < 7 || fabs(mom0->pdgId()) == 21) {
                    found = true;
                    break;
                }
            }
        }
        if (counter < moms.size() && !found) {
            mom = moms.at(counter);
            counter++;
        } else if (!found) mom = NULL;
    }
    if (found)
        return mom;
    else return NULL;
    
}

bool GenParticleManager::isPrompt( const GenParticle* p)
{
    const GenParticle* mom = getMother(&*p);
    while( mom && mom->pdgId() == p->pdgId() )
        mom = getMother(mom);
    
    if( !mom )
    {
        cout << "[ERROR] : Checking for Prompt/NonPrompt on a particle that has no mother" << endl;
        return false;
    }
    
    int id =  TMath::Abs( mom->pdgId() );
    
    if( id == 15 )
    {
        return isPrompt(mom);
    }
    
    switch(id) {
        case 22:
        case 23:
        case 24:
        case 25:
        case 1000022:
        case 1000023:
        case 1000024:
        case 1000025:
        case 1000035:
        case 1000039:
        case 1000011:
        case 1000012:
        case 1000014:
        case 1000015:
        case 1000016:
        case 2000011:
        case 2000013:
        case 2000015:
            return true;
        default:
            return false;
    }
}

void GenParticleManager::Classify()
{
    //loop through particles
    
    for(GenParticleCollection::const_reverse_iterator p = Collection->rbegin() ; p != Collection->rend() ; p++ )
    {
        int id = TMath::Abs(p->pdgId());
        //cout << id << endl;
        //continue;
        
        if( id == 11 )
        {
            if( isPrompt(&*p) )
                vPromptElectrons.push_back(&*p);
            else
                vNonPromptElectrons.push_back(&*p);
        }
        else if ( id == 13 )
        {
            if( isPrompt(&*p) )
                vPromptMuons.push_back(&*p);
            else
                vNonPromptMuons.push_back(&*p);
        }
        else if ( id == 15 )
        {
            if( isPrompt(&*p) )
                vPromptTaus.push_back(&*p);
            else
                vNonPromptTaus.push_back(&*p);
        }
        else if ( id == 23 )
        {
            vZBosons.push_back(&*p);
        }
        else if( id == 24 )
        {
            vWBosons.push_back(&*p);
        }
        else if ( id == 22 && p->mass() > 1.0)
        {
            vOffShellPhotons.push_back(&*p);
        }
        else if ( id == 25 )
        {
            vHiggsBosons.push_back(&*p);
        }
        else if ( id == 12 || id == 14 || id == 16 )
        {
            vInvisible.push_back(&*p);
        }
    }
}


void GenParticleManager::Reset()
{
    vPromptMuons.clear();
    vPromptElectrons.clear();
    vPromptTaus.clear();
    vNonPromptMuons.clear();
    vNonPromptElectrons.clear();
    vNonPromptTaus.clear();
    vInvisible.clear();
    vZBosons.clear();
    vWBosons.clear();
    vHiggsBosons.clear();
    vOffShellPhotons.clear();
    vCharginos.clear();
    vNeutralinos.clear();
}


std::vector<const GenParticle*> GenParticleManager::filterByStatus(std::vector<const GenParticle*>& input, int status)
{
    std::vector<const GenParticle*> v;
    for(std::vector<const GenParticle*>::iterator p = input.begin(); p != input.end(); p++ )
    {
        if( (*p)->status() == status) v.push_back(*(&*p));
    }
    return v;
}


void GenParticleManager::printInheritance(const GenParticle* p)
{
    cout << setw(10) << ParticleName(p->pdgId()  );
    const GenParticle* mom = getMother(&*p);
    while( mom )
    {
        cout << setw(10) << "  <--  " << ParticleName(mom->pdgId())<<" ("<<mom->status()<<")" ;
        if( mom->numberOfMothers() > 1 )
        {
            cout << setw(10) << "  <--  " << " MANY " ;
            break;
        }
        mom = getMother(mom);
    }
    cout << endl;
    
}

bool GenParticleManager::SameMother(const  GenParticle* p,  const GenParticle* part)
{
    //std::cout<<"Parton "<<part->pdgId()<<"; pt "<<part->pt()<<std::endl;
    if ((!p) || (!part)) return false;
    if (p->pt() == part->pt()) return true;
    //const GenParticle* mom = getMother(&*p);
    const GenParticle* mom = &*p;
    while( mom && mom->pt() != part->pt() ) {
        unsigned int k = 0;
        //std::cout<<"Number of mothers: "<<mom->numberOfMothers()<<std::endl;
        while (k < mom->numberOfMothers()) {
            const GenParticle* momK = getMother(mom, k);
            //std::cout<<momK->pdgId()<<", "<<momK->pt()<<std::endl;
            if (momK->pt() != part->pt()) k++;
            else return true;
        }
        mom = getMother(mom);
    }
    if (mom) return true;
    else return false;
}


std::vector<const GenParticle*> GenParticleManager::getAllMothers(const GenParticle* p)
{
    std::vector<const GenParticle*> moms;
    
    unsigned int counter = 0;
    moms.push_back(p);
    counter++;
    
    const GenParticle* mom = getMother(p);
    while( mom && mom->pdgId() == p->pdgId() )
    {
        mom = getMother(mom);
        //std::cout<<"itself? "<<mom->pdgId()<<std::endl;
    }
    if (mom) {
        moms.push_back(mom);
        counter++;
    }
    while( mom )
    {
        //mom = getMother(mom);
        const GenParticle* mom0;
        for (unsigned int i = 0; i<mom->numberOfMothers(); ++i) {
            mom0 = getMother(mom,i);
            if (mom0) {
                moms.push_back(mom0);
                //std::cout<<i<<" "<<mom0->pdgId()<<std::endl;
            }
        }
        if (counter < moms.size()) {
            mom = moms.at(counter);
            counter++;
        } else mom = NULL;
    }
    //std::cout<<"Full size "<<moms.size()<<std::endl;
    /*std::cout<<"************"<<std::endl;
    for (unsigned int i = 0; i!=moms.size(); ++i) {
        std::cout<<moms[i]->pdgId()<<" "<<std::endl;
    }
    std::cout<<"************"<<std::endl;
    */
    return moms;
}

enum decay {
    W_L,  // 0
    W_T_L, // 1
    W_B_L, // 2
    W_B_D_L, //3
    W_B_D_T_L, // 4
    W_B_T_L, // 5
    W_D_L, // 6
    W_D_T_L, //7
    B_L, // 8
    B_D_L, //9
    B_D_T_L, //10
    B_T_L,  // 11
    D_L, //12
    D_T_L, //13
    B_Baryon, // 14
    C_Baryon, //15
    pi_0, //16
    photon_, //17
    F_L, //18
    N_U_L_L // 19
};

bool GenParticleManager::fromTop(const GenParticle* p ) {
    if( !p ) return false;
    std::vector<const GenParticle*> moms = getAllMothers(p);
    
    bool res = false;
    //std::cout<<"****** from top ******"<<std::endl;
    for (unsigned int i = 0; i!=moms.size(); ++i ) {
        //std::cout<<moms.at(i)->pdgId()<<std::endl;
        if (fabs(moms.at(i)->pdgId()) == 6) {
            res = true;
            break;
        }
    }
    //std::cout<<"************"<<std::endl;
    return res;
}

bool GenParticleManager::fromID(const GenParticle* p, const int pdgID ) {
    if( !p ) return false;
    std::vector<const GenParticle*> moms = getAllMothers(p);
    
    bool res = false;
    //std::cout<<"****** from top ******"<<std::endl;
    for (unsigned int i = 0; i!=moms.size(); ++i ) {
        //std::cout<<moms.at(i)->pdgId()<<std::endl;
        if (fabs(moms.at(i)->pdgId()) == pdgID) {
            res = true;
            break;
        }
    }
    //std::cout<<"************"<<std::endl;
    return res;
}


int GenParticleManager::origin(const GenParticle *p)
{
    if( !p ) return N_U_L_L ;
    
    std::vector<const GenParticle*> moms = getAllMothers(p);
    
    if( comesFromBoson(moms) )
    {
        if( comesFromBMeson(moms) )
        {
            if( comesFromDMeson(moms) )
            {
                if( comesFromTau(moms) )
                    return W_B_D_T_L ;  //W-->B-->D-->Tau-->lepton
                
                return W_B_D_L ; // W-->B-->D-->lepton
            }
            if( comesFromTau(moms) )
                return W_B_T_L;  //W-->B-->Tau-->lepton
            
            return W_B_L; //W-->B-->lepton
        }
        if( comesFromDMeson(moms) ) {
            if( comesFromTau(moms) )
                return W_D_T_L ;  //W-->D-->Tau-->lepton
            
            return W_D_L ; // W-->D-->lepton
        }
        if( comesFromUDS(moms) ) {
            return pi_0;
        }
        if( comesFromTau(moms) )
            return W_T_L; //W-->Tau-->lepton
        
        return W_L; //W-->lepton
    }
    else if( comesFromBMeson(moms) )
    {
        if( comesFromDMeson(moms) )
        {
            if( comesFromTau(moms) )
                return B_D_T_L ;  //B-->D-->Tau-->lepton
            
            return B_D_L ; //B-->D-->lepton
        }
        if( comesFromTau(moms) )
            return B_T_L; //B-->Tau-->lepton
        
        return B_L; //B-->lepton
    }
    else if( comesFromDMeson(moms) )
    {
        if( comesFromTau(moms) )
            return D_T_L;  //D-->Tau-->lepton
        
        return D_L; //D-->lepton
    }
    else if( comesFromBBaryon(moms) )
        return B_Baryon;
    else if( comesFromCBaryon(moms) )
        return C_Baryon;
    else if ( comesFromPi0(moms) )
        return pi_0;
    else if( comesFromUDS(moms) )
        return pi_0;
    else if ( comesFromPhoton(moms) )
        return photon_;
    return F_L; //fake
    
}

int GenParticleManager::origin(const pat::Muon *p ) {  return origin(p->genLepton());}
int GenParticleManager::origin(const pat::Electron *p ) {  return origin(p->genLepton());}
//int GenParticleManager::origin(const reco::PFTau* p ) {}


bool GenParticleManager::comesFromBoson(std::vector<const GenParticle*>& m)
{
    for(uint i = 0 ; i < m.size(); i++ )
    {
        int id = TMath::Abs(m[i]->pdgId()) ;
        switch(id) {
            //case 22: //- photon
            case 23:
            case 24:
            case 25:
            case 1000022:
            case 1000023:
            case 1000024:
            case 1000025:
            case 1000035:
            case 1000039:
                return true;
            default : break;
        }
    }
    return false;
}

bool GenParticleManager::comesFromTau(std::vector<const GenParticle*>& m)
{
    for(uint i = 0 ; i < m.size(); i++ ) 
    {
        if( TMath::Abs(m[i]->pdgId()) == 15 ) return true;
    }
    return false;
}

bool GenParticleManager::comesFromBBaryon(std::vector<const GenParticle*>& m)
{
    for(uint i = 0 ; i < m.size(); i++ )
    {
        int id = TMath::Abs(m[i]->pdgId()) ;
        int mod = (id / 1000)%10;
        if( mod == 5 ) return true;
    }
    return false;
}

bool GenParticleManager::comesFromCBaryon(std::vector<const GenParticle*>& m)
{
    for(uint i = 0 ; i < m.size(); i++ )
    {
        int id = TMath::Abs(m[i]->pdgId()) ;
        int mod = (id / 1000)%10;
        if( mod == 4 ) return true;
    }
    return false;
}

bool GenParticleManager::comesFromBMeson(std::vector<const GenParticle*>& m)
{
    for(uint i = 0 ; i < m.size(); i++ ) 
    {
        int id = TMath::Abs(m[i]->pdgId()) ; 
        int mod = id % 1000;
        if( (mod >= 500 && mod < 600) || ((id / 1000)%10 == 5)  ) return true;
    }
    return false;
}

bool GenParticleManager::comesFromDMeson(std::vector<const GenParticle*>& m)
{
    for(uint i = 0 ; i < m.size(); i++ ) 
    {
        int id = TMath::Abs(m[i]->pdgId()) ; 
        int mod = id % 1000;
        if( (mod >= 400 && mod < 500) || ((id / 1000)%10 == 4)  ) return true;
    }
    return false;
}

bool GenParticleManager::comesFromPi0(std::vector<const GenParticle*>& m)
{
    for(uint i = 0 ; i < m.size(); i++ )
    {
        if( TMath::Abs(m[i]->pdgId()) == 111 ) return true;
    }
    return false;
}

bool GenParticleManager::comesFromUDS(std::vector<const GenParticle*>& m)
{
    for(uint i = 0 ; i < m.size(); i++ )
    {
        int id = TMath::Abs(m[i]->pdgId()) ;
        //std::cout<<"Checking "<<id<<std::endl;
        int mod = id % 1000;
        if( (mod >= 110) && (mod < 400) && ((id/1000)%10 < 4) && (id!=2212)) return true;
    }
    return false;
}

bool GenParticleManager::comesFromPhoton(std::vector<const GenParticle*>& m)
{
    for(uint i = 0 ; i < m.size(); i++ )
    {
        if( TMath::Abs(m[i]->pdgId()) == 22 ) return true;
    }
    return false;
}

int GenParticleManager::originReduced(int origin) {
    
    int originR; 
    if (origin <2 )
        originR = 0;
    else if ( ((origin>1) && (origin<6)) || ((origin>7) && (origin<12)) || (origin==14) )
        originR = 1;
    else if ((origin == 6 ) || (origin==7) || (origin==12) || (origin==13) || (origin==15))
        originR = 2;
    else originR = origin - 13;
    
    if ((originR == 4) || (originR == 5))
        originR = 3;
    else if (originR == 6) {
        originR = 4;
    }
/*
    0    "Prompt",
    1    "b-jets",
    2    "c-jets",
    3    "uds",
    4    "Unknown"
*/    
    return originR;
}

/*
 W_L,  // 0
 W_T_L, // 1
 W_B_L, // 2
 W_B_D_L, //3
 W_B_D_T_L, // 4
 W_B_T_L, // 5
 W_D_L, // 6
 W_D_T_L, //7
 B_L, // 8
 B_D_L, //9
 B_D_T_L, //10
 B_T_L,  // 11
 D_L, //12
 D_T_L, //13
 B_Baryon, // 14
 C_Baryon, //15
 pi_0, //16
 photon_, //17
 F_L, //18
 N_U_L_L // 19
 */

const GenParticle* GenParticleManager::matchedMC(const pat::Muon *pReco) {
    const GenParticle* mom = 0;
    TLorentzVector Gen1, Gen2;
    if (!pReco) return 0;
    Gen1.SetPtEtaPhiE(pReco->pt(),pReco->eta(),pReco->phi(),pReco->energy());
    double deltaRreco = 9999.;
    for(GenParticleCollection::const_reverse_iterator p = Collection->rbegin() ; p != Collection->rend() ; p++ ) {
        if (p->status()!=1) continue;
        Gen2.SetPtEtaPhiE(p->pt(),p->eta(),p->phi(),p->energy());
        double deltaRcur = Gen1.DeltaR(Gen2);
        if (deltaRcur < deltaRreco) {
            mom = &*p;
            deltaRreco = deltaRcur;
        }
    }
    return mom;
}
const GenParticle* GenParticleManager::matchedMC(const pat::Electron *pReco) {
    const GenParticle* mom = 0;
    if (!pReco) return 0;
    TLorentzVector Gen1, Gen2;
    Gen1.SetPtEtaPhiE(pReco->pt(),pReco->eta(),pReco->phi(),pReco->energy());
    double deltaRreco = 9999.;
    for(GenParticleCollection::const_reverse_iterator p = Collection->rbegin() ; p != Collection->rend() ; p++ ) {
        if (p->status()!=1) continue;
        Gen2.SetPtEtaPhiE(p->pt(),p->eta(),p->phi(),p->energy());
        double deltaRcur = Gen1.DeltaR(Gen2);
        if (deltaRcur < deltaRreco) {
            mom = &*p;
            deltaRreco = deltaRcur;
        }
    
    }
    return mom;
}
