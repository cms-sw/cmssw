#include "RecoVertex/KinematicFitPrimitives/interface/KinematicTree.h"

KinematicTree::KinematicTree()
{
 empt = true;
 treeWalker = 0;
}

 KinematicTree::~KinematicTree()
{
 delete treeWalker;
}


bool KinematicTree::isEmpty() const
 {return empt;}
 
bool KinematicTree::isConsistent() const
{
 movePointerToTheTop();
 bool des = false;
 if(!treeWalker->nextSibling()) des = true;
 return des;
}

void KinematicTree::addParticle(RefCountedKinematicVertex prodVtx, 
                                RefCountedKinematicVertex  decVtx, 
			         RefCountedKinematicParticle part)
{
 part->setTreePointer(this);
 treeGraph.addEdge(prodVtx,decVtx,part);
 empt = false;
 movePointerToTheTop();
 prodVtx->setTreePointer(this);
 decVtx->setTreePointer(this);
}

std::vector<RefCountedKinematicParticle> KinematicTree::finalStateParticles() const
{
 if(isEmpty() || !(isConsistent()))
 {
  throw VertexException("KinematicTree::finalStateParticles; tree is empty or not consistent");
 }else{
  RefCountedKinematicParticle initial = currentParticle();
  std::vector<RefCountedKinematicParticle> rs;
  movePointerToTheTop();
  if(!(leftFinalParticle()))
  {
   std::cout<<"top particle has no daughters, empty vector returned"<<std::endl;
  }else{
//now pointer is at the  most left final particle
   rs.push_back(currentParticle()); 
   bool next_right = true; 
   bool up = true;
   do
   {
    next_right = movePointerToTheNextChild();
    if(next_right)
    {
//if there's a way to the right,
//we go right and down possible    
     leftFinalParticle();
     rs.push_back(currentParticle()); 
    }else{
//once there's no way to right anymore
//trying to find a way upper    
     up = movePointerToTheMother();
    }
//loop stops when we are at the top:
//no way up, no way to the right    
   }while(up);   
  } 
//getting the pointer back  
  bool back = findParticle(initial);
  if(!back) throw VertexException("KinematicTree::FinalStateParticles; error occured while getting back");
  return rs;
 }
}

bool KinematicTree::leftFinalParticle() const
{
 bool res = false;
 if(movePointerToTheFirstChild())
 {
  res = true;
  bool next = true;
  do
  {
    next = movePointerToTheFirstChild(); 
  }while(next);
 }else{
  res =false;
 }
 return res;
}

 
 RefCountedKinematicParticle KinematicTree::topParticle() const
 {
   if(isEmpty()) throw VertexException("KinematicTree::topParticle; tree is empty!");
//putting pointer to the top of the tree 
   movePointerToTheTop();
   return treeWalker->current().second;					    
 }
 
 
 RefCountedKinematicVertex KinematicTree::currentDecayVertex() const
 {
   if(isEmpty()) throw VertexException("KinematicTree::currentDecayVertex; tree is empty!");
   return treeWalker->current().first;
 }
 
 std::pair<bool,RefCountedKinematicParticle>  KinematicTree::motherParticle() const
 {
  if(isEmpty()) throw VertexException("KinematicTree::motherParticle; tree is empty!");
  bool top = currentProductionVertex()->vertexIsValid();
   RefCountedKinematicParticle cr  = treeWalker->current().second;
   bool up = treeWalker->parent();
   std::pair<bool,RefCountedKinematicParticle> res; 
   if(up && top){
    RefCountedKinematicParticle pr = treeWalker->current().second;
    
//now putting the pointer back   
    bool fc = treeWalker->firstChild();
    if(!fc) throw VertexException("KinematicTree::motherParticle; tree is incorrect!");
    if(*(treeWalker->current().second) != *cr)
    {
     do{
      bool nx = treeWalker->nextSibling();
      if(!nx) throw VertexException("KinematicTree::motherParticle; tree is incorrect!");
     }while(*(treeWalker->current().second) != *cr);
    }
    res = std::pair<bool,RefCountedKinematicParticle>(true,pr);
    return res;
   }else{
    RefCountedKinematicParticle fk;
    return std::pair<bool,RefCountedKinematicParticle>(false,fk);
   }
 }
 
 std::vector<RefCountedKinematicParticle>  KinematicTree::daughterParticles() const
 {
  if(isEmpty()) throw VertexException("KinematicTree::daughterParticles; tree is empty!");
  std::vector<RefCountedKinematicParticle> sResult;
  RefCountedKinematicParticle initial = currentParticle();
  bool down  = treeWalker->firstChild();
  if(down)
  {
    sResult.push_back(treeWalker->current().second);
    bool sibling = true;
    do
    {
     sibling = treeWalker->nextSibling();
     if(sibling) sResult.push_back(treeWalker->current().second); 
    }while(sibling);
  }

//getting the pointer back to the mother  
  bool back = findParticle(initial);
  if(!back) throw VertexException("KinematicTree::daughterParticles; error occured while getting back");
  return sResult;
 }

void KinematicTree::movePointerToTheTop() const
{
 if(isEmpty()) throw VertexException("KinematicTree::movePointerToTheTop; tree is empty!");
 delete treeWalker;
 treeWalker = new graphwalker<RefCountedKinematicVertex,
                              RefCountedKinematicParticle>(treeGraph);
//now pointer is a pair: fake vertex and 
//icoming 0 pointer to the particle
//moving it to decayed particle
 bool move = treeWalker->firstChild();
 if(!move) throw VertexException("KinematicTree::movePointerToTheTop; non consistent tree?");
}

RefCountedKinematicVertex KinematicTree::currentProductionVertex() const
{
  if(isEmpty()) throw VertexException("KinematicTree::currentProductionVertex; tree is empty!");
//current particle
  RefCountedKinematicParticle initial = currentParticle();
  
  bool up;
  bool down;
  RefCountedKinematicVertex res;
  up = movePointerToTheMother();
  
 if(up)
 { 
  res = treeWalker->current().first;
		              		       
//pointer moved so we going back
  down = treeWalker->firstChild();
   
//_down_ variable is always TRUE here, if
//the tree is valid.
  if(down){
   if(initial == treeWalker->current().second)
   {
    return res;
   }else{
    bool next = true;
    do
    {
     next = treeWalker->nextSibling();
     if(treeWalker->current().second == initial) next = false;
    }while(next);
    return res;
   }
  }else{throw VertexException("KinematicTree::Navigation failed, tree invalid?");}			
 }else
 { 
//very unprobable case. This efectively means that user is
//already out of the tree. Moving back to the top
 delete treeWalker;
 treeWalker = new graphwalker<RefCountedKinematicVertex,
                              RefCountedKinematicParticle>
                                              (treeGraph);
 res = treeWalker->current().first;					      
//now pointer is a pair: fake vertex and 
//icoming 0 pointer to the particle
//moving it to decayed particle
 bool move = treeWalker->firstChild();
 if(!move) throw VertexException("KinematicTree::movePointerToTheTop; non consistent tree?");
 return res;
 } 
}
 
RefCountedKinematicParticle KinematicTree::currentParticle() const
{
 if(isEmpty()) throw VertexException("KinematicTree::currentParticle; tree is empty!");
 return treeWalker->current().second;
}

bool KinematicTree::movePointerToTheMother() const
{
 if(isEmpty()) throw VertexException("KinematicTree::movePointerToTheMother; tree is empty!");
 bool up = treeWalker->parent();
 bool cr = treeWalker->current().first->vertexIsValid();
 return (up && cr);
}

bool KinematicTree::movePointerToTheFirstChild() const
{
 if(isEmpty()) throw VertexException("KinematicTree::movePointerToTheFirstChild; tree is empty!");
 return treeWalker->firstChild();
}
 
bool KinematicTree::movePointerToTheNextChild() const
{
 if(isEmpty()) throw VertexException("KinematicTree::movePointerToTheNextChild; tree is empty!");
 bool res = treeWalker->nextSibling();
 return res;
}

bool KinematicTree::findParticle(const RefCountedKinematicParticle part) const
{
 if(isEmpty() || !(isConsistent()))
 {
  throw VertexException("KinematicTree::findParticle; tree is empty or not consistent");
 }else{
  bool res = false;
  movePointerToTheTop();
  if(currentParticle() == part)
  {
   res = true;
  }else if(leftBranchSearch(part)){
   res = true;
  }else{
   bool found = false;
   bool up = true;
   bool next_right = false;
   do
   {
//    if(*(currentParticle()) == *part) found = true;
    next_right = movePointerToTheNextChild();
    if(next_right)
    {
     found = leftBranchSearch(part);
    }else{
     up = movePointerToTheMother();
     if(currentParticle() == part) found = true;
    }
   }while(up && !found);
   res = found;
  } 
  return res;
 }
}


bool KinematicTree::leftBranchSearch(RefCountedKinematicParticle part) const
{
 bool found = false;
 bool next = true;
 if(currentParticle() == part)
 {
  found = true;
 }else{
  do
  {
   next = movePointerToTheFirstChild();
   if(currentParticle() == part)
   {
    found = true;
   }
  }while(next && !found);
  }
 return found;
}

bool KinematicTree::findDecayVertex(const RefCountedKinematicVertex vert)const
{
 if(isEmpty() || !(isConsistent()))
 {
  throw VertexException("KinematicTree::findParticle; tree is empty or not consistent");
 }else{
 bool res = false;
 movePointerToTheTop();
 if(currentDecayVertex() == vert)
 {
  res = true;
 }else if(leftBranchVertexSearch(vert)){
  res = true;
 }else{
  bool up = true;
  bool fnd = false;
  do
  {
   if(movePointerToTheNextChild())
   {
    fnd = leftBranchVertexSearch(vert);
   }else{
    up=movePointerToTheMother();
    if(currentDecayVertex() == vert) fnd = true;
   }   
  }while(up && !fnd);
  res = fnd;
 }
 return res;
 }
}

bool KinematicTree::findDecayVertex(KinematicVertex *vert)const
{
 if(isEmpty() || !(isConsistent()))
 {
  throw VertexException("KinematicTree::findParticle; tree is empty or not consistent");
 }else{
 bool res = false;
 movePointerToTheTop();
 if(*currentDecayVertex() == vert)
 {
  res = true;
 }else if(leftBranchVertexSearch(vert)){
  res = true;
 }else{
  bool up = true;
  bool fnd = false;
  do
  {
   if(movePointerToTheNextChild())
   {
    fnd = leftBranchVertexSearch(vert);
   }else{
    up=movePointerToTheMother();
    if(currentDecayVertex() == vert) fnd = true;
   }   
  }while(up && !fnd);
  res = fnd;
 }
 return res;
 }
}


bool KinematicTree::leftBranchVertexSearch(RefCountedKinematicVertex vtx) const
{
 bool found = false;
 if(currentDecayVertex() == vtx)
 {
  found = true;
 }else{
  bool next = true;
  bool res = false;
  do
  {
   next = movePointerToTheFirstChild();
   if(currentDecayVertex() == vtx) res = true;
  }while(next && !res); 
  found  = res;
 }
 return found;
}

void KinematicTree::leftBranchAdd(KinematicTree * otherTree, RefCountedKinematicVertex vtx)
{
  RefCountedKinematicVertex previous_decay = otherTree->currentDecayVertex(); 
 
//children of current particle of the
//other tree: in the end the whole 
//left branch should be added. 
  bool next = true;
  do
  {
   next = otherTree->movePointerToTheFirstChild();
   if(next)
   {
    RefCountedKinematicParticle par = otherTree->currentParticle();
    RefCountedKinematicVertex current_decay = otherTree->currentDecayVertex();
    addParticle(previous_decay, current_decay, par);
    previous_decay = current_decay;
   }
  }while(next);
}

void KinematicTree::replaceCurrentParticle(RefCountedKinematicParticle newPart) const
{
 RefCountedKinematicParticle cDParticle = currentParticle();
 bool replace = treeGraph.replaceEdge(cDParticle,newPart);
 if(!replace) throw VertexException("KinematicTree::Particle To Replace not found");
}
      
void KinematicTree::replaceCurrentVertex(RefCountedKinematicVertex newVert) const
{
 RefCountedKinematicVertex cDVertex = currentDecayVertex();
 bool replace = treeGraph.replace(cDVertex,newVert);
 if(! replace) throw VertexException("KinematicTree::Vertex To Replace not found");
}

			
void KinematicTree::addTree(RefCountedKinematicVertex vtx, KinematicTree * tr)
{
//adding new tree to the existing one:
 bool fnd = findDecayVertex(vtx);
 if(!fnd) throw VertexException("KinematicTree::addTree; Current tree does not contain the vertex passed");
 tr->movePointerToTheTop();
 
// adding the root of the tree: 
 RefCountedKinematicParticle mP = tr->currentParticle(); 
 RefCountedKinematicVertex   dec_vertex = tr->currentDecayVertex();
 addParticle(vtx,dec_vertex,mP);

// adding the left branch if any 
 leftBranchAdd(tr,dec_vertex);

//now the pointer is at the left down
//edge of the otherTree.
//current tree pointer is where the
//add operation was stoped last time.
 bool right = true;
 bool up = true;
 do
 {
  right = tr->movePointerToTheNextChild();
  if(right)
  {

//production vertex is already at the current tree  
//adding current partilce
   RefCountedKinematicVertex prodVertex = tr->currentProductionVertex();
   RefCountedKinematicParticle cPart = tr->currentParticle();
   RefCountedKinematicVertex decVertex = tr->currentDecayVertex();
   addParticle(prodVertex, decVertex, cPart);
   
//adding the rest of the branch   
   leftBranchAdd(tr,decVertex);
  }else{
   up = tr->movePointerToTheMother(); 
  }
 }while(up);
}			
			
