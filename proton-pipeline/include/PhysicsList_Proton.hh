#ifndef PhysicsList_Proton_h
#define PhysicsList_Proton_h 1

#include "G4VModularPhysicsList.hh"
#include "globals.hh"

class G4VPhysicsConstructor;

class PhysicsList_Proton : public G4VModularPhysicsList
{
  public:
    PhysicsList_Proton();
    ~PhysicsList_Proton() override;

    void ConstructParticle() override;
    void ConstructProcess() override;

    void AddPhysics(const G4String&);
    void SetTrackingCut(G4bool);

  private:
    void TrackingCut();

    G4VPhysicsConstructor* fEmPhysicsList = nullptr;
};

#endif
