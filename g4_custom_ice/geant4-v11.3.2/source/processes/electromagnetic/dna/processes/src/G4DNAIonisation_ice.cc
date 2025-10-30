//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
// Custom ice ionisation process (based on G4DNAIonisation)

#include "G4DNAIonisation._icehh"
#include "G4LEPTSIonisationModel.hh"
#include "G4SystemOfUnits.hh"
#include "G4LowEnergyEmProcessSubType.hh"

//SEB
#include "G4GenericIon.hh"
#include "G4Positron.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

using namespace std;

G4DNAIonisation_ice::G4DNAIonisation_ice(const G4String& processName,
                                 G4ProcessType type) :
    G4VEmProcess(processName, type)
{
  SetProcessSubType(fLowEnergyIonisation);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4DNAIonisation_ice::~G4DNAIonisation_ice()
= default;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4bool G4DNAIonisation_ice::IsApplicable(const G4ParticleDefinition& p)
{
  G4DNAGenericIonsManager *instance;
  instance = G4DNAGenericIonsManager::Instance();

  return (&p == G4Electron::Electron() || &p == G4Positron::Positron()
          || &p == G4Proton::Proton() || &p == instance->GetIon("hydrogen")
          || &p == instance->GetIon("alpha++")
          || &p == instance->GetIon("alpha+")
          || &p == instance->GetIon("helium")
          || &p == G4GenericIon::GenericIonDefinition());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4DNAIonisation_ice::InitialiseProcess(const G4ParticleDefinition* p)
{
  if(!isInitialised)
  {
    isInitialised = true;
    SetBuildTableFlag(false);

    G4String name = p->GetParticleName();

    if(name == "e-")
    {
      if(EmModel() == nullptr)
      {
        auto  born =
            new G4DNABornIonisationModel_ice();
        SetEmModel(born);
        born->SetLowEnergyLimit(2. * eV);
        born->SetHighEnergyLimit(100. * eV);
      }
      AddEmModel(1, EmModel());
    }
    else if(name == "e+")
    {
      if(EmModel() == nullptr)
      {
        auto  lepts =
            new G4LEPTSIonisationModel();
        SetEmModel(lepts);
        lepts->SetLowEnergyLimit(2. * eV);
        lepts->SetHighEnergyLimit(100. * eV);
      }
      AddEmModel(1, EmModel());
    }

    if(name == "proton")
    {
      if(EmModel(0) == nullptr)
      {
        auto  rudd =
             new G4DNARuddIonisationModel_ice();
        rudd->SetLowEnergyLimit(0 * eV);
        rudd->SetHighEnergyLimit(500 * keV); //Need to see if data can cover threshold between Rudd semi-empirical and Born
        SetEmModel(rudd);

        auto  born =
            new G4DNABornIonisationModel();
        born->SetLowEnergyLimit(500 * keV);
        born->SetHighEnergyLimit(100 * MeV);
        SetEmModel(born);
      }

      AddEmModel(1, EmModel());
      if(EmModel(1) != nullptr) AddEmModel(2, EmModel(1));
    }

    if(name == "hydrogen")
    {
      if(EmModel() == nullptr)
      {
        auto  rudd =
             new G4DNARuddIonisationModel_ice();
         SetEmModel(rudd);
        rudd->SetLowEnergyLimit(2 * eV);
        rudd->SetHighEnergyLimit(100 * MeV);
      }
      AddEmModel(1, EmModel());
    }

    if(name == "alpha" || name == "alpha+" || name == "helium")
    {
      if(EmModel() == nullptr)
      {
        auto  rudd =
            new G4DNARuddIonisationModel_ice();
        SetEmModel(rudd);
        rudd->SetLowEnergyLimit(0 * keV);
        rudd->SetHighEnergyLimit(100 * MeV);
      }
      AddEmModel(1, EmModel());
    }

    // Extension to HZE proposed by Z. Francis

    //SEB
    if(/*name == "carbon" || name == "nitrogen" || name == "oxygen" || name == "iron" ||*/
    name == "GenericIon")
    //
    {
      if(EmModel() == nullptr)
      {
        auto  ruddExt =
            new G4DNARuddIonisationExtendedModel_ice();
        SetEmModel(ruddExt);
        ruddExt->SetLowEnergyLimit(0 * keV);
        //SEB: 1e6*MeV by default - updated in model class
        //EmModel()->SetHighEnergyLimit(p->GetAtomicMass()*1e6*MeV);
        ruddExt->SetHighEnergyLimit(1e6 * MeV);
      }
      AddEmModel(1, EmModel());
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4DNAIonisation_ice::PrintInfo()
{
  if(EmModel(1) != nullptr)
  {
    G4cout << " Total cross sections computed from " << EmModel(0)->GetName()
           << " and " << EmModel(1)->GetName() << " models" << G4endl;
  }
  else
  {
    G4cout << " Total cross sections computed from "
           << EmModel()->GetName()
           << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
