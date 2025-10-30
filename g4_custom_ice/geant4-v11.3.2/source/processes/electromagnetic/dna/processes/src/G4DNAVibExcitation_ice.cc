//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
// Custom ice vibrational excitation process (based on G4DNAVibExcitation)
#include "G4DNAVibExcitation_ice.hh"
#include "G4DNAMichaudExcitationModel.hh"
#include "G4LEPTSVibExcitationModel.hh"
#include "G4SystemOfUnits.hh"
#include "G4Positron.hh"
#include "G4LowEnergyEmProcessSubType.hh"

using namespace std;

G4DNAVibExcitation_ice::G4DNAVibExcitation_ice(const G4String& processName,
  G4ProcessType type):G4VEmProcess (processName, type)
{
  SetProcessSubType(fLowEnergyVibrationalExcitation);
}

G4bool G4DNAVibExcitation_ice::IsApplicable(const G4ParticleDefinition& p)
{
  return (&p == G4Electron::Electron() || &p == G4Positron::Positron());
}

void G4DNAVibExcitation_ice::InitialiseProcess(const G4ParticleDefinition* p)
{
  if(!isInitialised)
  {
    isInitialised = true;
    SetBuildTableFlag(false);
    G4String name = p->GetParticleName();
    if(name == "e-")
    {
      if(nullptr == EmModel())
      {
  SetEmModel(new G4DNAMichaudExcitationModel);
        EmModel()->SetLowEnergyLimit(2*eV);
        EmModel()->SetHighEnergyLimit(100*eV);
      }
      AddEmModel(1, EmModel());
    }
    else if(name == "e+")
    {
      if(nullptr == EmModel())
      {
        SetEmModel(new G4LEPTSVibExcitationModel);
        EmModel()->SetLowEnergyLimit(2*eV);
        EmModel()->SetHighEnergyLimit(100*eV);
      }
      AddEmModel(1, EmModel());
    }
  }
}

void G4DNAVibExcitation_ice::ProcessDescription(std::ostream& out) const
{
  out << "  DNA Vibrational Excitation (ice)";
  G4VEmProcess::ProcessDescription(out);
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
