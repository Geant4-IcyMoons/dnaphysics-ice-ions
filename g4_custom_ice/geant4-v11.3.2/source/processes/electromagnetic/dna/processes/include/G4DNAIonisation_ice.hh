//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
// Custom ice ionisation process (based on G4DNAIonisation)


#ifndef G4DNAIonisation_ice_h
#define G4DNAIonisation_ice_h 1

#include "G4VEmProcess.hh"
#include "G4DNAGenericIonsManager.hh"
#include "G4Electron.hh"
#include "G4Proton.hh"

// Available models
#include "G4DNABornIonisationModel_ice.hh"
#include "G4DNARuddIonisationModel_ice.hh"
#include "G4DNARuddIonisationExtendedModel_ice.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class G4DNAIonisation_ice : public G4VEmProcess

{
public:

  G4DNAIonisation_ice(const G4String& processName ="DNAIonisation",
		     G4ProcessType type = fElectromagnetic);

  ~G4DNAIonisationi_ice() override;

  G4bool IsApplicable(const G4ParticleDefinition&) override;

  virtual void PrintInfo();

protected:

  void InitialiseProcess(const G4ParticleDefinition*) override;

private:

  G4bool       isInitialised{false};
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
