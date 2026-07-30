//
// Proton/alpha ionisation model for generated ice DCS/TCS tables.
//

#ifndef G4DNAEmfietzoglou_iceProtonIonisationModel_h
#define G4DNAEmfietzoglou_iceProtonIonisationModel_h 1

#include "G4DNAEmfietzoglou_iceProtonDcsTable.hh"
#include "G4DNAEmfietzoglou_iceProtonIonisationStructure.hh"
#include "G4ParticleChangeForGamma.hh"
#include "G4VEmModel.hh"

class G4DNAEmfietzoglou_iceProtonIonisationModel : public G4VEmModel
{
public:
  G4DNAEmfietzoglou_iceProtonIonisationModel(
      const G4ParticleDefinition* p = nullptr,
      const G4String& nam = "DNAEmfietzoglou_iceProtonIonisationModel");
  ~G4DNAEmfietzoglou_iceProtonIonisationModel() override = default;

  G4DNAEmfietzoglou_iceProtonIonisationModel&
  operator=(const G4DNAEmfietzoglou_iceProtonIonisationModel&) = delete;
  G4DNAEmfietzoglou_iceProtonIonisationModel(
      const G4DNAEmfietzoglou_iceProtonIonisationModel&) = delete;

  void Initialise(const G4ParticleDefinition*,
                  const G4DataVector& = *(new G4DataVector())) override;

  G4double CrossSectionPerVolume(const G4Material* material,
                                 const G4ParticleDefinition* p,
                                 G4double ekin,
                                 G4double emin,
                                 G4double emax) override;

  void SampleSecondaries(std::vector<G4DynamicParticle*>*,
                         const G4MaterialCutsCouple*,
                         const G4DynamicParticle*,
                         G4double tmin,
                         G4double maxEnergy) override;

  void SelectFasterComputation(G4bool) {}
  void SelectStationary(G4bool input) { fStationary = input; }

  static G4int GetLastShellIndex();
  static void ClearLastShellIndex();
  static G4double GetLastPartialSigma_cm2();
  static void ClearLastPartialSigma_cm2();

private:
  const G4ParticleDefinition* fProjectile = nullptr;
  const std::vector<G4double>* fpMolWaterDensity = nullptr;
  G4ParticleChangeForGamma* fParticleChangeForGamma = nullptr;
  G4DNAEmfietzoglou_iceProtonDcsTable fTable;
  G4DNAEmfietzoglou_iceProtonIonisationStructure fIonisationStructure;
  G4bool fInitialised = false;
  G4bool fStationary = false;
};

#endif
