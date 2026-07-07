//
// Proton-only excitation model for generated ice DCS/TCS tables.
//

#ifndef G4DNAEmfietzoglou_iceProtonExcitationModel_h
#define G4DNAEmfietzoglou_iceProtonExcitationModel_h 1

#include "G4DNAEmfietzoglou_iceProtonDcsTable.hh"
#include "G4ParticleChangeForGamma.hh"
#include "G4VEmModel.hh"

class G4DNAEmfietzoglou_iceProtonExcitationModel : public G4VEmModel
{
public:
  G4DNAEmfietzoglou_iceProtonExcitationModel(
      const G4ParticleDefinition* p = nullptr,
      const G4String& nam = "DNAEmfietzoglou_iceProtonExcitationModel");
  ~G4DNAEmfietzoglou_iceProtonExcitationModel() override = default;

  G4DNAEmfietzoglou_iceProtonExcitationModel&
  operator=(const G4DNAEmfietzoglou_iceProtonExcitationModel&) = delete;
  G4DNAEmfietzoglou_iceProtonExcitationModel(
      const G4DNAEmfietzoglou_iceProtonExcitationModel&) = delete;

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

  void SelectStationary(G4bool input) { fStationary = input; }

  static G4int GetLastExcitationIndex();
  static void ClearLastExcitationIndex();
  static G4double GetLastPartialSigma_cm2();
  static void ClearLastPartialSigma_cm2();

private:
  const std::vector<G4double>* fpMolWaterDensity = nullptr;
  G4ParticleChangeForGamma* fParticleChangeForGamma = nullptr;
  G4DNAEmfietzoglou_iceProtonDcsTable fTable;
  G4bool fInitialised = false;
  G4bool fStationary = false;
};

#endif
