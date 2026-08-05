#ifndef G4DNANLHHardElastic_h
#define G4DNANLHHardElastic_h 1

#include "G4DNANLHHardElasticTable.hh"
#include "G4ParticleChange.hh"
#include "G4VDiscreteProcess.hh"

#include <memory>
#include <string>

class G4Material;
class G4MaterialCutsCouple;

// Carbon-12 threshold-defined NLH hard elastic scattering on the independent
// H and O atoms of pure water ice. A discrete process is used deliberately:
// G4VEmProcess rescales arbitrary GenericIon energies to proton-equivalent
// energy, whereas these tables use total projectile kinetic energy.
class G4DNANLHHardElastic : public G4VDiscreteProcess
{
  public:
    explicit G4DNANLHHardElastic(
        const G4String& processName = "DNANLHCarbonHardElastic",
        G4ProcessType type = fElectromagnetic);
    ~G4DNANLHHardElastic() override = default;

    G4bool IsApplicable(const G4ParticleDefinition&) override;
    G4double GetCrossSection(G4double kineticEnergy,
                             const G4MaterialCutsCouple*) override;
    G4VParticleChange* PostStepDoIt(const G4Track&, const G4Step&) override;

    static G4double WaterMoleculeNumberDensity(const G4Material*);
    G4double MacroscopicCrossSection(const G4Material*, G4double) const;

  protected:
    G4double GetMeanFreePath(const G4Track&,
                             G4double previousStepSize,
                             G4ForceCondition*) override;

  private:
    static std::string ResolveDataPath();
    static std::shared_ptr<const G4DNANLHHardElasticTable> SharedTable(
        const std::string& path);
    static G4bool IsCarbon12(const G4ParticleDefinition*);
    G4ParticleChange fParticleChange;
    std::shared_ptr<const G4DNANLHHardElasticTable> fTable;
};

#endif
