#ifndef G4DNAZBLFullElastic_h
#define G4DNAZBLFullElastic_h 1

#include "G4ParticleChange.hh"
#include "G4VDiscreteProcess.hh"

#include <array>
#include <vector>

class G4Material;
class G4MaterialCutsCouple;
class G4ParticleDefinition;

// Full nuclear-elastic reference based on the universal ZBL screened
// potential. This is a validation-pending independent-atom baseline, not a
// charge-resolved molecular soft potential and not an additive correction to
// the retained-domain NLH hard process.
class G4DNAZBLFullElastic : public G4VDiscreteProcess
{
  public:
    enum class Target { Hydrogen = 0, Oxygen = 1 };

    explicit G4DNAZBLFullElastic(
        const G4ParticleDefinition* projectile,
        G4double minimumEnergy,
        G4double maximumEnergy,
        G4double minimumTransfer,
        G4double recoilThreshold,
        const G4String& processName = "DNAZBLFullElastic",
        G4ProcessType type = fElectromagnetic);
    ~G4DNAZBLFullElastic() override = default;

    G4bool IsApplicable(const G4ParticleDefinition&) override;
    G4double GetCrossSection(G4double kineticEnergy,
                             const G4MaterialCutsCouple*) override;
    G4VParticleChange* PostStepDoIt(const G4Track&, const G4Step&) override;

    G4double MacroscopicCrossSection(const G4Material*, G4double) const;
    static G4double WaterMoleculeNumberDensity(const G4Material*);

    // Public numerical primitives used by the independent C++ regression
    // test and the Python reference implementation.
    static G4double Screening(G4double reducedRadius);
    static G4double ScreeningDerivative(G4double reducedRadius);
    static G4double ScreeningLength(G4int projectileZ, G4int targetZ);
    static G4double MaximumRecoilEnergy(
        G4double projectileMass, G4double targetMass,
        G4double kineticEnergy);
    static G4double CosThetaCM(
        G4int projectileZ, G4double projectileMass,
        G4int targetZ, G4double targetMass,
        G4double kineticEnergy, G4double impactParameter);
    static G4double PairCrossSection(
        G4int projectileZ, G4double projectileMass,
        G4int targetZ, G4double targetMass,
        G4double kineticEnergy, G4double minimumTransfer);

  protected:
    G4double GetMeanFreePath(const G4Track&, G4double,
                             G4ForceCondition*) override;

  private:
    struct PairTable
    {
      G4int targetZ = 0;
      G4int targetA = 0;
      G4double targetMass = 0.;
      std::vector<G4double> logEnergy;
      std::vector<G4double> logCrossSection;
    };

    static G4int AtomicNumber(const G4ParticleDefinition*);
    static G4int MassNumber(const G4ParticleDefinition*);
    static G4double CenterOfMassKineticEnergy(
        G4double projectileMass, G4double targetMass,
        G4double kineticEnergy);
    static G4double PairNumberDensity(const G4Material*, Target);
    static G4bool ReadValidationOverride();

    PairTable BuildPairTable(Target) const;
    G4double InterpolatedPairCrossSection(
        const PairTable&, G4double kineticEnergy) const;
    G4bool MatchesProjectile(const G4ParticleDefinition*) const;
    void Scatter(const G4Track&, const PairTable&, G4double cosThetaCM);

    G4int fProjectileZ = 0;
    G4int fProjectileA = 0;
    G4double fProjectileMass = 0.;
    G4double fMinimumEnergy = 0.;
    G4double fMaximumEnergy = 0.;
    G4double fMinimumTransfer = 0.;
    G4double fRecoilThreshold = 0.;
    std::array<PairTable, 2> fPairTables;
    G4ParticleChange fParticleChange;
};

#endif
