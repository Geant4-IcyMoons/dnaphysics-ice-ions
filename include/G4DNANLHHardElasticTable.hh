#ifndef G4DNANLHHardElasticTable_h
#define G4DNANLHHardElasticTable_h 1

#include "globals.hh"

#include <array>
#include <string>
#include <vector>

// Immutable runtime representation of one checksum-linked NLH projectile
// product. Cross sections are evaluated analytically; only theta_cm is
// interpolated.
class G4DNANLHHardElasticTable
{
  public:
    enum class Target
    {
      Hydrogen = 0,
      Oxygen = 1
    };

    struct TwoBodyOutcome
    {
      G4double projectileEnergy = 0.;
      G4double recoilEnergy = 0.;
      G4double projectileTransverseMomentum = 0.;
      G4double projectileLongitudinalMomentum = 0.;
      G4double recoilTransverseMomentum = 0.;
      G4double recoilLongitudinalMomentum = 0.;
    };

    explicit G4DNANLHHardElasticTable(const std::string& path);

    G4bool InEnergyRange(Target target, G4double projectileEnergy) const;
    G4double EnergyMinimum(Target target) const;
    G4double EnergyMaximum(Target target) const;

    // Threshold-defined microscopic cross section:
    // pi*r_th^2*(1 - V_min/E_cm) above threshold, zero otherwise.
    // This method never interpolates the cross section.
    G4double HardCrossSection(Target target, G4double projectileEnergy) const;
    G4double ThetaCM(Target target,
                     G4double projectileEnergy,
                     G4double areaQuantile) const;
    TwoBodyOutcome Outcome(Target target,
                           G4double projectileEnergy,
                           G4double thetaCM) const;

    const std::string& Path() const { return fPath; }
    const std::string& SourceCsvSha256() const { return fSourceCsvSha256; }
    const std::string& ReleaseStatus() const { return fReleaseStatus; }
    G4double MinimumTurningPotential() const { return fMinimumTurningPotential; }

  private:
    struct EnergyNode
    {
      G4double energy = 0.;
      std::vector<G4double> quantiles;
      std::vector<G4double> thetaCM;
    };

    struct PairTable
    {
      G4double targetMass = 0.;
      G4double thresholdRadius = 0.;
      G4double energyMinimum = 0.;
      G4double energyMaximum = 0.;
      std::vector<EnergyNode> nodes;
    };

    struct PairKinematics
    {
      G4double invariantMass = 0.;
      G4double relativeKineticEnergy = 0.;
      G4double incomingMomentum = 0.;
      G4double momentumCM = 0.;
      G4double betaCM = 0.;
      G4double gammaCM = 0.;
    };

    static std::size_t Index(Target target);
    const PairTable& Pair(Target target) const;
    PairKinematics Kinematics(Target target, G4double projectileEnergy) const;
    static G4double AngleAtQuantile(const EnergyNode& node, G4double quantile);
    void Validate() const;

    std::string fPath;
    std::string fProjectile;
    std::string fSourceCsvSha256;
    std::string fReleaseStatus;
    G4double fProjectileMass = 0.;
    G4double fMinimumTurningPotential = 0.;
    std::array<PairTable, 2> fPairs;
};

#endif
