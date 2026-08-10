#include "G4DNAZBLFullElastic.hh"

#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4Exception.hh"
#include "G4IonTable.hh"
#include "G4LorentzVector.hh"
#include "G4LowEnergyEmProcessSubType.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4NucleiProperties.hh"
#include "G4ParticleDefinition.hh"
#include "G4PhysicalConstants.hh"
#include "G4Proton.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "ModelDataRegistry.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {
constexpr G4int kEnergyPoints = 161;
constexpr G4int kRootIterations = 64;

G4double ClampCosine(G4double value)
{
  return std::max(-1., std::min(1., value));
}

[[noreturn]] void ProcessError(const char* code, const std::string& message)
{
  G4ExceptionDescription description;
  description << message;
  G4Exception("G4DNAZBLFullElastic", code, FatalException, description);
  throw std::runtime_error(message);
}
}  // namespace

G4DNAZBLFullElastic::G4DNAZBLFullElastic(
    const G4ParticleDefinition* projectile,
    G4double minimumEnergy,
    G4double maximumEnergy,
    G4double minimumTransfer,
    G4double recoilThreshold,
    const G4String& processName,
    G4ProcessType type)
  : G4VDiscreteProcess(processName, type),
    fProjectileZ(AtomicNumber(projectile)),
    fProjectileA(MassNumber(projectile)),
    fProjectileMass(projectile ? projectile->GetPDGMass() : 0.),
    fMinimumEnergy(minimumEnergy),
    fMaximumEnergy(maximumEnergy),
    fMinimumTransfer(minimumTransfer),
    fRecoilThreshold(recoilThreshold)
{
  if (!projectile || fProjectileZ <= 0 || fProjectileA <= 0 ||
      fProjectileMass <= 0.) {
    ProcessError("zblProcess001", "ZBL requires an atomic projectile.");
  }
  if (!(minimumEnergy > 0. && maximumEnergy > minimumEnergy &&
        minimumTransfer > 0. && recoilThreshold > 0.)) {
    ProcessError("zblProcess002", "Invalid ZBL energy or transfer bounds.");
  }
  if (!ReadValidationOverride()) {
    ProcessError(
        "zblProcess003",
        "The full ZBL ion--ice elastic model is validation_pending. Set "
        "DNA_ZBL_ALLOW_VALIDATION_PENDING=1 only for diagnostic and "
        "validation runs.");
  }

  SetProcessSubType(fLowEnergyElastic);
  pParticleChange = &fParticleChange;
  fPairTables[static_cast<std::size_t>(Target::Hydrogen)] =
      BuildPairTable(Target::Hydrogen);
  fPairTables[static_cast<std::size_t>(Target::Oxygen)] =
      BuildPairTable(Target::Oxygen);

  auto& registry = ModelDataRegistry::Instance();
  registry.Record(std::string("model_mode:") + processName,
                  "full_zbl_screened_binary_collision");
  registry.Record(std::string("model_reference:") + processName,
                  "Mendenhall-Weller-2005; universal-ZBL");
  registry.Record(std::string("model_release_status:") + processName,
                  "validation_pending");
  registry.Record(std::string("model_energy_convention:") + processName,
                  "total_projectile_kinetic_energy");
  registry.Record(std::string("model_charge_convention:") + processName,
                  "nuclear_atomic_number_charge_state_independent");

  G4cout << "Initialized full ZBL screened nuclear elastic model for "
         << projectile->GetParticleName() << " [Z=" << fProjectileZ
         << ", A=" << fProjectileA << ", total energy="
         << fMinimumEnergy / keV << " keV-" << fMaximumEnergy / MeV
         << " MeV, minimum transfer=" << fMinimumTransfer / eV
         << " eV, recoil threshold=" << fRecoilThreshold / eV << " eV]."
         << G4endl;
}

G4bool G4DNAZBLFullElastic::ReadValidationOverride()
{
  const char* raw = std::getenv("DNA_ZBL_ALLOW_VALIDATION_PENDING");
  if (!raw || !*raw) return false;
  const std::string value(raw);
  return value == "1" || value == "true" || value == "on" ||
         value == "yes";
}

G4int G4DNAZBLFullElastic::AtomicNumber(const G4ParticleDefinition* particle)
{
  if (!particle) return 0;
  if (particle == G4Proton::ProtonDefinition()) return 1;
  return particle->GetAtomicNumber();
}

G4int G4DNAZBLFullElastic::MassNumber(const G4ParticleDefinition* particle)
{
  if (!particle) return 0;
  if (particle == G4Proton::ProtonDefinition()) return 1;
  return particle->GetAtomicMass();
}

G4double G4DNAZBLFullElastic::Screening(G4double x)
{
  if (x < 0.) return 0.;
  return 0.1818 * std::exp(-3.2 * x) +
         0.5099 * std::exp(-0.9423 * x) +
         0.2802 * std::exp(-0.4029 * x) +
         0.02817 * std::exp(-0.2016 * x);
}

G4double G4DNAZBLFullElastic::ScreeningDerivative(G4double x)
{
  if (x < 0.) return 0.;
  return -3.2 * 0.1818 * std::exp(-3.2 * x) -
         0.9423 * 0.5099 * std::exp(-0.9423 * x) -
         0.4029 * 0.2802 * std::exp(-0.4029 * x) -
         0.2016 * 0.02817 * std::exp(-0.2016 * x);
}

G4double G4DNAZBLFullElastic::ScreeningLength(
    G4int projectileZ, G4int targetZ)
{
  if (projectileZ <= 0 || targetZ <= 0) return 0.;
  return 0.88534 * Bohr_radius /
      (std::pow(static_cast<G4double>(projectileZ), 0.23) +
       std::pow(static_cast<G4double>(targetZ), 0.23));
}

G4double G4DNAZBLFullElastic::CenterOfMassKineticEnergy(
    G4double projectileMass, G4double targetMass, G4double kineticEnergy)
{
  if (projectileMass <= 0. || targetMass <= 0. || kineticEnergy <= 0.) return 0.;
  const G4double massSum = projectileMass + targetMass;
  const G4double s = massSum * massSum + 2. * targetMass * kineticEnergy;
  return std::sqrt(s) - massSum;
}

G4double G4DNAZBLFullElastic::MaximumRecoilEnergy(
    G4double projectileMass, G4double targetMass, G4double kineticEnergy)
{
  if (projectileMass <= 0. || targetMass <= 0. || kineticEnergy <= 0.) return 0.;
  const G4double momentumSquared =
      kineticEnergy * (kineticEnergy + 2. * projectileMass);
  const G4double s = projectileMass * projectileMass +
      targetMass * targetMass +
      2. * targetMass * (projectileMass + kineticEnergy);
  return 2. * targetMass * momentumSquared / s;
}

G4double G4DNAZBLFullElastic::CosThetaCM(
    G4int projectileZ, G4double projectileMass,
    G4int targetZ, G4double targetMass,
    G4double kineticEnergy, G4double impactParameter)
{
  if (impactParameter <= 0.) return -1.;
  const G4double length = ScreeningLength(projectileZ, targetZ);
  const G4double cmEnergy = CenterOfMassKineticEnergy(
      projectileMass, targetMass, kineticEnergy);
  if (length <= 0. || cmEnergy <= 0.) return 1.;
  const G4double epsilon = cmEnergy /
      (projectileZ * targetZ * elm_coupling / length);
  const G4double beta = impactParameter / length;

  auto turning = [epsilon, beta](G4double x) {
    return x * x - x * Screening(x) / epsilon - beta * beta;
  };
  G4double lower = 0.;
  G4double upper = std::max(1., beta + 1.);
  while (turning(upper) < 0. && upper < 1.e12) upper *= 2.;
  if (turning(upper) < 0.) return 1.;
  for (G4int i = 0; i < kRootIterations; ++i) {
    const G4double middle = 0.5 * (lower + upper);
    if (turning(middle) < 0.) lower = middle;
    else upper = middle;
  }
  const G4double x0 = 0.5 * (lower + upper);
  const G4double denominator = 0.5 + beta * beta / (2. * x0 * x0) -
      ScreeningDerivative(x0) / (2. * epsilon);
  if (denominator <= 0.) return 1.;

  G4double alpha = (1. + 1. / std::sqrt(denominator)) / 30.;
  constexpr G4double abscissa[] = {
      0.98302349, 0.84652241, 0.53235309, 0.18347974};
  constexpr G4double weight[] = {
      0.03472124, 0.14769029, 0.23485003, 0.18602489};
  for (G4int i = 0; i < 4; ++i) {
    const G4double x = x0 / abscissa[i];
    const G4double radicand = 1. - Screening(x) / (x * epsilon) -
        beta * beta / (x * x);
    if (radicand <= 0.) return 1.;
    alpha += weight[i] / std::sqrt(radicand);
  }
  const G4double complementaryAngle = pi * beta * alpha / x0;
  return ClampCosine(-std::cos(complementaryAngle));
}

G4double G4DNAZBLFullElastic::PairCrossSection(
    G4int projectileZ, G4double projectileMass,
    G4int targetZ, G4double targetMass,
    G4double kineticEnergy, G4double minimumTransfer)
{
  const G4double maximumTransfer = MaximumRecoilEnergy(
      projectileMass, targetMass, kineticEnergy);
  if (minimumTransfer <= 0. || maximumTransfer <= minimumTransfer) return 0.;
  const G4double targetCosine = ClampCosine(
      1. - 2. * minimumTransfer / maximumTransfer);
  const G4double length = ScreeningLength(projectileZ, targetZ);
  G4double lower = 0.;
  G4double upper = length;
  while (CosThetaCM(projectileZ, projectileMass, targetZ, targetMass,
                    kineticEnergy, upper) < targetCosine &&
         upper < 1.e8 * length) {
    upper *= 2.;
  }
  if (upper >= 1.e8 * length) return 0.;
  for (G4int i = 0; i < kRootIterations; ++i) {
    const G4double middle = 0.5 * (lower + upper);
    if (CosThetaCM(projectileZ, projectileMass, targetZ, targetMass,
                   kineticEnergy, middle) < targetCosine) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  const G4double maximumImpact = 0.5 * (lower + upper);
  return pi * maximumImpact * maximumImpact;
}

G4DNAZBLFullElastic::PairTable
G4DNAZBLFullElastic::BuildPairTable(Target target) const
{
  PairTable table;
  if (target == Target::Hydrogen) {
    table.targetZ = 1;
    table.targetA = 1;
    table.targetMass = G4Proton::ProtonDefinition()->GetPDGMass();
  } else {
    table.targetZ = 8;
    table.targetA = 16;
    table.targetMass = G4NucleiProperties::GetNuclearMass(16, 8);
  }
  table.logEnergy.reserve(kEnergyPoints);
  table.logCrossSection.reserve(kEnergyPoints);
  const G4double logMin = std::log(fMinimumEnergy);
  const G4double logMax = std::log(fMaximumEnergy);
  for (G4int i = 0; i < kEnergyPoints; ++i) {
    const G4double fraction = static_cast<G4double>(i) / (kEnergyPoints - 1);
    const G4double logEnergy = logMin + fraction * (logMax - logMin);
    const G4double energy = std::exp(logEnergy);
    const G4double crossSection = PairCrossSection(
        fProjectileZ, fProjectileMass, table.targetZ, table.targetMass,
        energy, fMinimumTransfer);
    if (!(crossSection > 0.) || !std::isfinite(crossSection)) {
      ProcessError("zblProcess004", "Non-positive ZBL table cross section.");
    }
    table.logEnergy.push_back(logEnergy);
    table.logCrossSection.push_back(std::log(crossSection));
  }
  return table;
}

G4double G4DNAZBLFullElastic::InterpolatedPairCrossSection(
    const PairTable& table, G4double kineticEnergy) const
{
  if (kineticEnergy < fMinimumEnergy || kineticEnergy > fMaximumEnergy) return 0.;
  const G4double x = std::log(kineticEnergy);
  const auto upper = std::lower_bound(
      table.logEnergy.begin(), table.logEnergy.end(), x);
  if (upper == table.logEnergy.begin()) return std::exp(table.logCrossSection.front());
  if (upper == table.logEnergy.end()) return std::exp(table.logCrossSection.back());
  const std::size_t hi = static_cast<std::size_t>(upper - table.logEnergy.begin());
  const std::size_t lo = hi - 1;
  const G4double fraction = (x - table.logEnergy[lo]) /
      (table.logEnergy[hi] - table.logEnergy[lo]);
  return std::exp(table.logCrossSection[lo] + fraction *
      (table.logCrossSection[hi] - table.logCrossSection[lo]));
}

G4double G4DNAZBLFullElastic::PairNumberDensity(
    const G4Material* material, Target target)
{
  if (!material) return 0.;
  const G4int desiredZ = target == Target::Hydrogen ? 1 : 8;
  const auto* elements = material->GetElementVector();
  const auto* densities = material->GetVecNbOfAtomsPerVolume();
  for (std::size_t i = 0; i < material->GetNumberOfElements(); ++i) {
    if (static_cast<G4int>(std::lround((*elements)[i]->GetZ())) == desiredZ) {
      return densities[i];
    }
  }
  return 0.;
}

G4double G4DNAZBLFullElastic::WaterMoleculeNumberDensity(
    const G4Material* material)
{
  const G4double hydrogen = PairNumberDensity(material, Target::Hydrogen);
  const G4double oxygen = PairNumberDensity(material, Target::Oxygen);
  if (oxygen <= 0. || std::abs(hydrogen / oxygen - 2.) > 1.e-10) return 0.;
  return oxygen;
}

G4double G4DNAZBLFullElastic::MacroscopicCrossSection(
    const G4Material* material, G4double kineticEnergy) const
{
  if (WaterMoleculeNumberDensity(material) <= 0.) return 0.;
  const auto& hydrogen = fPairTables[0];
  const auto& oxygen = fPairTables[1];
  return PairNumberDensity(material, Target::Hydrogen) *
             InterpolatedPairCrossSection(hydrogen, kineticEnergy) +
         PairNumberDensity(material, Target::Oxygen) *
             InterpolatedPairCrossSection(oxygen, kineticEnergy);
}

G4double G4DNAZBLFullElastic::GetCrossSection(
    G4double kineticEnergy, const G4MaterialCutsCouple* couple)
{
  return couple ? MacroscopicCrossSection(couple->GetMaterial(), kineticEnergy) : 0.;
}

G4bool G4DNAZBLFullElastic::MatchesProjectile(
    const G4ParticleDefinition* particle) const
{
  return AtomicNumber(particle) == fProjectileZ &&
         MassNumber(particle) == fProjectileA;
}

G4bool G4DNAZBLFullElastic::IsApplicable(
    const G4ParticleDefinition& particle)
{
  return particle.GetParticleType() == "nucleus" ||
         particle == *G4Proton::ProtonDefinition();
}

G4double G4DNAZBLFullElastic::GetMeanFreePath(
    const G4Track& track, G4double, G4ForceCondition* condition)
{
  *condition = NotForced;
  if (!MatchesProjectile(track.GetParticleDefinition())) return DBL_MAX;
  const G4double rate = MacroscopicCrossSection(
      track.GetMaterial(), track.GetKineticEnergy());
  return rate > 0. ? 1. / rate : DBL_MAX;
}

G4VParticleChange* G4DNAZBLFullElastic::PostStepDoIt(
    const G4Track& track, const G4Step& step)
{
  fParticleChange.Initialize(track);
  if (!MatchesProjectile(track.GetParticleDefinition())) {
    return G4VDiscreteProcess::PostStepDoIt(track, step);
  }
  const G4double energy = track.GetKineticEnergy();
  if (energy < fMinimumEnergy || energy > fMaximumEnergy) {
    return G4VDiscreteProcess::PostStepDoIt(track, step);
  }

  const G4Material* material = track.GetMaterial();
  const G4double rateH = PairNumberDensity(material, Target::Hydrogen) *
      InterpolatedPairCrossSection(fPairTables[0], energy);
  const G4double rateO = PairNumberDensity(material, Target::Oxygen) *
      InterpolatedPairCrossSection(fPairTables[1], energy);
  const G4double total = rateH + rateO;
  if (total <= 0.) return G4VDiscreteProcess::PostStepDoIt(track, step);

  const PairTable& target =
      G4UniformRand() * total < rateH ? fPairTables[0] : fPairTables[1];
  const G4double crossSection = InterpolatedPairCrossSection(target, energy);
  const G4double maximumImpact = std::sqrt(crossSection / pi);
  const G4double impact = maximumImpact * std::sqrt(G4UniformRand());
  const G4double cosine = CosThetaCM(
      fProjectileZ, fProjectileMass, target.targetZ, target.targetMass,
      energy, impact);
  Scatter(track, target, cosine);
  return G4VDiscreteProcess::PostStepDoIt(track, step);
}

void G4DNAZBLFullElastic::Scatter(
    const G4Track& track, const PairTable& target, G4double cosThetaCM)
{
  const G4double kineticEnergy = track.GetKineticEnergy();
  const G4double momentum = std::sqrt(
      kineticEnergy * (kineticEnergy + 2. * fProjectileMass));
  G4LorentzVector total(0., 0., momentum,
                        fProjectileMass + kineticEnergy + target.targetMass);
  G4LorentzVector scattered(0., 0., momentum,
                            fProjectileMass + kineticEnergy);
  const G4ThreeVector boost = total.boostVector();
  scattered.boost(-boost);
  const G4double momentumCM = scattered.vect().mag();
  const G4double sine = std::sqrt(std::max(
      0., (1. - cosThetaCM) * (1. + cosThetaCM)));
  const G4double azimuth = twopi * G4UniformRand();
  scattered.setVect(G4ThreeVector(
      momentumCM * sine * std::cos(azimuth),
      momentumCM * sine * std::sin(azimuth),
      momentumCM * cosThetaCM));
  scattered.boost(boost);
  G4LorentzVector recoil = total - scattered;

  G4ThreeVector projectileDirection = scattered.vect().unit();
  G4ThreeVector recoilDirection = recoil.vect().unit();
  projectileDirection.rotateUz(track.GetMomentumDirection());
  recoilDirection.rotateUz(track.GetMomentumDirection());
  const G4double finalEnergy = std::max(0., scattered.e() - fProjectileMass);
  const G4double recoilEnergy = std::max(0., recoil.e() - target.targetMass);
  fParticleChange.ProposeEnergy(finalEnergy);
  fParticleChange.ProposeMomentumDirection(projectileDirection);

  if (recoilEnergy >= fRecoilThreshold && recoilDirection.mag2() > 0.) {
    G4ParticleDefinition* recoilDefinition =
        target.targetZ == 1
            ? G4Proton::ProtonDefinition()
            : G4IonTable::GetIonTable()->GetIon(
                  target.targetZ, target.targetA, 0.);
    if (!recoilDefinition) {
      ProcessError("zblProcess005", "Could not construct ZBL recoil ion.");
    }
    fParticleChange.AddSecondary(new G4DynamicParticle(
        recoilDefinition, recoilDirection, recoilEnergy));
  } else if (recoilEnergy > 0.) {
    fParticleChange.ProposeLocalEnergyDeposit(recoilEnergy);
    fParticleChange.ProposeNonIonizingEnergyDeposit(recoilEnergy);
  }
}
