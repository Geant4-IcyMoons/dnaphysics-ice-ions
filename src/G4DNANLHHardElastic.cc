#include "G4DNANLHHardElastic.hh"

#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4Exception.hh"
#include "G4IonTable.hh"
#include "G4LowEnergyEmProcessSubType.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ParticleDefinition.hh"
#include "G4PhysicalConstants.hh"
#include "G4Proton.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "ModelDataRegistry.hh"
#include "Randomize.hh"

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
using Target = G4DNANLHHardElasticTable::Target;

G4bool FileExists(const std::string& path)
{
  std::ifstream input(path);
  return input.good();
}

G4bool ReadEnvFlag(const char* key)
{
  const char* raw = std::getenv(key);
  if (!raw || !*raw) return false;
  const std::string value(raw);
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

[[noreturn]] void ProcessError(const char* code, const std::string& message)
{
  G4ExceptionDescription description;
  description << message;
  G4Exception("G4DNANLHHardElastic", code, FatalException, description);
  throw std::runtime_error(message);
}
}  // namespace

G4DNANLHHardElastic::G4DNANLHHardElastic(const G4String& processName,
                                         G4ProcessType type)
  : G4VDiscreteProcess(processName, type),
    fTable(SharedTable(ResolveDataPath()))
{
  SetProcessSubType(fLowEnergyElastic);
  pParticleChange = &fParticleChange;
  if (fTable->ReleaseStatus() != "accepted" &&
      !ReadEnvFlag("DNA_NLH_ALLOW_VALIDATION_PENDING")) {
    ProcessError(
        "nlhProcess005",
        "The carbon NLH continuum model is atomistic_validation_pending. "
        "Set DNA_NLH_ALLOW_VALIDATION_PENDING=1 only for engineering and "
        "validation runs; production release requires the phase/orientation "
        "decision gate to pass.");
  }
  ModelDataRegistry::Instance().Record(
      std::string("model_ref:") + processName,
      ModelDataRegistry::NormalizeDatBasename(fTable->Path()));
  ModelDataRegistry::Instance().Record(
      std::string("model_source_sha256:") + processName,
      fTable->SourceCsvSha256());
  ModelDataRegistry::Instance().Record(
      std::string("model_mode:") + processName,
      "threshold_defined_nlh_hard_elastic");
  ModelDataRegistry::Instance().Record(
      std::string("model_release_status:") + processName,
      fTable->ReleaseStatus());
  G4cout << "Initialized carbon-12 threshold-defined NLH hard elastic process "
         << "from " << fTable->Path() << " [1 keV-100 MeV total energy, "
         << "V_min=" << fTable->MinimumTurningPotential() / eV << " eV]."
         << G4endl;
}

std::string G4DNANLHHardElastic::ResolveDataPath()
{
  constexpr const char* filename = "nlh_hard_elastic_C.dat";
  if (const char* explicitPath = std::getenv("DNA_NLH_HARD_ELASTIC_DATA")) {
    if (*explicitPath && FileExists(explicitPath)) return explicitPath;
    ProcessError("nlhProcess001",
                 "DNA_NLH_HARD_ELASTIC_DATA does not name a readable file.");
  }
  if (const char* directory = std::getenv("DNA_NLH_TABLE_DIR")) {
    if (*directory) {
      const std::string candidate = std::string(directory) + "/" + filename;
      if (FileExists(candidate)) return candidate;
      ProcessError("nlhProcess002",
                   "DNA_NLH_TABLE_DIR does not contain nlh_hard_elastic_C.dat.");
    }
  }
  const std::string candidates[] = {
      std::string("nlh_data/") + filename,
      std::string("python_scripts/physics_ice/nep_mbpol/collision_kernels/geant4/") +
          filename,
      std::string("../python_scripts/physics_ice/nep_mbpol/collision_kernels/geant4/") +
          filename,
#ifdef DNA_NLH_SOURCE_DATA_FILE
      DNA_NLH_SOURCE_DATA_FILE,
#endif
  };
  for (const auto& candidate : candidates) {
    if (!candidate.empty() && FileExists(candidate)) return candidate;
  }
  ProcessError("nlhProcess003",
               "Could not locate nlh_hard_elastic_C.dat. Set "
               "DNA_NLH_HARD_ELASTIC_DATA or DNA_NLH_TABLE_DIR.");
}

std::shared_ptr<const G4DNANLHHardElasticTable>
G4DNANLHHardElastic::SharedTable(const std::string& path)
{
  static std::mutex mutex;
  static std::map<std::string,
                  std::weak_ptr<const G4DNANLHHardElasticTable>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto existing = cache[path].lock()) return existing;
  auto table = std::make_shared<const G4DNANLHHardElasticTable>(path);
  cache[path] = table;
  return table;
}

G4bool G4DNANLHHardElastic::IsCarbon12(
    const G4ParticleDefinition* particle)
{
  return particle && particle->GetAtomicNumber() == 6 &&
         particle->GetAtomicMass() == 12;
}

G4bool G4DNANLHHardElastic::IsApplicable(
    const G4ParticleDefinition& particle)
{
  // C-12 and the other ions share GenericIon's process manager. The runtime
  // particle check below makes the process inactive for every other ion.
  return particle.GetParticleType() == "nucleus" || IsCarbon12(&particle);
}

G4double G4DNANLHHardElastic::WaterMoleculeNumberDensity(
    const G4Material* material)
{
  if (!material || material->GetNumberOfElements() != 2) return 0.;
  const auto* elements = material->GetElementVector();
  const auto* atomDensities = material->GetVecNbOfAtomsPerVolume();
  G4double hydrogenDensity = 0.;
  G4double oxygenDensity = 0.;
  for (std::size_t index = 0; index < material->GetNumberOfElements(); ++index) {
    const G4int z = static_cast<G4int>(std::lround((*elements)[index]->GetZ()));
    if (z == 1) {
      hydrogenDensity = atomDensities[index];
    } else if (z == 8) {
      oxygenDensity = atomDensities[index];
    } else {
      return 0.;
    }
  }
  if (oxygenDensity <= 0. ||
      std::abs(hydrogenDensity / oxygenDensity - 2.) > 1.e-10) {
    return 0.;
  }
  return oxygenDensity;
}

G4double G4DNANLHHardElastic::MacroscopicCrossSection(
    const G4Material* material, G4double kineticEnergy) const
{
  if (kineticEnergy < 1. * keV || kineticEnergy > 100. * MeV) return 0.;
  const G4double waterDensity = WaterMoleculeNumberDensity(material);
  if (waterDensity == 0.) return 0.;
  const G4double sigmaH =
      fTable->HardCrossSection(Target::Hydrogen, kineticEnergy);
  const G4double sigmaO =
      fTable->HardCrossSection(Target::Oxygen, kineticEnergy);
  // Pure water ice: Sigma_P^hard = n_H2O(2 sigma_PH^hard + sigma_PO^hard).
  return waterDensity * (2. * sigmaH + sigmaO);
}

G4double G4DNANLHHardElastic::GetCrossSection(
    G4double kineticEnergy, const G4MaterialCutsCouple* couple)
{
  return couple ? MacroscopicCrossSection(couple->GetMaterial(), kineticEnergy) : 0.;
}

G4double G4DNANLHHardElastic::GetMeanFreePath(
    const G4Track& track, G4double, G4ForceCondition* condition)
{
  *condition = NotForced;
  if (!IsCarbon12(track.GetParticleDefinition())) return DBL_MAX;
  const G4double crossSection =
      MacroscopicCrossSection(track.GetMaterial(), track.GetKineticEnergy());
  return crossSection > 0. ? 1. / crossSection : DBL_MAX;
}

G4VParticleChange* G4DNANLHHardElastic::PostStepDoIt(
    const G4Track& track, const G4Step& step)
{
  fParticleChange.Initialize(track);
  if (!IsCarbon12(track.GetParticleDefinition())) {
    return G4VDiscreteProcess::PostStepDoIt(track, step);
  }
  const G4double kineticEnergy = track.GetKineticEnergy();
  if (kineticEnergy < 1. * keV || kineticEnergy > 100. * MeV) {
    return G4VDiscreteProcess::PostStepDoIt(track, step);
  }

  const G4double sigmaH =
      fTable->HardCrossSection(Target::Hydrogen, kineticEnergy);
  const G4double sigmaO =
      fTable->HardCrossSection(Target::Oxygen, kineticEnergy);
  const G4double total = 2. * sigmaH + sigmaO;
  if (total <= 0.) return G4VDiscreteProcess::PostStepDoIt(track, step);

  const Target target =
      G4UniformRand() * total < 2. * sigmaH ? Target::Hydrogen : Target::Oxygen;
  const G4double thetaCM =
      fTable->ThetaCM(target, kineticEnergy, G4UniformRand());
  const auto outcome = fTable->Outcome(target, kineticEnergy, thetaCM);
  const G4double azimuth = CLHEP::twopi * G4UniformRand();
  const G4double cosine = std::cos(azimuth);
  const G4double sine = std::sin(azimuth);

  G4ThreeVector projectileMomentum(
      outcome.projectileTransverseMomentum * cosine,
      outcome.projectileTransverseMomentum * sine,
      outcome.projectileLongitudinalMomentum);
  G4ThreeVector recoilMomentum(outcome.recoilTransverseMomentum * cosine,
                               outcome.recoilTransverseMomentum * sine,
                               outcome.recoilLongitudinalMomentum);
  const G4ThreeVector incomingDirection = track.GetMomentumDirection();
  projectileMomentum.rotateUz(incomingDirection);
  recoilMomentum.rotateUz(incomingDirection);

  fParticleChange.ProposeEnergy(outcome.projectileEnergy);
  if (projectileMomentum.mag2() > 0.) {
    fParticleChange.ProposeMomentumDirection(projectileMomentum.unit());
  }
  if (outcome.recoilEnergy > 0. && recoilMomentum.mag2() > 0.) {
    G4ParticleDefinition* recoilDefinition =
        target == Target::Hydrogen
            ? G4Proton::ProtonDefinition()
            : G4IonTable::GetIonTable()->GetIon(8, 16, 0.);
    if (!recoilDefinition) {
      ProcessError("nlhProcess004", "Could not construct the target recoil ion.");
    }
    fParticleChange.AddSecondary(new G4DynamicParticle(
        recoilDefinition, recoilMomentum.unit(), outcome.recoilEnergy));
  }
  return G4VDiscreteProcess::PostStepDoIt(track, step);
}
