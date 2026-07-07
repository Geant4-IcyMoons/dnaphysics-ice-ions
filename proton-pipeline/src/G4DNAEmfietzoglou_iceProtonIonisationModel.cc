//
// Proton-only ionisation model for generated ice DCS/TCS tables.
//

#include "G4DNAEmfietzoglou_iceProtonIonisationModel.hh"

#include "G4DNAChemistryManager.hh"
#include "G4DNAMolecularMaterial.hh"
#include "G4DynamicParticle.hh"
#include "G4Electron.hh"
#include "G4Material.hh"
#include "G4Proton.hh"
#include "G4RandomDirection.hh"
#include "G4SystemOfUnits.hh"
#include "ModelDataRegistry.hh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace {
thread_local G4int g_lastIceProtonIonShell = -1;
thread_local G4double g_lastIceProtonIonSigma_cm2 = -1.0;

std::string ToLower(std::string value)
{
  for (auto& ch : value) ch = static_cast<char>(std::tolower(ch));
  return value;
}

std::string NormalizeIcePhase(const char* raw)
{
  if (!raw) return "hexagonal";
  const std::string phase = ToLower(raw);
  if (phase == "ice_hex" || phase == "hexagonal") return "hexagonal";
  if (phase == "ice_am" || phase == "amorphous") return "amorphous";
  return "hexagonal";
}

G4bool ReadEnvFlag(const char* key, G4bool defaultValue)
{
  const char* raw = std::getenv(key);
  if (!raw || !*raw) return defaultValue;
  const std::string value = ToLower(raw);
  if (value == "1" || value == "true" || value == "on" || value == "yes") {
    return true;
  }
  if (value == "0" || value == "false" || value == "off" || value == "no") {
    return false;
  }
  G4cout << "G4DNAEmfietzoglou_iceProtonIonisationModel: invalid "
         << key << "='" << raw << "', using default "
         << (defaultValue ? "on" : "off") << G4endl;
  return defaultValue;
}

G4bool UseBarkasDcsTables()
{
  const G4bool legacy = ReadEnvFlag("DNA_ICE_PROTON_BARKAS_DCS", false);
  return ReadEnvFlag("DNA_PROTON_BARKAS_DCS", legacy);
}

G4bool TrackSecondaryElectrons()
{
  return ReadEnvFlag("DNA_PROTON_TRACK_SECONDARY_ELECTRONS", false);
}
}  // namespace

G4DNAEmfietzoglou_iceProtonIonisationModel::
G4DNAEmfietzoglou_iceProtonIonisationModel(const G4ParticleDefinition*,
                                           const G4String& nam)
    : G4VEmModel(nam)
{
  SetLowEnergyLimit(0.1 * MeV);
  SetHighEnergyLimit(100. * MeV);
  SetDeexcitationFlag(true);
}

G4int G4DNAEmfietzoglou_iceProtonIonisationModel::GetLastShellIndex()
{
  return g_lastIceProtonIonShell;
}

void G4DNAEmfietzoglou_iceProtonIonisationModel::ClearLastShellIndex()
{
  g_lastIceProtonIonShell = -1;
  g_lastIceProtonIonSigma_cm2 = -1.0;
}

G4double G4DNAEmfietzoglou_iceProtonIonisationModel::GetLastPartialSigma_cm2()
{
  return g_lastIceProtonIonSigma_cm2;
}

void G4DNAEmfietzoglou_iceProtonIonisationModel::ClearLastPartialSigma_cm2()
{
  g_lastIceProtonIonSigma_cm2 = -1.0;
}

void G4DNAEmfietzoglou_iceProtonIonisationModel::Initialise(
    const G4ParticleDefinition* particle,
    const G4DataVector&)
{
  if (particle != G4Proton::ProtonDefinition()) {
    G4Exception("G4DNAEmfietzoglou_iceProtonIonisationModel::Initialise",
                "protonion001", FatalException,
                "Model is only applicable to protons.");
  }

  const std::string phase = NormalizeIcePhase(std::getenv("DNA_PHYSICS"));
  const std::string correction = UseBarkasDcsTables() ? "_barkas_dcs" : "";
  const std::string total =
      "sigma_ionisation_proton_" + phase + "_ice" + correction +
      "_emfietzoglou_kyriakou";
  const std::string diff =
      "sigmadiff_ionisation_proton_" + phase + "_ice" + correction +
      "_emfietzoglou_kyriakou.dat";

  fTable.Load(total, diff);
  ModelDataRegistry::Instance().Record(
      std::string("model_ref:") + GetName(),
      ModelDataRegistry::NormalizeDatBasename(total));
  ModelDataRegistry::Instance().Record(
      std::string("model_ref_diff:") + GetName(),
      ModelDataRegistry::NormalizeDatBasename(diff));
  ModelDataRegistry::Instance().Record(
      std::string("model_mode:") + GetName(),
      UseBarkasDcsTables() ? "born_plus_barkas_dcs" : "bare_born");

  fpMolWaterDensity =
      G4DNAMolecularMaterial::Instance()->GetNumMolPerVolTableFor(
          G4Material::GetMaterial("G4_WATER"));
  fParticleChangeForGamma = GetParticleChangeForGamma();
  fInitialised = true;

  G4cout << "Initialized proton ice ionisation model with "
         << fTable.TotalPath() << " and " << fTable.DiffPath() << G4endl;
}

G4double G4DNAEmfietzoglou_iceProtonIonisationModel::CrossSectionPerVolume(
    const G4Material* material,
    const G4ParticleDefinition* particleDefinition,
    G4double ekin,
    G4double,
    G4double)
{
  if (particleDefinition != G4Proton::ProtonDefinition()) return 0.;
  if (!fInitialised || ekin < LowEnergyLimit() || ekin > HighEnergyLimit()) {
    return 0.;
  }
  const G4double waterDensity = (*fpMolWaterDensity)[material->GetIndex()];
  return fTable.TotalCrossSection(ekin) * waterDensity;
}

void G4DNAEmfietzoglou_iceProtonIonisationModel::SampleSecondaries(
    std::vector<G4DynamicParticle*>* fvect,
    const G4MaterialCutsCouple*,
    const G4DynamicParticle* particle,
    G4double,
    G4double)
{
  const G4double kineticEnergy = particle->GetKineticEnergy();
  if (!fInitialised ||
      kineticEnergy < LowEnergyLimit() ||
      kineticEnergy > HighEnergyLimit()) {
    return;
  }

  const G4int shell = fTable.SelectComponent(kineticEnergy);
  g_lastIceProtonIonShell = shell;
  g_lastIceProtonIonSigma_cm2 =
      fTable.TotalCrossSection(kineticEnergy, shell) / (cm * cm);

  G4double transfer = fTable.SampleTransferEnergy(kineticEnergy, shell);
  if (transfer <= 0.) return;
  transfer = std::min(transfer, 0.999 * kineticEnergy);

  const G4double bindingEnergy = fIonisationStructure.IonisationEnergy(shell);
  G4double secondaryKinetic = transfer - bindingEnergy;
  if (secondaryKinetic < 0.) secondaryKinetic = 0.;

  const G4bool trackSecondaryElectrons = TrackSecondaryElectrons();
  if (trackSecondaryElectrons && secondaryKinetic > 0.) {
    auto* secondary = new G4DynamicParticle(G4Electron::Electron(),
                                            G4RandomDirection(),
                                            secondaryKinetic);
    fvect->push_back(secondary);
  }

  fParticleChangeForGamma->ProposeMomentumDirection(
      particle->GetMomentumDirection());
  fParticleChangeForGamma->SetProposedKineticEnergy(
      fStationary ? kineticEnergy : kineticEnergy - transfer);
  fParticleChangeForGamma->ProposeLocalEnergyDeposit(
      trackSecondaryElectrons ? transfer - secondaryKinetic : transfer);

  const G4Track* incomingTrack = fParticleChangeForGamma->GetCurrentTrack();
  G4DNAChemistryManager::Instance()->CreateWaterMolecule(eIonizedMolecule,
                                                         shell,
                                                         incomingTrack);
}
